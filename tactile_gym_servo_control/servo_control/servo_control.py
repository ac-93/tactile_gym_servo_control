"""
python servo_control.py -t surface_3d
python servo_control.py -t edge_2d
python servo_control.py -t edge_3d
python servo_control.py -t edge_5d
python servo_control.py -t surface_3d edge_2d edge_3d edge_5d
"""

import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import imageio

from tactile_gym.utils.general_utils import load_json_obj

from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.learning.learning_utils import decode_pose
from tactile_gym_servo_control.learning.learning_utils import POSE_LABEL_NAMES, POS_LABEL_NAMES, ROT_LABEL_NAMES
from tactile_gym_servo_control.cri_wrapper.cri_embodiment import quat2euler, euler2quat, transform, inv_transform
from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.learning.networks import create_model
from tactile_gym_servo_control.utils.image_transforms import process_image

from tactile_gym_servo_control.servo_control.setup_servo_control import setup_surface_3d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_2d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_3d_servo_control
from tactile_gym_servo_control.servo_control.setup_servo_control import setup_edge_5d_servo_control

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

stimuli_path = os.path.join(os.path.dirname(__file__), "../stimuli")
model_path = os.path.join(os.path.dirname(__file__), '../learned_models')


def load_embodiment_and_env(init_ref_pose, stim_name="square"):

    assert stim_name in ["square", "foil",
                         "clover", "circle",
                         "saddle"], "Invalid Stimulus"

    tactip_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    show_gui = True
    show_tactile = True

    # setup stimulus
    stimulus_pos = [0.6, 0.0, 0.0125]
    stimulus_rpy = [0, 0, 0]
    stim_path = os.path.join(
        stimuli_path,
        stim_name,
        stim_name + ".urdf"
    )

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.6, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    # setup robot data collection env
    embodiment, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stimulus_pos,
        stimulus_rpy,
        workframe_pos,
        workframe_rpy,
        show_gui,
        show_tactile,
    )

    # add user controllable ref pose to GUI
    ref_pose_ids = []
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('x', -2.0, 2.0, init_ref_pose[POSE_LABEL_NAMES.index('x')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('y', -2.0, 2.0, init_ref_pose[POSE_LABEL_NAMES.index('y')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('z', 2.0, 5.0, init_ref_pose[POSE_LABEL_NAMES.index('z')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Rx', -15.0, 15.0, init_ref_pose[POSE_LABEL_NAMES.index('Rx')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Ry', -15.0, 15.0, init_ref_pose[POSE_LABEL_NAMES.index('Ry')]))
    ref_pose_ids.append(embodiment._pb.addUserDebugParameter('Rz', -180.0, 180.0, init_ref_pose[POSE_LABEL_NAMES.index('Rz')]))

    return embodiment, ref_pose_ids


def get_prediction(
    trained_model,
    tactile_image,
    image_processing_params,
    label_names,
    pose_limits,
    device='cpu'
):

    # process image (as done for training)
    processed_image = process_image(
        tactile_image,
        gray=False,
        bbox=image_processing_params['bbox'],
        dims=image_processing_params['dims'],
        stdiz=image_processing_params['stdiz'],
        normlz=image_processing_params['normlz'],
        thresh=image_processing_params['thresh'],
    )

    # channel first for pytorch
    processed_image = np.rollaxis(processed_image, 2, 0)

    # add batch dim
    processed_image = processed_image[np.newaxis, ...]

    # perform inference with the trained model
    model_input = Variable(torch.from_numpy(processed_image)).float().to(device)
    raw_predictions = trained_model(model_input)

    # decode the prediction
    predictions_dict = decode_pose(raw_predictions, label_names, pose_limits)

    print("")
    print("Predictions")
    predictions_arr = np.zeros(6)
    for label_name in label_names:
        if label_name in POS_LABEL_NAMES:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() * 0.001
        if label_name in ROT_LABEL_NAMES:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy() * np.pi / 180

        print(label_name, predicted_val)
        predictions_arr[POSE_LABEL_NAMES.index(label_name)] = predicted_val

    return predictions_arr


def compute_target_pose(pred_pose, ref_pose, p_gains, tcp_pose):
    """
    Compute workframe pose for maintaining reference pose from predicted pose
    """
    # calculate deltas between reference and predicted pose
    ref_pose_q = euler2quat(ref_pose)
    pred_pose_q = euler2quat(pred_pose)
    pose_deltas = quat2euler(transform(ref_pose_q, pred_pose_q))

    # control signal in tcp frame (e.g frame of training data labels)
    control_signal = p_gains * pose_deltas

    # transform control signal from feature frame to TCP frame
    control_signal_q = euler2quat(control_signal)
    tcp_pose_q = euler2quat(tcp_pose)
    target_pose = quat2euler(inv_transform(control_signal_q, tcp_pose_q))

    return target_pose


def run_servo_control(
            embodiment,
            trained_model,
            image_processing_params,
            p_gains=np.zeros(6),
            label_names=[],
            pose_limits=[],
            ref_pose_ids=[],
            ep_len=400,
            quick_mode=True,
            record_vid=False,
        ):

    if record_vid:
        render_frames = []

    for i in range(ep_len):

        # get current tactile observation
        tactile_image = embodiment.get_tactile_observation()

        # get current TCP pose
        tcp_pose = embodiment.get_tcp_pose()

        ref_pose = []
        for j, label_name in enumerate(POSE_LABEL_NAMES):
            ref = embodiment._pb.readUserDebugParameter(ref_pose_ids[j])
            if ref is not None:
                if label_name in POS_LABEL_NAMES:
                    ref *= 1e-3
                if label_name in ROT_LABEL_NAMES:
                    ref *= np.pi/180
                if p_gains[j] == 0.0:
                    ref = 0
            ref_pose.append(ref)

        # predict pose from observation
        pred_pose = get_prediction(
            trained_model,
            tactile_image,
            image_processing_params,
            label_names,
            pose_limits,
            device
        )

        # compute new pose to move to
        target_pose = compute_target_pose(
            pred_pose, ref_pose, p_gains, tcp_pose
        )

        # move to new pose
        embodiment.move_linear(target_pose[:3], target_pose[3:], quick_mode=quick_mode)

        # draw TCP frame
        embodiment.arm.draw_TCP(lifetime=10.0)

        # render frames
        if record_vid:
            render_img = embodiment.render()
            render_frames.append(render_img)

        q_key = ord("q")
        keys = embodiment._pb.getKeyboardEvents()
        if q_key in keys and keys[q_key] & embodiment._pb.KEY_WAS_TRIGGERED:
            exit()

    embodiment.close()

    if record_vid:
        imageio.mimwrite(
            os.path.join(os.path.dirname(__file__), "../../example_videos", "render.mp4"),
            np.stack(render_frames),
            fps=24
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['surface_3d', 'edge_2d', 'edge_3d', 'edge_5d'].",
        default=['surface_3d']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cpu'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    device = args.device

    for task in tasks:

        # get the correct setup for the current task
        if task == "surface_3d":
            setup_servo_control = setup_surface_3d_servo_control
        if task == "edge_2d":
            setup_servo_control = setup_edge_2d_servo_control
        elif task == "edge_3d":
            setup_servo_control = setup_edge_3d_servo_control
        elif task == "edge_5d":
            setup_servo_control = setup_edge_5d_servo_control

        # set save dir
        save_dir_name = os.path.join(
            model_path,
            task,
            'tap',
        )

        # get limits and labels used during training
        out_dim, label_names = import_task(task)
        pose_limits_dict = load_json_obj(os.path.join(save_dir_name, 'pose_limits'))
        pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]

        # setup the task
        move_init_pose, stim_names, ep_len, init_ref_pose, p_gains = setup_servo_control()

        # perform the servo control
        for stim_name in stim_names:

            embodiment, ref_pose_ids = load_embodiment_and_env(init_ref_pose, stim_name)

            move_init_pose(embodiment, stim_name)

            # load params
            model_params = load_json_obj(os.path.join(save_dir_name, 'model_params'))
            learning_params = load_json_obj(os.path.join(save_dir_name, 'learning_params'))
            image_processing_params = load_json_obj(os.path.join(save_dir_name, 'image_processing_params'))

            trained_model = create_model(
                image_processing_params['dims'],
                out_dim,
                model_params,
                saved_model_dir=save_dir_name,
                device=device
            )
            trained_model.eval()

            run_servo_control(
                embodiment,
                trained_model,
                image_processing_params,
                p_gains=p_gains,
                label_names=label_names,
                pose_limits=pose_limits,
                ref_pose_ids=ref_pose_ids,
                ep_len=ep_len,
                quick_mode=False
            )
