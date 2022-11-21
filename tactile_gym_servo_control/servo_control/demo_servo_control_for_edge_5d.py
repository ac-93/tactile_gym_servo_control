import os
import numpy as np

from tactile_gym.utils.general_utils import load_json_obj
from tactile_gym_servo_control.learning.learning_utils import import_task
from tactile_gym_servo_control.servo_control.servo_control import load_robot_and_env
from tactile_gym_servo_control.servo_control.servo_control import load_nn_model
from tactile_gym_servo_control.servo_control.servo_control import run_servo_control

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def move_init_pose(robot, stim_name):

    if stim_name in ["saddle"]:
        init_pos = [-0.07, 0.0, -0.02]
        init_rpy = [0.0, 0.0, -np.pi/2]

    robot.move_linear(init_pos, init_rpy, quick_mode=False)


if __name__ == '__main__':
    task = 'edge_5d'
    data_collection = 'tap'

    # set save dir
    save_dir_name = os.path.join(
        os.path.dirname(__file__),
        '../learning/saved_models',
        task,
        data_collection,
    )

    # get limits and labels used during training
    out_dim, label_names = import_task(task)
    pose_limits_dict = load_json_obj(os.path.join(save_dir_name, 'pose_limits'))
    pose_limits = [pose_limits_dict['pose_llims'], pose_limits_dict['pose_ulims']]

    # set reference pose and gains
    ref_pose = np.array([
            0.001, 0.0, 0.004,  # meters
            np.deg2rad(0.0), np.deg2rad(0.0), np.deg2rad(0.0)  # rad
    ])
    p_gains = np.array([1.0, 1.0, 0.5, 0.25, 0.25, 0.25])

    stim_name = "saddle"

    robot = load_robot_and_env(stim_name)

    move_init_pose(robot, stim_name)

    trained_model, learning_params, image_processing_params = load_nn_model(
        save_dir_name, out_dim=out_dim
    )

    run_servo_control(
        robot,
        trained_model,
        image_processing_params,
        ref_pose=ref_pose,
        p_gains=p_gains,
        label_names=label_names,
        pose_limits=pose_limits,
        ep_len=500,
        quick_mode=False
    )
