import os
import numpy as np
import cv2

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_surface_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_5d_data_collection

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def collect_data(
    target_df,
    image_dir,
    stim_path,
    stimulus_pos,
    stimulus_rpy,
    workframe_pos,
    workframe_rpy,
    tactip_params,
    show_gui=True,
    show_tactile=True,
    quick_mode=False,
):

    # setup robot data collection env
    robot, _ = setup_pybullet_env(
        stim_path,
        tactip_params,
        stimulus_pos,
        stimulus_rpy,
        workframe_pos,
        workframe_rpy,
        show_gui,
        show_tactile,
    )

    hover_dist = 0.0075

    # move to work frame
    robot.move_linear([0, 0, 0], [0, 0, 0], quick_mode)

    # ==== data collection loop ====
    for index, row in target_df.iterrows():
        i_obj = int(row.loc["obj_id"])
        i_pose = int(row.loc["pose_id"])
        pose = row.loc["pose_1":"pose_6"].values.astype(np.float32)
        move = row.loc["move_1":"move_6"].values.astype(np.float32)
        obj_pose = row.loc["obj_pose"]
        sensor_image = row.loc["sensor_image"]

        # define the new pos and rpy
        # careful around offset for camera orientation
        obj_pose_array = np.array([float(i) for i in obj_pose])
        pose_array = np.array([float(i) for i in pose])
        move_array = np.array([float(i) for i in move])

        # combine relative pose and object pose
        new_pose = obj_pose_array + pose_array

        # convert to pybullet form
        final_pos = new_pose[:3] * 0.001  # to mm
        final_rpy = new_pose[3:] * np.pi / 180  # to rad
        move_pos = move_array[:3] * 0.001  # to mm
        move_rpy = move_array[3:] * np.pi / 180  # to rad

        with np.printoptions(precision=2, suppress=True):
            print(f"Collecting data for object {i_obj}, pose {i_pose}: ...")

        # move to slightly above new pose (avoid changing pose in contact with object)
        robot.move_linear(
            final_pos - move_pos - [0, 0, hover_dist],
            final_rpy - move_rpy,
            quick_mode
        )

        # move down to offset position
        robot.move_linear(
            final_pos - move_pos,
            final_rpy - move_rpy,
            quick_mode
        )
        # move to target positon inducing shear effects
        robot.move_linear(
            final_pos,
            final_rpy,
            quick_mode
        )

        # process frames and save
        img = robot.process_sensor()

        # raise tip before next move
        robot.move_linear(
            final_pos - [0, 0, hover_dist],
            final_rpy,
            quick_mode
        )

        # save tap img
        image_outfile = os.path.join(image_dir, sensor_image)
        cv2.imwrite(image_outfile, img)


if __name__ == "__main__":

    # tasks = ["surface_3d"]
    # tasks = ["edge_2d"]
    # tasks = ["edge_3d"]
    # tasks = ["edge_5d"]
    tasks = ["surface_3d", "edge_2d", "edge_3d", "edge_5d"]

    for task in tasks:

        if task == "surface_3d":
            setup_data_collection = setup_surface_3d_data_collection
        if task == "edge_2d":
            setup_data_collection = setup_edge_2d_data_collection
        elif task == "edge_3d":
            setup_data_collection = setup_edge_3d_data_collection
        elif task == "edge_5d":
            setup_data_collection = setup_edge_5d_data_collection

        tactip_params = {
            "name": "tactip",
            "type": "standard",
            "core": "no_core",
            "dynamics": {},
            "image_size": [256, 256],
            "turn_off_border": False,
        }

        show_gui = True
        show_tactile = True
        quick_mode = False
        num_samples = 10
        apply_shear = False
        collect_dir_name = "example_data"

        # setup stimulus
        stimulus_pos = [0.6, 0.0, 0.0125]
        stimulus_rpy = [0, 0, 0]
        stim_path = os.path.join(
            os.path.dirname(__file__),
            "../stimuli/circle/circle.urdf"
        )

        target_df, image_dir, workframe_pos, workframe_rpy = setup_data_collection(
            num_samples=num_samples,
            apply_shear=apply_shear,
            shuffle_data=False,
            collect_dir_name=collect_dir_name,
        )

        collect_data(
            target_df,
            image_dir,
            stim_path,
            stimulus_pos,
            stimulus_rpy,
            workframe_pos,
            workframe_rpy,
            tactip_params,
            show_gui=show_gui,
            show_tactile=show_tactile,
            quick_mode=quick_mode
        )
