import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_surface_3d_servo_control():

    def move_init_pose(embodiment, stim_name):

        if stim_name in ["saddle"]:
            init_pos = [-0.04, 0.0, -0.01]
            init_rpy = [0.0, 0.0, 0.0]

        embodiment.move_linear(init_pos, init_rpy, quick_mode=False)

    stim_names = ["saddle"]
    ep_len = 200

    # set reference pose and gains
    ref_pose = np.array([
        1.0, 0.0, 2.5,  # millimeters
        0.0, 0.0, 0.0  # degrees
    ])
    p_gains = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.0])

    return move_init_pose, stim_names, ep_len, ref_pose, p_gains


def setup_edge_2d_servo_control():

    def move_init_pose(embodiment, stim_name):

        # move to initial pose for starting episode
        if stim_name in ["square", "circle", "clover"]:
            init_pos = [0.0, -0.05, 0.004]
            init_rpy = [0.0, 0.0, 0.0]

        if stim_name in ["foil"]:
            init_pos = [0.0, -0.04, 0.004]
            init_rpy = [0.0, 0.0, 0.0]

        embodiment.move_linear(init_pos, init_rpy, quick_mode=False)

    stim_names = ["square", "circle", "clover", "foil"]
    ep_len = 350

    # set reference pose and gains
    ref_pose = np.array([
            1.0, 0.0, 2.5,  # millimeters
            0.0, 0.0, 0.0  # degrees
    ])
    p_gains = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.1])

    return move_init_pose, stim_names, ep_len, ref_pose, p_gains


def setup_edge_3d_servo_control():

    def move_init_pose(embodiment, stim_name):

        if stim_name in ["saddle"]:
            init_pos = [-0.07, 0.0, -0.02]
            init_rpy = [0.0, 0.0, -np.pi/2]

        embodiment.move_linear(init_pos, init_rpy, quick_mode=False)

    stim_names = ["saddle"]
    ep_len = 500

    # set reference pose and gains
    ref_pose = np.array([
            1.0, 0.0, 3.5,  # millimeters
            0.0, 0.0, 0.0  # degrees
    ])
    p_gains = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.05])

    return move_init_pose, stim_names, ep_len, ref_pose, p_gains


def setup_edge_5d_servo_control():

    def move_init_pose(embodiment, stim_name):

        if stim_name in ["saddle"]:
            init_pos = [-0.07, 0.0, -0.02]
            init_rpy = [0.0, 0.0, -np.pi/2]

        embodiment.move_linear(init_pos, init_rpy, quick_mode=False)

    stim_names = ["saddle"]
    ep_len = 500

    # set reference pose and gains
    ref_pose = np.array([
            1.0, 0.0, 4.0,  # millimeters
            0.0, 0.0, 0.0  # degrees
    ])
    p_gains = np.array([1.0, 1.0, 0.5, 0.05, 0.05, 0.05])

    return move_init_pose, stim_names, ep_len, ref_pose, p_gains


if __name__ == '__main__':
    pass
