import os
import numpy as np

from tactile_gym.utils.general_utils import save_json_obj

from tactile_gym_servo_control.data_collection.data_collection_utils import make_target_df_rand
from tactile_gym_servo_control.data_collection.data_collection_utils import create_data_dir


def setup_surface_3d_data_collection(
    num_samples=100,
    apply_shear=False,
    shuffle_data=False,
    collect_dir_name=None,
):

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.6, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [[0, 0, 0.5, -25, -25, 0], [0, 0, 5.5, 25, 25, 0]]

    if apply_shear:
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    else:
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "surface_3d",
        apply_shear=apply_shear,
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'workframe_pos': workframe_pos,
        'workframe_rpy': workframe_rpy,
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir, workframe_pos, workframe_rpy


def setup_edge_2d_data_collection(
    num_samples=100,
    apply_shear=False,
    shuffle_data=False,
    collect_dir_name=None,
):

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.545, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [[0, -6.0, 1.5, -2.5, -2.5, -180], [0, 6.0, 5.5, 2.5, 2.5, 180]]

    if apply_shear:
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    else:
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_2d",
        apply_shear=apply_shear,
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'workframe_pos': workframe_pos,
        'workframe_rpy': workframe_rpy,
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir, workframe_pos, workframe_rpy


def setup_edge_3d_data_collection(
    num_samples=100,
    apply_shear=False,
    shuffle_data=False,
    collect_dir_name=None,
):

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.545, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [[0, -6.0, 1.5, -25, -25, -180], [0, 6.0, 5.5, 25, 25, 180]]

    if apply_shear:
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    else:
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_3d",
        apply_shear=apply_shear,
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'workframe_pos': workframe_pos,
        'workframe_rpy': workframe_rpy,
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir, workframe_pos, workframe_rpy


def setup_edge_5d_data_collection(
    num_samples=100,
    apply_shear=False,
    shuffle_data=False,
    collect_dir_name=None,
):

    # set the work frame of the robot (relative to world frame)
    workframe_pos = [0.545, 0.0, 0.0525]
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]

    obj_poses = [[0, 0, 0, 0, 0, 0]]
    poses_rng = [[0, -6.0, 1.5, -25, -25, -180], [0, 6.0, 5.5, 25, 25, 180]]

    if apply_shear:
        moves_rng = [[-5, -5, 0, -5, -5, -5], [5, 5, 0, 5, 5, 5]]
    else:
        moves_rng = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

    target_df = make_target_df_rand(
        poses_rng, moves_rng, num_samples, obj_poses, shuffle_data
    )

    collect_dir, image_dir = create_data_dir(
        target_df,
        "edge_5d",
        apply_shear=apply_shear,
        collect_dir_name=collect_dir_name,
    )

    collection_params = {
        'workframe_pos': workframe_pos,
        'workframe_rpy': workframe_rpy,
        'obj_poses': obj_poses,
        'poses_rng': poses_rng,
        'moves_rng': moves_rng,
    }

    save_json_obj(collection_params, os.path.join(collect_dir, 'collection_params'))

    return target_df, image_dir, workframe_pos, workframe_rpy
