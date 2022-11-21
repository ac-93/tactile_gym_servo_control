import os
import time
import numpy as np
import pandas as pd
import json

from tactile_gym.utils.general_utils import check_dir


def make_target_df_rand(
    poses_rng, moves_rng, num_poses, obj_poses, shuffle_data=False
):
    # generate random poses
    np.random.seed()
    poses = np.random.uniform(
        low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6)
    )
    poses = poses[np.lexsort((poses[:, 1], poses[:, 5]))]
    moves = np.random.uniform(
        low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6)
    )

    # generate and save target data
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            "obj_id",
            "obj_pose",
            "pose_id",
            "pose_1",
            "pose_2",
            "pose_3",
            "pose_4",
            "pose_5",
            "pose_6",
            "move_1",
            "move_2",
            "move_3",
            "move_4",
            "move_5",
            "move_6",
        ]
    )

    # populate dateframe
    for i in range(num_poses * len(obj_poses)):
        image_file = "image_{:d}.png".format(i + 1)
        i_pose, i_obj = (int(i % num_poses), int(i / num_poses))
        pose = poses[i_pose, :]
        move = moves[i_pose, :]
        target_df.loc[i] = np.hstack(
            ((image_file, i_obj + 1, obj_poses[i_obj], i_pose + 1), pose, move)
        )

    if shuffle_data:
        target_df = target_df.sample(frac=1).reset_index(
            drop=True
        )

    return target_df


def create_data_dir(
    target_df,
    task_name,
    apply_shear=False,
    collect_dir_name=None,
):

    # experiment metadata
    home_dir = os.path.join(
        os.path.dirname(__file__),
        "../data",
        task_name,
        "shear" if apply_shear else "tap"
    )

    if collect_dir_name is None:
        collect_dir_name = "collect_tap_rand_" + time.strftime("%m%d%H%M")

    collect_dir = os.path.join(home_dir, collect_dir_name)
    image_dir = os.path.join(collect_dir, "images")
    target_file = os.path.join(collect_dir, "targets.csv")

    # check save dir exists
    check_dir(collect_dir)

    # create dirs
    os.makedirs(collect_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # save metadata (remove unneccesary non json serializable stuff)
    meta = locals().copy()
    del meta["collect_dir_name"], meta["target_df"]
    with open(os.path.join(collect_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # save target csv
    target_df.to_csv(target_file, index=False)

    return collect_dir, image_dir
