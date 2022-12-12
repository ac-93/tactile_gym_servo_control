import os

from tactile_gym_servo_control.utils.pybullet_utils import setup_pybullet_env
from tactile_gym_servo_control.data_collection.collect_data import collect_data
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_surface_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_2d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_3d_data_collection
from tactile_gym_servo_control.data_collection.setup_data_collection import setup_edge_5d_data_collection

stimuli_path = os.path.join(os.path.dirname(__file__), "../stimuli")

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
        "image_size": [128, 128],
        "turn_off_border": False,
    }

    show_gui = False
    show_tactile = False
    quick_mode = True

    # setup stimulus
    stimulus_pos = [0.6, 0.0, 0.0125]
    stimulus_rpy = [0, 0, 0]
    stim_path = os.path.join(
        stimuli_path, "square/square.urdf"
    )

    collection_params = {
        'train': 5000,
        'val': 2000
    }

    for collect_dir_name, num_samples in collection_params.items():

        target_df, image_dir, workframe_pos, workframe_rpy = setup_data_collection(
            num_samples=num_samples,
            apply_shear=False,
            shuffle_data=False,
            collect_dir_name=collect_dir_name,
        )

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

        collect_data(
            embodiment,
            target_df,
            image_dir,
            workframe_pos,
            workframe_rpy,
            tactip_params,
            show_gui=show_gui,
            show_tactile=show_tactile,
            quick_mode=quick_mode
        )
