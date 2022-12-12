import os
import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import pkgutil

from tactile_gym_servo_control.cri_wrapper.cri_embodiment import CRIEmbodiment
from tactile_gym.assets import add_assets_path

stimuli_path = os.path.join(os.path.dirname(__file__), "../stimuli")


def main(
    show_gui=True,
    show_tactile=True,
):
    time_step = 1.0 / 960  # low for small objects

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    else:
        pb = bc.BulletClient(connection_mode=p.DIRECT)
        egl = pkgutil.get_loader("eglRenderer")
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        else:
            p.loadPlugin("eglRendererPlugin")

    pb.setGravity(0, 0, -10)
    pb.setPhysicsEngineParameter(
        fixedTimeStep=time_step,
        numSolverIterations=150,  # 150 is good but slow
        numSubSteps=1,
        contactBreakingThreshold=0.0005,
        erp=0.05,
        contactERP=0.05,
        frictionERP=0.2,
        # need to enable friction anchors (maybe something to experiment with)
        solverResidualThreshold=1e-7,
        contactSlop=0.001,
        globalCFM=0.0001,
    )

    # set gui camera position
    cam_params = {
        'image_size': [512, 512],
        'dist': 0.25,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0,
    }

    if show_gui:
        p.resetDebugVisualizerCamera(
            cam_params['dist'],
            cam_params['yaw'],
            cam_params['pitch'],
            cam_params['pos']
        )

    # load the environment
    plane_id = p.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf")
    )

    stimulus_pos = [0.65, 0.0, 0.025]
    stimulus_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])

    stimulus_id = pb.loadURDF(
        os.path.join(
            stimuli_path,
            "square/square.urdf"
        ),
        stimulus_pos,
        stimulus_orn,
        useFixedBase=True,
        flags=pb.URDF_INITIALIZE_SAT_FEATURES,
    )

    # set up tactip
    tactip_params = {
        "name": "tactip",
        "type": "standard",
        "core": "no_core",
        "dynamics": {},
        "image_size": [256, 256],
        "turn_off_border": False,
    }

    # setup workspace
    workframe_pos = [0.65, 0.0, 0.15]  # relative to world frame
    workframe_rpy = [-np.pi, 0.0, np.pi / 2]  # relative to world frame

    embodiment = CRIEmbodiment(
        pb,
        workframe_pos=workframe_pos,
        workframe_rpy=workframe_rpy,
        image_size=[128, 128],
        arm_type="ur5",
        t_s_params=tactip_params,
        cam_params=cam_params,
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    # move to the workframe
    embodiment.move_linear([0, 0, 0], [0, 0, 0])
    embodiment.process_sensor()

    # move in different directions
    test_movement(embodiment)

    # move to sides of edge
    test_edge_pos(embodiment)

    if show_gui:
        while pb.isConnected():
            embodiment.arm.draw_workframe()
            embodiment.arm.draw_TCP()
            # embodiment.arm.print_joint_pos_vel()
            embodiment.step_sim()
            time.sleep(time_step)

            q_key = ord("q")
            keys = pb.getKeyboardEvents()
            if q_key in keys and keys[q_key] & pb.KEY_WAS_TRIGGERED:
                exit()


def test_edge_pos(embodiment, quick_mode=False):
    # move near
    embodiment.apply_action(
        [0, -0.05, 0.095, 0, 0, 0],
        control_mode="TCP_position_control",
        max_steps=1000,
    )
    embodiment.get_tactile_observation()
    print("move near edge")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move left
    embodiment.move_linear([0.05, 0, 0.095], [0, 0, 0], quick_mode)
    embodiment.get_tactile_observation()
    print("move left edge")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move far
    embodiment.move_linear([0, 0.05, 0.095], [0, 0, 0], quick_mode)
    embodiment.get_tactile_observation()
    print("move far edge")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move right
    embodiment.move_linear([-0.05, 0, 0.095], [0, 0, 0], quick_mode)
    embodiment.get_tactile_observation()
    print("move right edge")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)


def test_movement(embodiment, quick_mode=False):
    # move x
    embodiment.move_linear([0.05, 0, 0], [0, 0, 0], quick_mode)
    print("move +x")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([-0.05, 0, 0], [0, 0, 0], quick_mode)
    print("move -x")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move y
    embodiment.move_linear([0, +0.05, 0], [0, 0, 0], quick_mode)
    print("move +y")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([0, -0.05, 0], [0, 0, 0], quick_mode)
    print("move -y")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move z
    embodiment.move_linear([0, 0, 0.05], [0, 0, 0], quick_mode)
    print("move +z")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([0, 0, -0.05], [0, 0, 0], quick_mode)
    print("move -z")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move roll
    embodiment.move_linear([0, 0, 0], [0.785, 0, 0], quick_mode)
    print("move +roll")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [-0.785, 0, 0], quick_mode)
    print("move -roll")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move pitch
    embodiment.move_linear([0, 0, 0], [0, +0.785, 0], quick_mode)
    print("move +pitch")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, -0.785, 0], quick_mode)
    print("move -pitch")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)

    # move yaw
    embodiment.move_linear([0, 0, 0], [0, 0, +0.785], quick_mode)
    print("move +yaw")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, -0.785], quick_mode)
    print("move -yaw")
    time.sleep(1)
    embodiment.move_linear([0, 0, 0], [0, 0, 0], quick_mode)
    print("move center")
    time.sleep(1)


if __name__ == "__main__":

    # mode (gpu vs direct for comparison)
    show_gui = True
    show_tactile = True

    main(show_gui, show_tactile)
