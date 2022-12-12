import pybullet as p
import pybullet_utils.bullet_client as bc

import pkgutil

from tactile_gym_servo_control.cri_wrapper.cri_embodiment import CRIEmbodiment
from tactile_gym.assets import add_assets_path


def setup_pybullet_env(
    stim_path,
    tactip_params,
    stimulus_pos,
    stimulus_rpy,
    workframe_pos,
    workframe_rpy,
    show_gui,
    show_tactile,
):

    # ========= environment set up ===========
    time_step = 1.0 / 240

    if show_gui:
        pb = bc.BulletClient(connection_mode=p.GUI)
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
        numSolverIterations=300,
        numSubSteps=1,
        contactBreakingThreshold=0.0005,
        erp=0.05,
        contactERP=0.05,
        # need to enable friction anchors (something to experiment with)
        frictionERP=0.2,
        solverResidualThreshold=1e-7,
        contactSlop=0.001,
        globalCFM=0.0001,
    )

    pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/plane/plane.urdf"),
        [0, 0, -0.625],
    )
    pb.loadURDF(
        add_assets_path("shared_assets/environment_objects/table/table.urdf"),
        [0.50, 0.00, -0.625],
        [0.0, 0.0, 0.0, 1.0],
    )

    # set debug camera position
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
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.resetDebugVisualizerCamera(
            cam_params['dist'],
            cam_params['yaw'],
            cam_params['pitch'],
            cam_params['pos']
        )

    # add stimulus
    stimulus = p.loadURDF(
        stim_path,
        stimulus_pos,
        p.getQuaternionFromEuler(stimulus_rpy),
        useFixedBase=True,
    )

    # create the robot and sensor embodiment
    embodiment = CRIEmbodiment(
        pb,
        workframe_pos=workframe_pos,
        workframe_rpy=workframe_rpy,
        image_size=tactip_params["image_size"],
        arm_type="ur5",
        t_s_params=tactip_params,
        cam_params=cam_params,
        show_gui=show_gui,
        show_tactile=show_tactile,
    )

    return embodiment, stimulus
