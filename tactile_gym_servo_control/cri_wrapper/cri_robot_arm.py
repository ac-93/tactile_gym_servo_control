import numpy as np
import pybullet as pb
import cv2

from tactile_gym.robots.arms.robot import Robot
from tactile_gym.rl_envs.example_envs.example_arm_env.rest_poses import rest_poses_dict


def euler2quat(pose_e):
    """Converts an euler rotation pose to a quaternion rotation pose.
    """
    pose_e = np.array(pose_e, dtype=np.float64).ravel()
    assert pose_e.size == 6, "Invalid euler pose"
    rot_q = pb.getQuaternionFromEuler(pose_e[3:])
    pose_q = np.concatenate((pose_e[:3], rot_q))
    return pose_q


def quat2euler(pose_q):
    """Converts an euler rotation pose to a quaternion rotation pose.
    """
    pose_q = np.array(pose_q, dtype=np.float64).ravel()
    assert pose_q.size == 7, "Invalid quaternion pose"
    rot_e_rad = pb.getEulerFromQuaternion(pose_q[3:])
    pose_e = np.concatenate((pose_q[:3], rot_e_rad))
    return pose_e


def transform(pose_a, frame_b_a):
    """Transforms a quaternion pose between reference frames.

    Transforms a pose in reference frame A to a pose in reference frame
    B (B is expressed relative to reference frame A).
    """

    inv_frame_b_a_pos, inv_frame_b_a_rot = pb.invertTransform(
        frame_b_a[:3], frame_b_a[3:],
    )
    pos_b, rot_b = pb.multiplyTransforms(
        inv_frame_b_a_pos, inv_frame_b_a_rot,
        pose_a[:3], pose_a[3:]
    )

    return np.concatenate((pos_b, rot_b))


def inv_transform(pose_b, frame_b_a):
    """Inverse transforms  a quaternion pose between reference frames.

    Transforms a pose in reference frame B to a pose in reference frame
    A (B is expressed relative to A).
    """
    pos_a, rot_a = pb.multiplyTransforms(
        frame_b_a[:3], frame_b_a[3:],
        pose_b[:3], pose_b[3:]
    )

    return np.concatenate((pos_a, rot_a))


class CRIRobotArm(Robot):
    def __init__(
        self,
        pb,
        workframe_pos,
        workframe_rpy,
        image_size=[128, 128],
        turn_off_border=False,
        arm_type="ur5",
        t_s_params={},
        cam_params={},
        show_gui=True,
        show_tactile=True,
    ):

        self.cam_params = cam_params

        # initial joint positions used when reset
        rest_poses = rest_poses_dict[arm_type][t_s_params["name"]][t_s_params["type"]]

        # limits
        TCP_lims = np.zeros(shape=(6, 2))
        TCP_lims[0, 0], TCP_lims[0, 1] = -np.inf, +np.inf  # x lims
        TCP_lims[1, 0], TCP_lims[1, 1] = -np.inf, +np.inf  # y lims
        TCP_lims[2, 0], TCP_lims[2, 1] = -np.inf, +np.inf  # z lims
        TCP_lims[3, 0], TCP_lims[3, 1] = -np.inf, +np.inf  # roll lims
        TCP_lims[4, 0], TCP_lims[4, 1] = -np.inf, +np.inf  # pitch lims
        TCP_lims[5, 0], TCP_lims[5, 1] = -np.inf, +np.inf  # yaw lims

        super(CRIRobotArm, self).__init__(
            pb,
            rest_poses,
            workframe_pos,
            workframe_rpy,
            TCP_lims,
            image_size=image_size,
            turn_off_border=turn_off_border,
            arm_type=arm_type,
            t_s_name=t_s_params["name"],
            t_s_type=t_s_params["type"],
            t_s_core=t_s_params["core"],
            t_s_dynamics=t_s_params["dynamics"],
            show_gui=show_gui,
            show_tactile=show_tactile,
        )

    def close(self):
        if self._pb.isConnected():
            self._pb.disconnect()

    def move_linear(self, targ_pos, targ_rpy, quick_mode=True):
        self.arm.tcp_direct_workframe_move(targ_pos, targ_rpy)

        # slow but more realistic moves
        if not quick_mode:
            self.blocking_move(
                max_steps=10000,
                constant_vel=0.00025,
                pos_tol=1e-4,
                orn_tol=1e-4,
                jvel_tol=0.1,
            )

        # fast but unrealistic moves (bigger_moves = worse performance)
        else:
            self.blocking_move(
                max_steps=1000,
                constant_vel=None,
                pos_tol=1e-4,
                orn_tol=1e-4,
                jvel_tol=0.1,
            )

    def get_tcp_pose(self):
        """
        Returns pose of the Tool Center Point in world frame
        """
        (
                cur_TCP_pos,
                cur_TCP_rpy,
                _,
                _,
                _,
        ) = self.arm.get_current_TCP_pos_vel_workframe()

        return np.array([*cur_TCP_pos, *cur_TCP_rpy])

    def process_sensor(self):
        """
        Rename to be more like real CRI envs
        """
        return self.get_tactile_observation()

    def get_visual_observation(self):
        """
        Returns the rgb image from an environment camera.
        """

        # get an rgb image that matches the debug visualiser
        view_matrix = self._pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_params['pos'],
            distance=self.cam_params['dist'],
            yaw=self.cam_params['yaw'],
            pitch=self.cam_params['pitch'],
            roll=0,
            upAxisIndex=2,
        )

        proj_matrix = self._pb.computeProjectionMatrixFOV(
            fov=self.cam_params['fov'],
            aspect=float(self.cam_params['image_size'][0]) / self.cam_params['image_size'][1],
            nearVal=self.cam_params['near_val'],
            farVal=self.cam_params['far_val'],
        )

        (_, _, px, _, _) = self._pb.getCameraImage(
            width=self.cam_params['image_size'][0],
            height=self.cam_params['image_size'][1],
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._pb.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.cam_params['image_size'][0], self.cam_params['image_size'][1], 4))

        return rgb_array[:, :, :3]

    def render(self):
        """
        Return a concatenated tactile and visual image, useful for generating videos.
        """

        # get the rgb camera image
        rgb_array = self.get_visual_observation()

        # get the current tactile images and reformat to match rgb array
        tactile_array = self.get_tactile_observation()

        tactile_array = cv2.cvtColor(tactile_array, cv2.COLOR_GRAY2RGB)

        # resize tactile to match rgb if rendering in higher res
        if rgb_array.shape[:2] != tactile_array.shape[:2]:
            tactile_array = cv2.resize(tactile_array, tuple(rgb_array.shape[:2]))

        # concat the images into a single image
        render_array = np.concatenate([rgb_array, tactile_array], axis=1)

        return render_array
