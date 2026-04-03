import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import xrobotoolkit_sdk as xrt 

from constants_vuer import (
    T_robot_openxr,
    T_to_unitree_hand,
    grd_yup2grd_zup,
    hand2inspire,
    hand2inspire_l_arm,
    hand2inspire_l_finger,
    hand2inspire_r_arm,
    hand2inspire_r_finger,
)
from motion_utils import fast_mat_inv, mat_update

from robot_control.hand_retargeting import HandRetargeting, HandType

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import threading
import zmq

M_to_unitree_hand = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

def pose7_to_mat44(pose7):
    """
    pose7: [x, y, z, qx, qy, qz, qw]
    return: 4x4 np.ndarray
    """
    if pose7 is None or len(pose7) != 7:
        return np.eye(4)

    xyz = pose7[:3]
    quat = np.array(pose7[3:]) # qx, qy, qz, qw

    if np.all(quat == 0):
        mat = np.eye(4)
        mat[:3, 3] = xyz
        return mat

    try:
        mat = np.eye(4)
        rmat = R.from_quat(quat).as_matrix()
        mat[:3, :3] = rmat
        mat[:3, 3] = xyz
        return mat
    except ValueError:
        mat = np.eye(4)
        mat[:3, 3] = xyz
        return mat


class PicoReceiver:
    def __init__(self):
        self.T_corr = np.array([
            [-1,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  0,  1]
        ])

        try:
            print("[PICO] Initializing PICO SDK...")
            xrt.init()
            print("[PICO] PICO SDK Initialized successfully")
        except Exception as e:
            print(f"[PICO] Init Failed: {e}")
            print("[PICO] Continuing with existing SDK connection...")


    def get_latest_matrices(self):
        # 1. Get Head Pose (7-dim)
        head_pose = xrt.get_headset_pose()

        # 2. Get Hand Pose (7*26 dim)
        left_hand_pose = xrt.get_left_hand_tracking_state()
        right_hand_pose = xrt.get_right_hand_tracking_state()

        l_pose = left_hand_pose[1]
        r_pose = right_hand_pose[1]

        # Transform into 4*4 matrices

        head_mat = pose7_to_mat44(head_pose)
        left_mat = pose7_to_mat44(l_pose) if l_pose else None
        right_mat = pose7_to_mat44(r_pose) if r_pose else None

        return head_mat, left_mat, right_mat, np.array(left_hand_pose), np.array(right_hand_pose)
        

    def stop(self):
        xrt.close()


class VuerPreprocessor:
    def __init__(self, pico_receiver=None):
        self.pico_receiver = pico_receiver
        self.vuer_head_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]]
        )
        self.vuer_right_wrist_mat = np.array(
            [[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.vuer_left_wrist_mat = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)

        self.y_offset = None
        # Only adjust this parameter if g1's pelvis height is too higher or too lower than expected.
        self.target_height = 1.73
        self.calibration_enabled = False

    def trigger_calibration(self):
        self.y_offset = None
        self.calibration_enabled = True

    
    def process(self):
        p_head, p_left, p_right, p_left_hand, p_right_hand = self.pico_receiver.get_latest_matrices()
        head_mat = mat_update(self.vuer_head_mat, p_head)

        if self.calibration_enabled and head_mat is not None and self.y_offset is None:
            current_y = head_mat[1, 3]
            if not np.allclose(current_y, 0):
                self.y_offset = self.target_height - current_y
                print(f"Height Calibrated! Head Y: {current_y:.3f}, Offset: {self.y_offset:.3f}")

        height_offset = self.y_offset if self.y_offset is not None else 0.0

        if head_mat is not None:
            head_mat[1, 3] += height_offset

        self.vuer_head_mat = head_mat

        if p_left is not None:
            self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, p_left)
            self.vuer_left_wrist_mat[1, 3] += height_offset
            p_left_hand[:, 1] += height_offset
            p_left_hand = p_left_hand[:, :3]

        if p_right is not None:
            self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, p_right)
            self.vuer_right_wrist_mat[1, 3] += height_offset
            p_right_hand[:, 1] += height_offset
            p_right_hand = p_right_hand[:, :3]

        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        rel_left_wrist_mat = (
            fast_mat_inv(head_mat) @ left_wrist_mat @ hand2inspire_l_arm
        )
        rel_right_wrist_mat = (
            fast_mat_inv(head_mat) @ right_wrist_mat @ hand2inspire_r_arm
        )

        # homogeneous
        left_hand_vuer_mat = np.concatenate(
            [p_left_hand.copy().T, np.ones((1, p_left_hand.shape[0]))]
        )
        right_hand_vuer_mat = np.concatenate(
            [p_right_hand.copy().T, np.ones((1, p_right_hand.shape[0]))]
        )

        # change of basis
        left_hand_mat = T_robot_openxr @ left_hand_vuer_mat
        right_hand_mat = T_robot_openxr @ right_hand_vuer_mat

        left_hand_mat_wb = fast_mat_inv(left_wrist_mat) @ left_hand_mat
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat

        unitree_left_hand = (T_to_unitree_hand @ left_hand_mat_wb)[0:3, :].T
        unitree_right_hand = (T_to_unitree_hand @ right_hand_mat_wb)[0:3, :].T

        unitree_tip_indices = [5, 10, 15]  # [thumb, index, middle] in OpenXR

        # Check if hand data is initialized
        left_q_target, right_q_target = None, None

        if not np.all(left_hand_mat == 0.0):
            # Extract the relevant tip indices (assumed defined elsewhere)
            ref_left_value = unitree_left_hand[unitree_tip_indices].copy()
            ref_right_value = unitree_right_hand[unitree_tip_indices].copy()

            # Apply scaling factors to calibrate the values
            ref_left_value[0] *= 1.15
            ref_left_value[1] *= 1.05
            ref_left_value[2] *= 0.95

            ref_right_value[0] *= 1.15
            ref_right_value[1] *= 1.05
            ref_right_value[2] *= 0.95

            # Use the retargeting methods to convert reference values to qpos.
            left_q_target = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                self.hand_retargeting.right_dex_retargeting_to_hardware
            ]
            right_q_target = self.hand_retargeting.right_retargeting.retarget(
                ref_right_value
            )[self.hand_retargeting.right_dex_retargeting_to_hardware]


        return (
            head_mat,
            rel_left_wrist_mat,
            rel_right_wrist_mat,
            left_q_target,
            right_q_target,
        )
    

class PicoTeleop:
    def __init__(self):
        self.pico_receiver = PicoReceiver()

        self.processor = VuerPreprocessor(
            pico_receiver=self.pico_receiver,
        )


    def step(self, full_head=False):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_q, right_hand_q = (
            self.processor.process()
        )

        if full_head:
            head_rmat = head_mat
        else:
            head_rmat = head_mat[:3, :3]

        left_wrist_mat[2, 3] += 0.55
        right_wrist_mat[2, 3] += 0.55
        left_wrist_mat[0, 3] += 0.05
        right_wrist_mat[0, 3] += 0.05

        return head_rmat, left_wrist_mat, right_wrist_mat, left_hand_q, right_hand_q

    def shutdown(self):
        if self.pico_receiver:
            self.pico_receiver.stop()