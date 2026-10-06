"""실제 SO-101 URDF(TheRobotStudio/SO-ARM100) 래퍼. 천장 역거치(Inverted) 장착."""

import math
import os

import numpy as np
import pybullet as p

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "so101", "so101_new_calib.urdf")
CEILING_Z = 0.42          # 베이스(천장) 높이 [m]
ARM_JOINTS = [0, 1, 2, 3, 4]   # pan, lift, elbow, wrist_flex, wrist_roll
JAW_JOINT = 6
TOOL_LINK = 5             # gripper_frame_link (파지 중심)
CTRL_HZ = 20
SUBSTEPS = 12             # 240Hz / 20Hz


class SO101:
    def __init__(self, base_xy=(0.0, 0.0), yaw=0.0, client=0):
        self.cid = client
        self.base_pos = np.array([base_xy[0], base_xy[1], CEILING_Z])
        self.base_orn = p.getQuaternionFromEuler([math.pi, 0, yaw])  # roll=pi: 거꾸로 매달림
        self.id = p.loadURDF(URDF, self.base_pos.tolist(), self.base_orn, useFixedBase=True, physicsClientId=client)
        self.lo = np.array([p.getJointInfo(self.id, j, physicsClientId=client)[8] for j in ARM_JOINTS])
        self.hi = np.array([p.getJointInfo(self.id, j, physicsClientId=client)[9] for j in ARM_JOINTS])
        self.home_q = np.array([0.0, -0.9, 1.2, 0.9, 0.0])
        self.reset(self.home_q)
        for j in ARM_JOINTS + [JAW_JOINT]:
            p.changeDynamics(self.id, j, jointDamping=0.05, physicsClientId=client)
        self.q_target = self.home_q.copy()

    # ---- 상태 ----
    def q(self):
        return np.array([p.getJointState(self.id, j, physicsClientId=self.cid)[0] for j in ARM_JOINTS])

    def dq(self):
        return np.array([p.getJointState(self.id, j, physicsClientId=self.cid)[1] for j in ARM_JOINTS])

    def tool(self):
        ls = p.getLinkState(self.id, TOOL_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        z_axis = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)[:, 2]
        return np.array(ls[4]), z_axis

    def to_base(self, world_xyz):
        inv_p, inv_o = p.invertTransform(self.base_pos.tolist(), self.base_orn)
        return np.array(p.multiplyTransforms(inv_p, inv_o, list(world_xyz), [0, 0, 0, 1])[0])

    # ---- 제어 ----
    def reset(self, q):
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.id, j, float(v), 0.0, physicsClientId=self.cid)
        self.q_target = np.array(q, float)

    def command(self, q_target, jaw=None):
        self.q_target = np.clip(q_target, self.lo, self.hi)
        p.setJointMotorControlArray(self.id, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=self.q_target.tolist(),
                                    forces=[8.0] * 5, positionGains=[0.35] * 5, velocityGains=[1.0] * 5,
                                    physicsClientId=self.cid)
        if jaw is not None:
            p.setJointMotorControl2(self.id, JAW_JOINT, p.POSITION_CONTROL, targetPosition=float(jaw), force=3.0,
                                    physicsClientId=self.cid)
