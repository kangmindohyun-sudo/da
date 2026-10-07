"""실제 SO-101 URDF(TheRobotStudio/SO-ARM100) 래퍼. 천장 역거치(Inverted) 장착."""

import math
import os

import numpy as np
import pybullet as p

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "so101", "so101_grip.urdf")
CEILING_Z = 0.42          # 베이스(천장) 높이 [m]
ARM_JOINTS = [0, 1, 2, 3, 4]   # pan, lift, elbow, wrist_flex, wrist_roll
JAW_JOINT = 6
TOOL_LINK = 5             # gripper_frame_link (파지 중심)
CTRL_HZ = 20
SUBSTEPS = 12             # 240Hz / 20Hz


class SO101:
    def __init__(self, base_xy=(0.0, 0.0), yaw=0.0, client=0):
        self.cid = client
        self.yaw = yaw
        self.base_pos = np.array([base_xy[0], base_xy[1], CEILING_Z])
        self.base_orn = p.getQuaternionFromEuler([math.pi, 0, yaw])  # roll=pi: 거꾸로 매달림
        self.id = p.loadURDF(URDF, self.base_pos.tolist(), self.base_orn, useFixedBase=True, physicsClientId=client)
        self.lo = np.array([p.getJointInfo(self.id, j, physicsClientId=client)[8] for j in ARM_JOINTS])
        self.hi = np.array([p.getJointInfo(self.id, j, physicsClientId=client)[9] for j in ARM_JOINTS])
        self.home_q = np.array([1.6, -0.5, 0.8, 0.9, 0.0])  # 보관 자세: 두 팔 바깥쪽으로 접어 서로 간섭 없음
        self.reset(self.home_q)
        for j in ARM_JOINTS + [JAW_JOINT]:
            p.changeDynamics(self.id, j, jointDamping=0.05, physicsClientId=client)
        self.q_target = self.home_q.copy()
        self.max_force = 8.0   # 관절 토크 한계 [N·m]: 협동 압착 중에는 낮춰 접촉력을 물리적으로 제한(컴플라이언스)

    # ---- 상태 ----
    def q(self):
        return np.array([p.getJointState(self.id, j, physicsClientId=self.cid)[0] for j in ARM_JOINTS])

    def dq(self):
        return np.array([p.getJointState(self.id, j, physicsClientId=self.cid)[1] for j in ARM_JOINTS])

    def tool(self):
        ls = p.getLinkState(self.id, TOOL_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        z_axis = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)[:, 2]
        return np.array(ls[4]), z_axis

    def heading(self):
        """툴 x축(핀치축)의 수평 방향각 [rad], 베이스 yaw를 뺀 값(두 팔이 같은 좌표계로 학습되도록)."""
        ls = p.getLinkState(self.id, TOOL_LINK, computeForwardKinematics=True, physicsClientId=self.cid)
        Rm = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        h = math.atan2(Rm[1, 0], Rm[0, 0]) - self.yaw
        return (h + math.pi) % (2 * math.pi) - math.pi

    def to_base(self, world_xyz):
        inv_p, inv_o = p.invertTransform(self.base_pos.tolist(), self.base_orn)
        return np.array(p.multiplyTransforms(inv_p, inv_o, list(world_xyz), [0, 0, 0, 1])[0])

    # ---- 제어 ----
    def reset(self, q):
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(self.id, j, float(v), 0.0, physicsClientId=self.cid)
        self.q_target = np.array(q, float)

    def command(self, q_target, jaw=None):
        self.q_target = np.clip(q_target, self.lo + 0.10, self.hi - 0.10)   # 관절 한계 충격 방지 여유
        p.setJointMotorControlArray(self.id, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=self.q_target.tolist(),
                                    forces=[self.max_force] * 5, positionGains=[0.35] * 5, velocityGains=[1.0] * 5,
                                    physicsClientId=self.cid)
        if jaw is not None:
            p.setJointMotorControl2(self.id, JAW_JOINT, p.POSITION_CONTROL, targetPosition=float(jaw), force=3.0,
                                    physicsClientId=self.cid)
