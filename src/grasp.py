"""마찰 파지(구속 없음): 조 패드 접촉력으로 물체를 잡는다. 코인은 자석 팁(힘 기반 인력 모델)."""

import numpy as np
import pybullet as p

from .robot import JAW_JOINT, TOOL_LINK, ARM_JOINTS

# (조 개구 O[mm], 관절각 q[rad]) - 패드 박스 형상에서 측정
_OPEN_TBL = np.array([(10.3, -0.17), (20.0, 0.0), (37.2, 0.3), (53.0, 0.6), (69.8, 1.0), (79.0, 1.4), (80.0, 1.75)])
JAW_CLOSE_Q = -0.17
JAW_FORCE = 3.0
FIXED_FACE_X = 0.002     # 툴 프레임 x에서 고정 핑거 안쪽 면
PAD_FRICTION = 2.0
MAGNET_OFFSET = (0.0095, 0.006)   # 툴 프레임에서 자석면 위치 (x: 고정 패드 중심, z: 패드 하단)
JAW_WIDE_Q = 1.4


def jaw_q_for_opening(opening_m):
    return float(np.interp(opening_m * 1000, _OPEN_TBL[:, 0], _OPEN_TBL[:, 1]))


def setup_friction(env):
    for r in env.robots.values():
        for link in (4, JAW_JOINT):
            p.changeDynamics(r.id, link, lateralFriction=PAD_FRICTION, spinningFriction=0.02, rollingFriction=0.0,
                             physicsClientId=env.cid)


def tool_axes(robot):
    ls = p.getLinkState(robot.id, TOOL_LINK, computeForwardKinematics=True, physicsClientId=robot.cid)
    Rm = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
    return np.array(ls[4]), Rm


def contact_forces(env, arm, item_id):
    """고정 핑거(링크4) / 이동 조(링크6)와 물체 사이의 법선 접촉력 합 [N]."""
    r, uid = env.robots[arm], env.items[item_id]["uid"]
    out = {}
    for name, link in (("fixed", 4), ("jaw", JAW_JOINT)):
        pts = p.getContactPoints(r.id, uid, linkIndexA=link, physicsClientId=env.cid)
        out[name] = float(sum(c[9] for c in pts))
    return out


class MagnetTip:
    """자석 팁: 툴 프레임 근방 코인에 인력(힘)을 가한다. 구속이 아니라 외력 모델."""

    def __init__(self, env):
        self.env, self.active = env, {}

    def on(self, arm, item_id):
        self.active[arm] = item_id

    def off(self, arm):
        self.active.pop(arm, None)

    def apply(self):
        for arm, item_id in self.active.items():
            pos_t, Rm = tool_axes(self.env.robots[arm])
            tip = pos_t + Rm @ np.array([MAGNET_OFFSET[0], 0.0, MAGNET_OFFSET[1]])
            uid = self.env.items[item_id]["uid"]
            pos = np.array(p.getBasePositionAndOrientation(uid)[0])
            vel = np.array(p.getBaseVelocity(uid)[0])
            hz = self.env.items[item_id]["half"][2]
            target = tip + np.array([0, 0, -(hz + 0.0005)])   # 고정 핑거 패드 하단면에 코인 윗면이 붙는 위치
            d = target - pos
            if np.linalg.norm(d) < 0.02:
                m = p.getDynamicsInfo(uid, -1, physicsClientId=self.env.cid)[0]
                f = m * (3600.0 * d - 120.0 * vel) + m * 9.81 * np.array([0, 0, 1.0])   # 질량 정규화된 감쇠 스프링 + 중력 보상
                n = np.linalg.norm(f)
                if n > 0.6:
                    f = f / n * 0.6
                p.applyExternalForce(uid, -1, f.tolist(), pos.tolist(), p.WORLD_FRAME, physicsClientId=self.env.cid)
