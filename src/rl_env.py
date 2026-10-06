"""Gymnasium 환경: SO-101 (역거치) 목표점 도달 + 툴 하향 자세. PPO로 학습되는 저수준 정책용."""

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from .robot import ARM_JOINTS, CEILING_Z, SO101, SUBSTEPS

MAX_STEPS = 120
DQ_SCALE = 0.07     # 한 제어 스텝(50ms)당 최대 관절 목표 변화 [rad]
SUCCESS_DIST = 0.012
FINE_DIST = 0.008
TOOL_DOWN = 0.85


def sample_goal(robot, rng, lo_z=0.015, hi_z=0.28):
    """관절공간에서 샘플한 도달 가능한 FK 지점(툴 하향) -> 목표. 도달 불가능한 목표를 피함."""
    for i in range(4000):
        if i == 2000:
            hi_z = 0.28
        q = rng.uniform(robot.lo, robot.hi)
        for j, v in zip(ARM_JOINTS, q):
            p.resetJointState(robot.id, j, float(v), 0.0, physicsClientId=robot.cid)
        pos, zax = robot.tool()
        if zax[2] < -0.9 and lo_z <= pos[2] <= hi_z:
            return pos
    raise RuntimeError("goal sampling failed")


class ReachEnv(gym.Env):
    def __init__(self, seed=0, hold=False):
        super().__init__()
        self.hold = hold
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1 / 240, physicsClientId=self.cid)
        floor = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.01], physicsClientId=self.cid)
        p.createMultiBody(0, floor, basePosition=[0, 0, -0.01], physicsClientId=self.cid)
        self.robot = SO101(client=self.cid)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1, 1, (5,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (5 + 5 + 3 + 3 + 3 + 1,), np.float32)
        self.goal = np.zeros(3)

    def _obs(self):
        r = self.robot
        pos, zax = r.tool()
        gb, pb = r.to_base(self.goal), r.to_base(pos)
        return np.concatenate([r.q(), r.dq() * 0.1, pb, gb, gb - pb, [zax[2]]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.robot
        # 목표를 먼저 샘플(관절을 임의 자세로 흔들므로), 이후 홈 근처 + 노이즈로 시작
        self.goal = sample_goal(r, self.rng, hi_z=0.09 if (self.hold and self.rng.random() < 0.6) else 0.28)
        start = r.home_q + self.rng.normal(0, 0.25, 5)
        if self.rng.random() < 0.5:  # 절반은 임의 자세에서 시작 (연속 동작 중 재목표 상황 대응)
            start = np.clip(self.rng.uniform(r.lo, r.hi) * 0.6 + r.home_q * 0.4, r.lo, r.hi)
        r.reset(np.clip(start, r.lo, r.hi))
        r.command(r.q_target)
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.cid)
        self.t = 0
        self.ok_hist = []
        pos, _ = r.tool()
        self.prev_d = np.linalg.norm(pos - self.goal)
        return self._obs(), {}

    def step(self, action):
        r = self.robot
        a = np.clip(action, -1, 1)
        r.command(r.q_target + a * DQ_SCALE)
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.cid)
        self.t += 1
        pos, zax = r.tool()
        d = np.linalg.norm(pos - self.goal)
        down = -zax[2]
        if self.hold:
            # 목표 근처 '유지'를 보상: 정지/안정 수렴을 학습
            speed = float(np.linalg.norm(r.dq()))
            rew = 3.0 * (self.prev_d - d) - 0.2 * d + 0.05 * (down - 1) - 0.002 * float(np.sum(a * a))
            if d < 0.02:
                rew += 0.3 - 0.05 * speed
            if d < FINE_DIST and down > TOOL_DOWN:
                rew += 0.7
            self.prev_d = d
            if pos[2] < 0.003:
                rew -= 0.5
            ok = d < FINE_DIST and down > TOOL_DOWN
            self.ok_hist = (self.ok_hist + [ok])[-10:]
            final = self.t >= MAX_STEPS
            info = {"d": d, "down": down, "success": bool(final and np.mean(self.ok_hist) >= 0.8)}
            return self._obs(), float(rew), False, final, info
        rew = 4.0 * (self.prev_d - d) - 0.3 * d + 0.05 * (down - 1) - 0.002 * float(np.sum(a * a))
        self.prev_d = d
        success = d < SUCCESS_DIST and down > TOOL_DOWN
        low = pos[2] < 0.003  # 테이블 관통 방지
        if low:
            rew -= 0.5
        if success:
            rew += 5.0
        return self._obs(), float(rew), bool(success), self.t >= MAX_STEPS, {"d": d, "down": down, "success": success}

    def close(self):
        p.disconnect(self.cid)
