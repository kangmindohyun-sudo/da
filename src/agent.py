"""학습된 PPO 도달 정책을 폐루프로 실행하는 에이전트. 목표 좌표(월드)만 받고, 관절 명령은 정책이 생성."""

import os

import numpy as np
import pybullet as p

from .rl_env import DQ_SCALE, TOOL_DOWN
from .robot import SUBSTEPS


class LearnedReacher:
    def __init__(self, env, model_path="models/reach_ppo", vecnorm_path="models/reach_vecnorm.pkl"):
        import pickle
        from stable_baselines3 import PPO
        self.env = env
        self.model = PPO.load(model_path, device="cpu")
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.mean, self.var, self.clip, self.eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon
        self.steps_used = 0
        self.min_clearance = 1.0
        self.violations = 0

    def _obs(self, robot, goal):
        pos, zax = robot.tool()
        gb, pb = robot.to_base(goal), robot.to_base(pos)
        o = np.concatenate([robot.q(), robot.dq() * 0.1, pb, gb, gb - pb, [zax[2]]]).astype(np.float32)
        return np.clip((o - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def stow(self, arms, max_steps=80, hold=None):
        """보관(홈) 자세로 관절공간 복귀 (속도 제한). 작업 간 두 팔 간섭을 피하기 위한 파킹 동작."""
        env = self.env
        for _ in range(max_steps):
            err = 0.0
            for a in arms:
                r = env.robots[a]
                d = r.home_q - r.q_target
                err = max(err, float(np.abs(d).max()))
                r.command(r.q_target + np.clip(d, -DQ_SCALE, DQ_SCALE))
            if hold:
                hold()
            env.tick(SUBSTEPS)
            self.min_clearance = min(self.min_clearance, env.arm_clearance())
            if err < 0.02:
                break

    def reach(self, goals, tol=0.012, max_steps=140, settle=True, hold=None, lock_roll=False):
        """goals: {arm: xyz(world)}. 모든 팔을 동시에 구동. 반환: {arm: 최종 오차[m]}."""
        env = self.env
        arms = list(goals)
        done = {a: 0 for a in arms}
        for t in range(max_steps):
            for a in arms:
                r = env.robots[a]
                if done[a] >= 3 and settle:
                    r.command(r.q_target)
                    continue
                act, _ = self.model.predict(self._obs(r, goals[a]), deterministic=True)
                act = np.clip(act, -1, 1).astype(float)
                if lock_roll:
                    act[4] = 0.0
                r.command(r.q_target + act * DQ_SCALE)
            if hold:
                hold()
            env.tick(SUBSTEPS)
            self.steps_used += 1
            c = env.arm_clearance()
            self.min_clearance = min(self.min_clearance, c)
            self.violations += int(c < 0.01)
            for a in arms:
                pos, zax = env.robots[a].tool()
                ok = np.linalg.norm(pos - goals[a]) < tol and -zax[2] > TOOL_DOWN
                done[a] = done[a] + 1 if ok else 0
            if all(done[a] >= 3 for a in arms):
                break
        return {a: float(np.linalg.norm(env.robots[a].tool()[0] - goals[a])) for a in arms}
