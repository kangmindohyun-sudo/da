"""학습된 PPO 도달 정책을 폐루프로 실행하는 에이전트. 목표 좌표(월드)만 받고, 관절 명령은 정책이 생성."""

import os

import numpy as np
import pybullet as p

import pybullet as p
from .rl_env import DQ_SCALE, TOOL_DOWN
from .robot import SUBSTEPS, TOOL_LINK

WINDUP = 0.10   # 관절 목표-실제각 허용 편차 [rad]


class LearnedReacher:
    def __init__(self, env, model_path="models/reach_ppo", vecnorm_path="models/reach_vecnorm.pkl"):
        import pickle
        from stable_baselines3 import PPO
        self.env = env
        self.model = PPO.load(model_path, device="cpu")
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.mean, self.var, self.clip, self.eps = vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon
        self.use_head = len(self.mean) > 20
        self.steps_used = 0
        self.prev_act = {}
        self.min_clearance = 1.0
        self.violations = 0

    def _obs(self, robot, goal, heading=None):
        pos, zax = robot.tool()
        gb, pb = robot.to_base(goal), robot.to_base(pos)
        parts = [robot.q(), robot.dq() * 0.1, pb, gb, gb - pb, [zax[2]]]
        if self.use_head:
            tgt = robot.heading() if heading is None else (heading - robot.yaw)
            dh = (tgt - robot.heading() + np.pi) % (2 * np.pi) - np.pi
            parts.append([np.cos(dh), np.sin(dh)])
        o = np.concatenate(parts).astype(np.float32)
        return np.clip((o - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def resolved_rate(self, goals, tol=0.0015, max_steps=10, hold=None, gain=0.8):
        """접촉 중 정밀 동기 이송용: 툴 위치 오차를 감쇠 최소제곱 자코비안으로 관절 증분으로 변환(롤은 헤딩 유지)."""
        env = self.env
        arms = list(goals)
        hd = {a: env.robots[a].heading() for a in arms}
        for _ in range(max_steps):
            worst = 0.0
            for a in arms:
                r = env.robots[a]
                pos = r.tool()[0]
                e = np.asarray(goals[a], float) - pos
                worst = max(worst, float(np.linalg.norm(e)))
                en = float(np.linalg.norm(e))
                if en > 0.01:   # 스텝당 최대 1cm: 특이점 근처 폭주 방지
                    e = e * (0.01 / en)
                q_all = [p.getJointState(r.id, j, physicsClientId=r.cid)[0] for j in (0, 1, 2, 3, 4, 6)]
                Jl, _ = p.calculateJacobian(r.id, TOOL_LINK, [0, 0, 0], q_all, [0.0] * 6, [0.0] * 6, physicsClientId=r.cid)
                Rb = np.array(p.getMatrixFromQuaternion(r.base_orn)).reshape(3, 3)   # 자코비안은 베이스 프레임 기준 -> 월드로 변환
                J = (Rb @ np.array(Jl))[:, :5]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), gain * e)
                dq = np.clip(dq, -0.02, 0.02)
                dh = (hd[a] - r.heading() + np.pi) % (2 * np.pi) - np.pi
                dq[4] = float(np.clip(0.9 * dh, -0.03, 0.03))
                qa = r.q()
                r.command(np.clip(r.q_target + dq, qa - WINDUP, qa + WINDUP))
            if hold:
                hold()
            env.tick(SUBSTEPS)
            self.steps_used += 1
            if worst < tol:
                break

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

    def reach(self, goals, tol=0.012, max_steps=140, settle=True, hold=None, lock_roll=False, headings=None):
        """goals: {arm: xyz(world)}. 모든 팔을 동시에 구동. 반환: {arm: 최종 오차[m]}."""
        env = self.env
        arms = list(goals)
        done = {a: 0 for a in arms}
        if headings is None:   # 헤딩 미지정: 현재 방향 유지
            headings = {a: env.robots[a].heading() + env.robots[a].yaw for a in arms}
        for t in range(max_steps):
            for a in arms:
                r = env.robots[a]
                if done[a] >= 3 and settle:
                    r.command(r.q_target)
                    continue
                act, _ = self.model.predict(self._obs(r, goals[a], headings[a]), deterministic=True)
                act = np.clip(act, -1, 1).astype(float)
                act = 0.5 * self.prev_act.get(a, act) + 0.5 * act      # 가속 제한(저크 완화)
                self.prev_act[a] = act.copy()
                if lock_roll:
                    act[4] = 0.0
                elif self.use_head:
                    dh = (headings[a] - r.yaw - r.heading() + np.pi) % (2 * np.pi) - np.pi
                    act[4] = float(np.clip(0.9 * dh / DQ_SCALE, -1, 1))   # 롤: 헤딩 비례 서보
                qt = r.q_target + act * DQ_SCALE
                qa = r.q()
                r.command(np.clip(qt, qa - WINDUP, qa + WINDUP))   # 안티와인드업: 목표가 실제 관절각에서 멀어지지 않게
            if hold:
                hold()
            env.tick(SUBSTEPS)
            self.steps_used += 1
            if self.steps_used % 4 == 0:
                c = env.arm_clearance()
                self.min_clearance = min(self.min_clearance, c)
                self.violations += int(c < 0.01)
            for a in arms:
                pos, zax = env.robots[a].tool()
                ok = np.linalg.norm(pos - goals[a]) < tol and -zax[2] > TOOL_DOWN
                if self.use_head and headings is not None:
                    dh = (headings[a] - env.robots[a].yaw - env.robots[a].heading() + np.pi) % (2 * np.pi) - np.pi
                    ok = ok and abs(dh) < 0.15
                done[a] = done[a] + 1 if ok else 0
            if all(done[a] >= 3 for a in arms):
                break
        return {a: float(np.linalg.norm(env.robots[a].tool()[0] - goals[a])) for a in arms}
