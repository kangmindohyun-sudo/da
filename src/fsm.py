"""FSM 파이프라인: JEV Mock -> 라우팅 -> 접근/롤 정렬 -> 마찰 파지(접촉력 검증) -> 이송 -> (펄스 부하 R_int) -> 분류 투입.
FSM은 목표(어디로/어떤 파지)만 정하고, 팔 구동은 PPO 도달 정책(LearnedReacher)이 수행한다. 파지는 구속 없이 조 패드 마찰(코인은 자석 팁 외력)."""

import random
from enum import Enum, auto

import numpy as np
import pybullet as p

from .agent import LearnedReacher
from .environment import BIN_WALL_H, BINS
from .grasp import (FIXED_FACE_X, JAW_CLOSE_Q, JAW_FORCE, JAW_WIDE_Q, MAGNET_OFFSET, MagnetTip, contact_forces,
                    jaw_q_for_opening, tool_axes)
from .jev_mock import ItemType, JEVMockData, JEVProcessor
from .robot import JAW_JOINT

Z_HOVER = 0.13
Z_COOP = 0.16
Z_SAFE = 0.19
LARGE_SPAN = 0.05          # 이 이상이면 양팔 협동 파지
MIN_FORCE = 1.0            # 파지 판정: 패드(고정/이동) 각각 법선 접촉력 [N]
PINCH_HEADING = np.pi / 2  # 핀치축(툴 x)을 월드 y축 방향으로
PULSE_S = 0.1
SQUEEZE_TORQUE = 3.0   # 압착 중 관절 토크 한계 [N·m]
SQ_GAP = 0.016   # 압착 접근 시작 간격(측면에서 떨어진 거리)
SQUEEZE_TARGET, SQUEEZE_MAX, SQUEEZE_MIN = 4.0, 25.0, 2.0   # 협동 압착 접촉력 목표/상한/합격 [N]
PRESTOW = {"A": (-0.12, 0.08, 0.22), "B": (0.12, -0.08, 0.22)}   # 보관 전 자기 쪽 높은 경유점(상대 팔 보관 영역을 쓸지 않도록)
CELL_PARAMS = {ItemType.BATTERY_18650: (3.9, 2.0, 0.08), ItemType.BATTERY_AA_AAA: (1.5, 0.5, 0.30)}


class State(Enum):
    JEV_CLASSIFY = auto()
    APPROACH = auto()
    GRASP_VERIFY = auto()
    LIFT_TRANSPORT = auto()
    PULSE_TEST = auto()
    SORT_DECISION = auto()
    RELEASE = auto()
    RETURN_HOME = auto()
    FAULT = auto()
    DONE = auto()


class GraspFailure(Exception):
    pass


class SortingFSM:
    def __init__(self, env, reacher: LearnedReacher, ground_truth_r=None, seed=0):
        self.env, self.rc = env, reacher
        self.gt_r = ground_truth_r or {}
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.log = []
        self.counts = {k: 0 for k in BINS}
        self.jaw_q = {"A": JAW_CLOSE_Q, "B": JAW_CLOSE_Q}
        self.magnet = MagnetTip(env)
        env.hooks.append(self.magnet.apply)

    # ---------------- 유틸 ----------------
    def _hold(self):
        for a, r in self.env.robots.items():
            p.setJointMotorControl2(r.id, JAW_JOINT, p.POSITION_CONTROL, targetPosition=self.jaw_q[a],
                                    force=JAW_FORCE, physicsClientId=self.env.cid)

    def reach(self, goals, tol=0.012, max_steps=140, headings=None):
        self._hold()
        return self.rc.reach(goals, tol=tol, max_steps=max_steps, hold=self._hold, lock_roll=False, headings=headings)

    def move(self, goals, tol=0.012, step=0.05, max_steps=140, headings=None):
        """현재 툴 위치에서 목표까지 직선 웨이포인트(간격 step)를 만들고 각 점을 학습 정책이 추종."""
        goals = {a: np.asarray(g, float) for a, g in goals.items()}
        cur = {a: self.env.tool_pos(a) for a in goals}
        n = max(1, int(np.ceil(max(np.linalg.norm(goals[a] - cur[a]) for a in goals) / step)))
        err = {}
        for i in range(1, n + 1):
            wp = {a: cur[a] + (goals[a] - cur[a]) * i / n for a in goals}
            last = i == n
            err = self.reach(wp, tol=tol if last else 0.02, max_steps=max_steps if last else 50, headings=headings)
        return err

    def servo(self, goals, tol=0.004, iters=4, max_steps=100):
        """정책 도달 후 남은 위치 오차를 측정해 목표를 보정하며 재도달(폐루프 미세 보정)."""
        goals = {a: np.asarray(g, float) for a, g in goals.items()}
        cmd = {a: g.copy() for a, g in goals.items()}
        for _ in range(iters):
            err = self.reach(cmd, tol=tol, max_steps=max_steps)
            e = {a: goals[a] - self.env.tool_pos(a) for a in goals}
            if all(np.linalg.norm(v[:2]) < tol and abs(v[2]) < 2 * tol for v in e.values()):
                break
            for a in goals:
                cmd[a] = cmd[a] + e[a] * 0.8
        return {a: float(np.linalg.norm(goals[a] - self.env.tool_pos(a))) for a in goals}

    def _tick(self, n):
        for _ in range(n):
            self._hold()
            self.env.tick()

    def _align(self, arms, headings, periods=None, tol=0.03, steps=100):
        periods = periods or {}
        for _ in range(steps):
            worst = 0.0
            for a in arms:
                r = self.env.robots[a]
                _, Rm = tool_axes(r)
                h = np.arctan2(Rm[1, 0], Rm[0, 0])
                P = periods.get(a, np.pi)
                d = (headings[a] - h + P / 2) % P - P / 2
                worst = max(worst, abs(d))
                r.command(r.q_target + np.array([0, 0, 0, 0, np.clip(d, -0.1, 0.1)]))
            self._hold()
            self.env.tick(12)
            if worst < tol:
                break

    def _status(self, ctx, state):
        j = ctx["jev"]
        lines = [f"ITEM {j.item_id}  type={j.item_type.value}  visual={j.visual_status.value}"
                 + (f"  ocr={j.ocr_text}" if j.ocr_text else ""),
                 f"STATE {state.name}   route={ctx.get('route', '-')}   arms={ctx.get('arms', '-')}  grasp={ctx.get('mode', '-')}"]
        if "r_int" in ctx:
            lines.append(f"PULSE 0.1s  dV={ctx['dv']*1000:.0f}mV  I={ctx['i_load']:.1f}A  R_int={ctx['r_int']*1000:.0f} mOhm"
                         f"  (thr {ctx['thr']*1000:.0f}) -> {ctx.get('dest', '?')}")
        if "forces" in ctx:
            lines.append("CONTACT FORCE " + "  ".join(f"{a}: fixed {f['fixed']:.1f}N jaw {f['jaw']:.1f}N" if 'fixed' in f else f"{a}: magnet"
                                                       for a, f in ctx["forces"].items()))
        lines.append("BINS " + "  ".join(f"{k}={v}" for k, v in self.counts.items() if v))
        lines.append(f"policy steps {self.rc.steps_used}   arm clearance min {self.rc.min_clearance*100:.1f}cm")
        self.env.status["lines"] = lines

    # ---------------- 상태 핸들러 ----------------
    def s_JEV_CLASSIFY(self, ctx):
        j: JEVMockData = ctx["jev"]
        hx, hy, hz = ctx["half"]
        large = max(hx, hy) * 2 >= LARGE_SPAN
        if JEVProcessor.is_fire_hazard(j):
            ctx.update(route="FIRE_ISOLATION", dest="isolation", cell=False)
        elif JEVProcessor.is_foreign_object(j):
            ctx.update(route="MANUAL_DIVERT", dest="manual", cell=False)
        elif JEVProcessor.is_coin_battery(j):
            ctx.update(route="COIN_SORT", dest=None, cell=False)
        else:
            ctx.update(route="PULSE_TEST", dest=None, cell=True)
        if large:
            ctx["arms"] = "AB"
        elif ctx["route"] in ("FIRE_ISOLATION", "MANUAL_DIVERT"):
            ctx["arms"] = "B"
        else:
            ctx["arms"] = "A"
        ctx["mode"] = "magnet" if ctx["route"] == "COIN_SORT" else ("coop" if large else "pinch")
        return State.APPROACH

    def _plan(self, ctx):
        """arm -> 파지 계획. pinch: A가 물체 중앙을 핀치. support: B가 +x쪽 측면에 닫힌 패드로 접촉(양팔 접점 검증/지지)."""
        item = self.env.items[ctx["id"]]
        c = self.env.item_pos(ctx["id"])
        hx, hy, hz = item["half"]
        full_h = 2 * hz
        zs = float(c[2] - hz)   # 물체가 놓인 면의 높이(플랫폼)
        plan = {}
        for a in ctx["arms"]:
            if ctx["mode"] == "magnet":
                plan[a] = dict(kind="magnet", xy=c[:2].copy(), z=zs + 2 * hz + 0.008 + MAGNET_OFFSET[1], heading=self._pick_heading(a, PINCH_HEADING, (0.0, np.pi)))
            elif ctx["mode"] == "coop":
                # 두 팔이 닫힌 패드로 양쪽에서 압착(헤딩 0: 툴 +x = 월드 +x).
                # A: 고정 패드 바깥면(+0.017)이 물체 -x 측면에, B: 이동 조 패드 바깥면(-0.0225)이 +x 측면에 닿음
                zt = zs + min(max(0.5 * full_h, 0.012), 0.07) + 0.006
                h = self._pick_heading(a, 0.0)                     # 롤 한계 여유 기준으로 헤딩 0 또는 π 선택
                s = np.cos(h)
                if a == "A":   # 월드 +x 방향으로 미는 면
                    pad, off = ("fixed", 0.017) if s > 0 else ("jaw", 0.0225)
                    plan[a] = dict(kind="squeeze", xy=np.array([c[0] - hx - off - SQ_GAP, c[1]]), z=zt, heading=h, period=2 * np.pi, inward=1.0, push=0.0, pad=pad)
                else:          # B: 월드 -x 방향으로 미는 면
                    pad, off = ("jaw", 0.0225) if s > 0 else ("fixed", 0.017)
                    plan[a] = dict(kind="squeeze", xy=np.array([c[0] + hx + off + SQ_GAP, c[1]]), z=zt, heading=h, period=2 * np.pi, inward=-1.0, push=0.0, pad=pad)
            else:
                chord = 2 * hy if ctx["jev"].item_type not in (ItemType.CAN_ALUMINUM, ItemType.CAN_FERROUS) else 2 * hx
                O = float(np.clip(chord + 0.050, 0.012, 0.079))
                zb = 0.004 if full_h < 0.03 else 0.010
                plan[a] = dict(kind="pinch", xy=c[:2].copy(), O=O, qo=jaw_q_for_opening(O), z=zs + zb + 0.006, heading=self._pick_heading(a, PINCH_HEADING, (0.0, np.pi)), period=2 * np.pi)
        return plan

    def _pick_heading(self, a, base, offsets=(0.0, np.pi)):
        """등가 헤딩 후보(base+offset) 중 롤 관절 한계 여유가 있고 회전량이 작은 쪽의 월드 헤딩."""
        r = self.env.robots[a]
        cur, q4 = r.heading() + r.yaw, r.q()[4]
        best = None
        for off in offsets:
            dh = (base + off - cur + np.pi) % (2 * np.pi) - np.pi
            ok = r.lo[4] + 0.25 < q4 + dh < r.hi[4] - 0.25
            key = (0 if ok else 1, abs(dh))
            if best is None or key < best[0]:
                best = (key, cur + dh)
        return best[1]

    def _target_heading(self, a, pl):
        """계획 헤딩(주기 P)과 등가인 각 중, 현재 헤딩에서 가장 가까운 월드 헤딩."""
        r = self.env.robots[a]
        cur = r.heading() + r.yaw
        P = pl.get("period", np.pi)
        return cur + ((pl["heading"] - cur + P / 2) % P - P / 2)

    def _tool_goal(self, a, pl, z, h=None):
        """정렬된 툴 자세에서 핀치(자석) 지점 또는 지지 접점이 목표 xy가 되도록 툴 프레임 목표 계산."""
        if pl["kind"] == "squeeze":
            return np.array([pl["xy"][0] + pl["inward"] * pl["push"], pl["xy"][1], z])
        off = (MAGNET_OFFSET[0] if pl["kind"] == "magnet" else FIXED_FACE_X - pl["O"] / 2)
        if h is None:
            _, Rm = tool_axes(self.env.robots[a])
            ax = Rm[:2, 0]
        else:
            ax = np.array([np.cos(h), np.sin(h)])
        xy = pl["xy"] - ax * off
        return np.array([xy[0], xy[1], z])

    def _grasp_err(self, a, pl, z_target):
        """(수평 오차 벡터, 방향 오차 [rad], 3D 보정용 오차 벡터)."""
        pos, Rm = tool_axes(self.env.robots[a])
        if pl["kind"] == "squeeze":
            pt = pos[:2] - np.array([pl["inward"] * pl["push"], 0.0])
        else:
            off = (MAGNET_OFFSET[0] if pl["kind"] == "magnet" else FIXED_FACE_X - pl["O"] / 2)
            pt = pos[:2] + Rm[:2, 0] * off
        h = np.arctan2(Rm[1, 0], Rm[0, 0])
        P = pl.get("period", np.pi)
        dh = abs((pl["heading"] - h + P / 2) % P - P / 2)
        e = np.array([pt[0] - pl["xy"][0], pt[1] - pl["xy"][1], pos[2] - z_target])
        return e[:2], float(dh), e

    def s_APPROACH(self, ctx):
        plan = self._plan(ctx)
        ctx["plan"] = plan
        arms = ctx["arms"]
        z_h = min(0.165, max(Z_HOVER, 2 * ctx["half"][2] + 0.04)) if len(arms) == 1 else Z_HOVER   # 정책 도달 범위(≈17cm) 내에서 물체 윗면 위
        ctx["z_h"] = z_h
        for a in arms:
            self.jaw_q[a] = JAW_CLOSE_Q   # 이동 중에는 조를 닫아 부피를 줄임
        self.move({a: [plan[a]["xy"][0], plan[a]["xy"][1], z_h] for a in arms}, tol=0.015)
        for a in arms:
            self.jaw_q[a] = plan[a].get("qo", JAW_CLOSE_Q)   # support/magnet: 닫힘 유지
        self._tick(70)
        corr = {a: np.zeros(3) for a in arms}      # 정책 편향 보정량(측정 오차를 누적해 목표에 더함)
        errs = {}
        for z in (z_h, 0.09, 0.05, None):          # 단계 하강: 매 단계 롤 재정렬 + 물체 기준 폐루프 보정
            for it in range(6):
                hd = {a: self._target_heading(a, plan[a]) for a in arms}   # 헤딩 목표를 정책에 직접 전달(스크립트 롤 정렬 없음)
                zz = {a: (plan[a]["z"] if z is None else max(z, plan[a]["z"])) for a in arms}
                self.reach({a: self._tool_goal(a, plan[a], zz[a], hd[a]) + corr[a] for a in arms}, tol=0.003, max_steps=60, headings=hd)
                errs = {a: self._grasp_err(a, plan[a], zz[a]) for a in arms}
                if all(np.linalg.norm(e[0]) < 0.003 and abs(e[2][2]) < 0.004 and e[1] < 0.08 for e in errs.values()):
                    break
                for a in arms:
                    corr[a] = np.clip(corr[a] - 0.8 * errs[a][2], -0.012, 0.012)   # 발산 방지: 보정량 상한
        err = {a: float(np.linalg.norm(errs[a][2])) for a in arms}
        ctx["reach_err_mm"] = {a: round(v * 1000, 1) for a, v in err.items()}
        ctx["tool_xy_off"] = {a: self.env.tool_pos(a)[:2] - self.env.item_pos(ctx["id"])[:2] for a in arms}
        return State.GRASP_VERIFY

    def s_GRASP_VERIFY(self, ctx):
        arms, plan = ctx["arms"], ctx["plan"]
        for a in arms:
            if plan[a]["kind"] == "magnet":
                self.magnet.on(a, ctx["id"])
        # 조를 천천히 닫으며 양 패드 접촉력이 충분해지면 멈춤 (급하게 닫으면 물체가 튕김)
        for _ in range(80):
            for a in arms:
                if plan[a]["kind"] == "pinch":
                    f_ = contact_forces(self.env, a, ctx["id"])
                    if min(f_["fixed"], f_["jaw"]) < 3.0:
                        self.jaw_q[a] = max(JAW_CLOSE_Q, self.jaw_q[a] - 0.02)
            self._tick(5)
        self._tick(40)
        forces = {}
        if any(plan[a]["kind"] == "squeeze" for a in arms):
            for a in arms:
                self.env.robots[a].max_force = SQUEEZE_TORQUE
            self._squeeze(ctx, plan, arms)
        for a in arms:
            if plan[a]["kind"] in ("pinch", "squeeze"):
                forces[a] = contact_forces(self.env, a, ctx["id"])
        ctx["forces"] = forces
        item_xy = self.env.item_pos(ctx["id"])[:2]
        for a in arms:   # 압착 오프셋(안쪽으로 미는 량)을 이후 리프트/이송 목표에 유지
            if plan[a]["kind"] == "squeeze":
                ctx["tool_xy_off"][a] = np.array([plan[a]["xy"][0] + plan[a]["inward"] * plan[a]["push"], plan[a]["xy"][1]]) - item_xy
        ctx["tool_off3"] = {a: np.array([ctx["tool_xy_off"][a][0], ctx["tool_xy_off"][a][1], self.env.tool_pos(a)[2] - self.env.item_pos(ctx["id"])[2]]) for a in arms}
        bad = {}
        for a, f_ in forces.items():
            if plan[a]["kind"] == "pinch":
                ok = min(f_["fixed"], f_["jaw"]) >= MIN_FORCE
            else:
                ok = f_[plan[a]["pad"]] >= SQUEEZE_MIN
            if not ok:
                bad[a] = f_
        if bad:
            raise GraspFailure(f"contact force check failed: { {a: {k: round(v, 2) for k, v in f_.items()} for a, f_ in bad.items()} }")
        return State.LIFT_TRANSPORT

    def _squeeze(self, ctx, plan, arms):
        """두 팔을 안쪽으로 밀며 양쪽 패드 접촉력을 목표(SQUEEZE_TARGET N)로 맞춤. 접촉 전에는 큰 걸음, 접촉 후에는 0.4mm 미세 조절."""
        for a in arms:
            plan[a]["push"] = max(plan[a]["push"], SQ_GAP - 0.004)   # 접촉 직전까지 빠르게 접근
        self.reach({a: self._tool_goal(a, plan[a], plan[a]["z"]) for a in arms}, tol=0.0015, max_steps=40)
        for _ in range(60):
            fs = {a: contact_forces(self.env, a, ctx["id"])[plan[a]["pad"]] for a in arms}
            if all(SQUEEZE_TARGET <= v <= SQUEEZE_MAX for v in fs.values()):
                break
            for a in arms:
                if fs[a] < SQUEEZE_TARGET:
                    plan[a]["push"] = min(plan[a]["push"] + (0.0004 if fs[a] > 0 else 0.001), 0.034)
                elif fs[a] > SQUEEZE_MAX:
                    plan[a]["push"] = max(plan[a]["push"] - 0.0006, 0.0)
            self.reach({a: self._tool_goal(a, plan[a], plan[a]["z"]) for a in arms}, tol=0.001, max_steps=30)
            self._tick(24)

    def servo_move(self, goals, step=0.004):
        """현재 툴 위치에서 목표까지 직선(step 간격)으로 자코비안 서보 이동 - 접촉 직후 안전한 후퇴/상승용."""
        goals = {a: np.asarray(g, float) for a, g in goals.items()}
        cur = {a: self.env.tool_pos(a) for a in goals}
        n = max(1, int(np.ceil(max(np.linalg.norm(goals[a] - cur[a]) for a in goals) / step)))
        for i in range(1, n + 1):
            self._hold()
            self.rc.resolved_rate({a: cur[a] + (goals[a] - cur[a]) * i / n for a in goals}, tol=0.002, max_steps=8, hold=self._hold)

    def coop_move(self, ctx, target, step=0.003, tol=0.004):
        """양팔 동기 이송: 물체 중심을 기준 좌표계로 두고 목표까지 step씩 전진, 각 팔 목표 = 물체 경로점 + 파지 오프셋."""
        arms, iid = ctx["arms"], ctx["id"]
        item0 = self.env.item_pos(iid)
        target = np.asarray(target, float)
        offs = ctx["tool_off3"]
        n = max(1, int(np.ceil(np.linalg.norm(target - item0) / step)))
        corr = {a: np.zeros(3) for a in arms}      # 각 팔의 추종 오차 적분 보상(두 팔 간 어긋남으로 인한 전단 제거)
        for i in range(1, n + 1):
            wp = item0 + (target - item0) * i / n
            goals = {a: wp + offs[a] + corr[a] for a in arms}
            self._hold()
            self.rc.resolved_rate(goals, tol=0.0015, max_steps=10, hold=self._hold)   # 접촉 중 두 팔 정밀 동기(자코비안 서보)
            for a in arms:
                e = self.env.tool_pos(a) - (wp + offs[a])
                corr[a] = np.clip(corr[a] - 0.5 * e, -0.008, 0.008)

    def _lift(self, ctx):
        arms, plan = ctx["arms"], ctx["plan"]
        z0 = self.env.item_pos(ctx["id"])[2]
        z_up = ctx["z_h"]
        if len(arms) > 1:
            i0 = self.env.item_pos(ctx["id"])
            self.coop_move(ctx, i0 + np.array([0, 0, z_up - self.env.tool_pos(arms[0])[2]]))
        else:
            goals = {}
            ixy = self.env.item_pos(ctx["id"])[:2]
            for a in arms:
                xy = ixy + ctx["tool_xy_off"][a]
                goals[a] = np.array([xy[0], xy[1], z_up])
            self.move(goals, tol=0.010, step=0.02, max_steps=100)
        self._tick(40)
        z1 = self.env.item_pos(ctx["id"])[2]
        if z1 - z0 < 0.6 * (z_up - plan[arms[0]]["z"]):
            raise GraspFailure(f"slip during lift: item rose {z1 - z0:.3f}m")
        hz = self.env.items[ctx["id"]]["half"][2]
        ctx["bottom_off"] = self.env.tool_pos(arms[0])[2] - (z1 - hz)   # 툴 z - 물체 바닥 z

    def _carry_to(self, ctx, dest):
        arms = ctx["arms"]
        z = BIN_WALL_H + 0.045 + ctx["bottom_off"]   # 수거함 벽 위 4.5cm에 물체 바닥 (작업영역 안쪽의 낮은 호버)
        cx, cy = self._bin_xy(dest)
        ctx["drop_xy"] = (cx, cy)
        goals = {a: np.array([cx + ctx["tool_xy_off"][a][0], cy + ctx["tool_xy_off"][a][1], z]) for a in arms}
        ctx["hover"] = goals
        if len(arms) > 1:
            i0 = self.env.item_pos(ctx["id"])
            self.coop_move(ctx, np.array([cx, cy, i0[2]]), step=0.005)
        else:
            self.move(goals, tol=0.012, max_steps=200)

    def _bin_xy(self, dest):
        cx, cy = BINS[dest]["center"]
        j = BINS[dest]["half"] * 0.3
        return cx + self.rng.uniform(-j, j), cy + self.rng.uniform(-j, j)

    def s_LIFT_TRANSPORT(self, ctx):
        self._lift(ctx)
        if ctx["cell"]:
            return State.PULSE_TEST
        if ctx["route"] == "COIN_SORT":
            return State.SORT_DECISION
        self._carry_to(ctx, ctx["dest"])
        return State.RELEASE

    def s_PULSE_TEST(self, ctx):
        j = ctx["jev"]
        ocv, i_load, thr = CELL_PARAMS[j.item_type]
        r_true = self.gt_r.get(j.item_id, 0.1)
        n = int(PULSE_S * 240)
        v = []
        for k in range(n):
            t = (k + 1) / 240
            v.append(ocv - i_load * r_true * (1 - np.exp(-t / 0.01)) + self.np_rng.normal(0, 0.002))
            self._tick(1)
        dv = ocv - float(np.mean(v[-6:]))
        ctx.update(i_load=i_load, dv=dv, r_int=dv / i_load, thr=thr, r_true=r_true)
        ctx["dest"] = "reuse" if ctx["r_int"] < thr else "recycle"
        return State.SORT_DECISION

    def s_SORT_DECISION(self, ctx):
        j = ctx["jev"]
        if ctx["route"] == "COIN_SORT":
            prefix = (j.ocr_text or "")[:2].upper()
            dest = {"CR": "coin_CR", "SR": "coin_SR", "LR": "coin_LR"}.get(prefix)
            if dest is None:
                dest = "manual"
                ctx["route"] = "COIN_OCR_FAIL->MANUAL"
            ctx["dest"] = dest
        self._carry_to(ctx, ctx["dest"])
        return State.RELEASE

    def s_RELEASE(self, ctx):
        arms = ctx["arms"]
        z_drop = BIN_WALL_H + 0.02 + ctx["bottom_off"]
        if len(arms) > 1:
            hz = self.env.items[ctx["id"]]["half"][2]
            cx, cy = ctx["drop_xy"]
            self.coop_move(ctx, np.array([cx, cy, BIN_WALL_H + 0.02 + hz]), step=0.005)   # 양팔 동기 하강
            for a in arms:
                self.magnet.off(a)
            plan = ctx["plan"]
            out = {a: self.env.tool_pos(a) + np.array([-plan[a].get("inward", 0.0) * 0.035, 0.0, 0.0]) for a in arms}
            for r in self.env.robots.values():
                r.max_force = 8.0
            self.servo_move(out)                         # 압착 해제: 바깥으로 천천히
            self._tick(40)
            self.servo_move({a: out[a] + np.array([0.0, 0.0, 0.07]) for a in arms}, step=0.005)   # 수직 상승
        else:
            drop = {a: np.array([h[0], h[1], z_drop]) for a, h in ctx["hover"].items()}
            self.move(drop, tol=0.010, max_steps=120)
            for a in arms:
                self.jaw_q[a] = JAW_WIDE_Q
                self.magnet.off(a)
            self._tick(90)
            self.move({a: ctx["hover"][a] for a in arms}, tol=0.015, max_steps=80)
        for a in arms:
            self.jaw_q[a] = JAW_CLOSE_Q
        self._tick(40)
        self.counts[ctx["dest"]] += 1
        return State.RETURN_HOME

    def _go_stow(self, arms):
        for r in self.env.robots.values():
            r.max_force = 8.0
        for a in arms:
            self.jaw_q[a] = JAW_CLOSE_Q
        self.move({a: PRESTOW[a] for a in arms}, tol=0.03, max_steps=80)
        self.rc.stow("AB", hold=self._hold)

    def s_RETURN_HOME(self, ctx):
        self._go_stow(ctx["arms"])
        return State.DONE

    def s_FAULT(self, ctx):
        for a in "AB":
            self.jaw_q[a] = JAW_WIDE_Q
            self.magnet.off(a)
        self._tick(60)
        arms = ctx["arms"]
        goals = {}
        for a in arms:
            t = self.env.tool_pos(a)
            goals[a] = np.array([t[0], t[1], ctx.get("z_h", Z_COOP)])
        self.move(goals, tol=0.02, max_steps=80)
        self._go_stow(arms)
        # 작업자가 수거: 실패한 물체를 셀 밖으로 치움 (정상 분류로 집계하지 않음)
        p.resetBasePositionAndOrientation(self.env.items[ctx["id"]]["uid"], [0.8, 0.8, 0.05], [0, 0, 0, 1])
        ctx["dest"] = None
        return State.DONE

    # ---------------- 실행 ----------------
    def process(self, jev: JEVMockData):
        uid = self.env.spawn_item(jev.item_id, jev.item_type)
        ctx = {"jev": jev, "id": jev.item_id, "half": self.env.items[jev.item_id]["half"], "attempts": 0}
        state = State.JEV_CLASSIFY
        while state != State.DONE:
            self._status(ctx, state)
            try:
                state = getattr(self, f"s_{state.name}")(ctx)
            except GraspFailure as e:
                ctx["attempts"] += 1
                ctx["fault"] = str(e)
                if ctx["attempts"] < 2 and state in (State.GRASP_VERIFY, State.LIFT_TRANSPORT):
                    for a in "AB":   # 1회 재시도: 놓고 다시 접근
                        self.jaw_q[a] = JAW_WIDE_Q
                        self.magnet.off(a)
                    self._tick(60)
                    state = State.APPROACH
                else:
                    state = State.FAULT
        self._status(ctx, State.DONE)
        self.env.tick(30)
        pos = p.getBasePositionAndOrientation(uid)[0]
        expected = ctx["dest"]
        inside = False
        if expected:
            b = BINS[expected]
            inside = (abs(pos[0] - b["center"][0]) < b["half"] and abs(pos[1] - b["center"][1]) < b["half"]
                      and pos[2] < BIN_WALL_H + 0.02)
        rec = dict(item_id=jev.item_id, type=jev.item_type.value, visual=jev.visual_status.value, ocr=jev.ocr_text,
                   route=ctx["route"], arms=ctx["arms"], grasp=ctx["mode"], dest=expected, in_bin=bool(inside),
                   r_int_mohm=round(ctx["r_int"] * 1000, 1) if "r_int" in ctx else None,
                   r_true_mohm=round(ctx["r_true"] * 1000, 1) if "r_true" in ctx else None,
                   forces={a: {k: round(v, 2) for k, v in f.items()} for a, f in ctx.get("forces", {}).items()},
                   final_xyz=[round(float(x), 3) for x in pos], drop_xy=[round(float(x), 3) for x in ctx.get("drop_xy", [])],
                   reach_err_mm=ctx.get("reach_err_mm"), attempts=ctx["attempts"] + 1,
                   fault=ctx.get("fault") if (expected is None or ctx["attempts"]) else None)
        self.log.append(rec)
        return rec
