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

    def reach(self, goals, tol=0.012, max_steps=140):
        self._hold()
        return self.rc.reach(goals, tol=tol, max_steps=max_steps, hold=self._hold, lock_roll=True)

    def move(self, goals, tol=0.012, step=0.05, max_steps=140):
        """현재 툴 위치에서 목표까지 직선 웨이포인트(간격 step)를 만들고 각 점을 학습 정책이 추종."""
        goals = {a: np.asarray(g, float) for a, g in goals.items()}
        cur = {a: self.env.tool_pos(a) for a in goals}
        n = max(1, int(np.ceil(max(np.linalg.norm(goals[a] - cur[a]) for a in goals) / step)))
        err = {}
        for i in range(1, n + 1):
            wp = {a: cur[a] + (goals[a] - cur[a]) * i / n for a in goals}
            last = i == n
            err = self.reach(wp, tol=tol if last else 0.02, max_steps=max_steps if last else 50)
        return err

    def _tick(self, n):
        for _ in range(n):
            self._hold()
            self.env.tick()

    def _align(self, arms, heading, tol=0.03, steps=80):
        for _ in range(steps):
            worst = 0.0
            for a in arms:
                r = self.env.robots[a]
                _, Rm = tool_axes(r)
                h = np.arctan2(Rm[1, 0], Rm[0, 0])
                d = (heading - h + np.pi / 2) % np.pi - np.pi / 2
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
        ctx["mode"] = "magnet" if ctx["route"] == "COIN_SORT" else ("coop-pinch" if large else "pinch")
        return State.APPROACH

    def _plan(self, ctx):
        """arm -> 파지 계획(핀치 위치, 개구, 툴 z). 파지 중심은 물체 중심 기준."""
        item = self.env.items[ctx["id"]]
        c = self.env.item_pos(ctx["id"])
        hx, hy, hz = item["half"]
        full_h = 2 * hz
        plan = {}
        for a in ctx["arms"]:
            if ctx["mode"] == "magnet":
                plan[a] = dict(kind="magnet", xy=c[:2].copy(), z=2 * hz + 0.003 + MAGNET_OFFSET[1])
                continue
            dx = 0.0
            if ctx["mode"] == "coop-pinch":
                delta = {ItemType.AUX_PACK: 0.035, ItemType.CAN_ALUMINUM: 0.026, ItemType.CAN_FERROUS: 0.026}.get(ctx["jev"].item_type, 0.022)
                dx = -delta if a == "A" else delta
            if item["half"][0] == item["half"][1] and ctx["jev"].item_type in (ItemType.CAN_ALUMINUM, ItemType.CAN_FERROUS):
                chord = 2 * np.sqrt(max(hx ** 2 - dx ** 2, 1e-6))
            else:
                chord = 2 * hy       # 핀치축=월드 y -> y방향 폭
            O = float(np.clip(chord + 0.030, 0.012, 0.079))
            zb = 0.004 if full_h < 0.03 else 0.010   # 패드 하단 높이
            plan[a] = dict(kind="pinch", xy=c[:2] + np.array([dx, 0.0]), O=O, qo=jaw_q_for_opening(O), z=zb + 0.006)
        return plan

    def _tool_goal(self, a, pl, z):
        """정렬된 툴 자세에서 핀치(자석) 지점이 목표 xy가 되도록 툴 프레임 목표 계산."""
        _, Rm = tool_axes(self.env.robots[a])
        off = (MAGNET_OFFSET[0] if pl["kind"] == "magnet" else FIXED_FACE_X - pl["O"] / 2)
        xy = pl["xy"] - Rm[:2, 0] * off
        return np.array([xy[0], xy[1], z])

    def s_APPROACH(self, ctx):
        plan = self._plan(ctx)
        ctx["plan"] = plan
        arms = ctx["arms"]
        z_h = max(Z_COOP if len(arms) > 1 else Z_HOVER, 2 * ctx["half"][2] + 0.075)   # 물체 윗면보다 충분히 위
        ctx["z_h"] = z_h
        for a in arms:
            self.jaw_q[a] = JAW_CLOSE_Q   # 이동 중에는 조를 닫아 부피를 줄임
        self.move({a: [plan[a]["xy"][0], plan[a]["xy"][1], z_h] for a in arms}, tol=0.015)
        for a in arms:
            self.jaw_q[a] = plan[a].get("qo", JAW_CLOSE_Q)
        self._tick(70)
        self._align(arms, PINCH_HEADING)
        self.move({a: self._tool_goal(a, plan[a], z_h) for a in arms}, tol=0.008)
        self.move({a: self._tool_goal(a, plan[a], 0.09) for a in arms}, tol=0.006, max_steps=120)
        err = self.move({a: self._tool_goal(a, plan[a], plan[a]["z"]) for a in arms}, tol=0.005, max_steps=140)
        ctx["reach_err_mm"] = {a: round(v * 1000, 1) for a, v in err.items()}
        ctx["tool_xy_off"] = {a: self.env.tool_pos(a)[:2] - self.env.item_pos(ctx["id"])[:2] for a in arms}
        return State.GRASP_VERIFY

    def s_GRASP_VERIFY(self, ctx):
        arms, plan = ctx["arms"], ctx["plan"]
        for a in arms:
            if plan[a]["kind"] == "pinch":
                self.jaw_q[a] = JAW_CLOSE_Q
            else:
                self.magnet.on(a, ctx["id"])
        self._tick(150)
        forces = {}
        for a in arms:
            if plan[a]["kind"] == "pinch":
                forces[a] = contact_forces(self.env, a, ctx["id"])
        ctx["forces"] = forces
        bad = {a: f for a, f in forces.items() if min(f["fixed"], f["jaw"]) < MIN_FORCE}
        if bad:
            raise GraspFailure(f"contact force below {MIN_FORCE}N: { {a: {k: round(v, 2) for k, v in f.items()} for a, f in bad.items()} }")
        return State.LIFT_TRANSPORT

    def _lift(self, ctx):
        arms, plan = ctx["arms"], ctx["plan"]
        z0 = self.env.item_pos(ctx["id"])[2]
        z_up = ctx["z_h"]
        goals = {}
        for a in arms:
            t = self.env.tool_pos(a)
            goals[a] = np.array([t[0], t[1], z_up])
        self.move(goals, tol=0.010, max_steps=100)
        self._tick(40)
        z1 = self.env.item_pos(ctx["id"])[2]
        if z1 - z0 < 0.6 * (z_up - plan[arms[0]]["z"]):
            raise GraspFailure(f"slip during lift: item rose {z1 - z0:.3f}m")
        hz = self.env.items[ctx["id"]]["half"][2]
        ctx["bottom_off"] = self.env.tool_pos(arms[0])[2] - (z1 - hz)   # 툴 z - 물체 바닥 z

    def _carry_to(self, ctx, dest):
        arms = ctx["arms"]
        z = ctx["z_h"]
        cx, cy = self._bin_xy(dest)
        ctx["drop_xy"] = (cx, cy)
        goals = {a: np.array([cx + ctx["tool_xy_off"][a][0], cy + ctx["tool_xy_off"][a][1], z]) for a in arms}
        ctx["hover"] = goals
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

    def s_RETURN_HOME(self, ctx):
        self.rc.stow("AB", hold=self._hold)
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
        for a in 'AB':
            self.jaw_q[a] = JAW_CLOSE_Q
        self.rc.stow("AB", hold=self._hold)
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
