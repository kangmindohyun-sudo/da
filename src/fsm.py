"""FSM 파이프라인: JEV Mock -> 라우팅 -> (학습 정책으로) 접근/파지(접점 검증) -> 이송 -> (펄스 부하 R_int) -> 분류 투입.
FSM은 '어디로 갈지(목표 좌표)'만 정하고, 관절 구동은 PPO로 학습된 도달 정책(LearnedReacher)이 수행한다."""

import random
from enum import Enum, auto

import numpy as np
import pybullet as p

from .agent import LearnedReacher
from .environment import BIN_WALL_H, BINS
from .jev_mock import ItemType, JEVMockData, JEVProcessor
from .robot import JAW_JOINT, TOOL_LINK

Z_HOVER = 0.13
Z_CARRY_COOP = 0.15
LARGE_SPAN = 0.05          # 이 이상이면 양팔 협동 파지 (SO-101 조 개구 한계)
SIDE_GAP = 0.010           # 협동 파지: 툴 프레임-물체 측면 간격 [m]
CONTACT_TOL = 0.008        # 그리퍼-물체 접점 판정 거리 [m]
JAW_OPEN, JAW_CLOSED = 1.0, 0.1
PULSE_S = 0.1              # 동적 펄스 부하 시간
# 종류별 (OCV[V], 펄스전류[A], 재사용 R_int 임계[Ω])
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
        self.constraints = []
        self.jaw = {"A": JAW_OPEN, "B": JAW_OPEN}

    # ---------------- 유틸 ----------------
    def _hold(self):
        for a, r in self.env.robots.items():
            p.setJointMotorControl2(r.id, JAW_JOINT, p.POSITION_CONTROL, targetPosition=self.jaw[a], force=3.0,
                                    physicsClientId=self.env.cid)

    def reach(self, goals, tol=0.012, max_steps=140):
        self._hold()
        return self.rc.reach(goals, tol=tol, max_steps=max_steps, hold=self._hold)

    def _status(self, ctx, state):
        j = ctx["jev"]
        lines = [f"ITEM {j.item_id}  type={j.item_type.value}  visual={j.visual_status.value}"
                 + (f"  ocr={j.ocr_text}" if j.ocr_text else ""),
                 f"STATE {state.name}   route={ctx.get('route', '-')}   arms={ctx.get('arms', '-')}"]
        if "r_int" in ctx:
            lines.append(f"PULSE 0.1s  dV={ctx['dv']*1000:.0f}mV  I={ctx['i_load']:.1f}A  R_int={ctx['r_int']*1000:.0f} mOhm"
                         f"  (thr {ctx['thr']*1000:.0f}) -> {ctx.get('dest', '?')}")
        if "contacts" in ctx:
            lines.append("CONTACT " + "  ".join(f"{a}:{'OK' if v else 'NO'}({d*1000:.1f}mm)" for a, (v, d) in ctx["contacts"].items()))
        lines.append("BINS " + "  ".join(f"{k}={v}" for k, v in self.counts.items() if v))
        lines.append(f"policy steps {self.rc.steps_used}   arm clearance min {self.rc.min_clearance*100:.1f}cm")
        self.env.status["lines"] = lines

    def _attach(self, arm, item_id):
        e = self.env
        r = e.robots[arm]
        item = e.items[item_id]["uid"]
        ls = p.getLinkState(r.id, TOOL_LINK, computeForwardKinematics=True, physicsClientId=e.cid)
        pad, ee_orn = ls[4], ls[5]
        ipos, iorn = p.getBasePositionAndOrientation(item)
        inv_p, inv_o = p.invertTransform(ipos, iorn)
        local, q = p.multiplyTransforms(inv_p, inv_o, list(pad), ee_orn)
        c = p.createConstraint(r.id, TOOL_LINK, item, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], local,
                               childFrameOrientation=p.invertTransform([0, 0, 0], q)[1])
        p.changeConstraint(c, maxForce=300)
        self.constraints.append(c)

    def _release_all(self):
        for c in self.constraints:
            p.removeConstraint(c)
        self.constraints.clear()

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
        return State.APPROACH

    def _grasp_points(self, ctx):
        item = self.env.items[ctx["id"]]
        c = self.env.item_pos(ctx["id"])
        hx, hy, hz = item["half"]
        if ctx["arms"] == "AB":
            off = hx + SIDE_GAP
            zc = min(max(hz, 0.03), 0.07)
            return {"A": np.array([c[0] - off, c[1], zc]), "B": np.array([c[0] + off, c[1], zc])}
        return {ctx["arms"]: np.array([c[0], c[1], max(2 * hz - 0.012, 0.012)])}

    def s_APPROACH(self, ctx):
        pts = self._grasp_points(ctx)
        ctx["pts"] = pts
        for a in ctx["arms"]:
            self.jaw[a] = JAW_OPEN
        z_h = Z_CARRY_COOP if ctx["arms"] == "AB" else Z_HOVER
        self.reach({a: [pt[0], pt[1], z_h] for a, pt in pts.items()}, tol=0.012)
        err = self.reach(pts, tol=0.008, max_steps=100)
        ctx["reach_err_mm"] = {a: round(v * 1000, 1) for a, v in err.items()}
        return State.GRASP_VERIFY

    def s_GRASP_VERIFY(self, ctx):
        res = {}
        for a in ctx["arms"]:
            d = self.env.contact_distance(a, ctx["id"])
            res[a] = (d <= CONTACT_TOL, d)
        ctx["contacts"] = res
        if not all(v for v, _ in res.values()):
            raise GraspFailure(f"contact check failed { {a: round(d*1000,1) for a,(v,d) in res.items()} }")
        for a in ctx["arms"]:
            self.jaw[a] = JAW_CLOSED
            self._attach(a, ctx["id"])
        self._hold()
        self.env.tick(30)
        return State.LIFT_TRANSPORT

    def _bin_xy(self, dest):
        cx, cy = BINS[dest]["center"]
        j = BINS[dest]["half"] * 0.35
        return cx + self.rng.uniform(-j, j), cy + self.rng.uniform(-j, j)

    def _carry_to(self, ctx, dest):
        pts = ctx["pts"]
        z = Z_CARRY_COOP if ctx["arms"] == "AB" else Z_HOVER
        c0 = self.env.item_pos(ctx["id"])
        self.reach({a: [pt[0], pt[1], z] for a, pt in pts.items()}, tol=0.012)  # 수직 상승
        cx, cy = self._bin_xy(dest)
        ctx["drop_xy"] = (cx, cy)
        hover = {a: np.array([cx + (pt[0] - c0[0]), cy + (pt[1] - c0[1]), z]) for a, pt in pts.items()}
        ctx["hover"] = hover
        self.reach(hover, tol=0.012, max_steps=180)

    def s_LIFT_TRANSPORT(self, ctx):
        if ctx["cell"]:
            self.reach({a: [pt[0], pt[1], Z_HOVER] for a, pt in ctx["pts"].items()}, tol=0.012)
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
            self.env.tick()
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
        hz = self.env.items[ctx["id"]]["half"][2]
        coop = ctx["arms"] == "AB"
        # 물체 바닥이 수거함 벽 위 약 2cm가 되도록 하강
        grasp_z = max(hz, 0.03) if coop else max(2 * hz - 0.012, 0.012)
        dz = BIN_WALL_H + 0.02 + (grasp_z if coop else grasp_z)
        drop = {a: np.array([h[0], h[1], dz]) for a, h in ctx["hover"].items()}
        self.reach(drop, tol=0.010, max_steps=100)
        self._release_all()
        for a in ctx["arms"]:
            self.jaw[a] = JAW_OPEN
        self._hold()
        self.env.tick(60)
        self.reach({a: h for a, h in ctx["hover"].items()}, tol=0.015, max_steps=60)
        self.env.tick(60)
        self.counts[ctx["dest"]] += 1
        return State.RETURN_HOME

    def s_RETURN_HOME(self, ctx):
        self.rc.stow("AB", hold=self._hold)
        return State.DONE

    def s_FAULT(self, ctx):
        self._release_all()
        for a in "AB":
            self.jaw[a] = JAW_OPEN
        self.rc.stow("AB", hold=self._hold)
        mx, my = BINS["manual"]["center"]
        p.resetBasePositionAndOrientation(self.env.items[ctx["id"]]["uid"], [mx, my, 0.03], [0, 0, 0, 1])
        ctx["dest"] = "manual"
        self.counts["manual"] += 1
        self.env.tick(60)
        return State.DONE

    # ---------------- 실행 ----------------
    def process(self, jev: JEVMockData):
        uid = self.env.spawn_item(jev.item_id, jev.item_type)
        ctx = {"jev": jev, "id": jev.item_id, "half": self.env.items[jev.item_id]["half"]}
        state = State.JEV_CLASSIFY
        while state != State.DONE:
            self._status(ctx, state)
            try:
                state = getattr(self, f"s_{state.name}")(ctx)
            except GraspFailure as e:
                ctx["fault"] = str(e)
                state = State.FAULT
        self._status(ctx, State.DONE)
        self.env.tick(30)
        pos = p.getBasePositionAndOrientation(uid)[0]
        b = BINS[ctx["dest"]]
        inside = (abs(pos[0] - b["center"][0]) < b["half"] and abs(pos[1] - b["center"][1]) < b["half"]
                  and pos[2] < BIN_WALL_H + 0.02)
        rec = dict(item_id=jev.item_id, type=jev.item_type.value, visual=jev.visual_status.value, ocr=jev.ocr_text,
                   route=ctx["route"], arms=ctx["arms"], dest=ctx["dest"], in_bin=bool(inside),
                   r_int_mohm=round(ctx["r_int"] * 1000, 1) if "r_int" in ctx else None,
                   r_true_mohm=round(ctx["r_true"] * 1000, 1) if "r_true" in ctx else None,
                   contacts={a: round(d * 1000, 2) for a, (v, d) in ctx.get("contacts", {}).items()},
                   reach_err_mm=ctx.get("reach_err_mm"), fault=ctx.get("fault"))
        self.log.append(rec)
        return rec
