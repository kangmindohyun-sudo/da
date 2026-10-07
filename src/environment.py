"""PyBullet 셀 환경: 천장 역거치 Dual SO-101(실제 URDF), 수거함, 배터리/이물질 스폰, 렌더링."""

import math

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from .jev_mock import ItemType
from .robot import CEILING_Z, SO101, SUBSTEPS, TOOL_LINK

SPAWN_XY = (0.0, 0.0)
ARM_BASES = {"A": ((-0.22, 0.0), 0.0), "B": ((0.22, 0.0), math.pi)}  # A는 +x, B는 -x를 향함

# (형상, 치수[반지름,높이]|[반치수], 색, 질량)  코인은 시각화를 위해 실물 대비 1.5배
ITEM_SPECS = {
    ItemType.BATTERY_18650: ("cyl", (0.009, 0.065), (0.2, 0.4, 0.9, 1), 0.045),
    ItemType.BATTERY_AA_AAA: ("cyl", (0.007, 0.050), (0.25, 0.75, 0.3, 1), 0.023),
    ItemType.COIN_CR: ("cyl", (0.015, 0.005), (0.95, 0.6, 0.15, 1), 0.003),
    ItemType.COIN_SR: ("cyl", (0.012, 0.004), (0.85, 0.85, 0.9, 1), 0.0015),
    ItemType.COIN_LR: ("cyl", (0.009, 0.006), (0.6, 0.35, 0.8, 1), 0.001),
    ItemType.AUX_PACK: ("box", (0.050, 0.027, 0.010), (0.3, 0.3, 0.35, 1), 0.12),
    ItemType.CAN_ALUMINUM: ("cyl", (0.033, 0.123), (0.8, 0.8, 0.85, 1), 0.015),
    ItemType.CAN_FERROUS: ("cyl", (0.033, 0.123), (0.55, 0.45, 0.4, 1), 0.02),
    ItemType.UNKNOWN_WASTE: ("box", (0.030, 0.030, 0.030), (0.6, 0.5, 0.2, 1), 0.05),
}

BINS = {
    "isolation": dict(center=(0.0, -0.16), half=0.065, color=(0.85, 0.15, 0.15, 1), label="FIRE ISOLATION"),
    "manual": dict(center=(0.0, 0.26), half=0.065, color=(0.95, 0.8, 0.1, 1), label="MANUAL LINE"),
    "reuse": dict(center=(-0.095, 0.185), half=0.042, color=(0.2, 0.7, 0.3, 1), label="REUSE"),
    "recycle": dict(center=(-0.095, -0.055), half=0.040, color=(0.2, 0.45, 0.85, 1), label="RECYCLE"),
    "coin_CR": dict(center=(-0.095, 0.285), half=0.028, color=(0.95, 0.5, 0.1, 1), label="CR Li"),
    "coin_SR": dict(center=(-0.095, 0.10), half=0.028, color=(0.75, 0.75, 0.8, 1), label="SR Ag"),
    "coin_LR": dict(center=(-0.095, 0.035), half=0.028, color=(0.6, 0.35, 0.8, 1), label="LR Alk"),
}
BIN_WALL_H = 0.04


class Environment:
    def __init__(self, gui=False, width=960, height=640, frame_every=12):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(1 / 240, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(numSolverIterations=150, enableConeFriction=1, contactBreakingThreshold=0.001,
                                    physicsClientId=self.cid)
        self.width, self.height, self.frame_every = width, height, frame_every
        self.frames, self.tick_count = [], 0
        self.status = {}
        self.items = {}
        self.robots = {}
        self.hooks = []
        self._build_scene()

    def _build_scene(self):
        he = [0.9, 0.7, 0.01]
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.9, 0.7, 0.10])   # 두꺼운 바닥(고속 물체 터널링 방지)
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=[0.86, 0.86, 0.84, 1])
        p.createMultiBody(0, cs, vs, [0, 0, -0.10])
        p.createMultiBody(0, -1, vs, [0, 0, -0.01])
        pv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.002, rgbaColor=[0.25, 0.25, 0.28, 1])
        p.createMultiBody(0, -1, pv, [*SPAWN_XY, 0.001])  # 진단 플레이트(투입 위치)
        fv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.40, 0.07, 0.008], rgbaColor=[0.35, 0.35, 0.4, 1])
        p.createMultiBody(0, -1, fv, [0, 0, CEILING_Z + 0.010])  # 천장 프레임
        for n, (xy, yaw) in ARM_BASES.items():
            self.robots[n] = SO101(xy, yaw, self.cid)
        for name, b in BINS.items():
            self._make_bin(b["center"], b["half"], b["color"])
        from .grasp import setup_friction
        setup_friction(self)

    def _make_bin(self, center, half, color):
        t = 0.004
        parts = [((0, 0, t / 2), (half, half, t / 2))]
        for sx, sy, hx, hy in [(1, 0, t / 2, half), (-1, 0, t / 2, half), (0, 1, half, t / 2), (0, -1, half, t / 2)]:
            parts.append(((sx * half, sy * half, BIN_WALL_H / 2), (hx, hy, BIN_WALL_H / 2)))
        for off, he in parts:
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=color)
            p.createMultiBody(0, cs, vs, [center[0] + off[0], center[1] + off[1], off[2]])

    def spawn_item(self, item_id, item_type, xy=SPAWN_XY, yaw=0.0):
        shape, dims, color, mass = ITEM_SPECS[item_type]
        if shape == "cyl":
            r, h = dims
            cs = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
            vs = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
            half = (r, r, h / 2)
        else:
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=color)
            half = dims
        uid = p.createMultiBody(mass, cs, vs, [xy[0], xy[1], half[2] + 0.001], p.getQuaternionFromEuler([0, 0, yaw]))
        p.changeDynamics(uid, -1, lateralFriction=3.0, linearDamping=0.05, angularDamping=0.1, restitution=0.0,
                         ccdSweptSphereRadius=min(half) * 0.8, contactProcessingThreshold=0.0)
        self.items[item_id] = dict(uid=uid, half=half, type=item_type)
        self.tick(30)
        return uid

    def item_pos(self, item_id):
        return np.array(p.getBasePositionAndOrientation(self.items[item_id]["uid"])[0])

    def tool_pos(self, arm):
        return self.robots[arm].tool()[0]

    def tick(self, n=1):
        for _ in range(n):
            for h in self.hooks:
                h()
            p.stepSimulation(physicsClientId=self.cid)
            self.tick_count += 1
            if self.frame_every and self.tick_count % self.frame_every == 0:
                self.frames.append(self.render())
                if len(self.frames) > 2400:   # 메모리 보호: 프레임을 절반으로 솎고 간격을 2배로
                    self.frames = self.frames[::2]
                    self.frame_every *= 2

    def arm_clearance(self):
        pts = p.getClosestPoints(self.robots["A"].id, self.robots["B"].id, 0.12, physicsClientId=self.cid)
        return min((c[8] for c in pts), default=0.12)

    def contact_distance(self, arm, item_id):
        """그리퍼(전 링크) - 물체 최근접 거리 [m]. 접점 검증용."""
        pts = p.getClosestPoints(self.robots[arm].id, self.items[item_id]["uid"], 0.05, physicsClientId=self.cid)
        return min((c[8] for c in pts if c[3] >= 4), default=1.0)  # 링크 4(wrist_roll)~: 손목/그리퍼/조

    def _project(self, pt, view, proj):
        v = np.array(view).reshape(4, 4).T @ np.array([*pt, 1.0])
        c = np.array(proj).reshape(4, 4).T @ v
        return None if c[3] <= 0 else ((c[0] / c[3] + 1) / 2 * self.width, (1 - c[1] / c[3]) / 2 * self.height)

    def render(self):
        view = p.computeViewMatrixFromYawPitchRoll([-0.02, 0.0, 0.17], 0.95, 0, -38, 0, 2)
        proj = p.computeProjectionMatrixFOV(55, self.width / self.height, 0.05, 5)
        img = p.getCameraImage(self.width, self.height, view, proj, renderer=p.ER_TINY_RENDERER,
                               physicsClientId=self.cid)[2]
        im = Image.fromarray(np.reshape(img, (self.height, self.width, 4))[:, :, :3].astype(np.uint8))
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        for b in BINS.values():
            pt = self._project((b["center"][0], b["center"][1], BIN_WALL_H + 0.01), view, proj)
            if pt:
                d.text((pt[0] - 24, pt[1] - 4), b["label"], fill=(0, 0, 0), font=font)
        y = 8
        for line in self.status.get("lines", []):
            d.text((10, y), line, fill=(0, 0, 0), font=font)
            y += 14
        return np.array(im)

    def save_video(self, path, fps=20):
        import imageio
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
        for f in self.frames:
            w.append_data(f)
        w.close()

    def close(self):
        p.disconnect(self.cid)
