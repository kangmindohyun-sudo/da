"""시뮬레이션 실행: python simulation/run_sim.py [--gui] [--out out/sim.mp4]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Environment
from src.agent import LearnedReacher
from src.fsm import SortingFSM
from src.jev_mock import ItemType as T, JEVMockData, VisualStatus as V


def scenario():
    """(JEV Mock, 숨겨진 실제 내부저항[Ω] - 시뮬레이션 전용 ground truth)"""
    mk = lambda i, t, v=V.NORMAL, ocr=None: JEVMockData(f"ITEM_2026_{i:04d}", t, v, ocr)
    return [
        (mk(101, T.BATTERY_18650), 0.045),                       # 건강 -> REUSE
        (mk(102, T.BATTERY_18650), 0.140),                       # 노화 -> RECYCLE
        (mk(103, T.BATTERY_AA_AAA), 0.150),                      # 알칼라인 양호 -> REUSE
        (mk(104, T.BATTERY_AA_AAA), 0.650),                      # 방전 -> RECYCLE
        (mk(105, T.BATTERY_18650, V.SWELLING), None),            # 스웰링 -> 소화 격리함
        (mk(106, T.COIN_CR, ocr="CR2032"), None),                # 리튬 코인
        (mk(107, T.COIN_SR, ocr="SR626SW"), None),               # 은 코인
        (mk(108, T.COIN_LR, ocr="LR44"), None),                  # 알칼라인 코인
        (mk(109, T.CAN_ALUMINUM), None),                         # 캔 -> 양팔 협동 -> 수작업
        (mk(110, T.AUX_PACK), None),                             # 보조배터리 -> 양팔 협동 -> 수작업
        (mk(111, T.CAN_FERROUS), None),
        (mk(112, T.COIN_CR, ocr="??"), None),                    # OCR 실패 -> 수작업
        (mk(113, T.UNKNOWN_WASTE), None),                        # 미확인 이물질 -> 양팔 -> 수작업
        (mk(114, T.AUX_PACK, V.SWELLING), None),                 # 부푼 보조배터리 -> 양팔 -> 소화 격리
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default="out/sim.mp4")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated item numbers, e.g. 101,106,109")
    ap.add_argument("--model", default="models/reach_ppo")
    ap.add_argument("--vecnorm", default="models/reach_vecnorm.pkl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    env = Environment(gui=a.gui, width=480, height=320, frame_every=0 if (a.gui or a.no_video) else 36)
    rc = LearnedReacher(env, a.model, a.vecnorm)
    sc = scenario()
    if a.only:
        keep = {f"ITEM_2026_{int(x):04d}" for x in a.only.split(",")}
        sc = [(j, r) for j, r in sc if j.item_id in keep]
    fsm = SortingFSM(env, rc, {j.item_id: r for j, r in sc if r is not None})
    events = []
    import numpy as np, pybullet as pb

    def watch():
        if env.tick_count % 12 == 0 and 'clr' not in seen:
            c = env.arm_clearance()
            if c < 0.0:
                seen.add('clr')
                events.append(dict(arm_collision=round(c * 1000, 1), tick=env.tick_count, tools={a: [round(float(x), 3) for x in env.tool_pos(a)] for a in 'AB'},
                                   q={a: [round(float(x), 2) for x in env.robots[a].q()] for a in 'AB'}))
        for iid, it in env.items.items():
            if iid in seen:
                continue
            v = np.linalg.norm(pb.getBaseVelocity(it["uid"], physicsClientId=env.cid)[0])
            if v > 3.0:
                seen.add(iid)
                cps = pb.getContactPoints(bodyA=it["uid"], physicsClientId=env.cid)
                events.append(dict(item=iid, tick=env.tick_count, speed=round(float(v), 1), pos=[round(float(x), 3) for x in env.item_pos(iid)],
                                   contacts=[(c[2], c[4], round(c[9], 1)) for c in cps][:5],
                                   tools={a: [round(float(x), 3) for x in env.tool_pos(a)] for a in 'AB'}))
    seen = set()
    env.hooks.append(watch)
    t0 = time.time()
    for jev, _ in sc:
        r = fsm.process(jev)
        print(f"{r['item_id']} {r['type']:12s} {r['route']:22s} arms={r['arms']:2s} {r['grasp']:10s} -> {str(r['dest']):10s} "
              f"in_bin={r['in_bin']} R={r['r_int_mohm']} forces={r['forces']} err={r['reach_err_mm']} final={r['final_xyz']} drop={r['drop_xy']} try={r['attempts']} {r['fault'] or ''}", flush=True)
    ok = sum(r["in_bin"] for r in fsm.log)
    print(f"\n{ok}/{len(fsm.log)} items verified inside destination bin | min arm clearance "
          f"{rc.min_clearance*100:.1f} cm | policy steps {rc.steps_used} | wall {time.time()-t0:.0f}s")
    for ev in events:
        print('BLOWUP', ev)
    json.dump(fsm.log, open(a.out.replace(".mp4", ".json"), "w"), indent=2, ensure_ascii=False)
    if env.frames:
        from PIL import Image
        gif = a.out.replace(".mp4", ".gif")
        ims = [Image.fromarray(f) for f in env.frames[::2]]
        ims[0].save(gif, save_all=True, append_images=ims[1:], duration=100, loop=0, optimize=True)
        print("gif:", gif, len(ims), "frames", flush=True)
        try:
            env.save_video(a.out)
            print("video:", a.out, len(env.frames), "frames")
        except Exception as e:
            print("mp4 skipped:", e)


if __name__ == "__main__":
    main()
