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
    ap.add_argument("--model", default="models/reach_ppo")
    ap.add_argument("--vecnorm", default="models/reach_vecnorm.pkl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    env = Environment(gui=a.gui, width=640, height=428, frame_every=0 if (a.gui or a.no_video) else 16)
    rc = LearnedReacher(env, a.model, a.vecnorm)
    sc = scenario()
    fsm = SortingFSM(env, rc, {j.item_id: r for j, r in sc if r is not None})
    t0 = time.time()
    for jev, _ in sc:
        r = fsm.process(jev)
        print(f"{r['item_id']} {r['type']:10s} {r['route']:22s} arms={r['arms']:2s} -> {r['dest']:10s} "
              f"in_bin={r['in_bin']} R={r['r_int_mohm']} contacts={r['contacts']} err={r['reach_err_mm']} {r['fault'] or ''}")
    ok = sum(r["in_bin"] for r in fsm.log)
    print(f"\n{ok}/{len(fsm.log)} items verified inside destination bin | min arm clearance "
          f"{rc.min_clearance*100:.1f} cm | policy steps {rc.steps_used} | wall {time.time()-t0:.0f}s")
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
