"""작업 영역 도달 가능 목표 집합: 무작위 관절 FK(툴 하향) 샘플을 3D 복셀로 밀도 평탄화. 출력 models/goals.npy"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pybullet as p
from src.robot import SO101, ARM_JOINTS

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
    cid = p.connect(p.DIRECT)
    r = SO101(client=cid)
    rng = np.random.default_rng(0)
    vox, keep = {}, []
    for i in range(n):
        q = rng.uniform(r.lo, r.hi)
        for j, v in zip(ARM_JOINTS, q): p.resetJointState(r.id, j, v, 0.0, physicsClientId=cid)
        pos, zax = r.tool()
        hd = r.heading()
        if zax[2] > -0.9 or not (-0.2 <= pos[0] <= 0.36 and -0.32 <= pos[1] <= 0.32 and 0.005 <= pos[2] <= 0.2):
            continue
        k = tuple((pos // 0.015).astype(int))
        if vox.get(k, 0) < 8:
            vox[k] = vox.get(k, 0) + 1; keep.append(np.array([pos[0], pos[1], pos[2], hd]))
        if i % 300000 == 0: print(i, len(keep), flush=True)
    np.save("models/goals_head.npy", np.array(keep))
    print("saved", len(keep), "goals; voxels", len(vox))
