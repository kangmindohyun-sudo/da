"""헤딩 조건부 도달 정책 평가: 위치 오차와 헤딩 오차 분포"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from src.rl_env import ReachEnv
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="models/v7"); ap.add_argument("--n", type=int, default=60)
a = ap.parse_args()
m = PPO.load(a.model + "_ppo", device="cpu"); vn = pickle.load(open(a.model + "_vecnorm.pkl", "rb"))
env = ReachEnv(seed=7, hold=True, head=True)
pe, he = [], []
for ep in range(a.n):
    o, _ = env.reset()
    for t in range(120):
        on = np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -10, 10)
        act, _ = m.predict(on, deterministic=True)
        o, r, te, tr, info = env.step(act)
        if tr: break
    pe.append(info["d"] * 1000); he.append(abs(env._dh()))
pe, he = np.array(pe), np.degrees(np.array(he))
print(f"pos err mm: median {np.median(pe):.1f} p75 {np.percentile(pe,75):.1f} | within 8mm {np.mean(pe<8)*100:.0f}%")
print(f"heading err deg: median {np.median(he):.1f} p75 {np.percentile(he,75):.1f} | within 9deg {np.mean(he<9)*100:.0f}%")
print(f"both: {np.mean((pe<8)&(he<9))*100:.0f}%")
