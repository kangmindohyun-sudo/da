"""도달 정책 정량 평가: python simulation/eval_reach.py [--n 200]"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from src.rl_env import ReachEnv

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--model", default="models/reach_ppo")
ap.add_argument("--vecnorm", default="models/reach_vecnorm.pkl")
ap.add_argument("--steps", type=int, default=140)
a = ap.parse_args()
m = PPO.load(a.model, device="cpu")
vn = pickle.load(open(a.vecnorm, "rb"))
env = ReachEnv(seed=123)
errs, downs, tsteps = [], [], []
for ep in range(a.n):
    o, _ = env.reset()
    for t in range(a.steps):
        on = np.clip((o - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + vn.epsilon), -10, 10)
        act, _ = m.predict(on, deterministic=True)
        o, r, te, tr, info = env.step(act)
        if tr: break
    errs.append(info["d"]); downs.append(info["down"])
errs = np.array(errs) * 1000
print(f"n={a.n} final error mm: median={np.median(errs):.1f} mean={errs.mean():.1f} p90={np.percentile(errs,90):.1f}")
for th in (12, 8, 5, 3): print(f"  within {th}mm: {np.mean(errs<th)*100:.0f}%")
print(f"  tool-down >0.85: {np.mean(np.array(downs)>0.85)*100:.0f}%")
