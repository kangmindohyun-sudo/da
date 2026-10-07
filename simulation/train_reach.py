"""SO-101 도달 정책 PPO 학습: python simulation/train_reach.py --steps 3000000"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from src.rl_env import ReachEnv


TAG = "reach"


class Log(BaseCallback):
    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.model.ep_hist = getattr(self.model, "ep_hist", [])
                self.model.ep_hist.append(bool(info.get("success", False)))
        if self.n_calls % 2000 == 0 and getattr(self.model, "ep_hist", None):
            h = self.model.ep_hist[-200:]
            print(f"steps={self.num_timesteps} success_rate(last {len(h)} eps)={np.mean(h):.2f}", flush=True)
        if self.n_calls % 20000 == 0:
            self.model.save(f"models/{TAG}_ppo"); self.training_env.save(f"models/{TAG}_vecnorm.pkl")
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--resume", default="")
    ap.add_argument("--hold", action="store_true")
    ap.add_argument("--lock-roll", action="store_true")
    ap.add_argument("--head", action="store_true", help="목표 헤딩(핀치축 방향) 조건부 정책")
    ap.add_argument("--tag", default="reach")
    a = ap.parse_args()
    TAG = a.tag
    torch.set_num_threads(1)
    env = make_vec_env(lambda: ReachEnv(hold=a.hold, lock_roll=a.lock_roll, head=a.head), n_envs=a.envs, vec_env_cls=SubprocVecEnv, seed=0)
    model = None
    if a.resume:
        import pickle
        old_vn = pickle.load(open(a.resume + "_vecnorm.pkl", "rb"))
        n_old = len(old_vn.obs_rms.mean)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        n_new = env.observation_space.shape[0]
        env.obs_rms.mean[:n_old] = old_vn.obs_rms.mean
        env.obs_rms.var[:n_old] = old_vn.obs_rms.var
        env.obs_rms.count = old_vn.obs_rms.count
        old = PPO.load(a.resume + "_ppo", device="cpu")
        model = PPO("MlpPolicy", env, n_steps=512, batch_size=512, n_epochs=8, gamma=0.98, gae_lambda=0.95,
                    learning_rate=1.5e-4, ent_coef=0.0, clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256]),
                    device="cpu", seed=0, verbose=0)
        sd_old, sd_new = old.policy.state_dict(), model.policy.state_dict()
        for k, v in sd_new.items():
            if k in sd_old and sd_old[k].shape == v.shape:
                sd_new[k] = sd_old[k].clone()
            elif k in sd_old and v.dim() == 2 and v.shape[1] == n_new and sd_old[k].shape[1] == n_old:
                z = v.clone() * 0.0   # 새 입력(헤딩) 가중치는 0에서 시작
                z[:, :n_old] = sd_old[k]
                sd_new[k] = z
        model.policy.load_state_dict(sd_new)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = model or PPO("MlpPolicy", env, n_steps=512, batch_size=512, n_epochs=8, gamma=0.98, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256]),
                device="cpu", seed=0, verbose=0)
    model.learn(a.steps, callback=Log())
    model.save(f"models/{a.tag}_ppo"); env.save(f"models/{a.tag}_vecnorm.pkl")
    print("done")
