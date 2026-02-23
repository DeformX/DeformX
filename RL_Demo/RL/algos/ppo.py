"""
Pure PyTorch PPO (no Isaac/Omni imports).
- Vectorized storage (stores batches across num_envs)
- Fixed diagonal Gaussian policy (std from cfg.action_std)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device

        self.lr = float(cfg.lr)
        self.gamma = float(cfg.gamma)
        self.eps_clip = float(cfg.eps_clip)
        self.k_epochs = int(cfg.k_epochs)

        self.entropy_coef = float(cfg.entropy_coef)
        self.vf_coef = float(cfg.vf_coef)
        self.grad_clip = float(cfg.grad_clip)

        self.action_std = float(getattr(cfg, "action_std", 0.5))

        # Actor-Critic
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, self.act_dim),
            nn.Tanh(),
        ).to(self.device)

        self.critic = nn.Sequential(
            nn.Linear(self.obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        ).to(self.device)

        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr},
                {"params": self.critic.parameters(), "lr": self.lr},
            ]
        )
        self.mse = nn.MSELoss()

        self.clear_buffer()

    def save(self, path: str, *, step: int | None = None, extra: dict | None = None):
        payload = {
            "step": step,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path: str, *, map_location: str | torch.device | None = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt

    def clear_buffer(self):
        self.buffer_obs: List[torch.Tensor] = []
        self.buffer_act: List[torch.Tensor] = []
        self.buffer_logp: List[torch.Tensor] = []
        self.buffer_rew: List[torch.Tensor] = []
        self.buffer_done: List[torch.Tensor] = []

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Args:
            obs_np: (N, obs_dim)
        Returns:
            actions_np: (N, act_dim) in [-1, 1] due to tanh + sampling
            logp_cpu: (N,) torch tensor on CPU
        """
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        mean = self.actor(obs)  # (N, act_dim)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)

        # Fixed diagonal covariance
        cov = torch.diag(torch.full((self.act_dim,), self.action_std, device=self.device))
        dist = torch.distributions.MultivariateNormal(mean, cov)

        actions = dist.sample()
        logp = dist.log_prob(actions)  # (N,)

        return actions.detach().cpu().numpy(), logp.detach().cpu()

    def store(
        self,
        obs_np: np.ndarray,
        act_np: np.ndarray,
        logp_cpu: torch.Tensor,
        rew_np: np.ndarray,
        done_np: np.ndarray,
    ):
        # store as CPU tensors first; move to GPU in update()
        self.buffer_obs.append(torch.as_tensor(obs_np, dtype=torch.float32))
        self.buffer_act.append(torch.as_tensor(act_np, dtype=torch.float32))
        self.buffer_logp.append(logp_cpu.to(torch.float32))
        self.buffer_rew.append(torch.as_tensor(rew_np, dtype=torch.float32))
        self.buffer_done.append(torch.as_tensor(done_np, dtype=torch.float32))

    def update(self):
        """
        Flatten (T, N, ...) into (T*N, ...)
        Use simple reward normalization for stability (like your original).
        """
        if len(self.buffer_obs) == 0:
            return

        # Shapes:
        # obs: (T, N, obs_dim)
        obs = torch.stack(self.buffer_obs, dim=0).to(self.device)
        act = torch.stack(self.buffer_act, dim=0).to(self.device)
        logp_old = torch.stack(self.buffer_logp, dim=0).to(self.device)  # (T, N)
        rew = torch.stack(self.buffer_rew, dim=0).to(self.device)        # (T, N)
        done = torch.stack(self.buffer_done, dim=0).to(self.device)      # (T, N)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        act = torch.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)
        logp_old = torch.nan_to_num(logp_old, nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        done = torch.nan_to_num(done, nan=1.0, posinf=1.0, neginf=1.0)

        T, N, _ = obs.shape

        # ----- compute returns (discounted, reset on done) -----
        returns = torch.zeros((T, N), device=self.device, dtype=torch.float32)
        running = torch.zeros((N,), device=self.device, dtype=torch.float32)
        for t in reversed(range(T)):
            running = rew[t] + self.gamma * running * (1.0 - done[t])
            returns[t] = running

        # normalize returns (helps stability in this simple PPO)
        flat_returns = returns.view(-1)
        ret_std = flat_returns.std(unbiased=False)
        if torch.isfinite(ret_std) and float(ret_std.item()) > 0.0:
            flat_returns = (flat_returns - flat_returns.mean()) / (ret_std + 1e-7)
        returns = torch.nan_to_num(flat_returns.view(T, N), nan=0.0, posinf=0.0, neginf=0.0)

        # flatten
        obs_f = obs.view(T * N, -1)
        act_f = act.view(T * N, -1)
        logp_old_f = logp_old.view(T * N)
        ret_f = returns.view(T * N)

        # PPO epochs
        for _ in range(self.k_epochs):
            mean = self.actor(obs_f)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
            cov = torch.diag(torch.full((self.act_dim,), self.action_std, device=self.device))
            dist = torch.distributions.MultivariateNormal(mean, cov)

            logp = dist.log_prob(act_f)
            entropy = dist.entropy().mean()

            values = self.critic(obs_f).squeeze(-1)
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            ratios = torch.exp(logp - logp_old_f)
            ratios = torch.nan_to_num(ratios, nan=1.0, posinf=10.0, neginf=0.0)

            adv = ret_f - values.detach()
            adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.mse(values, ret_f)

            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy
            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            finite_grad = True
            for p in list(self.actor.parameters()) + list(self.critic.parameters()):
                if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                    finite_grad = False
                    break
            if not finite_grad:
                self.optimizer.zero_grad(set_to_none=True)
                continue
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.optimizer.step()

        self.clear_buffer()
