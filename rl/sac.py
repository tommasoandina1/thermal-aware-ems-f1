"""
Soft Actor-Critic, with automatic entropy tuning and an optional Lagrangian
extension for the constrained (CMDP) formulation.

Why SAC rather than DDPG / TD3 / PPO
------------------------------------
* the action is continuous (electrical power command), which rules out the
  value-based family without an arbitrary discretisation;
* the algorithm is off-policy, so the replay buffer is reused across updates -
  this matters because one five-lap episode is 3886 plant evaluations;
* the policy is stochastic and trained with a maximum-entropy objective. The
  cost landscape of this problem has genuine discontinuities (deploy budget
  exhaustion, activation of thermal derating): structured entropic exploration
  is a better fit than the additive Ornstein-Uhlenbeck noise of DDPG, which
  tends to collapse onto whichever side of the discontinuity it starts on;
* it is empirically the least hyperparameter-sensitive member of the family.

TD3 remains the mandatory comparison baseline, and it is worth stating plainly
that SAC is not universally superior: at least one study in the process-control
literature reports an entropy-maximising TD3 variant outperforming both TD3 and
SAC on sample efficiency and convergence speed.

What is implemented
-------------------
* twin critics with a shared target and the clipped double-Q target (the "take
  the smaller of the two" rule that limits the overestimation bias);
* target networks updated by Polyak averaging;
* tanh-squashed Gaussian policy with the exact log-probability correction for
  the change of variables;
* automatic temperature tuning against a target entropy of -dim(A). The review
  presents SAC with a fixed alpha; in practice the automatic version is what is
  used, and it removes the single most sensitive hyperparameter;
* optional cost critic + Lagrange multiplier, which turns the objective into

      max_pi  E[sum r]   s.t.   E[sum c] <= d

  solved by dual gradient ascent on lambda. This is the practical form of the
  CMDP formulation: it removes the need to hand-calibrate the relative weight
  of the shortfall penalty inside the reward, replacing an arbitrary weight
  with an interpretable budget d expressed in MJ.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SACConfig

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
        elif out_act is not None:
            layers.append(out_act())
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.body = mlp([obs_dim, *hidden], act=nn.ReLU, out_act=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logp=True):
        h = self.body(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)

        a = torch.tanh(u)
        if not with_logp:
            return a, None

        # log-probability with the tanh change-of-variables correction, in the
        # numerically stable form: log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u))
        logp = (-0.5 * ((u - mu) / (std + 1e-8)) ** 2
                - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2.0) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return a, logp


class Critic(nn.Module):
    """Twin Q network. A single module holding both heads keeps the two
    optimisers, target copies and losses in step."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, obs_dim, act_dim, cfg: SACConfig | None = None, device=None):
        self.cfg = cfg or SACConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        h = tuple(self.cfg.hidden_sizes)

        self.actor = SquashedGaussianActor(obs_dim, act_dim, h).to(self.device)
        self.critic = Critic(obs_dim, act_dim, h).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # temperature
        self.target_entropy = (self.cfg.target_entropy
                               if self.cfg.target_entropy is not None else -float(act_dim))
        self.log_alpha = torch.tensor(np.log(self.cfg.init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=self.cfg.autotune_alpha)
        self.opt_alpha = (torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
                          if self.cfg.autotune_alpha else None)

        # constrained mode
        self.use_lag = self.cfg.use_lagrangian
        if self.use_lag:
            self.cost_critic = Critic(obs_dim, act_dim, h).to(self.device)
            self.cost_critic_targ = copy.deepcopy(self.cost_critic).requires_grad_(False)
            self.opt_cost = torch.optim.Adam(self.cost_critic.parameters(), lr=self.cfg.lr_critic)
            self.log_lambda = torch.tensor(np.log(self.cfg.lambda_init), dtype=torch.float32,
                                           device=self.device, requires_grad=True)
            self.opt_lambda = torch.optim.Adam([self.log_lambda], lr=self.cfg.lr_lambda)
            # running estimate of the undiscounted episodic cost, used for the
            # dual update (the multiplier must react to the *episode* budget,
            # not to a per-step quantity)
            self.ep_cost_est = 0.0

        self._it = 0

    # -- properties --------------------------------------------------------

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @property
    def lam(self):
        if not self.use_lag:
            return torch.zeros((), device=self.device)
        return torch.clamp(self.log_lambda.exp(), max=self.cfg.lambda_max).detach()

    # -- interaction -------------------------------------------------------

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logp=False)
        return a.squeeze(0).cpu().numpy()

    def policy(self, deterministic=True):
        """Adapter so the agent can be passed to rl.evaluate.rollout."""
        def _p(env, obs):
            return self.act(obs, deterministic=deterministic)
        return _p

    # -- learning ----------------------------------------------------------

    def update(self, batch):
        cfg, dev = self.cfg, self.device
        to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        obs, act, rew = to(batch["obs"]), to(batch["act"]), to(batch["rew"])
        nobs, done, w = to(batch["next_obs"]), to(batch["done"]), to(batch["weights"])
        cost = to(batch["cost"])

        # ---- critic -------------------------------------------------------
        with torch.no_grad():
            na, nlogp = self.actor(nobs)
            q1t, q2t = self.critic_targ(nobs, na)
            qt = torch.min(q1t, q2t) - self.alpha * nlogp
            y = rew + cfg.gamma * (1.0 - done) * qt

        q1, q2 = self.critic(obs, act)
        td1, td2 = q1 - y, q2 - y
        loss_q = (w * (td1 ** 2 + td2 ** 2)).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        info = dict(loss_q=float(loss_q.detach()), q_mean=float(q1.mean().detach()))
        td_err = 0.5 * (td1.abs() + td2.abs()).detach().cpu().numpy()

        # ---- cost critic --------------------------------------------------
        if self.use_lag:
            with torch.no_grad():
                na, _ = self.actor(nobs)
                c1t, c2t = self.cost_critic_targ(nobs, na)
                # for costs the *pessimistic* choice is the maximum
                yc = cost + cfg.gamma * (1.0 - done) * torch.max(c1t, c2t)
            c1, c2 = self.cost_critic(obs, act)
            loss_c = ((c1 - yc) ** 2 + (c2 - yc) ** 2).mean()
            self.opt_cost.zero_grad(set_to_none=True)
            loss_c.backward()
            self.opt_cost.step()
            info["loss_c"] = float(loss_c.detach())

        # ---- actor (delayed) ----------------------------------------------
        self._it += 1
        if self._it % max(1, cfg.policy_delay) == 0 and self._it >= cfg.actor_start_it:
            for p in self.critic.parameters():
                p.requires_grad_(False)
            a_pi, logp = self.actor(obs)
            q1_pi, q2_pi = self.critic(obs, a_pi)
            q_pi = torch.min(q1_pi, q2_pi)
            obj = q_pi - self.alpha * logp
            if self.use_lag:
                for p in self.cost_critic.parameters():
                    p.requires_grad_(False)
                c1_pi, c2_pi = self.cost_critic(obs, a_pi)
                c_pi = torch.max(c1_pi, c2_pi)
                obj = (obj - self.lam * c_pi) / (1.0 + float(self.lam))
                for p in self.cost_critic.parameters():
                    p.requires_grad_(True)
            loss_pi = -obj.mean()
            self.opt_actor.zero_grad(set_to_none=True)
            loss_pi.backward()
            self.opt_actor.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)
            info["loss_pi"] = float(loss_pi.detach())

            # ---- temperature ----------------------------------------------
            if self.opt_alpha is not None:
                loss_alpha = -(self.log_alpha
                               * (logp.detach() + self.target_entropy)).mean()
                self.opt_alpha.zero_grad(set_to_none=True)
                loss_alpha.backward()
                self.opt_alpha.step()
                info["alpha"] = float(self.alpha)
                info["entropy"] = float(-logp.mean().detach())

        # ---- targets -------------------------------------------------------
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
            if self.use_lag:
                for p, pt in zip(self.cost_critic.parameters(),
                                 self.cost_critic_targ.parameters()):
                    pt.mul_(1 - cfg.tau).add_(cfg.tau * p)

        info["td_err"] = td_err
        return info

    def update_lambda(self, episode_cost):
        """
        Dual ascent on the Lagrange multiplier, driven by the *episodic*
        constraint violation. Called once per finished episode, not per step:
        the constraint is a budget over the whole race, so the multiplier must
        respond to the episode total.
        """
        if not self.use_lag:
            return None
        self.ep_cost_est = 0.9 * self.ep_cost_est + 0.1 * float(episode_cost)
        violation = self.ep_cost_est - self.cfg.cost_limit
        loss = -self.log_lambda * violation
        self.opt_lambda.zero_grad(set_to_none=True)
        loss.backward()
        self.opt_lambda.step()
        with torch.no_grad():
            self.log_lambda.clamp_(max=float(np.log(self.cfg.lambda_max)))
        return dict(lam=float(self.lam), ep_cost=self.ep_cost_est, violation=violation)

    # -- persistence -------------------------------------------------------

    def save(self, path):
        payload = dict(actor=self.actor.state_dict(),
                       critic=self.critic.state_dict(),
                       log_alpha=self.log_alpha.detach().cpu(),
                       cfg=self.cfg.to_dict())
        if self.use_lag:
            payload["cost_critic"] = self.cost_critic.state_dict()
            payload["log_lambda"] = self.log_lambda.detach().cpu()
        torch.save(payload, path)

    def load(self, path, actor_only=False):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        if actor_only:
            return
        self.critic.load_state_dict(ck["critic"])
        self.critic_targ = copy.deepcopy(self.critic).requires_grad_(False)
        with torch.no_grad():
            self.log_alpha.copy_(ck["log_alpha"].to(self.device))
        if self.use_lag and "cost_critic" in ck:
            self.cost_critic.load_state_dict(ck["cost_critic"])
            self.cost_critic_targ = copy.deepcopy(self.cost_critic).requires_grad_(False)
            with torch.no_grad():
                self.log_lambda.copy_(ck["log_lambda"].to(self.device))
