"""
Replay buffer.

Two features beyond the textbook version, both motivated by the structure of
this problem:

COST CHANNEL. Every transition stores a scalar cost alongside the reward, so
the same buffer serves both the scalarised objective and the constrained (CMDP)
formulation. In the constrained formulation the shortfall is a constraint, not
a reward term, and the cost critic needs its own target.

PRIORITISATION. The episodes are long (3886 steps for five laps) and the
interesting transitions are rare: the steps where the deploy budget runs out,
where the thermal ceiling becomes the binding constraint (261 events at nominal
cooling), or where the demand cannot be met. Uniform sampling drowns them.
Prioritised replay (Schaul et al.) samples in proportion to a priority p_i,
here the TD error, with the standard exponent/importance-sampling correction to
avoid the bias that pure greedy prioritisation introduces:

    P(i) = p_i^alpha / sum_j p_j^alpha
    w_i  = (1 / (N * P(i)))^beta   , normalised by max_i w_i

beta is annealed to 1 over training, since the bias only matters near
convergence.

The implementation uses a plain array with argpartition-free sampling
(np.random.choice on the normalised probabilities). For buffers up to a few
1e6 transitions this is fast enough and far easier to audit than a sum-tree;
if it ever becomes the bottleneck, replace it, not the semantics.
"""

from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=1_000_000,
                 prioritized=False, alpha=0.6, beta0=0.4, eps=1e-3):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.cost = np.zeros((capacity, 1), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.ptr, self.size = 0, 0

        self.prioritized = prioritized
        self.alpha, self.beta0, self.eps = alpha, beta0, eps
        self.prio = np.zeros(capacity, np.float32)
        self._max_prio = 1.0

    def add(self, obs, act, rew, next_obs, done, cost=0.0):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.cost[i] = cost
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.prio[i] = self._max_prio
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=None):
        if self.prioritized:
            p = self.prio[: self.size] ** self.alpha
            p = p / p.sum()
            idx = np.random.choice(self.size, batch_size, p=p)
            beta = self.beta0 if beta is None else beta
            w = (self.size * p[idx]) ** (-beta)
            w = (w / w.max()).astype(np.float32)
        else:
            idx = np.random.randint(0, self.size, batch_size)
            w = np.ones(batch_size, np.float32)
        batch = dict(obs=self.obs[idx], act=self.act[idx], rew=self.rew[idx],
                     cost=self.cost[idx], next_obs=self.next_obs[idx],
                     done=self.done[idx], weights=w[:, None], idx=idx)
        return batch

    def update_priorities(self, idx, td_errors):
        if not self.prioritized:
            return
        p = np.abs(np.asarray(td_errors).ravel()) + self.eps
        self.prio[idx] = p
        self._max_prio = max(self._max_prio, float(p.max()))

    def __len__(self):
        return self.size
