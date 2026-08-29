"""
Behaviour cloning: warm-start the SAC actor from an existing controller.

Rationale. Sample efficiency is the acknowledged bottleneck of RL, and the
usual remedy in the process-control literature is to train offline on a model
and then transfer online. Here the situation is better than usual: an *optimal*
teacher already exists. The DP policy is the clairvoyant optimum on this exact
plant, so its trajectories are the best possible demonstration data. Cloning it
first and refining with SAC afterwards is the same imitation-plus-refinement
pattern used, for example, in the two-stage TD3 work on fuel-cell control.

Two caveats that must be stated in the write-up:

1. The DP is clairvoyant and the agent is not. Perfect cloning is therefore
   impossible whenever the preview window is shorter than the horizon the DP
   exploits. The residual cloning error is not a bug: it is a measurement of
   how much of the DP's advantage comes from information rather than from
   optimisation.

2. Cloning transfers the teacher's blind spots. The baseline multi-lap DP
   drains the pack in the first two laps; an actor cloned from it inherits that
   behaviour, and the subsequent SAC phase has to unlearn it. Prefer cloning
   the DP variant with the end-of-lap SoC constraint.

The dataset is built by replaying the teacher inside EMSEnv, so observations
and actions are in exactly the representation the agent will see - including
the action normalisation induced by the safety layer.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def collect_demonstrations(env_fn, policy, n_episodes=1, seeds=None):
    """Replay a teacher policy and record (obs, action) pairs."""
    seeds = seeds if seeds is not None else range(n_episodes)
    OBS, ACT = [], []
    for s in seeds:
        env = env_fn()
        obs, _ = env.reset(seed=int(s))
        while True:
            a = np.asarray(policy(env, obs), dtype=np.float32).ravel()
            OBS.append(np.asarray(obs, np.float32))
            ACT.append(a)
            obs, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
    return np.stack(OBS), np.stack(ACT)


def clone(agent, obs, act, epochs=50, batch_size=256, lr=1e-3, verbose=True):
    """
    Fit the actor's deterministic output (tanh of the mean) to the teacher's
    actions by MSE.

    Only the mean head is supervised. The log-std head is left to its
    initialisation so that the subsequent SAC phase starts with a policy that
    is confident about *where* to act but still free to explore *how much*.
    Supervising the std as well would produce a near-deterministic policy and
    defeat the entropy term in the first thousands of SAC updates.
    """
    dev = agent.device
    X = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    Y = torch.as_tensor(act, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(list(agent.actor.body.parameters())
                           + list(agent.actor.mu.parameters()), lr=lr)
    n = len(X)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            h = agent.actor.body(X[idx])
            pred = torch.tanh(agent.actor.mu(h))
            loss = F.mse_loss(pred, Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        hist.append(tot / n)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [bc] epoch {ep:3d}  mse={hist[-1]:.5f}")
    return hist
