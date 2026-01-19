# ================================================================================
# Dependencies: torch, numpy, pandas (optional but recommended)
# ================================================================================
import os, math, time, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Utility: joint action enumeration / probing
# --------------------------
def all_joint_actions(n_agents: int, n_actions: int, device: torch.device) -> torch.Tensor:
    K = n_actions ** n_agents
    acts = torch.zeros((K, n_agents), dtype=torch.long, device=device)
    for j in range(K):
        x = j
        for i in range(n_agents):
            acts[j, i] = x % n_actions
            x //= n_actions
    return acts

def sample_joint_actions(n_agents: int, n_actions: int, device: torch.device, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, n_actions, (k, n_agents), device=device, generator=g, dtype=torch.long)

import torch
import torch.nn.functional as F

def counts_hash_index(a_joint: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    a_joint: (N, n_agents) long
    returns: (N,) int64 hash of counts histogram (robust for large n)
    """
    dev = a_joint.device
    counts = F.one_hot(a_joint, num_classes=n_actions).sum(dim=1).to(torch.int64)  # (N,A)

    mult = torch.arange(1, n_actions + 1, device=dev, dtype=torch.int64) * 0x9e3779b97f4a7c15
    h = (counts * mult).sum(dim=1)

    # final mixing
    h ^= (h >> 33)
    h *= 0xff51afd7ed558ccd
    h ^= (h >> 33)
    h *= 0xc4ceb9fe1a85ec53
    h ^= (h >> 33)
    return h

def joint_index_safe(a_joint: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    Exact base-A index. ONLY safe when n is small enough that A^n fits in int64.
    """
    B, n_agents = a_joint.shape
    dev = a_joint.device
    idx = torch.zeros(B, dtype=torch.long, device=dev)
    base = 1
    for i in range(n_agents):
        idx += a_joint[:, i] * base
        base *= n_actions
    return idx

def build_support_vocab(actions_train: torch.Tensor, n_actions: int,
                        max_contexts: int = 500000) -> torch.Tensor:
    """
    Build a vocabulary of context IDs for a_t from training data.
    Context = joint action at time t.
    Returns a 1D sorted tensor vocab_ids on the same device.
    """
    dev = actions_train.device
    E, T, n = actions_train.shape
    ctx = actions_train[:, :-1, :].reshape(-1, n)  # all a_t contexts

    if ctx.shape[0] > max_contexts:
        perm = torch.randperm(ctx.shape[0], device=dev)[:max_contexts]
        ctx = ctx[perm]

    # exact index if safe; otherwise counts-hash
    if (n <= 10) and (n_actions ** n <= 2_000_000):
        ids = joint_index_safe(ctx, n_actions)
    else:
        ids = counts_hash_index(ctx, n_actions)

    vocab = torch.unique(ids)
    vocab, _ = torch.sort(vocab)
    return vocab

def in_support_mask(context_ids: torch.Tensor, vocab_sorted: torch.Tensor) -> torch.Tensor:
    """
    Returns boolean mask of context_ids that are in vocab_sorted.
    Uses searchsorted for speed.
    """
    pos = torch.searchsorted(vocab_sorted, context_ids)
    pos = torch.clamp(pos, 0, vocab_sorted.numel() - 1)
    return vocab_sorted[pos] == context_ids

def joint_index(a_joint: torch.Tensor, n_actions: int) -> torch.Tensor:
    B, n_agents = a_joint.shape
    idx = torch.zeros(B, dtype=torch.long, device=a_joint.device)
    base = 1
    for i in range(n_agents):
        idx += a_joint[:, i] * base
        base *= n_actions
    return idx

def one_hot_actions(a: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(a, num_classes=n_actions).float()


# --------------------------
# Configs
# --------------------------
@dataclass
class TrainCfg:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    trunc_bptt: int = 50
    verbose_every: int = 10
    learn_init_q: bool = False
    reg_l2_pay: float = 1e-4
    reg_budget: float = 0.0  # optional

@dataclass
class LearnerCfg:
    alpha: float = 0.25
    beta: float = 3.0
    eps: float = 0.05

@dataclass
class DataCfg:
    # For small/medium tasks, you requested fewer episodes because convergence happens early.
    n_episodes: int = 60
    horizon: int = 50
    # Time-OOD training truncation
    train_frac_time: float = 1.0
    # Static-only ablation: keep last frac of timesteps for training
    static_only_last_frac: float = 0.0  # 0.0 => disabled; e.g., 0.3 => use last 30%

@dataclass
class EvalCfg:
    # Probes for off-support MSE when |A|^n is large
    probe_k: int = 5000
    # Counterfactual shifts
    cf_alpha_scale: float = 0.6
    cf_beta_scale: float = 1.4
    cf_eps_scale: float = 1.5
    # Init-Q shift magnitude
    cf_initq_std: float = 0.5
    # Calibration bins
    ece_bins: int = 15

@dataclass
class SuiteCfg:
    seed: int = 0
    out_dir: str = "./diml_results"
    device: str = "auto"


# ================================================================================
# Learner model: differentiable Logit-Q
# ================================================================================
class LogitQRule:
    def __init__(self, n_agents: int, n_actions: int, alpha: float, beta: float, eps: float):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def init_state(self, B: int, device: torch.device, init_q: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Q: (B, n_agents, n_actions)
        if init_q is None:
            return torch.zeros((B, self.n_agents, self.n_actions), device=device)
        return init_q.unsqueeze(0).expand(B, -1, -1).contiguous()

    def policy(self, Q: torch.Tensor) -> torch.Tensor:
        # Softmax policy with epsilon mixing
        pi = F.softmax(self.beta * Q, dim=-1)
        if self.eps > 0:
            uni = torch.full_like(pi, 1.0 / self.n_actions)
            pi = (1 - self.eps) * pi + self.eps * uni
            pi = pi / pi.sum(dim=-1, keepdim=True)
        return pi

    def sample_actions(self, pi: torch.Tensor) -> torch.Tensor:
        # pi: (B, n_agents, n_actions) -> actions: (B, n_agents)
        B, n_agents, n_actions = pi.shape
        a = torch.zeros((B, n_agents), dtype=torch.long, device=pi.device)
        for i in range(n_agents):
            a[:, i] = torch.multinomial(pi[:, i, :], 1).squeeze(-1)
        return a

    def update(self, Q: torch.Tensor, u_cf: torch.Tensor) -> torch.Tensor:
        # u_cf: (B, n_agents, n_actions)
        return (1 - self.alpha) * Q + self.alpha * u_cf


# ================================================================================
# Mechanisms: (A) small/medium unstructured neural; (B) scalable symmetric count-based
# ================================================================================
class NeuralJointMechanism(nn.Module):
    """Unstructured neural mechanism M(a_joint) -> payouts (B, n_agents)."""
    def __init__(self, n_agents: int, n_actions: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        emb_dim = min(24, max(6, n_actions))
        self.embs = nn.ModuleList([nn.Embedding(n_actions, emb_dim) for _ in range(n_agents)])
        layers = []
        d = n_agents * emb_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, n_agents)]
        self.net = nn.Sequential(*layers)

    def forward(self, a_joint: torch.Tensor) -> torch.Tensor:
        # a_joint: (B, n_agents)
        xs = [self.embs[i](a_joint[:, i]) for i in range(self.n_agents)]
        x = torch.cat(xs, dim=-1)
        return self.net(x)


class CongestionParamMechanism(nn.Module):
    """
    Structured congestion mechanism (correctly specified baseline):
    Each action is a resource; payoff_i = -(c1[a]*load + c2[a]*load^2) + tau[a]*load.
    """
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        # initialize small positive
        self.c1 = nn.Parameter(torch.rand(n_actions) * 0.3 + 0.1)
        self.c2 = nn.Parameter(torch.rand(n_actions) * 0.1 + 0.05)
        self.tau = nn.Parameter(torch.rand(n_actions) * 0.2)

    def forward(self, a_joint: torch.Tensor) -> torch.Tensor:
        B = a_joint.shape[0]
        dev = a_joint.device
        payouts = torch.zeros((B, self.n_agents), device=dev)
        # vectorize via one-hot counts
        oh = one_hot_actions(a_joint, self.n_actions)   # (B,n,A)
        counts = oh.sum(dim=1)                           # (B,A)
        # payoff for agent i depends on its chosen action
        for i in range(self.n_agents):
            ai = a_joint[:, i]                           # (B,)
            load = counts.gather(1, ai.view(-1,1)).squeeze(1)  # (B,)
            c1 = self.c1[ai]
            c2 = self.c2[ai]
            tau = self.tau[ai]
            cost = c1 * load + c2 * (load ** 2)
            toll = tau * load
            payouts[:, i] = -cost + toll
        return payouts


class PublicGoodsParamMechanism(nn.Module):
    """
    Structured public goods mechanism baseline:
    action -> contribution level in [0,1]; payoff_i = benefit(total) - cost(a_i) + subsidy(total, a_i).
    """
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        levels = torch.linspace(0.0, 1.0, steps=n_actions)
        self.register_buffer("levels", levels)
        self.k = nn.Parameter(torch.tensor(2.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.cost_scale = nn.Parameter(torch.tensor(0.8))
        self.sub = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, a_joint: torch.Tensor) -> torch.Tensor:
        B = a_joint.shape[0]
        lv = self.levels[a_joint]                       # (B,n)
        total = lv.sum(dim=1, keepdim=True)             # (B,1)
        benefit = self.scale * (1.0 - torch.exp(-self.k * total))  # (B,1)
        cost = self.cost_scale * (lv ** 2)              # (B,n)
        base = benefit.expand(-1, self.n_agents) - cost
        sub_in = torch.stack([total.expand(-1, self.n_agents), lv], dim=-1)  # (B,n,2)
        sub_val = self.sub(sub_in.reshape(B * self.n_agents, 2)).reshape(B, self.n_agents)
        return base + sub_val


class SymmetricCountMechanism(nn.Module):
    """
    Scalable symmetric mechanism for large n:
      payoff_i = f_theta(a_i, counts_-i)
    where counts_-i is the histogram of opponent actions (excluding agent i).

    Supports n=100-300, A=20-50 with O(B*n*A) counterfactual compute (chunked).
    """
    def __init__(self, n_actions: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        self.n_actions = n_actions
        a_emb = min(32, max(12, n_actions))
        self.a_emb = nn.Embedding(n_actions, a_emb)
        layers = []
        d = a_emb + n_actions  # embed(a_i) + counts vector
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def payoff(self, a_i, counts_vec):
        # counts_vec: (N, A) int/float in [0, n-1]
        # Normalize to [0,1]
        counts_norm = counts_vec / (counts_vec.sum(dim=1, keepdim=True).clamp_min(1.0))
        # or: counts_norm = counts_vec / (self.n_agents_minus_1)  # if you store it
        e = self.a_emb(a_i)
        x = torch.cat([e, counts_norm], dim=-1)
        return self.net(x).squeeze(-1)


    def forward(self, a_joint: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        """
        a_joint: (B, n) long
        returns payouts: (B, n) float
        Computes counts_-i for each agent and evaluates payoff_i = f(a_i, counts_-i).
        Chunked for memory safety.
        """
        B, n = a_joint.shape
        A = self.n_actions
        dev = a_joint.device

        # one-hot actions + counts
        oh = F.one_hot(a_joint, num_classes=A).float()   # (B,n,A)
        counts = oh.sum(dim=1)                          # (B,A)
        counts_minus = counts.unsqueeze(1) - oh         # (B,n,A)

        # flatten to (B*n,)
        a_flat = a_joint.reshape(-1)                    # (BN,)
        c_flat = counts_minus.reshape(-1, A)            # (BN,A)
        BN = a_flat.shape[0]

        out = torch.empty((BN,), device=dev)
        for s in range(0, BN, chunk):
            e = min(s + chunk, BN)
            out[s:e] = self.payoff(a_flat[s:e], c_flat[s:e])

        return out.view(B, n)



# ================================================================================
# Counterfactual payoff computation
# ================================================================================
def counterfactual_payoffs_small(mech: nn.Module, a_joint: torch.Tensor, n_actions: int) -> torch.Tensor:
    """
    Generic stateless mechanism counterfactuals:
      u_cf[b,i,a] = mech(a_i=a, a_-i fixed)[i]
    Vectorized per-agent over actions; loops over agents (small n).
    Returns (B, n_agents, n_actions).
    """
    B, n_agents = a_joint.shape
    dev = a_joint.device
    u_cf = torch.empty((B, n_agents, n_actions), device=dev)
    # Loop over agents; vectorize over actions
    for i in range(n_agents):
        a_rep = a_joint.unsqueeze(1).expand(B, n_actions, n_agents).reshape(B * n_actions, n_agents)
        a_rep[:, i] = torch.arange(n_actions, device=dev).repeat(B)
        pay = mech(a_rep).reshape(B, n_actions, n_agents)  # (B,A,n)
        u_cf[:, i, :] = pay[:, :, i]
    return u_cf


def counterfactual_payoffs_symmetric_count(
    mech: SymmetricCountMechanism,
    a_joint: torch.Tensor,
    n_actions: int,
    chunk: int = 65536
) -> torch.Tensor:
    """
    Scalable counterfactuals for symmetric count mechanism:
      u_cf[b,i,a] = f_theta(a, counts_-i^a)
    where counts_-i^a is opponent histogram (excluding i), unchanged by i's counterfactual action.

    So counts_-i does NOT depend on a. That means we can reuse counts_-i across all a,
    and only vary the action embedding.
    """
    B, n = a_joint.shape
    A = n_actions
    dev = a_joint.device

    # Compute counts_-i once
    oh = F.one_hot(a_joint, num_classes=A).float()      # (B,n,A)
    counts = oh.sum(dim=1)                              # (B,A)
    counts_minus = counts.unsqueeze(1) - oh             # (B,n,A)

    BN = B * n
    counts_flat = counts_minus.reshape(BN, A)           # (BN,A)

    # For each (b,i), we need payoff for all a in [0..A-1]
    # Build (BN*A,) action ids and repeat counts accordingly
    a_ids = torch.arange(A, device=dev).repeat(BN)      # (BN*A,)
    counts_rep = counts_flat.repeat_interleave(A, dim=0)  # (BN*A, A)

    out = torch.empty((BN * A,), device=dev)
    for s in range(0, BN * A, chunk):
        e = min(s + chunk, BN * A)
        out[s:e] = mech.payoff(a_ids[s:e], counts_rep[s:e])

    return out.view(BN, A).view(B, n, A)


# ================================================================================
# Data generation
# ================================================================================
@torch.no_grad()
def generate_dataset_stateless(
    mech_true: nn.Module,
    learner: LogitQRule,
    n_agents: int,
    n_actions: int,
    n_episodes: int,
    horizon: int,
    device: torch.device,
    scalable_count: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      actions: (E,T,n) long
    """
    actions = torch.zeros((n_episodes, horizon, n_agents), dtype=torch.long, device=device)
    Q = learner.init_state(n_episodes, device=device)

    actions[:, 0, :] = learner.sample_actions(learner.policy(Q))

    for t in range(horizon - 1):
        a_t = actions[:, t, :]
        if scalable_count:
            u_cf = counterfactual_payoffs_symmetric_count(mech_true, a_t, n_actions)
        else:
            u_cf = counterfactual_payoffs_small(mech_true, a_t, n_actions)
        Q = learner.update(Q, u_cf)
        actions[:, t + 1, :] = learner.sample_actions(learner.policy(Q))

    return {"actions": actions}


# ================================================================================
# Baselines: Behavior-only imitation (JointBC)
# ================================================================================
class JointBC(nn.Module):
    def __init__(self, n_agents: int, n_actions: int, window: int = 5, hidden: int = 256):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.window = window
        in_dim = window * n_agents * n_actions
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents * n_actions)
        )

    def forward(self, hist_actions: torch.Tensor) -> torch.Tensor:
        # hist_actions: (B, window, n_agents)
        B = hist_actions.shape[0]
        x = F.one_hot(hist_actions, num_classes=self.n_actions).float().reshape(B, -1)
        logits = self.net(x).reshape(B, self.n_agents, self.n_actions)
        return logits

def train_jointbc(actions: torch.Tensor, window: int, epochs: int = 40, lr: float = 3e-3, batch_size: int = 512, verbose: bool=True):
    dev = actions.device
    E, T, n = actions.shape
    A = int(actions.max().item() + 1)
    model = JointBC(n, A, window=window).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Build dataset
    X, Y = [], []
    for t in range(window, T):
        X.append(actions[:, t-window:t, :])
        Y.append(actions[:, t, :])
    X = torch.cat(X, dim=0)  # (E*(T-w), w, n)
    Y = torch.cat(Y, dim=0)  # (E*(T-w), n)
    N = X.shape[0]
    idx = torch.arange(N, device=dev)

    for ep in range(1, epochs+1):
        perm = idx[torch.randperm(N)]
        losses = []
        for s in range(0, N, batch_size):
            b = perm[s:s+batch_size]
            logits = model(X[b])
            loss = 0.0
            for i in range(n):
                loss = loss + F.cross_entropy(logits[:, i, :], Y[b, i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if verbose and (ep == 1 or ep % 10 == 0):
            print(f"[JointBC] epoch {ep:03d} | loss {np.mean(losses):.4f}")
    return model

@torch.no_grad()
def jointbc_accuracy(model: JointBC, actions: torch.Tensor) -> float:
    dev = actions.device
    E, T, n = actions.shape
    w = model.window
    X, Y = [], []
    for t in range(w, T):
        X.append(actions[:, t-w:t, :])
        Y.append(actions[:, t, :])
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    logits = model(X)
    pred = logits.argmax(dim=-1)
    return float((pred == Y).float().mean().item())


# ================================================================================
# Baseline: Equilibrium-only inverse (EOI) generalized to n agents (myopic logit inversion)
# ================================================================================
@torch.no_grad()
def equilibrium_only_inverse_u(actions: torch.Tensor, beta: float, gauge: str = "mean0") -> torch.Tensor:
    """
    Estimate per-agent conditional utilities u_i(a_i, a_-i) from empirical conditionals p(a_i | a_-i)
    using logit inversion. This is not a full mechanism reconstruction when n>2 because a_-i is huge;
    we implement the feasible variant:
      - condition only on a summary statistic of opponents: opponent action counts histogram.
    This is meaningful in symmetric / anonymous environments and serves as a strong "static" baseline.

    Returns:
      u_hat: (n_agents, n_actions, n_actions) if using summary "counts mode"? No—histogram has dimension A.
    We instead return a function represented as a table:
      u_hat[a_i, k] where k is "most common opponent action" (mode), for simplicity.
      This is crude but illustrates equilibrium-only limitations.
    """
    # For general n and A, full a_-i conditioning is infeasible.
    # We condition on opponent mode action m_-i(t) = argmax count among opponents.
    dev = actions.device
    E, T, n = actions.shape
    A = int(actions.max().item() + 1)
    # counts[i, a_i, m] = number of times agent i took action a_i when opponent-mode was m
    counts = torch.zeros((n, A, A), device=dev)
    # compute modes per timestep per episode
    # actions: (E,T,n)
    for t in range(T):
        a_t = actions[:, t, :]                        # (E,n)
        oh = F.one_hot(a_t, num_classes=A).float()    # (E,n,A)
        total = oh.sum(dim=1)                         # (E,A)
        # opponent mode per agent: subtract own one-hot
        for i in range(n):
            opp_counts = total - oh[:, i, :]
            m = opp_counts.argmax(dim=-1)             # (E,)
            ai = a_t[:, i]                            # (E,)
            # accumulate counts
            counts[i].index_put_((ai, m), torch.ones_like(ai, dtype=torch.float), accumulate=True)

    # estimate p(a_i | m)
    u_hat = torch.zeros((n, A, A), device=dev)  # u_hat[i, a, m]
    for i in range(n):
        denom = counts[i].sum(dim=0, keepdim=True) + 1e-12  # (1,A)
        p = counts[i] / denom                                # (A,A)
        for m in range(A):
            logp = torch.log(p[:, m] + 1e-12)
            # differences relative to action 0
            u = (logp - logp[0]) / beta
            if gauge == "mean0":
                u = u - u.mean()
            u_hat[i, :, m] = u
    return u_hat  # not a full mechanism; used for behavioral prediction only in static settings


# ================================================================================
# Tabular MLE baseline (only for small joint action spaces)
# ================================================================================
class TabularMechanism(nn.Module):
    def __init__(self, n_agents: int, n_actions: int):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.K = n_actions ** n_agents
        self.U = nn.Parameter(torch.zeros(self.K, n_agents))

    def forward(self, a_joint: torch.Tensor) -> torch.Tensor:
        idx = joint_index(a_joint, self.n_actions)
        return self.U[idx]


# ================================================================================
# DIML Trainer (small/medium generic) with vectorized batching
# ================================================================================
class DIMLTrainer:
    def __init__(
        self,
        mech: nn.Module,
        learner: LogitQRule,
        n_agents: int,
        n_actions: int,
        device: torch.device,
        train_cfg: TrainCfg,
        scalable_count: bool = False
    ):
        self.mech = mech.to(device)
        self.learner = learner
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.device = device
        self.cfg = train_cfg
        self.scalable_count = scalable_count

        self.init_q = nn.Parameter(torch.zeros(n_agents, n_actions, device=device)) if train_cfg.learn_init_q else None

        params = list(self.mech.parameters()) + ([self.init_q] if self.init_q is not None else [])
        self.opt = torch.optim.Adam(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    def regularizer(self, a_joint_flat: torch.Tensor) -> torch.Tensor:
      reg = torch.tensor(0.0, device=self.device)
      if self.cfg.reg_l2_pay <= 0 and self.cfg.reg_budget <= 0:
          return reg

      # For large-scale, subsample to keep this cheap
      if self.scalable_count and a_joint_flat.shape[0] > 5000:
          idx = torch.randint(0, a_joint_flat.shape[0], (5000,), device=self.device)
          a_joint_flat = a_joint_flat[idx]

      pay = self.mech(a_joint_flat)
      if self.cfg.reg_l2_pay > 0:
          reg = reg + self.cfg.reg_l2_pay * (pay ** 2).mean()
      if self.cfg.reg_budget > 0:
          reg = reg + self.cfg.reg_budget * (pay.sum(dim=-1) ** 2).mean()
      return reg

    def nll_batch(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B,T,n)
        B, T, n = actions.shape
        Q = self.learner.init_state(B, self.device, init_q=self.init_q if self.init_q is not None else None)
        loss = torch.tensor(0.0, device=self.device)

        # unroll
        for t in range(T - 1):
            a_t = actions[:, t, :]
            if self.scalable_count:
                u_cf = counterfactual_payoffs_symmetric_count(self.mech, a_t, self.n_actions)
            else:
                u_cf = counterfactual_payoffs_small(self.mech, a_t, self.n_actions)

            Q = self.learner.update(Q, u_cf)
            pi = self.learner.policy(Q)   # (B,n,A)
            a_next = actions[:, t+1, :]   # (B,n)

            # gather probs
            # vectorized gather: probs = pi.gather(2, a_next.unsqueeze(-1)).squeeze(-1) => (B,n)
            probs = torch.gather(pi, 2, a_next.unsqueeze(-1)).squeeze(-1)
            loss = loss - torch.log(probs + 1e-12).mean() * n  # scale like sum over agents (for comparability)

        # add reg sampled on visited actions
        loss = loss + self.regularizer(actions.reshape(B*T, n))
        return loss

    def fit(self, actions_train: torch.Tensor, tag: str, curves: List[Dict[str, Any]]):
        E, T, n = actions_train.shape
        idx = torch.arange(E, device=self.device)

        for ep in range(1, self.cfg.epochs + 1):
            t_ep0 = time.time()
            perm = idx[torch.randperm(E)]
            losses = []
            for s in range(0, E, self.cfg.batch_size):
                b = perm[s:s+self.cfg.batch_size]
                batch = actions_train[b]
                if self.cfg.trunc_bptt < T:
                    t0 = torch.randint(0, T - self.cfg.trunc_bptt, (1,), device=self.device).item()
                    batch = batch[:, t0:t0+self.cfg.trunc_bptt, :]

                self.opt.zero_grad(set_to_none=True)
                loss = self.nll_batch(batch)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.mech.parameters(), self.cfg.grad_clip)
                self.opt.step()
                losses.append(loss.item())

            dt = time.time() - t_ep0
            mean_loss = float(np.mean(losses))
            if (ep == 1) or (ep % self.cfg.verbose_every == 0):
                print(f"[{tag}] epoch {ep:04d} | loss {mean_loss:.4f} | sec {dt:.2f}")

            curves.append({
                "tag": tag,
                "epoch": ep,
                "train_loss": mean_loss,
                "epoch_seconds": dt,
            })


# ================================================================================
# Evaluation metrics
# ================================================================================
@torch.no_grad()
def behavior_nll_per_agent_step(mech: nn.Module, actions: torch.Tensor, learner: LogitQRule, n_actions: int,
                                init_q: Optional[torch.Tensor] = None, scalable_count: bool = False) -> float:
    dev = actions.device
    E, T, n = actions.shape
    Q = learner.init_state(E, dev, init_q=init_q)
    total = 0.0
    count = 0
    for t in range(T - 1):
        a_t = actions[:, t, :]
        if scalable_count:
            u_cf = counterfactual_payoffs_symmetric_count(mech, a_t, n_actions)
        else:
            u_cf = counterfactual_payoffs_small(mech, a_t, n_actions)
        Q = learner.update(Q, u_cf)
        pi = learner.policy(Q)
        a_next = actions[:, t+1, :]
        probs = torch.gather(pi, 2, a_next.unsqueeze(-1)).squeeze(-1)  # (E,n)
        total += float((-torch.log(probs + 1e-12)).mean().item())
        count += 1
    # This is already averaged over agents via mean; divide by steps.
    return total / max(count, 1)

@torch.no_grad()
def behavior_nll_in_support(
    mech: torch.nn.Module,
    actions: torch.Tensor,
    learner,
    n_actions: int,
    support_vocab_sorted: torch.Tensor,
    init_q: torch.Tensor = None,
    scalable_count: bool = False,
    max_eval_steps: int = 500000
) -> float:
    """
    In-support NLL: only include timesteps whose context a_t is in the training support.
    Returns average negative log-likelihood per agent-step over included steps.
    """
    dev = actions.device
    E, T, n = actions.shape

    Q = learner.init_state(E, dev, init_q=init_q)
    total_nll = 0.0
    total_count = 0

    # Optionally subsample episodes to cap eval compute
    if E * (T - 1) > max_eval_steps:
        # sample a subset of episodes
        keepE = max(1, max_eval_steps // max(1, (T - 1)))
        permE = torch.randperm(E, device=dev)[:keepE]
        actions = actions[permE]
        E = actions.shape[0]
        Q = learner.init_state(E, dev, init_q=init_q)

    for t in range(T - 1):
        a_t = actions[:, t, :]      # (E,n)
        a_next = actions[:, t+1, :] # (E,n)

        # compute context IDs for support test
        if (n <= 10) and (n_actions ** n <= 2_000_000):
            ctx_ids = joint_index_safe(a_t, n_actions)
        else:
            ctx_ids = counts_hash_index(a_t, n_actions)

        mask = in_support_mask(ctx_ids, support_vocab_sorted)  # (E,)
        if not mask.any():
            # still update Q so dynamics remain consistent, but skip scoring
            if scalable_count:
                u_cf = counterfactual_payoffs_symmetric_count(mech, a_t, n_actions)
            else:
                u_cf = counterfactual_payoffs_small(mech, a_t, n_actions)
            Q = learner.update(Q, u_cf)
            continue

        # update using all episodes (to keep Q consistent), but only score masked ones
        if scalable_count:
            u_cf = counterfactual_payoffs_symmetric_count(mech, a_t, n_actions)
        else:
            u_cf = counterfactual_payoffs_small(mech, a_t, n_actions)

        Q = learner.update(Q, u_cf)
        pi = learner.policy(Q)  # (E,n,A)

        probs = torch.gather(pi, 2, a_next.unsqueeze(-1)).squeeze(-1)  # (E,n)
        # Score only masked contexts
        probs_m = probs[mask]  # (Em,n)
        nll = (-torch.log(probs_m + 1e-12)).mean().item()  # mean over masked agents+episodes
        total_nll += nll
        total_count += 1

    if total_count == 0:
        # Nothing in support (should not happen if support built from similar data)
        return float("nan")

    return total_nll / total_count


@torch.no_grad()
def ece_from_logits(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    probs: (N, A) probabilities
    targets: (N,) long
    Expected calibration error (multiclass) using confidence = max prob and accuracy.
    """
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    bins = torch.linspace(0, 1, steps=n_bins+1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    N = probs.shape[0]
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        mask = (conf >= lo) & (conf < hi) if b < n_bins-1 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += (mask.float().mean()) * torch.abs(acc[mask].mean() - conf[mask].mean())
    return float(ece.item())

@torch.no_grad()
def mechanism_mse_offsupport(mech_est: nn.Module, mech_true: nn.Module, n_agents: int, n_actions: int,
                             device: torch.device, probe_k: int = 5000) -> float:
    K = n_actions ** n_agents
    if K <= 20000:
        ja = all_joint_actions(n_agents, n_actions, device)
    else:
        ja = sample_joint_actions(n_agents, n_actions, device, k=probe_k, seed=123)
    return float(((mech_est(ja) - mech_true(ja)) ** 2).mean().item())

@torch.no_grad()
def mechanism_mse_onsupport(mech_est: nn.Module, mech_true: nn.Module, actions: torch.Tensor) -> float:
    # Use empirical visitation distribution over joint actions encountered in data.
    dev = actions.device
    E, T, n = actions.shape
    A = int(actions.max().item() + 1)
    flat = actions.reshape(E*T, n)
    # unique visited joint actions
    # For small n, we can just sample visited set; for large n, we sample a subset.
    if flat.shape[0] > 200000:
        flat = flat[torch.randperm(flat.shape[0], device=dev)[:200000]]
    # compute MSE on these visited points
    return float(((mech_est(flat) - mech_true(flat)) ** 2).mean().item())

@torch.no_grad()
def payoff_difference_mse(mech_est: nn.Module, mech_true: nn.Module, actions: torch.Tensor, n_actions: int,
                          scalable_count: bool = False, max_samples: int = 20000) -> float:
    """
    Measures error in payoff differences across own actions given observed opponents:
      E_{(a_-i)} E_{a,a'} [ (u(a,a_-i)-u(a',a_-i)) - (û(...) - û(...)) ]^2
    Approximate by sampling:
      - sample (episode,t,agent) contexts and random (a,a') pairs.
    """
    dev = actions.device
    E, T, n = actions.shape
    # sample contexts
    Nctx = min(max_samples, E*(T-1)*n)
    # sample indices
    e_idx = torch.randint(0, E, (Nctx,), device=dev)
    t_idx = torch.randint(0, T-1, (Nctx,), device=dev)
    i_idx = torch.randint(0, n, (Nctx,), device=dev)
    a_obs = actions[e_idx, t_idx, :]  # (Nctx, n)
    # opponents fixed per context; sample a,a'
    a1 = torch.randint(0, n_actions, (Nctx,), device=dev)
    a2 = torch.randint(0, n_actions, (Nctx,), device=dev)

    # build two counterfactual joint actions for each context
    aj1 = a_obs.clone()
    aj2 = a_obs.clone()
    aj1[torch.arange(Nctx, device=dev), i_idx] = a1
    aj2[torch.arange(Nctx, device=dev), i_idx] = a2

    u1_true = mech_true(aj1)[torch.arange(Nctx, device=dev), i_idx]
    u2_true = mech_true(aj2)[torch.arange(Nctx, device=dev), i_idx]
    u1_est  = mech_est(aj1)[torch.arange(Nctx, device=dev), i_idx]
    u2_est  = mech_est(aj2)[torch.arange(Nctx, device=dev), i_idx]
    diff_true = u1_true - u2_true
    diff_est = u1_est - u2_est
    return float(((diff_true - diff_est) ** 2).mean().item())

@torch.no_grad()
def simulate_actions(mech: nn.Module, learner: LogitQRule, n_agents: int, n_actions: int,
                     E: int, T: int, device: torch.device,
                     init_q: Optional[torch.Tensor] = None,
                     scalable_count: bool = False) -> torch.Tensor:
    actions = torch.zeros((E, T, n_agents), dtype=torch.long, device=device)
    Q = learner.init_state(E, device=device, init_q=init_q)
    actions[:, 0, :] = learner.sample_actions(learner.policy(Q))
    for t in range(T - 1):
        a_t = actions[:, t, :]
        if scalable_count:
            u_cf = counterfactual_payoffs_symmetric_count(mech, a_t, n_actions)
        else:
            u_cf = counterfactual_payoffs_small(mech, a_t, n_actions)
        Q = learner.update(Q, u_cf)
        actions[:, t+1, :] = learner.sample_actions(learner.policy(Q))
    return actions

@torch.no_grad()
def counterfactual_kl(mech_true: nn.Module, mech_test: nn.Module, learner_cf: LogitQRule,
                      n_agents: int, n_actions: int, device: torch.device,
                      E: int = 400, T: int = 50,
                      scalable_count: bool = False) -> float:
    a_true = simulate_actions(mech_true, learner_cf, n_agents, n_actions, E, T, device, scalable_count=scalable_count)
    a_test = simulate_actions(mech_test, learner_cf, n_agents, n_actions, E, T, device, scalable_count=scalable_count)

    flat_true = a_true.reshape(E*T, n_agents)
    flat_test = a_test.reshape(E*T, n_agents)

    eps = 1e-12

    # Small n: safe to use exact joint index
    if (n_agents <= 10) and (n_actions ** n_agents <= 2_000_000):
        idx_true = joint_index(flat_true, n_actions)
        idx_test = joint_index(flat_test, n_actions)
        K = n_actions ** n_agents
        p = torch.bincount(idx_true, minlength=K).float().to(device); p = p / p.sum()
        q = torch.bincount(idx_test, minlength=K).float().to(device); q = q / q.sum()
        return float((p * (torch.log(p+eps) - torch.log(q+eps))).sum().item())

    # Large n: use counts-hash distribution (appropriate for symmetric-count mechanisms)
    h_true = counts_hash_index(flat_true, n_actions)
    h_test = counts_hash_index(flat_test, n_actions)

    # Map hashes to contiguous ids to bincount
    # (We avoid huge minlen by using unique+inverse)
    uniq_true, inv_true = torch.unique(h_true, return_inverse=True)
    uniq_test, inv_test = torch.unique(h_test, return_inverse=True)

    # Build a joint vocabulary of hashes
    # We'll concatenate and re-unique to align supports
    all_h = torch.cat([uniq_true, uniq_test], dim=0)
    vocab, _ = torch.unique(all_h, sorted=False, return_inverse=True)

    # Re-map each hash to vocab id via searchsorted-like dictionary
    # Use CPU dict if vocab is huge; otherwise do vector trick
    # For our scales, vocab size is typically manageable.
    # We'll do a torch-based mapping using sorting.
    vocab_sorted, perm = torch.sort(vocab)
    # indices in sorted vocab
    true_pos = torch.searchsorted(vocab_sorted, h_true)
    test_pos = torch.searchsorted(vocab_sorted, h_test)

    Kc = vocab_sorted.numel()
    p = torch.bincount(true_pos, minlength=Kc).float().to(device); p = p / p.sum()
    q = torch.bincount(test_pos, minlength=Kc).float().to(device); q = q / q.sum()

    return float((p * (torch.log(p+eps) - torch.log(q+eps))).sum().item())

@torch.no_grad()
def counterfactual_kl_in_support(
    mech_true: torch.nn.Module,
    mech_est: torch.nn.Module,
    learner_cf,
    n_agents: int,
    n_actions: int,
    device: torch.device,
    support_vocab_sorted: torch.Tensor,
    E: int = 400,
    T: int = 50,
    scalable_count: bool = False
) -> float:
    """
    In-support CF-KL: only include transitions whose context a_t is in the training support.
    KL is computed over the distribution of next-step outcomes among included transitions.
    """
    a_true = simulate_actions(mech_true, learner_cf, n_agents, n_actions, E, T, device, scalable_count=scalable_count)
    a_est  = simulate_actions(mech_est,  learner_cf, n_agents, n_actions, E, T, device, scalable_count=scalable_count)

    # contexts and next actions
    ctx_true = a_true[:, :-1, :].reshape(-1, n_agents)     # a_t
    nxt_true = a_true[:, 1:, :].reshape(-1, n_agents)      # a_{t+1}

    ctx_est  = a_est[:, :-1, :].reshape(-1, n_agents)
    nxt_est  = a_est[:, 1:, :].reshape(-1, n_agents)

    # compute context IDs and masks
    if (n_agents <= 10) and (n_actions ** n_agents <= 2_000_000):
        ctx_ids_true = joint_index_safe(ctx_true, n_actions)
        ctx_ids_est  = joint_index_safe(ctx_est,  n_actions)
        nxt_ids_true = joint_index_safe(nxt_true, n_actions)
        nxt_ids_est  = joint_index_safe(nxt_est,  n_actions)
    else:
        ctx_ids_true = counts_hash_index(ctx_true, n_actions)
        ctx_ids_est  = counts_hash_index(ctx_est,  n_actions)
        nxt_ids_true = counts_hash_index(nxt_true, n_actions)
        nxt_ids_est  = counts_hash_index(nxt_est,  n_actions)

    mask_true = in_support_mask(ctx_ids_true, support_vocab_sorted)
    mask_est  = in_support_mask(ctx_ids_est,  support_vocab_sorted)

    nxt_true = nxt_ids_true[mask_true]
    nxt_est  = nxt_ids_est[mask_est]

    if nxt_true.numel() == 0 or nxt_est.numel() == 0:
        return float("nan")

    # Build joint vocabulary for next-outcome IDs to compare distributions
    all_ids = torch.cat([nxt_true, nxt_est], dim=0)
    vocab, _ = torch.unique(all_ids, sorted=True, return_inverse=True)

    # Map to contiguous bins via searchsorted
    pos_true = torch.searchsorted(vocab, nxt_true)
    pos_est  = torch.searchsorted(vocab, nxt_est)

    K = vocab.numel()
    p = torch.bincount(pos_true, minlength=K).float().to(device); p = p / p.sum()
    q = torch.bincount(pos_est,  minlength=K).float().to(device); q = q / q.sum()

    eps = 1e-12
    return float((p * (torch.log(p + eps) - torch.log(q + eps))).sum().item())


# ================================================================================
# Experiment utilities: train/test splits and time/static ablations
# ================================================================================
def split_train_test(actions: torch.Tensor, train_frac: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    E = actions.shape[0]
    Etr = max(1, int(train_frac * E))
    return actions[:Etr].contiguous(), actions[Etr:].contiguous()

def apply_time_ood(actions: torch.Tensor, train_frac_time: float) -> torch.Tensor:
    if train_frac_time >= 0.999:
        return actions
    E, T, n = actions.shape
    Ttr = max(2, int(train_frac_time * T))
    return actions[:, :Ttr, :].contiguous()

def apply_static_only(actions: torch.Tensor, last_frac: float) -> torch.Tensor:
    if last_frac <= 1e-9:
        return actions
    E, T, n = actions.shape
    keep = max(2, int(last_frac * T))
    return actions[:, T-keep:, :].contiguous()


# ================================================================================
# CSV logging
# ================================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(rows: List[Dict[str, Any]], filepath: str):
    if not rows:
        return
    ensure_dir(os.path.dirname(filepath))
    if PANDAS_OK:
        pd.DataFrame(rows).to_csv(filepath, index=False)
    else:
        # minimal csv writer
        keys = sorted({k for r in rows for k in r.keys()})
        with open(filepath, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


# ================================================================================
# Main experiment runner
# ================================================================================
def run_small_medium_experiment(
    name: str,
    mech_true: nn.Module,
    mech_class_for_diml: nn.Module,   # instantiated model
    mech_class_for_structured: Optional[nn.Module],  # structured param MLE baseline (optional)
    n_agents: int,
    n_actions: int,
    data_cfg: DataCfg,
    train_cfg: TrainCfg,
    gen_cfg: LearnerCfg,
    infer_cfg: LearnerCfg,
    eval_cfg: EvalCfg,
    device: torch.device,
    enable_tabular: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      curves_rows: per-epoch training logs (+ per-epoch evaluation metrics)
      results_rows: evaluation summary rows
    """
    curves_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    # Generator learner
    gen_learner = LogitQRule(n_agents, n_actions, gen_cfg.alpha, gen_cfg.beta, gen_cfg.eps)

    print(f"\n=== [{name}] Generate data (n={n_agents}, A={n_actions}, E={data_cfg.n_episodes}, T={data_cfg.horizon}) ===")
    t0 = time.time()
    ds = generate_dataset_stateless(mech_true, gen_learner, n_agents, n_actions, data_cfg.n_episodes, data_cfg.horizon, device)
    gen_sec = time.time() - t0

    actions = ds["actions"]
    train_actions_full, test_actions = split_train_test(actions, 0.8)

    # Apply time-OOD training truncation and/or static-only ablation
    train_actions = apply_time_ood(train_actions_full, data_cfg.train_frac_time)
    train_actions = apply_static_only(train_actions, data_cfg.static_only_last_frac)

    # Build support vocab for in-support evaluation (based on actual training slice)
    support_vocab = build_support_vocab(train_actions, n_actions)

    # Inference learner (DIML assumed model)
    infer_learner = LogitQRule(n_agents, n_actions, infer_cfg.alpha, infer_cfg.beta, infer_cfg.eps)

    # ----------------------------
    # Helper: per-epoch eval
    # ----------------------------
    @torch.no_grad()
    def eval_epoch(mech_est: nn.Module, init_q: Optional[torch.Tensor], tag: str) -> Dict[str, float]:
        # Use test_actions for eval; for speed, optionally subsample episodes
        Etest = test_actions.shape[0]
        if Etest > 200:
            perm = torch.randperm(Etest, device=device)[:200]
            te = test_actions[perm]
        else:
            te = test_actions

        # MSE on-support (visited joint actions in te)
        mse_on = mechanism_mse_onsupport(mech_est, mech_true, te)
        # payoff-difference MSE
        diff_mse = payoff_difference_mse(mech_est, mech_true, te, n_actions, max_samples=20000)

        # NLL (full, not in-support restricted)
        nll = behavior_nll_per_agent_step(mech_est, te, infer_learner, n_actions, init_q=init_q, scalable_count=False)

        # NLL restricted to in-support contexts
        nll_in = behavior_nll_in_support(
            mech_est, te, infer_learner, n_actions,
            support_vocab_sorted=support_vocab,
            init_q=init_q,
            scalable_count=False
        )

        # Counterfactual learner shift
        cf_learner = LogitQRule(
            n_agents, n_actions,
            alpha=gen_cfg.alpha * eval_cfg.cf_alpha_scale,
            beta=gen_cfg.beta * eval_cfg.cf_beta_scale,
            eps=min(0.5, gen_cfg.eps * eval_cfg.cf_eps_scale)
        )

        # CF KL (unrestricted)
        cfkl = counterfactual_kl(
            mech_true, mech_est, cf_learner,
            n_agents, n_actions, device,
            E=200, T=min(35, data_cfg.horizon),
            scalable_count=False
        )

        # CF KL restricted to in-support contexts
        cfkl_in = counterfactual_kl_in_support(
            mech_true, mech_est, cf_learner,
            n_agents, n_actions, device,
            support_vocab_sorted=support_vocab,
            E=200, T=min(35, data_cfg.horizon),
            scalable_count=False
        )

        return {
            "mse_on": float(mse_on),
            "diff_mse": float(diff_mse),
            "nll_test": float(nll),
            "nll_test_in_support": float(nll_in),
            "cfkl_params": float(cfkl),
            "cfkl_params_in_support": float(cfkl_in),
        }

    # ----------------------------
    # Train + per-epoch log wrapper
    # ----------------------------
    def fit_with_epoch_eval(trainer: DIMLTrainer, tag: str):
        """
        Runs training exactly like DIMLTrainer.fit, but with per-epoch evaluation appended to curves_rows.
        """
        Etr, Ttr, _ = train_actions.shape
        idx = torch.arange(Etr, device=device)

        for ep in range(1, train_cfg.epochs + 1):
            t_ep0 = time.time()
            perm = idx[torch.randperm(Etr)]
            losses = []

            for s in range(0, Etr, train_cfg.batch_size):
                b = perm[s:s + train_cfg.batch_size]
                batch = train_actions[b]

                if train_cfg.trunc_bptt < Ttr:
                    t0 = torch.randint(0, Ttr - train_cfg.trunc_bptt, (1,), device=device).item()
                    batch = batch[:, t0:t0 + train_cfg.trunc_bptt, :]

                trainer.opt.zero_grad(set_to_none=True)
                loss = trainer.nll_batch(batch)
                loss.backward()
                if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainer.mech.parameters(), train_cfg.grad_clip)
                trainer.opt.step()
                losses.append(loss.item())

            dt = time.time() - t_ep0
            mean_loss = float(np.mean(losses))

            # Compute epoch evals
            ev = eval_epoch(trainer.mech, trainer.init_q, tag=tag)

            # Append to curves
            row = {
                "tag": tag,
                "epoch": ep,
                "train_loss": mean_loss,
                "epoch_seconds": float(dt),
                "train_s": float(dt),
                **ev
            }
            curves_rows.append(row)

            # Print
            if (ep == 1) or (ep % train_cfg.verbose_every == 0) or (ep == train_cfg.epochs):
                print(
                    f"[{tag}] epoch {ep:04d} | loss {mean_loss:.4f} | "
                    f"mse_on {ev['mse_on']:.4f} | diff_mse {ev['diff_mse']:.4f} | "
                    f"nll {ev['nll_test']:.4f} | nll_in {ev['nll_test_in_support']:.4f} | "
                    f"cfkl {ev['cfkl_params']:.4f} | cfkl_in {ev['cfkl_params_in_support']:.4f} | "
                    f"sec {dt:.2f}"
                )

    # ---- DIML ----
    print(f"\n[{name}] Train DIML (neural/param mechanism class)...")
    diml = DIMLTrainer(
        mech=mech_class_for_diml,
        learner=infer_learner,
        n_agents=n_agents,
        n_actions=n_actions,
        device=device,
        train_cfg=train_cfg,
        scalable_count=False
    )
    fit_with_epoch_eval(diml, tag=f"{name}/DIML")

    # ---- DIML Wrong learner ----
    print(f"\n[{name}] Train DIML-WrongLearner (beta mismatch)...")
    infer_wrong = LogitQRule(n_agents, n_actions, infer_cfg.alpha, infer_cfg.beta * 0.6, infer_cfg.eps)
    diml_wrong = DIMLTrainer(
        mech=type(mech_class_for_diml)(*getattr(mech_class_for_diml, "init_args", ())),
        learner=infer_wrong,
        n_agents=n_agents,
        n_actions=n_actions,
        device=device,
        train_cfg=train_cfg,
        scalable_count=False
    )
    fit_with_epoch_eval(diml_wrong, tag=f"{name}/DIML_Wrong")

    # ---- Tabular MLE (oracle-ish) ----
    tab = None
    tab_time = None
    K = n_actions ** n_agents
    if enable_tabular and (K <= 20000):
        print(f"\n[{name}] Train Tabular MLE baseline (K={K})...")
        tab_mech = TabularMechanism(n_agents, n_actions).to(device)
        tab_trainer = DIMLTrainer(
            mech=tab_mech,
            learner=infer_learner,
            n_agents=n_agents,
            n_actions=n_actions,
            device=device,
            train_cfg=train_cfg,
            scalable_count=False
        )
        tt0 = time.time()
        fit_with_epoch_eval(tab_trainer, tag=f"{name}/TabularMLE")
        tab_time = time.time() - tt0
        tab = tab_trainer
    else:
        tab_time = None

    # ---- Structured param MLE baseline (if provided) ----
    structured = None
    structured_time = None
    if mech_class_for_structured is not None:
        print(f"\n[{name}] Train Structured Param MLE baseline...")
        struct_trainer = DIMLTrainer(
            mech=mech_class_for_structured,
            learner=infer_learner,
            n_agents=n_agents,
            n_actions=n_actions,
            device=device,
            train_cfg=train_cfg,
            scalable_count=False
        )
        st0 = time.time()
        fit_with_epoch_eval(struct_trainer, tag=f"{name}/StructMLE")
        structured_time = time.time() - st0
        structured = struct_trainer

    # ---- JointBC (behavior only) ----
    print(f"\n[{name}] Train JointBC baseline...")
    jbc = train_jointbc(train_actions, window=5, epochs=35, lr=3e-3, batch_size=512, verbose=True)

    # ---- Equilibrium-only inverse (static) ----
    u_hat_eoi = equilibrium_only_inverse_u(train_actions, beta=gen_cfg.beta)

    # ---- Final evaluation summary ----
    def eval_model(tag: str, mech_est: Optional[nn.Module], init_q: Optional[torch.Tensor]):
        row: Dict[str, Any] = {
            "exp": name,
            "model": tag,
            "n_agents": n_agents,
            "n_actions": n_actions,
            "episodes": data_cfg.n_episodes,
            "horizon": data_cfg.horizon,
            "train_frac_time": data_cfg.train_frac_time,
            "static_only_last_frac": data_cfg.static_only_last_frac,
            "gen_alpha": gen_cfg.alpha, "gen_beta": gen_cfg.beta, "gen_eps": gen_cfg.eps,
            "infer_alpha": infer_cfg.alpha, "infer_beta": infer_cfg.beta, "infer_eps": infer_cfg.eps,
            "gen_seconds": gen_sec,
            "K_joint": float(n_actions ** n_agents),
        }
        if mech_est is None:
            return row

        row["mse_off"] = mechanism_mse_offsupport(mech_est, mech_true, n_agents, n_actions, device, probe_k=eval_cfg.probe_k)
        row["mse_on"] = mechanism_mse_onsupport(mech_est, mech_true, test_actions)
        row["diff_mse"] = payoff_difference_mse(mech_est, mech_true, test_actions, n_actions, max_samples=20000)

        row["nll_test"] = behavior_nll_per_agent_step(mech_est, test_actions, infer_learner, n_actions, init_q=init_q, scalable_count=False)
        row["nll_test_in_support"] = behavior_nll_in_support(
            mech_est, test_actions, infer_learner, n_actions,
            support_vocab_sorted=support_vocab,
            init_q=init_q,
            scalable_count=False
        )

        cf_learner = LogitQRule(n_agents, n_actions,
                                alpha=gen_cfg.alpha * eval_cfg.cf_alpha_scale,
                                beta=gen_cfg.beta * eval_cfg.cf_beta_scale,
                                eps=min(0.5, gen_cfg.eps * eval_cfg.cf_eps_scale))
        row["cfkl_params"] = counterfactual_kl(mech_true, mech_est, cf_learner, n_agents, n_actions, device, E=300, T=min(50, data_cfg.horizon))
        row["cfkl_params_in_support"] = counterfactual_kl_in_support(
            mech_true, mech_est, cf_learner,
            n_agents, n_actions, device,
            support_vocab_sorted=support_vocab,
            E=300, T=min(50, data_cfg.horizon),
            scalable_count=False
        )
        return row

    # DIML row
    results_rows.append(eval_model("DIML", diml.mech, diml.init_q))
    results_rows[-1]["train_time_sec"] = sum(r["epoch_seconds"] for r in curves_rows if r["tag"] == f"{name}/DIML")

    # Wrong learner
    results_rows.append(eval_model("DIML_Wrong", diml_wrong.mech, diml_wrong.init_q))
    results_rows[-1]["train_time_sec"] = sum(r["epoch_seconds"] for r in curves_rows if r["tag"] == f"{name}/DIML_Wrong")

    # Tabular
    if tab is not None:
        results_rows.append(eval_model("TabularMLE", tab.mech, tab.init_q))
        results_rows[-1]["train_time_sec"] = tab_time
    else:
        results_rows.append({
            "exp": name, "model": "TabularMLE",
            "n_agents": n_agents, "n_actions": n_actions,
            "episodes": data_cfg.n_episodes, "horizon": data_cfg.horizon,
            "tabular_status": "SKIPPED_OR_TIMEOUT",
            "K_joint": float(n_actions ** n_agents),
        })

    # Structured
    if structured is not None:
        results_rows.append(eval_model("StructMLE", structured.mech, structured.init_q))
        results_rows[-1]["train_time_sec"] = structured_time

    # JointBC (behavior only)
    results_rows.append({
        "exp": name, "model": "JointBC",
        "n_agents": n_agents, "n_actions": n_actions,
        "episodes": data_cfg.n_episodes, "horizon": data_cfg.horizon,
        "jointbc_acc": jointbc_accuracy(jbc, test_actions),
        "note": "behavior_only_no_mechanism"
    })

    # EOI (behavior-only-ish; not a mechanism)
    results_rows.append({
        "exp": name, "model": "EOI_StaticInverse",
        "n_agents": n_agents, "n_actions": n_actions,
        "episodes": data_cfg.n_episodes, "horizon": data_cfg.horizon,
        "note": "partial_inverse_u_i(a_i, mode(a_-i)); no full mechanism"
    })

    # Print quick summary (final)
    print(f"\n--- [{name}] Summary (key metrics, final) ---")
    for r in results_rows:
        if r.get("exp") != name:
            continue
        if r["model"] in ["DIML", "TabularMLE", "StructMLE", "DIML_Wrong"]:
            print(r["model"],
                  "mse_on=", f"{r.get('mse_on', float('nan')):.4f}",
                  "mse_off=", f"{r.get('mse_off', float('nan')):.4f}",
                  "diff_mse=", f"{r.get('diff_mse', float('nan')):.4f}",
                  "nll_test=", f"{r.get('nll_test', float('nan')):.4f}",
                  "nll_test_in_support=", f"{r.get('nll_test_in_support', float('nan')):.4f}",
                  "cfkl_params=", f"{r.get('cfkl_params', float('nan')):.4f}",
                  "cfkl_params_in_support=", f"{r.get('cfkl_params_in_support', float('nan')):.4f}",
                  "train_s=", f"{r.get('train_time_sec', float('nan')):.2f}")
        if r["model"] == "JointBC":
            print("JointBC acc=", f"{r['jointbc_acc']:.4f}")

    return curves_rows, results_rows


def run_large_scale_scalability_experiment(
    name: str,
    n_agents: int,
    n_actions: int,
    n_episodes: int,
    horizon: int,
    train_cfg: TrainCfg,
    gen_cfg: LearnerCfg,
    infer_cfg: LearnerCfg,
    eval_cfg: EvalCfg,
    device: torch.device
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Large-scale experiment designed to show DIML scales while Tabular must be skipped.
    Uses SymmetricCountMechanism (scalable counterfactuals).
    Adds per-epoch evaluation into curves_rows.
    """
    curves_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    print(f"\n=== [{name}] LARGE SCALE (n={n_agents}, A={n_actions}) ===")

    # true mechanism
    mech_true = SymmetricCountMechanism(n_actions=n_actions, hidden=256, depth=2).to(device)
    for p in mech_true.parameters():
        p.requires_grad = False

    gen_learner = LogitQRule(n_agents, n_actions, gen_cfg.alpha, gen_cfg.beta, gen_cfg.eps)

    # generate data
    t0 = time.time()
    ds = generate_dataset_stateless(mech_true, gen_learner, n_agents, n_actions, n_episodes, horizon, device, scalable_count=True)
    gen_sec = time.time() - t0
    actions = ds["actions"]
    train_actions, test_actions = split_train_test(actions, 0.8)

    support_vocab = build_support_vocab(train_actions, n_actions)

    infer_learner = LogitQRule(n_agents, n_actions, infer_cfg.alpha, infer_cfg.beta, infer_cfg.eps)

    # DIML: learn same symmetric count mechanism class
    mech_diml = SymmetricCountMechanism(n_actions=n_actions, hidden=256, depth=2).to(device)
    diml = DIMLTrainer(
        mech=mech_diml,
        learner=infer_learner,
        n_agents=n_agents,
        n_actions=n_actions,
        device=device,
        train_cfg=train_cfg,
        scalable_count=True
    )

    # ----------------------------
    # Helper: per-epoch eval (large)
    # ----------------------------
    @torch.no_grad()
    def eval_epoch_large(mech_est: nn.Module, init_q: Optional[torch.Tensor], tag: str) -> Dict[str, float]:
        # subsample test episodes for speed
        Etest = test_actions.shape[0]
        if Etest > 80:
            perm = torch.randperm(Etest, device=device)[:80]
            te = test_actions[perm]
        else:
            te = test_actions

        mse_on = mechanism_mse_onsupport(mech_est, mech_true, te)
        diff_mse = payoff_difference_mse(mech_est, mech_true, te, n_actions, max_samples=50000)

        nll = behavior_nll_per_agent_step(mech_est, te, infer_learner, n_actions, init_q=init_q, scalable_count=True)
        nll_in = behavior_nll_in_support(
            mech_est, te, infer_learner, n_actions,
            support_vocab_sorted=support_vocab,
            init_q=init_q,
            scalable_count=True
        )

        cf_learner = LogitQRule(
            n_agents, n_actions,
            alpha=gen_cfg.alpha * eval_cfg.cf_alpha_scale,
            beta=gen_cfg.beta * eval_cfg.cf_beta_scale,
            eps=min(0.5, gen_cfg.eps * eval_cfg.cf_eps_scale)
        )
        cfkl = counterfactual_kl(mech_true, mech_est, cf_learner, n_agents, n_actions, device,
                                 E=120, T=min(30, horizon), scalable_count=True)
        cfkl_in = counterfactual_kl_in_support(
            mech_true, mech_est, cf_learner,
            n_agents, n_actions, device,
            support_vocab_sorted=support_vocab,
            E=160, T=min(30, horizon),
            scalable_count=True
        )

        return {
            "mse_on": float(mse_on),
            "diff_mse": float(diff_mse),
            "nll_test": float(nll),
            "nll_test_in_support": float(nll_in),
            "cfkl_params": float(cfkl),
            "cfkl_params_in_support": float(cfkl_in),
        }

    # ----------------------------
    # Training loop with per-epoch eval
    # ----------------------------
    print(f"[{name}] Train DIML (scalable count-based) ...")
    Etr, Ttr, _ = train_actions.shape
    idx = torch.arange(Etr, device=device)

    for ep in range(1, train_cfg.epochs + 1):
        t_ep0 = time.time()
        perm = idx[torch.randperm(Etr)]
        losses = []

        for s in range(0, Etr, train_cfg.batch_size):
            b = perm[s:s + train_cfg.batch_size]
            batch = train_actions[b]

            if train_cfg.trunc_bptt < Ttr:
                t0 = torch.randint(0, Ttr - train_cfg.trunc_bptt, (1,), device=device).item()
                batch = batch[:, t0:t0 + train_cfg.trunc_bptt, :]

            diml.opt.zero_grad(set_to_none=True)
            loss = diml.nll_batch(batch)
            loss.backward()
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(diml.mech.parameters(), train_cfg.grad_clip)
            diml.opt.step()
            losses.append(loss.item())

        dt = time.time() - t_ep0
        mean_loss = float(np.mean(losses))

        ev = eval_epoch_large(diml.mech, diml.init_q, tag=f"{name}/DIML_Large")

        row = {
            "tag": f"{name}/DIML_Large",
            "epoch": ep,
            "train_loss": mean_loss,
            "epoch_seconds": float(dt),
            "train_s": float(dt),
            **ev
        }
        curves_rows.append(row)

        if (ep == 1) or (ep % train_cfg.verbose_every == 0) or (ep == train_cfg.epochs):
            print(
                f"[{name}/DIML_Large] epoch {ep:04d} | loss {mean_loss:.4f} | "
                f"mse_on {ev['mse_on']:.4f} | diff_mse {ev['diff_mse']:.4f} | "
                f"nll {ev['nll_test']:.4f} | nll_in {ev['nll_test_in_support']:.4f} | "
                f"cfkl {ev['cfkl_params']:.4f} | cfkl_in {ev['cfkl_params_in_support']:.4f} | "
                f"sec {dt:.2f}"
            )

    # Tabular must be skipped; record why
    K_joint = float(n_actions ** n_agents) if n_actions ** n_agents < 1e308 else float("inf")

    # Final evaluation row
    row = {
        "exp": name,
        "model": "DIML_Large",
        "n_agents": n_agents,
        "n_actions": n_actions,
        "episodes": n_episodes,
        "horizon": horizon,
        "gen_seconds": gen_sec,
        "K_joint": K_joint,
        "train_time_sec": sum(r["epoch_seconds"] for r in curves_rows if r["tag"] == f"{name}/DIML_Large"),
        "mse_on": mechanism_mse_onsupport(diml.mech, mech_true, test_actions),
        "diff_mse": payoff_difference_mse(diml.mech, mech_true, test_actions, n_actions, max_samples=50000),
        "nll_test": behavior_nll_per_agent_step(diml.mech, test_actions, infer_learner, n_actions,
                                               init_q=diml.init_q, scalable_count=True),
        "nll_test_in_support": behavior_nll_in_support(
            diml.mech, test_actions, infer_learner, n_actions,
            support_vocab_sorted=support_vocab,
            init_q=diml.init_q,
            scalable_count=True
        )
    }

    cf_learner = LogitQRule(n_agents, n_actions,
                            alpha=gen_cfg.alpha * eval_cfg.cf_alpha_scale,
                            beta=gen_cfg.beta * eval_cfg.cf_beta_scale,
                            eps=min(0.5, gen_cfg.eps * eval_cfg.cf_eps_scale))
    row["cfkl_params"] = counterfactual_kl(mech_true, diml.mech, cf_learner, n_agents, n_actions, device,
                                          E=120, T=min(35, horizon), scalable_count=True)
    row["cfkl_params_in_support"] = counterfactual_kl_in_support(
        mech_true, diml.mech, cf_learner,
        n_agents, n_actions, device,
        support_vocab_sorted=support_vocab,
        E=200, T=min(35, horizon),
        scalable_count=True
    )

    results_rows.append(row)
    results_rows.append({
        "exp": name,
        "model": "TabularMLE",
        "n_agents": n_agents,
        "n_actions": n_actions,
        "episodes": n_episodes,
        "horizon": horizon,
        "K_joint": K_joint,
        "tabular_status": "TIMEOUT_SKIPPED_A^n_ASTRONOMICAL",
        "note": "Tabular requires parameters O(|A|^n * n); infeasible at large n."
    })

    print(f"\n--- [{name}] Summary ---")
    print("DIML_Large",
          "mse_on=", f"{row['mse_on']:.4f}",
          "diff_mse=", f"{row['diff_mse']:.4f}",
          "nll_test=", f"{row['nll_test']:.4f}",
          "nll_test_in_support=", f"{row['nll_test_in_support']:.4f}",
          "cfkl_params=", f"{row['cfkl_params']:.4f}",
          "cfkl_params_in_support=", f"{row['cfkl_params_in_support']:.4f}",
          "train_s=", f"{row['train_time_sec']:.2f}")
    print("TabularMLE skipped:", results_rows[-1]["tabular_status"])

    return curves_rows, results_rows


# ================================================================================
# Main: run suite + save CSV
# ================================================================================
def main():
    suite = SuiteCfg(seed=0, out_dir="./diml_results", device="auto")
    set_seed(suite.seed)
    ensure_dir(suite.out_dir)

    if suite.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(suite.device)
    print("Device:", device)

    # Global configs
    eval_cfg = EvalCfg(probe_k=8000, cf_alpha_scale=0.6, cf_beta_scale=1.4, cf_eps_scale=1.5, cf_initq_std=0.6, ece_bins=15)
    gen_cfg  = LearnerCfg(alpha=0.25, beta=3.0, eps=0.06)
    infer_cfg = LearnerCfg(alpha=0.25, beta=3.0, eps=0.06)

    # For small/medium tasks, keep episodes low as you requested
    train_cfg_small = TrainCfg(epochs=100, batch_size=32, lr=1e-3, trunc_bptt=50, verbose_every=10, reg_l2_pay=1e-4)
    data_small = DataCfg(n_episodes=60, horizon=60, train_frac_time=1.0, static_only_last_frac=0.0)

    all_curves: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []

    # -------------------------
    # EXP A: Unstructured neural (small joint space => Tabular feasible)
    # -------------------------
    n_agents, n_actions = 3, 6
    mech_true = NeuralJointMechanism(n_agents, n_actions, hidden=128, depth=2).to(device)
    for p in mech_true.parameters():
        p.requires_grad = False

    mech_diml = NeuralJointMechanism(n_agents, n_actions, hidden=128, depth=2).to(device)
    # "init_args" so DIML-Wrong can re-instantiate; used in runner
    mech_diml.init_args = (n_agents, n_actions, 128, 2)

    curves, results = run_small_medium_experiment(
        name="E1_UnstructuredNeural_3A",
        mech_true=mech_true,
        mech_class_for_diml=mech_diml,
        mech_class_for_structured=None,
        n_agents=n_agents, n_actions=n_actions,
        data_cfg=data_small,
        train_cfg=train_cfg_small,
        gen_cfg=gen_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        device=device,
        enable_tabular=True
    )
    all_curves += curves
    all_results += results

    # -------------------------
    # EXP B: Congestion (structured baseline available)
    # -------------------------
    n_agents, n_actions = 4, 5
    mech_true = CongestionParamMechanism(n_agents, n_actions).to(device)
    for p in mech_true.parameters():
        p.requires_grad = False

    mech_diml = NeuralJointMechanism(n_agents, n_actions, hidden=128, depth=2).to(device)
    mech_diml.init_args = (n_agents, n_actions, 128, 2)

    mech_struct = CongestionParamMechanism(n_agents, n_actions).to(device)

    data = DataCfg(n_episodes=60, horizon=60, train_frac_time=1.0, static_only_last_frac=0.0)
    curves, results = run_small_medium_experiment(
        name="E2_Congestion_4A",
        mech_true=mech_true,
        mech_class_for_diml=mech_diml,
        mech_class_for_structured=mech_struct,
        n_agents=n_agents, n_actions=n_actions,
        data_cfg=data,
        train_cfg=train_cfg_small,
        gen_cfg=gen_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        device=device,
        enable_tabular=True  # K=625 => feasible
    )
    all_curves += curves
    all_results += results

    # -------------------------
    # EXP C: Public goods (structured baseline available)
    # -------------------------
    n_agents, n_actions = 3, 7
    mech_true = PublicGoodsParamMechanism(n_agents, n_actions).to(device)
    for p in mech_true.parameters():
        p.requires_grad = False

    mech_diml = NeuralJointMechanism(n_agents, n_actions, hidden=128, depth=2).to(device)
    mech_diml.init_args = (n_agents, n_actions, 128, 2)

    mech_struct = PublicGoodsParamMechanism(n_agents, n_actions).to(device)

    data = DataCfg(n_episodes=60, horizon=60, train_frac_time=1.0, static_only_last_frac=0.0)
    curves, results = run_small_medium_experiment(
        name="E3_PublicGoods_3A",
        mech_true=mech_true,
        mech_class_for_diml=mech_diml,
        mech_class_for_structured=mech_struct,
        n_agents=n_agents, n_actions=n_actions,
        data_cfg=data,
        train_cfg=train_cfg_small,
        gen_cfg=gen_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        device=device,
        enable_tabular=False  # K=343 => feasible but not needed; keep runtime down
    )
    all_curves += curves
    all_results += results

    # -------------------------
    # EXP G: Large-scale scalability (hundreds of agents, dozens of actions)
    # -------------------------
    # Use smaller episodes to keep runtime manageable, but large enough to demonstrate scaling.
    train_cfg_large = TrainCfg(epochs=200, batch_size=8, lr=1e-3, trunc_bptt=30, verbose_every=10, reg_l2_pay=1e-4)
    gen_cfg_large = LearnerCfg(alpha=0.20, beta=6.0, eps=0.02)
    infer_cfg_large = LearnerCfg(alpha=0.20, beta=6.0, eps=0.02)

    for (n_big, A_big, E_big, T_big) in [
        (40, 10, 24, 30),
        (80, 20, 24, 30),
        (120, 25, 24, 30),
        (200, 30, 20, 30),
        (300, 40, 16, 25),
    ]:
        curves, results = run_large_scale_scalability_experiment(
            name=f"E7_LargeScale_n{n_big}_A{A_big}",
            n_agents=n_big,
            n_actions=A_big,
            n_episodes=E_big,
            horizon=T_big,
            train_cfg=train_cfg_large,
            gen_cfg=gen_cfg_large,
            infer_cfg=infer_cfg_large,
            eval_cfg=eval_cfg,
            device=device
        )
        all_curves += curves
        all_results += results

    # Save CSVs
    curves_path = os.path.join(suite.out_dir, "training_curves.csv")
    results_path = os.path.join(suite.out_dir, "evaluation_results.csv")
    save_csv(all_curves, curves_path)
    save_csv(all_results, results_path)

    print("\nSaved:")
    print(" -", curves_path)
    print(" -", results_path)

    # Print compact summary
    print("\n================== COMPACT SUMMARY ==================")
    for r in all_results:
        if r.get("model") in ["DIML", "TabularMLE", "StructMLE", "DIML_Wrong", "DIML_Large"]:
            exp = r.get("exp")
            model = r.get("model")
            print(exp, "|", model,
                  "| mse_on:", f"{r.get('mse_on', float('nan')):.4f}" if "mse_on" in r else "NA",
                  "| mse_off:", f"{r.get('mse_off', float('nan')):.4f}" if "mse_off" in r else "NA",
                  "| diff_mse:", f"{r.get('diff_mse', float('nan')):.4f}" if "diff_mse" in r else "NA",
                  "| nll:", f"{r.get('nll_test', float('nan')):.4f}" if "nll_test" in r else "NA",
                  "| cfkl_params:", f"{r.get('cfkl_params', float('nan')):.4f}" if "cfkl_params" in r else "NA",
                  "| train_s:", f"{r.get('train_time_sec', float('nan')):.2f}")
        if r.get("model") == "TabularMLE" and r.get("tabular_status"):
            print(r["exp"], "| TabularMLE |", r["tabular_status"], "| K_joint=", r.get("K_joint"))
    print("=====================================================")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mech = SymmetricCountMechanism(n_actions=25).to(device)
    a = torch.randint(0, 25, (4, 120), device=device)
    p = mech(a)  # should work: (4,120)
    u = counterfactual_payoffs_symmetric_count(mech, a, 25)
    print(p.shape, u.shape)  # (4,120) and (4,120,25)

    main()
