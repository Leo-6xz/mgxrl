"""Attribution over the RL policy.

Two corrections to the naive approach:

1. KernelSHAP is the wrong tool. It needs thousands of forward passes per
   decision. You are explaining a control loop that runs at 10 ms and you will
   want attributions for tens of thousands of decisions across an evaluation
   sweep. Use gradient-based attribution (Integrated Gradients / DeepSHAP) on
   the actor network -- exact enough, three orders of magnitude cheaper.

2. The policy output is a 20-dimensional action, not a scalar. SHAP over a
   vector output is ill-posed unless you say what you are explaining. Three
   well-defined targets are supported:
       "action_norm" -- total control effort (why did the agent act at all?)
       "action_dim"  -- one specified actuator (why did DG3's zeta move?)
       "value"       -- the critic (why is this state judged bad?)
   We state the one we used.
"""

from __future__ import annotations

import numpy as np


class PolicyAttributor:
    def __init__(self, model, feature_names, target="action_norm", dim=None, baseline=None):
        self.model = model
        self.feature_names = list(feature_names)
        self.target = target
        self.dim = dim
        self.baseline = baseline      # (d,) reference state; default = zeros

    # ---------------------------------------------------------------- scalar

    def _scalarize(self, obs_batch):
        """Map a batch of observations to the scalar being explained.

        Accepts a torch tensor OR a numpy array. This used to unconditionally
        call np.asarray(obs_batch), which RAISES on the requires_grad tensor
        that integrated_gradients passes in ("Can't call numpy() on Tensor that
        requires grad") -- and had it not raised it would have severed the
        autograd graph, so torch.autograd.grad would then have failed with
        "one of the differentiated Tensors appears not to have been used".
        Integrated gradients could never have run; Phase 6 would have died on
        its first call.
        """
        import torch

        policy = self.model.policy
        if torch.is_tensor(obs_batch):
            obs_t = obs_batch.to(device=policy.device, dtype=torch.float32)
        else:
            obs_t = torch.as_tensor(np.asarray(obs_batch, np.float32),
                                    device=policy.device)
        if self.target == "value":
            return policy.predict_values(obs_t).squeeze(-1)
        dist = policy.get_distribution(obs_t)
        mean = dist.distribution.mean
        if self.target == "action_dim":
            return mean[:, int(self.dim)]
        # clamp before sqrt: d/dx sqrt(x) is infinite at 0, and the mean action
        # is exactly 0 wherever the policy has learned to do nothing -- which
        # under trust weights is the common case, not an edge case.
        return (mean ** 2).sum(-1).clamp_min(1e-12).sqrt()

    def integrated_gradients(self, obs, steps=64):
        """Integrated Gradients: exact, complete, and cheap.

        Completeness (sum of attributions == f(x) - f(baseline)) is a property
        you can assert in the paper and verify numerically -- see
        `check_completeness` below. Do assert it; it is free credibility.
        """
        import torch

        policy = self.model.policy
        x = np.asarray(obs, np.float32).reshape(-1)
        b = np.zeros_like(x) if self.baseline is None else np.asarray(self.baseline, np.float32)

        # Midpoint rule. np.linspace(0, 1, steps) includes both endpoints, and
        # averaging over them biases the Riemann sum -- which shows up directly
        # as completeness error.
        
        alphas = ((np.arange(steps) + 0.5) / steps).reshape(-1, 1)
        path = b + alphas * (x - b)
        # torch.tensor, not torch.as_tensor: as_tensor has no requires_grad
        # parameter and raises TypeError. as_tensor would also be wrong in
        # principle here -- it may alias the input rather than create the leaf
        # that autograd.grad below differentiates with respect to.
        path_t = torch.tensor(path, dtype=torch.float32,
                              device=policy.device, requires_grad=True)
        out = self._scalarize(path_t)
        grads = torch.autograd.grad(out.sum(), path_t)[0].cpu().numpy()
        return (x - b) * grads.mean(axis=0)

    def check_completeness(self, obs, attributions, tol=0.05):
        x = np.asarray(obs, np.float32)
        b = np.zeros_like(x) if self.baseline is None else np.asarray(self.baseline, np.float32)
        with_ = float(self._scalarize(x[None]).detach().cpu().numpy()[0])
        without = float(self._scalarize(b[None]).detach().cpu().numpy()[0])
        err = abs(attributions.sum() - (with_ - without)) / (abs(with_ - without) + 1e-9)
        return {"relative_error": float(err), "passes": bool(err < tol)}

    def top_k(self, attributions, k=5):
        idx = np.argsort(-np.abs(attributions))[:k]
        return [
            {"feature": self.feature_names[i], "attribution": float(attributions[i]),
             "rank": r + 1}
            for r, i in enumerate(idx)
        ]