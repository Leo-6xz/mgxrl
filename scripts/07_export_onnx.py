#!/usr/bin/env python3
"""Export the trained actor to ONNX for import into Simulink.

Exports the DETERMINISTIC MEAN action only -- not the stochastic policy. PPO's
actor outputs a distribution; Simulink needs a function. Deploying the sampled
policy would inject exploration noise into a live controller, which is both
unsafe and unreproducible.

Also exports the VecNormalize statistics. Forgetting these is the single most
common sim-to-sim transfer bug: the agent was trained on normalized
observations, so feeding it raw ones produces confident nonsense rather than an
obvious failure.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, help="e.g. runs/ppo/agent_final.zip")
    ap.add_argument("--vecnorm", default=None, help="runs/ppo/vecnormalize.pkl")
    ap.add_argument("--out", default="runs/ppo/actor.onnx")
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO

    model = PPO.load(a.agent, device="cpu")
    obs_dim = int(model.observation_space.shape[0])

    class ActorMean(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            if isinstance(features, tuple):
                features = features[0]
            latent = self.policy.mlp_extractor.forward_actor(features)
            # CLAMP, not tanh. SB3's MlpPolicy uses squash_output=False for a
            # Box action space: predict() clips the mean action to the bounds,
            # it does not squash it. Exporting tanh therefore deployed a
            # DIFFERENT controller from the one evaluated -- and the error is
            # worst exactly where it matters, since tanh(-1) = -0.76, so a
            # commanded full isolation would arrive in Simulink as 24% trust
            # rather than 0.
            return torch.clamp(self.policy.action_net(latent), -1.0, 1.0)

    net = ActorMean(model.policy).eval()
    dummy = torch.zeros(1, obs_dim)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.onnx.export(
        net, dummy, a.out,
        input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=13,
    )
    print(f"Wrote {a.out}   obs_dim={obs_dim}  act_dim={model.action_space.shape[0]}")
    print(f"Set obsInfo = rlNumericSpec([{obs_dim} 1]) in matlab/deploy_policy.m")
    print("Action semantics: the 20 outputs are TRUST OFFSETS, not derivative")
    print("residuals. Simulink must apply trust = min(max(1 + a, 0), 1) as a")
    print("per-source weight on the consensus graph -- see matlab/mg_dynamics.m.")
    print("Summing them into dxi/dzeta is the ABANDONED action space and will")
    print("produce a controller unrelated to the one that was trained.")

    if a.vecnorm and os.path.exists(a.vecnorm):
        # Unpickle directly: VecNormalize.load(path, venv) calls
        # set_venv(venv) unconditionally and dereferences venv.num_envs, so
        # venv=None raises AttributeError. The pickle holds all the statistics.
        import pickle
        with open(a.vecnorm, "rb") as fh:
            vn = pickle.load(fh)
        got = int(np.asarray(vn.obs_rms.mean).reshape(-1).shape[0])
        if got != obs_dim:
            raise SystemExit(
                f"ERROR: {a.vecnorm} has {got} features but the network takes "
                f"{obs_dim}. These are from different observation layouts."
            )
        stats = {"mean": vn.obs_rms.mean.tolist(),
                 "var": vn.obs_rms.var.tolist(),
                 "clip_obs": float(vn.clip_obs), "epsilon": float(vn.epsilon)}
        path = os.path.splitext(a.out)[0] + "_obsnorm.json"
        json.dump(stats, open(path, "w"))
        print(f"Wrote {path}")
        print("Apply in Simulink BEFORE the network: "
              "o = clip((o - mean)/sqrt(var + eps), -clip_obs, clip_obs)")
    else:
        print("\nWARNING: no VecNormalize stats exported. If you trained with "
              "VecNormalize (the default), the ONNX policy WILL misbehave in "
              "Simulink without them. Pass --vecnorm runs/ppo/vecnormalize.pkl")


if __name__ == "__main__":
    main()
