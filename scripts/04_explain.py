#!/usr/bin/env python3
"""Stage 4: attributions + LLM explanations + faithfulness battery.

Three fixes over the previous version, all of which silently corrupted the
output rather than raising:

1. OBSERVATION NORMALISATION. The policy was trained under
   VecNormalize(norm_obs=True) and was being given raw observations here, so
   both the actions being explained and the attributions computed around them
   described a controller that does not exist. Same bug as 03_evaluate.py.

2. FEATURE NAMES. `feature_names` listed six quantities per DG per step while
   the observation now carries eight (the consensus innovation was added on
   both channels). Every attribution label after the sixth block was attached
   to the wrong feature. There is now an assertion against env.obs_dim so this
   cannot drift again unnoticed.

3. STALE MEASUREMENT. `meas` was read once AFTER the rollout finished and then
   paired with all 25 records, so every explanation described the final state
   while claiming to explain an earlier action. That also invalidated the
   unverified-number rate, which is reported as a safety metric -- it was
   counting mismatches the model never caused.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from mgxrl.envs.resilient_env import ResilientMicrogridEnv
from mgxrl.xai.attribution import PolicyAttributor
from mgxrl.xai.faithfulness import aggregate
from mgxrl.llm.explain import (OllamaExplainer, TemplateExplainer, build_prompt,
                               numeric_consistency_check,
                               checker_false_positive_rate,
                               coverage_manifest, coverage_score,
                               relational_consistency_check,
                               GridContext, SYSTEM_PROMPT,
                               SYSTEM_PROMPT_ENUMERATE)


def feature_names(env):
    """Must match ResilientMicrogridEnv._observe exactly, in order."""
    quantities = ("V", "f", "P", "Q", "xi", "zeta", "inn_xi", "inn_zeta")
    base = [f"{q}_DG{i+1}" for q in quantities for i in range(env.n)]
    names = [f"t-{h}:{b}" for h in range(env.cfg.history_len - 1, -1, -1)
             for b in base]
    names += [f"detector_p{c}" for c in range(env._det_dim)]
    assert len(names) == env.obs_dim, (
        f"feature_names produces {len(names)} labels but obs_dim is "
        f"{env.obs_dim}. _observe changed and this list did not; every "
        f"attribution would be mislabelled."
    )
    return names


def load_vecnorm(path, expect_dim=None):
    """Unpickle the VecNormalize statistics.

    NOT VecNormalize.load(path, venv=None): that helper is just
    `pickle.load` followed by `set_venv(venv)`, and set_venv dereferences
    venv.num_envs unconditionally, so passing None raises AttributeError. The
    pickle holds everything needed -- obs_rms, clip_obs, epsilon -- and
    normalize_obs never touches the venv.
    """
    import pickle

    with open(path, "rb") as fh:
        vn = pickle.load(fh)
    if not hasattr(vn, "obs_rms") or vn.obs_rms is None:
        raise SystemExit(f"{path} holds no observation statistics.")
    vn.training = False          # freeze the running statistics
    vn.norm_reward = False
    got = int(np.asarray(vn.obs_rms.mean).reshape(-1).shape[0])
    if expect_dim is not None and got != expect_dim:
        raise SystemExit(
            f"\nERROR: {path} carries statistics for {got} observation "
            f"features, but the environment produces {expect_dim}.\n"
            f"These are from a different observation layout -- most likely a "
            f"run predating the consensus-innovation features, which took the "
            f"per-step frame from 6 to 8 quantities per DG. Retrain; the "
            f"checkpoint cannot be evaluated against this environment.\n"
        )
    return vn



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--allow-unnormalized", action="store_true")
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--n-explanations", type=int, default=25,
                    help="how many states to generate explanations for. This "
                         "was hardcoded to 25, so --n-states raised the cost of "
                         "the faithfulness battery without improving the LLM "
                         "numbers at all. At 25 the run-to-run spread in "
                         "weighted coverage is about 0.04, which is the same "
                         "size as the differences being compared.")
    ap.add_argument("--llm", default="llama3.1:8b")
    ap.add_argument("--timeout", type=float, default=300.0,
                    help="seconds per LLM call. The FIRST call also pays the "
                         "cold load of the model into memory -- ~5 GB for an "
                         "8B model -- which routinely exceeds the 60 s "
                         "default. Warm it once with `ollama run <model> hi` "
                         "and this matters less.")
    ap.add_argument("--reuse-faithfulness", action="store_true",
                    help="load faithfulness.json from --out instead of "
                         "recomputing. The battery is 100 states of integrated "
                         "gradients plus deletion curves; there is no reason to "
                         "pay for it again when only the LLM stage failed.")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--prompt-variant", default="baseline",
                    choices=["baseline", "enumerate"],
                    help="baseline: three least-trusted links shown, model "
                         "chooses what to mention. enumerate: only links below "
                         "full trust are shown AND the system prompt requires "
                         "every one of them to be named. Scoring is identical "
                         "in both, so the difference is presentation.")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from stable_baselines3 import PPO
    env = ResilientMicrogridEnv()
    model = PPO.load(a.agent)

    vn_path = a.vecnorm or os.path.join(os.path.dirname(a.agent), "vecnormalize.pkl")
    if os.path.exists(vn_path):
        vn = load_vecnorm(vn_path, env.obs_dim)
    elif a.allow_unnormalized:
        vn = None
    else:
        raise SystemExit(
            f"ERROR: no VecNormalize statistics at {vn_path}. Attributions "
            f"computed on unnormalised observations describe a controller that "
            f"was never trained. Pass --vecnorm or --allow-unnormalized."
        )

    def norm(o):
        o = np.asarray(o, dtype=np.float32).reshape(1, -1)
        return vn.normalize_obs(o) if vn is not None else o

    attributor = PolicyAttributor(model, feature_names(env), target="action_norm")

    obs, _ = env.reset(seed=0)
    states, records = [], []
    for _ in range(a.n_states):
        o = norm(obs)
        act, _ = model.predict(o, deterministic=True)
        act = np.asarray(act).reshape(-1)
        # The measurement is captured BEFORE stepping, so it is the state this
        # action was taken in -- not the state it led to.
        meas = env.mg.measure(noisy=False)
        flat = o.reshape(-1)
        states.append(flat.copy())
        attr = attributor.integrated_gradients(flat)
        records.append({"obs": flat.copy(), "action": act, "meas": meas,
                        "top": attributor.top_k(attr, 5),
                        "completeness": attributor.check_completeness(flat, attr)})
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()

    fa_path = os.path.join(a.out, "faithfulness.json")
    if a.reuse_faithfulness and os.path.exists(fa_path):
        fa = json.load(open(fa_path))
        print(f"Faithfulness battery: reusing {fa_path} "
              f"(n_states={fa.get('n_states')})")
    else:
        print("Faithfulness battery ...")
        fa = aggregate(attributor, model, states[:100])
        json.dump(fa, open(fa_path, "w"), indent=2)
    print(json.dumps(fa, indent=2))

    if not a.no_llm:
        ctx = GridContext(action_block="reduced" if a.prompt_variant == "enumerate"
                          else "top3")
        system = (SYSTEM_PROMPT_ENUMERATE if a.prompt_variant == "enumerate"
                  else SYSTEM_PROMPT)
        print(f"Prompt variant: {a.prompt_variant}")
        gen = OllamaExplainer(a.llm)
        gen.timeout = a.timeout      # set on the instance: the constructor
                                     # signature varies, generate() reads this
        tmpl = TemplateExplainer()

        # Warm-up. The first call loads the model into memory, so a timeout
        # here is a cold-start problem, not a broken server or a bad prompt.
        print(f"Warming {a.llm} (first call includes model load) ...")
        try:
            gen.generate("Reply with the single word: ready")
        except Exception as e:
            raise SystemExit(
                f"\nERROR: could not reach the model.\n  {type(e).__name__}: {e}\n"
                f"  404      -> model not pulled: ollama pull {a.llm}\n"
                f"  timeout  -> cold load exceeded {a.timeout:.0f}s; warm it "
                f"first with: ollama run {a.llm} \"ready\"\n"
                f"  refused  -> ollama is not running\n"
            )

        outs, failures = [], []
        n_explain = min(a.n_explanations, len(records))
        for i, rec in enumerate(records[:n_explain]):
            prompt = build_prompt(rec["meas"], rec["action"], rec["top"],
                                  None, ctx=ctx)
            try:
                text = gen.generate(prompt, system=system)
            except Exception as e:
                # One slow call must not discard the other 24.
                failures.append({"index": i, "error": f"{type(e).__name__}: {e}"})
                continue
            check = numeric_consistency_check(text, rec["meas"],
                                              action=rec["action"],
                                              prompt=prompt)
            manifest = coverage_manifest(rec["meas"], rec["action"], rec["top"],
                                         ctx=ctx)
            tmpl_text = tmpl.generate(prompt, meas=rec["meas"],
                                      action=rec["action"],
                                      attributions=rec["top"], ctx=ctx)
            outs.append({"explanation": text,
                         "template_baseline": tmpl_text,
                         "numeric_check": check,
                         "template_check": numeric_consistency_check(
                             tmpl_text, rec["meas"], action=rec["action"],
                             prompt=prompt),
                         "relational_check": relational_consistency_check(text),
                         "coverage": coverage_score(text, manifest),
                         "template_coverage": coverage_score(tmpl_text, manifest),
                         "top_features": rec["top"]})

        if failures:
            print(f"\n  {len(failures)} of {n_explain} generations failed:")
            for f in failures[:3]:
                print(f"    [{f['index']}] {f['error']}")

        # Show what was actually flagged. A rate this high is either a real
        # safety finding or a broken checker, and the difference is visible in
        # two examples -- a rounded value scored against an exact measurement,
        # or the digit in "DG3" parsed as a numeric claim, would put most of
        # the rate down to false positives.
        bad = [o for o in outs if not o["numeric_check"]["passes"]]
        for o in bad[:2]:
            print("\n  --- flagged generation ---")
            print("  " + str(o["explanation"])[:400].replace("\n", "\n  "))
            print(f"  check: {o['numeric_check']}")
            if o["numeric_check"]["unsourced"]:
                print(f"  (unsourced but true: {o['numeric_check']['unsourced']})")

        if outs:
            # Over SUCCESSFUL generations only. Counting a failed call as a
            # pass or a violation would corrupt a number reported as a safety
            # metric.
            viol = float(np.mean([not o["numeric_check"]["passes"] for o in outs]))
            unsrc = float(np.mean([bool(o["numeric_check"]["unsourced"])
                                   for o in outs]))
            tmpl_viol = float(np.mean([not o["template_check"]["passes"]
                                       for o in outs]))
            print(f"\nUnverified-number rate: {viol:.1%} over {len(outs)} "
                  f"generations  (report this; it is a safety metric)")
            rel_bad = [o for o in outs if not o["relational_check"]["passes"]]
            n_rel = sum(o["relational_check"]["n_relations"] for o in outs)
            print(f"\nFalse-comparison rate: {len(rel_bad) / len(outs):.1%} over "
                  f"{n_rel} explicit comparisons  (both numbers correct, the "
                  f"relation between them false -- invisible to the numeric check)")
            for o in rel_bad[:2]:
                v = o["relational_check"]["violations"][0]
                print(f"    claimed {v['lhs']} {v['relation']} {v['rhs']}: "
                      f"\"{v['claim'][:90]}\"")
            print(f"Unsourced-but-true rate: {unsrc:.1%}  (not fabrication, but "
                  f"not checkable by an operator reading the same block)")
            print(f"TemplateExplainer baseline: {tmpl_viol:.1%}  (the ablation; "
                  f"a template that states only supplied values should be 0%)")

            # The floor. A violation rate is uninterpretable without it: an
            # earlier revision of the checker verified 50-60% of entirely
            # fabricated numbers by chance, which made a measured 0.0% vacuous.
            r0 = records[0]
            floor = checker_false_positive_rate(
                r0["meas"], action=r0["action"],
                prompt=build_prompt(r0["meas"], r0["action"], r0["top"],
                                    None, ctx=ctx))
            print(f"Checker floor: {floor['false_verification_rate']:.1%} of "
                  f"RANDOM decoy numbers verify by chance "
                  f"({floor['n_allowed_values']} permitted values). Read every "
                  f"rate above against this.")

            # The second axis. Numeric consistency alone rewards saying
            # nothing, it is reported beside coverage or the ablation is
            # measuring terseness.
            cov = float(np.mean([o["coverage"]["coverage"] for o in outs]))
            cov_t = float(np.mean([o["template_coverage"]["coverage"] for o in outs]))
            n_num = float(np.mean([o["numeric_check"]["n_numbers"] for o in outs]))
            empty = sum(1 for o in outs if o["numeric_check"]["n_numbers"] == 0)
            print(f"\nWeighted coverage: LLM {cov:.2f} vs template {cov_t:.2f} "
                  f"(fraction of supplied facts conveyed, weighted by "
                  f"operational importance)")
            print(f"  numbers per generation: {n_num:.2f}; "
                  f"{empty} of {len(outs)} generations stated none")
            missed = {}
            for o in outs:
                for lab in o["coverage"]["missed_important"]:
                    missed[lab] = missed.get(lab, 0) + 1
            for lab, k in sorted(missed.items(), key=lambda kv: -kv[1])[:3]:
                print(f"  most-missed important fact: {lab} ({k}/{len(outs)})")
        else:
            print("\nNo generations succeeded; nothing to report.")

        # Tag the filename with the model. Comparing models by re-running with
        # --llm and the same --out silently overwrote each previous run, so a
        # four-model sweep left only the last one's text on disk.
        tag = "".join(c if c.isalnum() or c in "-._" else "-" for c in a.llm)
        out_path = os.path.join(a.out,
                                f"explanations_{tag}_{a.prompt_variant}.json")
        json.dump({"model": a.llm, "prompt_variant": a.prompt_variant,
                   "n_generated": len(outs), "n_failed": len(failures),
                   "checker_floor": checker_false_positive_rate(
                       records[0]["meas"], action=records[0]["action"],
                       prompt=build_prompt(records[0]["meas"],
                                           records[0]["action"],
                                           records[0]["top"], None, ctx=ctx)),
                   "weighted_coverage_llm":
                       float(np.mean([o["coverage"]["coverage"] for o in outs]))
                       if outs else None,
                   "weighted_coverage_template":
                       float(np.mean([o["template_coverage"]["coverage"]
                                      for o in outs])) if outs else None,
                   "unverified_number_rate":
                       float(np.mean([not o["numeric_check"]["passes"]
                                      for o in outs])) if outs else None,
                   "failures": failures, "explanations": outs},
                  open(out_path, "w"), indent=2, default=str)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
