"""Natural-language operator interface.

The generator is grounded in three things it must not contradict: the measured
state, the attributions, and the detector posterior. Everything numeric in the
prompt comes from the system; the model's only job is to verbalise. That
restriction is what makes the output auditable.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass

import numpy as np

SYSTEM_PROMPT = """You are a microgrid control-room assistant. You explain the \
actions of an automated secondary-control agent to a human operator.

Rules you must follow exactly:
- Use ONLY the numbers supplied in the state block. Never invent a value.
- If the evidence is ambiguous, say so. Do not manufacture confidence.
- Never assert an attack is occurring unless the detector block says so; \
report the detector's confidence as given.
- Be concrete about the physical mechanism, not the model internals. The \
operator cares about power sharing and voltage, not about network layers.
- Trust weights are PER SOURCE: each scales what one DG broadcasts, applied at \
every neighbour. Never describe a weight for a path between two DGs ("from DG10 \
to DG5") -- no such quantity exists, and an operator told to inspect one would be \
looking for something that is not there.
- Trust is bounded at 1.00 and can only be REDUCED from it. Never describe the \
agent as increasing a trust weight.
- Limit status is supplied in the state block. Report it as given; do not compare \
a measurement against a limit yourself.
- Maximum 4 sentences unless the requested level is 'detailed'."""


ENUMERATION_RULE = """
- Name EVERY link in the agent-action block whose trust is below 1.00, each \
with its numeric trust value. Do not summarise them as "several links" and do \
not mention only the most affected one; an operator has to know which \
communication paths to inspect, and one you omit is one they will not check."""

SYSTEM_PROMPT_ENUMERATE = SYSTEM_PROMPT + ENUMERATION_RULE


@dataclass
class GridContext:
    n_dg: int = 10
    f_nom: float = 50.0
    v_limits: tuple = (0.95, 1.05)
    f_tolerance: float = 0.05
    action_block: str = "top3"
    """How build_prompt presents the action.

    "top3"    the three least-trusted links, whatever their trust. The model
              must decide which of them are worth mentioning, and measurement
              showed it usually mentions one: with two links genuinely below
              full trust, 19 of 25 generations named only the deeper one.
    "reduced" only links below full trust, so there is no selection decision
              left to make. If every link is at full trust the block says so
              explicitly rather than listing three uninteresting rows."""

    topology: str = "10-DG islanded AC microgrid, ring communication graph"
    control_layer: str = "distributed consensus secondary control (voltage + frequency restoration)"


def _action_rows(action, ctx, k: int = 3, mode: str = None):
    """The links the prompt shows, least-trusted first.

    Shared by build_prompt and coverage_manifest so the set of supplied values
    and the set scored for coverage cannot drift apart.
    """
    a = np.asarray(action, dtype=float)
    t_xi = np.clip(1.0 + a[: ctx.n_dg], 0.0, 1.0)
    t_zeta = np.clip(1.0 + a[ctx.n_dg :], 0.0, 1.0)
    order = np.argsort(t_xi + t_zeta)
    mode = mode or getattr(ctx, "action_block", "top3")
    if mode == "reduced":
        # CAPPED AT k, exactly as "top3" is. Without the cap this returned
        # every link below full trust -- eight or ten rows where the baseline
        # showed three -- while the manifest still scored only three. The model
        # then spread a four-sentence budget across links that were not being
        # scored, and the variant looked worse for a reason that had nothing to
        # do with the enumeration instruction: DG5 went from missed 5/25 to
        # 24/25. The two variants must differ ONLY in whether full-trust rows
        # are suppressed.
        sel = [int(i) for i in order[:k] if min(t_xi[i], t_zeta[i]) < 0.99]
        return np.asarray(sel, dtype=int), t_xi, t_zeta
    return order[:k], t_xi, t_zeta


def build_prompt(meas, action, attributions, detector_posterior, ctx=None, level="standard"):
    """Assemble a fully grounded prompt. Every number is system-supplied."""
    ctx = ctx or GridContext()
    v, f, p, q = meas["V"], meas["f"], meas["P"], meas["Q"]
    from ..plant.attacks import ATTACK_CLASSES

    det = ""
    if detector_posterior is not None:
        post = np.asarray(detector_posterior, float)
        order = np.argsort(-post)[:3]
        det = "\n".join(
            f"  {ATTACK_CLASSES.get(int(i), str(i))}: {post[i]:.2f}" for i in order
        )
    else:
        det = "  (detector not available)"

    attr = "\n".join(
        f"  {a['rank']}. {a['feature']}: {a['attribution']:+.4f}" for a in attributions
    ) or "  (none)"

    # The comparison against each limit is evaluated HERE and stated as a
    # verdict. A model asked to compare two supplied numbers gets it wrong:
    # llama3.1:8b wrote that a mean frequency of 50.0034 Hz "is above the upper
    # limit of 50.05 Hz" and justified isolating a source on that basis. Both
    # figures were correct and both were supplied; only the comparison was
    # false, so every numeric check passed on a materially misleading
    # explanation. Supplying the verdict removes the need to compute it.
    v_ok = bool(v.min() >= ctx.v_limits[0] and v.max() <= ctx.v_limits[1])
    f_dev = float(np.abs(f - ctx.f_nom).max())
    f_ok = bool(f_dev <= ctx.f_tolerance)
    status = (
        f"    Voltage within band: {'YES' if v_ok else 'NO'}"
        f" (min {v.min():.4f}, max {v.max():.4f} against "
        f"{ctx.v_limits[0]}-{ctx.v_limits[1]} pu)\n"
        f"    Frequency within tolerance: {'YES' if f_ok else 'NO'}"
        f" (max deviation {f_dev:.4f} Hz against {ctx.f_tolerance} Hz)")

    # The action is a per-SOURCE TRUST WEIGHT, w = clip(1 + a, 0, 1): it scales
    # what one DG broadcasts, applied identically at every neighbour. Calling
    # it a "link" invited models to invent per-edge weights -- llama3.2:3b
    # described "the link from DG10 to DG5" and quoted a trust value for it,
    # which is a quantity that does not exist in this system.
    movers, t_xi, t_zeta = _action_rows(action, ctx)
    act = "\n".join(
        f"  broadcast from DG{int(i)+1}: frequency-path trust {t_xi[i]:.2f}, "
        f"voltage-path trust {t_zeta[i]:.2f}"
        f"{'   <- ISOLATED' if min(t_xi[i], t_zeta[i]) < 0.1 else ''}"
        for i in movers
    ) or "  all links at full trust (1.00); the agent made no change"

    return textwrap.dedent(f"""
    [GRID]
    {ctx.topology}; {ctx.control_layer}
    Voltage band {ctx.v_limits[0]}-{ctx.v_limits[1]} pu; frequency {ctx.f_nom} +/- {ctx.f_tolerance} Hz

    [LIMIT STATUS - already evaluated; do not recompute]
{status}

    [MEASURED STATE @ t={meas['t']:.3f} s]
    Voltage    mean {v.mean():.4f} pu, min {v.min():.4f}, max {v.max():.4f}
    Frequency  mean {f.mean():.4f} Hz, max deviation {np.abs(f - ctx.f_nom).max():.4f} Hz
    Active P   mean {p.mean():.4f} pu, dispersion {p.std():.4f}
    Reactive Q mean {q.mean():.4f} pu, dispersion {q.std():.4f}

    [DETECTOR POSTERIOR]
{det}

    [AGENT ACTION - per-SOURCE trust weights on the consensus network]
    (1.00 = full trust, the unmodified published controller; 0.00 = source isolated)
    Each weight scales the signal that DG transmits, applied identically at every
    neighbour. These are NOT per-edge weights: there is no separate weight for a
    path "from DGi to DGj", and no such quantity exists in this system.
{act}

    [ATTRIBUTIONS - Integrated Gradients over the policy, target = total trust reduction]
{attr}

    [TASK] Explanation level: {level}
    Explain (1) what the agent did, (2) which measurements drove it,
    (3) the expected physical effect, (4) what the operator should watch next.
    """).strip()


class OllamaExplainer:
    """Local LLM via Ollama. Swap in any chat client with the same interface."""

    def __init__(self, model="llama3.1:8b", host="http://localhost:11434",
                 timeout=300):
        # 60 s was not enough for the FIRST call, which also pays the cold load
        # of the model into memory (~5 GB for an 8B). Warm it once with
        # `ollama run <model> "ready"` and later calls are fast.
        self.model, self.host, self.timeout = model, host, timeout

    def generate(self, prompt, system=SYSTEM_PROMPT, temperature=0.2):
        import requests

        r = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model, "stream": False,
                "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt}],
            },
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            # raise_for_status() discards the body, and the body is where
            # Ollama says WHY -- a 404 on /api/chat almost always means the
            # model is not pulled, which the status code alone does not tell
            # you.
            raise RuntimeError(
                f"Ollama returned {r.status_code} for model {self.model!r}: "
                f"{r.text[:300]}"
            )
        return r.json()["message"]["content"].strip()


class TemplateExplainer:
    """Deterministic rule-based baseline.

    We included this as an ablation.

    The previous version returned a fixed sentence containing NO NUMBERS, and
    described a "residual correction" -- the abandoned action space. As an
    ablation it measured nothing: it could not lose on numeric consistency
    (zero numbers, zero violations, trivially) and could not win on
    informativeness (it stated no facts). A baseline has to be good enough to
    lose fairly, so this one reports the same grounded quantities the LLM is
    given.
    """

    def generate(self, prompt, meas=None, action=None, attributions=None,
                 ctx=None, **kw):
        if meas is None or action is None:
            return ("Template baseline: no structured state supplied; pass "
                    "meas and action to generate a grounded explanation.")
        ctx = ctx or GridContext()
        a = np.asarray(action, dtype=float)
        t_xi = np.clip(1.0 + a[: ctx.n_dg], 0.0, 1.0)
        t_zeta = np.clip(1.0 + a[ctx.n_dg :], 0.0, 1.0)
        order = np.argsort(t_xi + t_zeta)[:3]

        parts = []
        cut = [i for i in order if min(t_xi[i], t_zeta[i]) < 0.9]
        if cut:
            links = "; ".join(
                f"DG{int(i)+1} (frequency-path trust {t_xi[i]:.2f}, "
                f"voltage-path trust {t_zeta[i]:.2f})" for i in cut)
            parts.append(f"The agent reduced trust in the broadcast from {links}.")
        else:
            parts.append("The agent left all consensus links at full trust; "
                         "this is the unmodified published controller.")

        v, f = meas["V"], meas["f"]
        parts.append(
            f"Voltage mean {v.mean():.4f} pu (min {v.min():.4f}, "
            f"max {v.max():.4f}); frequency mean {f.mean():.4f} Hz, "
            f"maximum deviation {np.abs(f - ctx.f_nom).max():.4f} Hz.")

        if attributions:
            top = attributions[0]
            parts.append(f"The dominant measurement was {top['feature']} "
                         f"(attribution {top['attribution']:+.4f}).")
        parts.append(
            "Operator action: inspect the communication path of any link shown "
            "below full trust, and watch power sharing dispersion for recovery.")
        return " ".join(parts)


_NUMBER = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)(?![\w])")
_IDENTIFIER = re.compile(r"\b(?:DG|bus|node|link|unit|phase)\s*#?\d+\b", re.I)
_RANGE = re.compile(r"(\d)\s*-\s*(\d)")


def extract_numbers(text: str) -> list:
    r"""Numbers a reader would take as factual claims about the system.

    Two corrections over the previous `re.findall(r"-?\d+\.\d+", ...)`:

      RANGES. build_prompt writes "Voltage band 0.95-1.05 pu", and that regex
      reads the range dash as a minus sign, yielding -1.05. No such value
      exists anywhere, so it can never verify: every explanation that quoted
      the band -- which the prompt invites -- took an automatic violation.

      IDENTIFIERS. "DG9" and "link 3" are labels, not numeric claims. The old
      pattern happened to miss them by requiring a decimal point, which also
      meant it missed integers generally, including a genuinely fabricated
      "50 Hz" or "12 percent".
    """
    t = _IDENTIFIER.sub(" ", text or "")
    t = _RANGE.sub(r"\1 to \2", t)
    out = []
    for m in _NUMBER.finditer(t):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _state_values(meas: dict, action=None, ctx=None) -> list:
    """Everything true of the system, whether or not the prompt supplied it."""
    vals = []
    for k, v in (meas or {}).items():
        if k in ("collapsed",):
            continue
        arr = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
        vals.extend(float(x) for x in arr if np.isfinite(x))
    if action is not None:
        a = np.asarray(action, dtype=float).ravel()
        vals.extend(float(x) for x in a if np.isfinite(x))
        trust = np.clip(1.0 + a, 0.0, 1.0)
        vals.extend(float(x) for x in trust)
        vals.extend([float(np.mean(1.0 - trust)), float(np.max(1.0 - trust))])
    if ctx is not None:
        vals.extend([float(ctx.f_nom), float(ctx.f_tolerance),
                     float(ctx.v_limits[0]), float(ctx.v_limits[1]),
                     float(ctx.n_dg)])
    vals.extend([0.0, 1.0, 100.0])
    return vals


def _matches(x: float, truth) -> bool:
    """Match at the PRECISION WRITTEN, and no wider.

    A claim of 1.00 is satisfied by a true 1.0010, because rounding is
    formatting rather than fabrication. An ADDITIONAL relative tolerance is not
    applied: at 5% of the value, the windows around ~136 permitted values
    covered most of the plausible range, and a decoy-number experiment showed
    50-60% of entirely fabricated figures verifying by chance. A tolerance wide
    enough to accept anything measures nothing.
    """
    if len(truth) == 0:
        return False
    decimals = len(str(x).split(".")[1]) if "." in str(x) else 0
    return bool(np.any(np.round(truth, decimals) == round(x, decimals)))


_PERCENT = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)


def _percent_values(text: str) -> set:
    """Numbers written AS PERCENTAGES, so they can also be tested as fractions.

    A model that writes "voltage-path trust at 61%" for a supplied trust of
    0.61 is correct; comparing 61.0 against 0.61 flagged it as a fabrication.
    Same family as the range-dash defect: the model was right and the
    instrument was wrong.

    Gated on an adjacent % or the word "percent" rather than testing every
    number at both scales, which would widen the allowed set for free and lift
    the chance floor the metric is measured against.
    """
    return {float(m.group(1)) for m in _PERCENT.finditer(text or "")}


def numeric_consistency_check(explanation: str, meas: dict, action=None,
                              ctx=None, prompt: str = None) -> dict:
    """Verify every number in the text against what the system supplied.

    Hallucinated figures in a control-room interface are a safety issue, not a
    quality issue.

    WHY THE PREVIOUS VERSION REPORTED 52-84% ACROSS FOUR MODELS
    Its `allowed` set was V, f, P, Q, t and four aggregates. But SYSTEM_PROMPT
    instructs the model to "Use ONLY the numbers supplied in the state block",
    and build_prompt supplies the voltage band, the frequency tolerance and the
    per-link TRUST WEIGHTS -- none of which were in `allowed`. The verification
    set was narrower than the set the prompt hands the model, so following the
    instruction exactly scored as hallucinating. A correct "trust weight of
    0.00" was a violation; so was quoting the operating envelope.

    The set is now anchored on the prompt itself, which is the only defensible
    construction: every number in the prompt is system-supplied by definition.
    Two counts are returned, because they are different failures:

      n_unverified  numbers that match nothing -- invented. The safety metric.
      n_unsourced   numbers that are true of the state but were NOT in the
                    prompt. Not fabrication, but not auditable by an operator
                    reading the same block either.
    """
    stated = extract_numbers(explanation)
    supplied = np.asarray(extract_numbers(prompt), dtype=float) if prompt else np.empty(0)
    state = np.asarray(_state_values(meas, action, ctx), dtype=float)

    # VERIFIED means "appears in the prompt". That is what SYSTEM_PROMPT
    # demands, and it is the only set an operator reading the same block can
    # audit -- about 25 numbers rather than the ~136 of the full internal
    # state. Values true of the state but absent from the prompt are not
    # fabrication and get their own bucket rather than widening this one.
    as_percent = _percent_values(explanation)

    unverified, unsourced = [], []
    for x in stated:
        if abs(x) < 20 and float(x).is_integer():
            continue                                  # index or count
        if _matches(x, supplied):
            continue
        if x in as_percent and _matches(x / 100.0, supplied):
            continue                                  # a fraction stated as %
        if _matches(x, state) or (x in as_percent and _matches(x / 100.0, state)):
            unsourced.append(x)
            continue
        unverified.append(x)

    n = len(stated)
    return {
        "n_numbers": n,
        "n_unverified": len(unverified),
        "unverified": unverified,
        "n_unsourced": len(unsourced),
        "unsourced": unsourced,
        "rate": (len(unverified) / n) if n else 0.0,
        "passes": len(unverified) == 0,
    }


def checker_false_positive_rate(meas, action=None, ctx=None, prompt=None,
                                n_decoys: int = 500, seed: int = 0) -> dict:
    """How often does a MADE-UP number verify by chance?

    A violation rate is uninterpretable without this. The allowed set contains
    every prompt number, ~60 measurements, 20 raw actions, 20 trust weights and
    the plant constants; with a 5% relative tolerance those windows can cover a
    large part of the plausible range, and then any number a model writes
    verifies and a 0% violation rate is vacuous.

    Decoys are drawn in the ranges an operator explanation actually uses: pu
    quantities near 0-1.2, percentages, and frequencies near 50 Hz. Read the
    reported violation rate against this floor -- if decoys verify 60% of the
    time, only violations far above 40% mean anything.
    """
    rng = np.random.default_rng(seed)
    decoys = np.concatenate([
        np.round(rng.uniform(0.0, 1.2, n_decoys // 3), 2),      # pu / trust
        np.round(rng.uniform(0.0, 100.0, n_decoys // 3), 1),    # percentages
        np.round(rng.uniform(49.0, 51.0, n_decoys - 2 * (n_decoys // 3)), 3),
    ])
    supplied = np.asarray(extract_numbers(prompt), dtype=float) if prompt else np.empty(0)
    state = np.asarray(_state_values(meas, action, ctx), dtype=float)

    verified = 0
    for x in decoys:
        if abs(x) < 20 and float(x).is_integer():
            continue
        if _matches(x, supplied):
            verified += 1
    return {
        "n_decoys": int(len(decoys)),
        "false_verification_rate": float(verified / max(1, len(decoys))),
        "n_allowed_values": int(len(supplied)),
    }


# ---------------------------------------------------------------- coverage

_ISOLATION_WORDS = re.compile(
    r"\b(isolat\w*|distrust\w*|cut off|disconnect\w*|down-?weight\w*|"
    r"reduc\w+ trust|zero trust|no trust)\b", re.I)


def coverage_manifest(meas, action, attributions=None, detector_posterior=None,
                      ctx=None) -> list:
    """What the explanation OUGHT to convey, weighted by operational importance.

    Numeric consistency alone rewards saying nothing: an explanation that
    states no figures cannot state a wrong one. Measured on this project,
    mistral-nemo scored 0% violations while citing 1.88 numbers per generation
    with 6 of 25 numerically empty, against a template stating 13.0 -- the
    safety metric ranked it level with a model conveying four times as much.
    Coverage is the second axis that makes the ablation honest.

    Weights are operational, not statistical. An isolated link is the action
    being explained and an operator must be told which one; the reactive-power
    dispersion is context. An explanation reciting every aggregate while
    omitting the isolated link would score well on unweighted coverage and be
    useless, which is why unweighted coverage is reported only as a secondary
    figure.
    """
    ctx = ctx or GridContext()
    items = []
    # Always top3 for SCORING, whatever the prompt variant shows, so an A/B on
    # presentation is not confounded by a change in the denominator.
    movers, t_xi, t_zeta = _action_rows(action, ctx, mode="top3")

    # Importance scales CONTINUOUSLY with isolation depth: 0.5 at full trust,
    # 3.0 at full isolation. A binary "anything below 0.9 is important" made a
    # link at 0.85 count the same as one at 0.00, so every model was reported
    # as missing two mildly down-weighted links while correctly naming the
    # isolated one -- the metric was penalising the right behaviour.
    # Links at FULL TRUST are excluded from scoring entirely, in every prompt
    # variant. Two reasons, and the second is the one that matters:
    #
    #   * There is nothing for an operator to inspect on a link at 1.00, so an
    #     explanation deserves neither credit for mentioning it nor blame for
    #     omitting it.
    #   * It removes a confound. The "reduced" prompt variant withholds
    #     full-trust links, so scoring them made models look worse for not
    #     stating values they were never given -- roughly 24% of the total
    #     weight in a low-action state, which is most of the 0.15 coverage drop
    #     first measured for that variant. The denominator is now the
    #     intersection of what every variant supplies.
    for i in movers:
        i = int(i)
        if min(float(t_xi[i]), float(t_zeta[i])) >= 0.99:
            continue
        for path, t, scale in (("frequency", t_xi, 1.0), ("voltage", t_zeta, 0.5)):
            trust = float(t[i])
            worst = min(float(t_xi[i]), float(t_zeta[i]))
            items.append({
                "label": f"DG{i+1} {path}-path trust", "value": trust,
                "weight": (0.5 + 2.5 * (1.0 - trust)) * scale,
                "dg": i + 1,
                "kind": ("isolated" if worst < 0.1
                         else "reduced" if worst < 0.9 else "full"),
            })

    v, f = meas["V"], meas["f"]
    breach_v = bool(v.min() < ctx.v_limits[0] or v.max() > ctx.v_limits[1])
    breach_f = bool(np.abs(f - ctx.f_nom).max() > ctx.f_tolerance)
    items += [
        {"label": "voltage mean", "value": float(v.mean()), "weight": 1.0},
        {"label": "voltage min", "value": float(v.min()),
         "weight": 3.0 if breach_v else 1.0},
        {"label": "voltage max", "value": float(v.max()),
         "weight": 3.0 if breach_v else 1.0},
        {"label": "frequency mean", "value": float(f.mean()), "weight": 1.0},
        {"label": "max frequency deviation",
         "value": float(np.abs(f - ctx.f_nom).max()),
         "weight": 3.0 if breach_f else 1.0},
        {"label": "P mean", "value": float(meas["P"].mean()), "weight": 0.5},
        {"label": "P dispersion", "value": float(meas["P"].std()), "weight": 0.5},
        {"label": "Q mean", "value": float(meas["Q"].mean()), "weight": 0.5},
        {"label": "Q dispersion", "value": float(meas["Q"].std()), "weight": 0.5},
    ]

    if detector_posterior is not None:
        post = np.asarray(detector_posterior, float)
        top = int(np.argmax(post))
        items.append({"label": "detector top-class confidence",
                      "value": float(post[top]), "weight": 2.0})

    if attributions:
        items.append({"label": "top attribution",
                      "value": float(attributions[0]["attribution"]),
                      "weight": 0.25})
    return items


def _covers(value: float, stated) -> bool:
    """Did the text state THIS quantity?

    Absolute tolerance, not the precision matching used for verification. Those
    are different questions and need different rules: accepting a loose match
    when verifying only risks passing a true statement, but accepting one when
    scoring coverage INVENTS credit. Measured concretely -- a trust weight of
    0.0 renders as "0.0", one decimal, so any small number in the text rounded
    onto it and a P dispersion of 0.0113 was scored as having conveyed that a
    link was isolated.
    """
    if len(stated) == 0:
        return False
    tol = max(0.005, 0.0005 * abs(value))
    return bool(np.any(np.abs(np.asarray(stated, dtype=float) - value) <= tol))


def coverage_score(explanation: str, manifest: list) -> dict:
    """Weighted fraction of the supplied facts the explanation conveys.

    An action item counts as conveyed if its numeric value appears OR the DG is
    named alongside an isolation/trust phrase -- "isolated the link from DG9"
    tells the operator what they need without quoting 0.00, and penalising that
    would measure formatting rather than content.
    """
    text = explanation or ""
    stated = np.asarray(extract_numbers(text), dtype=float)

    got, total, missed = 0.0, 0.0, []
    for it in manifest:
        total += it["weight"]
        hit = _covers(it["value"], stated)
        if not hit and "dg" in it and it["kind"] in ("isolated", "reduced"):
            named = re.search(rf"\bDG\s*#?{it['dg']}\b", text, re.I)
            if named and _ISOLATION_WORDS.search(text):
                hit = True
        if hit:
            got += it["weight"]
        elif it["weight"] >= 2.0:
            missed.append(it["label"])

    n_hit = sum(1 for it in manifest if _covers(it["value"], stated))
    return {
        "coverage": float(got / total) if total else 0.0,
        "coverage_unweighted": float(n_hit / len(manifest)) if manifest else 0.0,
        "missed_important": missed,
        "n_items": len(manifest),
    }


_RELATION = re.compile(
    # Commas ARE allowed inside the gap: the observed false comparison reads
    # "the mean frequency (50.0034 Hz), which is above the upper limit of
    # 50.05 Hz", and excluding commas missed it entirely. The "or"-clause false
    # positive is handled by the clause split below instead, which is the right
    # place for it -- a comma separates a clause from its own qualifier, a
    # conjunction separates two independent claims.
    r"([-+]?\d+(?:\.\d+)?)\s*(?:Hz|pu|%)?[^.;]{0,45}?\b"
    r"(above|below|exceeds?|under|greater than|less than|higher than|lower than)"
    r"\b[^.;]{0,45}?([-+]?\d+(?:\.\d+)?)",
    re.I)

_GREATER = {"above", "exceed", "exceeds", "greater than", "higher than"}
_LESSER = {"below", "under", "less than", "lower than"}

# Text that must be removed before the relation parser runs, or it invents
# comparisons out of things that are not numeric claims at all. Both false
# positives observed came from here:
#   "DG10's voltage was below the minimum limit at t-2"
#       -> parsed 10 (from DG10) below -2 (from the history index t-2)
#   "rises above 1.05 pu or falls below 0.95"
#       -> paired 1.05 with the "below" belonging to the second clause
# The first is fixed by stripping identifiers and history indices; the second
# by refusing to cross a comma or a conjunction.
_STRIP_FOR_RELATIONS = re.compile(
    r"\b(?:DG|bus|node|link|unit|phase)\s*#?\d+\b|\bt-\d+\b", re.I)
_CLAUSE_BREAK = re.compile(r"\b(?:or|and|while|whereas|but)\b", re.I)


def relational_consistency_check(explanation: str) -> dict:
    """Verify comparisons between two stated numbers.

    numeric_consistency_check verifies each number against what was supplied.
    It cannot see a false RELATION between two numbers that are both correct,
    and that failure is more dangerous than a wrong digit: llama3.1:8b wrote
    that a mean frequency of 50.0034 Hz "is above the upper limit of 50.05 Hz"
    and justified isolating a source on it. Every figure verified; the
    explanation was still wrong, and an operator would have believed a
    frequency violation had occurred.

    Deliberately conservative -- only explicit binary comparisons with both
    operands present are examined. Comparisons in words alone, or against an
    implied limit, are out of scope and are NOT counted as passing; they are
    simply not seen. Reported as a lower bound on relational error.
    """
    text = " ".join((explanation or "").split())
    text = _STRIP_FOR_RELATIONS.sub(" ", text)
    # Split on clause boundaries so a comparison cannot pair an operand from
    # one clause with a relation from the next.
    segments = [seg for seg in _CLAUSE_BREAK.split(text) if seg.strip()]

    violations = []
    for m in (mm for seg in segments for mm in _RELATION.finditer(seg)):
        try:
            lhs, rhs = float(m.group(1)), float(m.group(3))
        except (TypeError, ValueError):
            continue
        rel = m.group(2).lower().strip()
        if rel in _GREATER:
            ok = lhs > rhs
        elif rel in _LESSER:
            ok = lhs < rhs
        else:
            continue                      # "within"/"outside" need a range
        if not ok:
            violations.append({"claim": m.group(0)[:120], "lhs": lhs,
                               "relation": rel, "rhs": rhs})
    return {"n_relations": sum(1 for seg in segments
                               for _ in _RELATION.finditer(seg)),
            "n_violations": len(violations),
            "violations": violations,
            "passes": len(violations) == 0}
