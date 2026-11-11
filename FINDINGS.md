# FINDINGS — the IEEE DataPort microgrid release

Evidence gathered while building this project against *Labeled Cyber–Physical
Microgrid Dataset for Secondary-Control Cyberattack Detection*
(doi:10.21227/bmtr-fr76), the release accompanying Ogiesoba-Eguakun & Rath,
*Cyberattack Detection in Virtualized Microgrids Using LightGBM and
Knowledge-Distilled Classifiers*.

Every number below is reproducible from the script named beside it, run on the
public release. Nothing is inferred from the paper's text alone.

**Correspondence.** Findings 1–3 and the integrity output of
`scripts/11_check_csv_integrity.py` were sent to the dataset authors before
publication. They replied that their paper is a preprint under revision, and
confirmed that the release's secondary control restores frequency but slowly,
with set-point tracking running beyond one second. That reply is the basis of
the wording of findings 4 and 5. The figures in findings 2 and 3 were revised
after that correspondence; see the corrections at the end.

---

## Summary

| # | Finding | Evidence |
|---|---|---|
| 1 | Solver divergence in 6 of 7 files, with a per-file signature | `11_check_csv_integrity.py` |
| 2 | The corruption artifacts are an exploitable per-file fingerprint | `12_shortcut_probe.py`, `01_detector_benchmark.py` |
| 3 | Sample-level leakage inflates accuracy by tens of points | `01_detector_benchmark.py` |
| 4 | 1 s per scenario, one run each, cold start, bounded attack window | `10_inspect_release.py` |
| 5 | The release cannot validate a secondary control layer | `10_inspect_release.py`, `17_settling_timescale.py` |
| 6 | The DoS file is not the same simulation as the others | `14_preonset_identity_probe.py` |

Findings 1–3 bear on detection benchmarks built on the release. Findings 4–6
bound what a fidelity comparison against it can establish. The two subtlest
attack classes, Coordinated Stealth and SlowRamp, fall from above 0.98 F1 under
random splitting to 0.39 and 0.46 under block splitting on physical features
alone, and much of the SlowRamp score a full-feature detector retains comes
from the corrupted channels.

**Settings for every detector number below:** `--decimate 500` (1 ms), the
first 0.15 s of each file trimmed (energisation), ten repeats, LightGBM.

---

## 1. Solver divergence

`scripts/11_check_csv_integrity.py`

The CSVs are structurally perfect: 500,001 rows × 38 fields, no ragged lines,
no parse failures. The values are not.

| Scenario | Channel(s) | max \|x\| | Samples > 1e7 | NaN |
|---|---|---|---|---|
| Normal | V3 | 2.312e152 | 108,534 | 0 |
| Additive | V1, V3, I1 | 2.023e152 | 232,282 | 0 |
| Ramp | V3 | 4.361e131 | 53,821 | 0 |
| SlowRamp | V3, I1 | 8.230e152 | 63,122 | 413,408 |
| Sinusoid | V1 | 6.550e152 | 149,214 | 0 |
| Stealth | V2 | 2.467e8 | 47 | 0 |
| DoS | — | — | 0 | 0 |

This is not file corruption. Consecutive values are smooth and monotone —
differences constant to 4e-12 relative, with an ~8 ms time constant.
It is the solver diverging inside the source model, faithfully recorded.

Divergence begins around t = 0.029 s, just after energisation, and in the worst
case spans 46 % of the recording. SlowRamp is 83 % NaN. DoS is the only clean
file.

The affected channels are bus voltages and currents (V1–V3, I1). The per-DG
active power, reactive power and frequency channels (`P_DG*`, `Q_DG*`,
`f_DG*`) are finite throughout, and every later comparison in this project uses
only those.

**Consequence.** The published ablation reports bus voltage as the most
informative feature group (1.01 % macro-F1 drop on removal). That figure was
computed over channels that reach 1e152.

---

## 2. The artifacts are a per-file fingerprint

`scripts/12_shortcut_probe.py`

Which channel diverges, by how much and over how many samples differs file by
file. Mean log10|x| on diverged samples:

| Scenario | Channel | Mean log10\|x\| |
|---|---|---|
| SlowRamp | V3 | 74.9 |
| Additive | V1 | 74.5 |
| Sinusoid | V1 | 44.5 |
| Normal | V3 | 39.6 |
| Ramp | V3 | 23.9 |

Normal and Ramp differ by 16 decades in the same channel: one tree split.

To test whether this is exploitable rather than merely present, the same
LightGBM was trained on **corruption features only** — `is_diverged`,
`is_missing`, and log10|x| on diverged samples. No voltages, currents, powers
or frequencies: nothing a controller could measure or an attacker influence.

Artifact-only probe (`12_shortcut_probe.py --mode magnitude`), multiclass,
chance = 0.143:

| Split | Macro-F1 | Rows flagged | Accuracy on flagged rows | SlowRamp F1 | Stealth F1 |
|---|---|---|---|---|---|
| Random | 0.431 | 33.3 % | 94.0 % | 0.94 | 0.35 |
| Block | 0.385 | 38.5 % | 84.4 % | 0.93 | 0.26 |

On rows where any flag fires, the source file is identified 84 % of the time
under a block split, against 14 % by chance. SlowRamp is identified almost
perfectly from the artifacts alone.

**Does a detector exploit it?** The same detector at the same settings, block
split, with and without the three corrupted channels:

| Features | Block multiclass macro-F1 | Stealth F1 | SlowRamp F1 |
|---|---|---|---|
| 33 clean physical (`results/release_clean`) | 0.658 ± 0.097 | 0.39 | 0.46 |
| 36, corrupted channels masked (`results/release_detector`) | 0.782 ± 0.078 | 0.46 | 0.76 |
| 9 artifact flags only (probe above) | 0.385 | 0.26 | 0.93 |

Keeping the corrupted channels adds 12 points of block-split macro-F1 and
30 points on SlowRamp, the class the artifacts identify best on their own. A
detector reported on all 36 channels is therefore partly a detector of which
file it was given. Masking the diverged values (setting them to NaN) does not
remove the fingerprint, because the detector treats missingness as information;
only dropping the channels does.

The probe is capped: Stealth and DoS carry no flags at all and are mutually
indistinguishable to it, and it still reaches more than half the full
detector's block-split score.

---

## 3. Leakage, and what survives removing the artifacts

`scripts/01_detector_benchmark.py --drop-corrupt`

Dropping V1, V3 and I1 entirely — whole channels, not affected rows, because
the missingness pattern is itself a fingerprint — leaves 33 clean features
including every P, Q and f channel.

Random against block splitting on the same 33 clean features, same settings
(`results/release_clean`):

| Protocol | Binary macro-F1 | Multiclass macro-F1 | Stealth F1 | SlowRamp F1 |
|---|---|---|---|---|
| Random | 0.994 ± 0.002 | 0.990 ± 0.005 | 0.98 | 0.98 |
| Block | 0.929 ± 0.033 | 0.658 ± 0.097 | 0.39 | 0.46 |

Three readings:

- **Random splitting inflates multiclass macro-F1 by about 33 points**, and the
  Stealth and SlowRamp scores by more than 50. At 2 µs adjacent samples are
  near-identical, so a stratified random split places near-duplicates on both
  sides of the partition. This needs no artifact to occur.
- **Stealth and SlowRamp are the classes the source paper reports above 0.996
  F1.** On physical features under block splitting they reach 0.39 and 0.46:
  detectable in part, nowhere near the published figures, and with high
  variance.
- **The startup transient is itself a confound.** Without the 0.15 s trim the
  same clean-feature benchmark gave block multiclass macro-F1 of only 0.50–0.54
  in three earlier runs, because the energisation event, common to every file,
  dominates block-split results on a one-second recording. Quote the trimmed
  figure.

**Statistical caveat.** The block-split standard deviation is about 0.1 over
ten repeats, and the split-coverage guard fired (SlowRamp absent from a
training split in one repeat). Block splitting strands whole classes on a
one-second recording, and more repeats do not fix it. The direction is
unambiguous; the point estimate is not, and should be quoted only with this
caveat.

**The constructive counterpart.** On the synthetic corpus generated by this
project's plant (`scripts/09_generate_detector_dataset.py`; archived at
doi:10.5281/zenodo.24082859) — 40 independent runs per class with randomised
onset, severity, targets, load profile and benign load steps, and no corrupted
channels — all three split protocols can be formed and the leakage effect is
measured cleanly: multiclass macro-F1 0.9987 (random) → 0.9289 (block) →
**0.7568** (run-level holdout), three repeats. That result is unaffected by
anything in this document.

---

## 4. Structure: one second, one run each, bounded attack

`scripts/10_inspect_release.py`

- **1.00 s per scenario, one run each — 7 s of simulated time in total.** The
  3.5 M rows are those 7 s sampled every 2 µs. At a control-relevant 10 ms rate
  the release is 700 rows, about 31 attacked samples per class.
- **Run-level splitting is impossible.** Holding out a run requires more than
  one run. A block split with purge gaps is the ceiling, and it does not address
  duplication across files.
- **The attack window is bounded: t ∈ [0.5005, 0.801] s.** The label goes
  0 → 1 → 0, so the final ~20 % is the grid recovering, labelled normal.
  Pre-attack and post-attack operation are physically different states sharing
  one label; `dataio.py` emits a `phase` column so they can be separated.
- **Cold start.** Every file begins from zero initial conditions. Voltage
  settles at about 0.014 s; earlier samples are an energisation event, not
  normal operation.
- **Effective rank ≈ 22 of 36 channels.** During energisation all ten DGs are
  numerically identical; at steady state DG1 (+8 % active power) and DG6
  (+37 % reactive power) separate from the rest.
- **Frequency has not restored within the record.** The release ends at
  49.8757 Hz, 0.124 Hz below nominal. The authors confirm that their secondary
  restoration runs beyond one second, so this is a slow secondary layer
  observed over too short a record, not an absent one.

---

## 5. The release cannot validate a secondary layer

`scripts/10_inspect_release.py`, `scripts/17_settling_timescale.py`

Because restoration outlasts the one-second record, the attack's effect on
secondary control never develops inside the data. What the release can
validate is the primary, droop-dominated response in the first tens of
milliseconds after onset. This project's Tier B fidelity comparison is
restricted to that layer accordingly: over a 50 ms window the plant reproduces
the release's active-power response to an additive consensus bias in direction
and shape (correlation 0.9368 against a 0.90 criterion).

The fast (~0.05 s) redistribution visible in the release after onset is
primary-loop behaviour, not secondary restoration, so it does not imply a
mismatch in secondary-layer timescale between the release and this plant.

**A modelling difference, stated rather than elided.** In this project the
attack corrupts the consensus value *transmitted* by a DG, leaving that DG's own
dynamics untouched (communication-path threat model). In the release's
Additive file the attacked DG sheds 36.60 W while the other nine gain 35.19 W
in near-equal shares, and its ring neighbours are not preferentially disturbed:
a node-local signature. The two systems appear to inject the attack at
different points, so localisation cannot be validated against the release, and
only aggregate, direction-and-shape agreement is claimed.

---

## 6. The DoS file is not the same simulation

`scripts/14_preonset_identity_probe.py`

The differential comparison (attacked file minus Normal file, per timestep)
is licensed by measurement rather than assumed. On every `P_DG*`, `Q_DG*` and
`f_DG*` channel, five of the six attacked files agree with the Normal file to
between 1e-6 and 1e-13 before onset, against post-onset differences three to
eight orders of magnitude larger: those runs are the same solve up to the
attack, and the energisation transient cancels exactly.

DoS fails this gate. Its post-onset to pre-onset differential ratio is 0.92
against a floor of 10, and it differs from the Normal file from the first
sample, with no step at onset. It is not the same simulation as the others, and
no differential against it is defensible. It is excluded from the fidelity
comparison rather than averaged in.

---

## What to claim, and what not to

**Supportable:**

- The release contains solver divergence with a per-file signature; the
  signature alone identifies SlowRamp almost perfectly, and keeping the
  corrupted channels adds about 12 points to a detector's block-split score.
- Standard random splitting on 2 µs data inflates multiclass macro-F1 by about
  33 points.
- The release does not support the published near-perfect scores for
  Coordinated Stealth and SlowRamp: on physical features under block splitting
  they reach 0.39 and 0.46.
- The release supports a primary-layer, direction-and-shape fidelity check on
  five of six attack classes, and nothing at the secondary layer.
- Detection benchmarks on high-rate single-trajectory simulation data need
  run-level holdout and a data-integrity check as standard practice.

**Not supportable:**

- That the published 99.72 % is entirely artifact. Genuine signal exists for
  the overt classes.
- Any localisation claim. The label is global, so nothing supervises it, and
  the injection point differs between the two systems (finding 5).
- A precise point estimate for clean block-split performance on the release.
- Any statement about the release's secondary controller beyond "frequency has
  not restored within the one-second record".

---

## Corrections made during the analysis

Recorded so that earlier drafts and correspondence can be read correctly.

- **Detector figures in findings 2 and 3, as first circulated** — superseded.
  The earlier text compared the artifact probe (0.3852) with a full detector at
  0.3814, reported the clean block-split score as 0.5429 ± 0.165 with Stealth
  0.042 and SlowRamp 0.000, and concluded that removing the artifacts raises
  block F1 and that both subtle classes sit at the floor. Those runs used three
  repeats and no startup trim, so the energisation transient dominated the
  block split; the comparison was also not matched across feature sets. The
  matched reruns above replace them. What survives: the artifacts are an
  exploitable fingerprint, and random splitting inflates accuracy. What
  changes: Stealth and SlowRamp are low and unstable rather than at the floor,
  and the corrupted channels raise, not lower, block-split performance.

- **"The secondary controller is not restoring frequency"** — withdrawn after
  the authors' reply; replaced by the wording in finding 4.
- **"The release's secondary layer settles about 15× faster than the plant"** —
  withdrawn. The 0.051 s timescale measured in the release is primary-loop
  redistribution, not secondary restoration (finding 5).
- **"The release holds all DGs on a common bus"** — withdrawn.
  `scripts/15_release_channel_identity.py` shows all ten DG columns are
  distinct, and `scripts/17_settling_timescale.py` shows ring-compatible
  distance structure that resolves within the attack window.
