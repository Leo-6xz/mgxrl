# reproduce_all.ps1
# ---------------------------------------------------------------------------
# Every experiment behind the manuscript, in dependency order, from a clean
# checkout. Written against the argparse signatures as they actually are, not
# from memory.
#
# WHAT THIS IS FOR
# No single pass has ever been run over the whole chain. Individual stages were
# run as they were developed, several of them under code that was later fixed.
# This script exists so that one clean execution produces every number the
# paper reports, in an order where each stage's inputs are outputs of the ones
# before it.
#
# WHAT IT IS NOT
# It is not a correctness check on its own. Six metrics in this project have
# reported clean numbers while measuring the wrong quantity, and a script that
# runs end to end without erroring would have reported all six as fine. Compare
# the values it produces against the ones in the manuscript; a stage that runs
# and disagrees is the useful outcome.
#
# RUNTIME. Roughly 8-11 hours end to end on a commodity workstation, dominated
# by three 5M-step training runs (~90 min each) and the detector benchmark on
# the synthetic corpus. Use -SkipTraining to reuse runs/ppo3_s0..2, which is
# what you want unless the environment itself changed.
#
#   .\reproduce_all.ps1 -Dataset C:\research\mgxrl\Dataset
#   .\reproduce_all.ps1 -Dataset C:\research\mgxrl\Dataset -SkipTraining
#   .\reproduce_all.ps1 -Dataset C:\research\mgxrl\Dataset -Only 4
# ---------------------------------------------------------------------------

param(
    [string]$Dataset      = "C:\research\mgxrl\Dataset",
    [string]$Synthetic    = "data\synthetic",
    [string]$Results      = "results",
    [int]   $Timesteps    = 5000000,
    [int]   $NEnvs        = 8,
    [switch]$SkipTraining,
    [switch]$SkipMatlab,
    [switch]$SkipLLM,
    [int[]] $Only         = @()
)

$ErrorActionPreference = "Stop"
$started = Get-Date
$log = Join-Path $Results ("reproduce_" + $started.ToString("yyyyMMdd_HHmmss") + ".log")
New-Item -ItemType Directory -Force -Path $Results | Out-Null
Start-Transcript -Path $log | Out-Null

# Every stage goes through here. A stage that fails STOPS the run rather than
# letting later stages consume a missing or stale input -- the failure mode
# that produced evaluation numbers from an unnormalised policy.
function Step($phase, $name, $block) {
    if ($Only.Count -gt 0 -and $Only -notcontains $phase) { return }
    Write-Host "`n=== [$phase] $name" -ForegroundColor Cyan
    $t0 = Get-Date
    & $block
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        Stop-Transcript | Out-Null
        throw "[$phase] $name FAILED (exit $LASTEXITCODE). Nothing downstream was run."
    }
    Write-Host ("    done in {0:n1} min" -f ((Get-Date) - $t0).TotalMinutes)
}

# --------------------------------------------------------------------------
# PHASE 0  Environment and regression tests
# --------------------------------------------------------------------------
# The tests come FIRST and the run stops if they fail. A silently broken plant
# produces a beautifully converged agent that has learned nothing, and every
# number below would then be internally consistent and wrong.

Step 0 "regression tests (expect 23 passed)" { python -m pytest tests\ -q }

Step 0 "record environment" {
    python -c "import sys,numpy,pandas;print('python',sys.version.split()[0]);print('numpy',numpy.__version__);print('pandas',pandas.__version__)"
    python -c "import stable_baselines3 as s,torch,lightgbm as l;print('sb3',s.__version__);print('torch',torch.__version__);print('lightgbm',l.__version__)"
}

# --------------------------------------------------------------------------
# PHASE 1  Release forensics
# --------------------------------------------------------------------------
# These read the public release only and depend on nothing else, so they run
# first and their outputs bound what later fidelity work is allowed to claim.

Step 1 "release structure and steady state" {
    python scripts\10_inspect_release.py --root $Dataset --settle 0.15 --out $Results
}

Step 1 "CSV integrity (send this output to the dataset authors)" {
    python scripts\11_check_csv_integrity.py --root $Dataset
}

Step 1 "corruption-artifact shortcut probe" {
    python scripts\12_shortcut_probe.py --root $Dataset --mode magnitude --out $Results
}

Step 1 "pre-onset identity gate (licenses the differential comparison)" {
    python scripts\14_preonset_identity_probe.py --dataset $Dataset --out "$Results\preonset_identity.csv"
}

Step 1 "per-DG channel independence in the release" {
    python scripts\15_release_channel_identity.py --dataset $Dataset --out "$Results\release_channel_identity.csv"
}

# --------------------------------------------------------------------------
# PHASE 2  Detector corpus and leakage-corrected benchmark
# --------------------------------------------------------------------------
# The synthetic corpus is what makes run-level holdout possible at all; the
# release has one run per scenario, so scenario_split cannot be formed on it.

Step 2 "generate synthetic corpus (40 runs x 7 scenarios)" {
    python scripts\09_generate_detector_dataset.py --out $Synthetic --runs 40 --seconds 8.0 --log-dt 0.001 --seed 0
}

Step 2 "detector benchmark on the synthetic corpus, 3 protocols" {
    python scripts\01_detector_benchmark.py --root $Synthetic --decimate 1 --windowed --repeats 3 --out $Results
}

Step 2 "detector benchmark on the release, for contrast" {
    python scripts\01_detector_benchmark.py --root $Dataset --decimate 500 --repeats 10 --trim 0.15 --out "$Results\release_detector"
}

Step 2 "detector benchmark with corrupt channels dropped" {
    python scripts\01_detector_benchmark.py --root $Dataset --decimate 500 --drop-corrupt --repeats 10 --trim 0.15 --out "$Results\release_clean""
}

# --------------------------------------------------------------------------
# PHASE 3  Plant fidelity
# --------------------------------------------------------------------------
# 08 must run before the MATLAB stages: the YAML is the single source of truth
# and editing it alone changes MATLAB while leaving Python untouched.

Step 3 "sync MATLAB parameters from the YAML" {
    python scripts\08_sync_matlab_params.py
}

Step 3 "export reduced-order reference trajectories" {
    python scripts\05_export_reference_traj.py --out $Results --seconds 12.0
}

if (-not $SkipMatlab) {
    Step 3 "MATLAB: Tier A, reduced vs stiff-solver reference" {
        matlab -batch "cd matlab; validate_fidelity"
    }
    Step 3 "MATLAB: inner-loop contribution to the Tier A gap" {
        matlab -batch "cd matlab; check_inner_loops"
    }
}

Step 3 "fidelity against the release, primary layer, differential" {
    python scripts\06_validate_against_dataset.py --root $Dataset --decimate 100 --primary 0.05 --out $Results
}

# --------------------------------------------------------------------------
# PHASE 4  Mechanism: what the attack does and what no gain can undo
# --------------------------------------------------------------------------
# These are plant-only measurements. They produce the scaling law, which is the
# analytic case for acting on the signal before the consensus update rather
# than tuning the secondary layer.

Step 4 "injection point and synchronism" {
    python scripts\16_injection_point_probe.py --root $Dataset --long 3.0 --out $Results
}

Step 4 "settling timescale, release vs plant" {
    python scripts\17_settling_timescale.py --root $Dataset --attack Additive --span 3.0 --out $Results
}

Step 4 "consensus gain sweep (expect saturation, not 1/c_f)" {
    python scripts\18_cf_sweep.py --root $Dataset --values 10,20,50,100,200,300 --span 3.0 --out $Results
}

Step 4 "restoration gain sweep (expect tau to RISE)" {
    python scripts\19_param_sweep.py --root $Dataset --param k_p --values 5,10,20,40,80 --span 20 --out $Results
}

Step 4 "droop sweep (expect exact 1/m scaling of the steady-state effect)" {
    python scripts\19_param_sweep.py --root $Dataset --param m_droop --values 0.785,1.571,3.142 --span 20 --out $Results
}

Step 4 "saturation limits are not the binding constraint" {
    python scripts\19_param_sweep.py --root $Dataset --param zeta_limit --values 0.1,0.3,1.0 --span 20 --out $Results
    python scripts\19_param_sweep.py --root $Dataset --param xi_limit --values 31.4,100,300 --span 20 --out $Results
}

Step 4 "attack-amplitude scaling (expect log-log slope 1.0000)" {
    python scripts\20_attack_scaling.py --attr bias_xi --values 0.05,0.1,0.2,0.3,0.6,1.0 --span 20 --out $Results
}

Step 4 "ring-size scaling, even and odd" {
    python scripts\21_topology_scaling.py --values 6,8,10,12,14 --span 200 --out $Results
    python scripts\21_topology_scaling.py --values 7,9,11,13 --span 200 --out $Results
}

# --------------------------------------------------------------------------
# PHASE 5  Pre-training admissibility, then training
# --------------------------------------------------------------------------
# The admissibility tests run BEFORE training, which is the whole point of
# them: all three fail silently and look identical from the training curve.
# Two of the three are in tests\test_plant.py (expressiveness, exploration
# affordability) and already ran in Phase 0.

Step 5 "reward calibration and class separation" {
    python scripts\00_calibrate_reward.py --episodes 30 --sep-seeds 10 --out $Results
}

Step 5 "observability: is the attacked source identifiable at all" {
    python scripts\13_observability_probe.py --episodes 60 --repeats 3 --out $Results
}

if (-not $SkipTraining) {
    foreach ($s in 0,1,2) {
        Step 5 "train seed $s ($Timesteps steps)" {
            python scripts\02_train_agent.py --timesteps $Timesteps --n-envs $NEnvs --seed $s --outdir "runs\ppo3_s$s"
        }
    }
} else {
    Write-Host "`n=== [5] training skipped, reusing runs\ppo3_s0..2" -ForegroundColor Yellow
}

# Provenance guard. runs/ppo2_s0..2 hold policies bit-identical to ppo3 -- the
# same weights from an earlier campaign -- but they predate train_config.json,
# so they cannot be shown to be current from their own directory. results/eval2_*
# is older still: it evaluated a checkpoint that predates ppo2 entirely, and
# mixing it with eval3_* is what put a stale 0.40 into the reported range for
# frac_components_in_flat_region. Only eval3_* is reported.
foreach ($s in 0,1,2) {
    $cfg = "runs\ppo3_s$s\train_config.json"
    if (-not (Test-Path $cfg)) {
        throw "Missing $cfg. Every reported run must record its own configuration; a run directory that cannot identify itself is how a stale evaluation set entered the results once already."
    }
}

foreach ($s in 0,1,2) {
    Step 5 "evaluate seed $s against the nominal controller, 20 seeds" {
        python scripts\03_evaluate.py --agent "runs\ppo3_s$s\agent_final.zip" --vecnorm "runs\ppo3_s$s\vecnormalize.pkl" --seeds 20 --out "$Results\eval3_s$s"
    }
}

# --------------------------------------------------------------------------
# PHASE 6  Explanation: faithfulness battery and the two-axis LLM ablation
# --------------------------------------------------------------------------
# --reuse-faithfulness on every run after the first: the battery is 100 states
# of integrated gradients plus deletion curves and does not depend on which
# generator is being scored.

Step 6 "faithfulness battery + template baseline" {
    python scripts\04_explain.py --agent "runs\ppo3_s0\agent_final.zip" --vecnorm "runs\ppo3_s0\vecnormalize.pkl" --n-states 100 --n-explanations 100 --no-llm --out "$Results\explain_template"
}

if (-not $SkipLLM) {
    foreach ($m in "llama3:8b","llama3.1:8b","llama3.2:3b","mistral-nemo") {
        $tag = $m -replace "[:.]", "_"
        Step 6 "LLM ablation: $m (baseline prompt)" {
            python scripts\04_explain.py --agent "runs\ppo3_s0\agent_final.zip" --vecnorm "runs\ppo3_s0\vecnormalize.pkl" --n-states 100 --n-explanations 100 --llm $m --timeout 300 --prompt-variant baseline --reuse-faithfulness --out "$Results\explain_$tag"
        }
    }
    Step 6 "prompt-design variant on the 3B model" {
        python scripts\04_explain.py --agent "runs\ppo3_s0\agent_final.zip" --vecnorm "runs\ppo3_s0\vecnormalize.pkl" --n-states 100 --n-explanations 100 --llm "llama3.2:3b" --timeout 300 --prompt-variant enumerate --reuse-faithfulness --out "$Results\explain_enumerate"
    }
}

# --------------------------------------------------------------------------
# PHASE 7  Export
# --------------------------------------------------------------------------
# Needs onnxscript: newer torch routes torch.onnx.export through the dynamo
# exporter, which imports it unconditionally and fails with a bare
# ModuleNotFoundError that gives no hint the dependency is optional.

Step 7 "ONNX export with VecNormalize statistics" {
    # NOT `python -c "import onnxscript" 2>$null`. With $ErrorActionPreference
    # set to Stop, PowerShell turns a native command's stderr into a
    # terminating error, so the ImportError traceback aborted the run even
    # though the redirection was meant to swallow it -- a probe that killed
    # the script it was probing for. find_spec returns None instead of
    # raising, so nothing reaches stderr and the exit code carries the answer.
    python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('onnxscript') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    onnxscript missing; installing (torch.onnx.export needs it)"
        pip install onnxscript
    }
    python scripts\07_export_onnx.py --agent "runs\ppo3_s0\agent_final.zip" --vecnorm "runs\ppo3_s0\vecnormalize.pkl" --out "runs\ppo3_s0\actor.onnx"
}

# --------------------------------------------------------------------------

$mins = ((Get-Date) - $started).TotalMinutes
Write-Host ("`nComplete in {0:n1} min. Transcript: {1}" -f $mins, $log) -ForegroundColor Green
Write-Host "Compare against the manuscript before concluding anything. A stage"
Write-Host "that ran without erroring has not thereby been verified."
Stop-Transcript | Out-Null
