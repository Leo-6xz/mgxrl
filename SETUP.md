# SETUP — Python and MATLAB, from nothing

Written for someone who has used neither. Every step ends with a verification
command and its expected output; if a verification fails, stop and fix it
there, because errors compound silently otherwise.

Commands are Windows PowerShell, which is what the results were produced on.
macOS/Linux equivalents are given where they differ.

---

## What you need, and when

| Stage (see `README.md`) | Needs |
|---|---|
| Tests, forensics, detector benchmark, mechanism sweeps, training, evaluation | Python only |
| Tier A fidelity (`validate_fidelity`, `check_inner_loops`) | base MATLAB, no toolboxes |
| Explanation ablation with language models | Ollama |
| ONNX export | Python + `onnxscript` |
| Simulink deployment scripts (not used by any reported result) | Simulink, Reinforcement Learning Toolbox, Deep Learning Toolbox |

No GPU is required. Everything runs on CPU: one 5M-step training run takes
about 48 minutes with eight workers on a commodity workstation.

Hardware: 16 GB RAM (8 GB works with a larger `--decimate` on the release),
4+ CPU cores, about 10 GB free disk without MATLAB.

---

## Part 1 — Python

### 1.1 Install Python

The reported results were produced with **Python 3.14.6**, and
`requirements-lock.txt` pins the package versions that go with it. Use that
version if you want to load the archived agents and reproduce the numbers
exactly.

Windows: download the 64-bit installer from python.org, and on its first screen
tick **"Add python.exe to PATH"**. Missing that box is the most common setup
failure.

Verify in a new PowerShell window:

```powershell
python --version
```

Expected: `Python 3.14.6`.

### 1.2 Place the project

Use a path without spaces or non-ASCII characters, for example
`C:\research\mgxrl`. Then:

```powershell
cd C:\research\mgxrl
```

`dir` should show `README.md`, `reproduce_all.ps1`, `requirements-lock.txt` and
the folders `configs matlab scripts src tests figs runs results`.

### 1.3 Create and activate a virtual environment

```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt should now start with `(.venv)`. Activate it in every new terminal;
a `ModuleNotFoundError` later almost always means the environment is not
active, or a different project's environment is.

If PowerShell says running scripts is disabled, run this once and activate
again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`.

### 1.4 Install the packages

```powershell
python -m pip install --upgrade pip
```
```powershell
pip install -r requirements-lock.txt
```

| Package | Role |
|---|---|
| numpy, scipy | plant mathematics |
| pandas | dataset loading, results tables |
| scikit-learn | metrics, nearest-neighbour leakage probe |
| lightgbm | detector |
| gymnasium | RL environment interface |
| stable-baselines3, torch | PPO agent |
| matplotlib | figures |
| pyyaml | reading `configs/microgrid.yaml` |
| requests | Ollama client |
| tensorboard | training curves |
| pytest | test suite |

Verify:

```powershell
python -c "import numpy, pandas, scipy, sklearn, lightgbm, gymnasium, stable_baselines3, torch, yaml; print('all imports OK')"
```

Expected: `all imports OK`.

### 1.5 Verify the project

```powershell
python -m pytest tests\ -q
```

Expected: **`23 passed`**. This is the real gate: it runs the plant, all six
attacks, the reward checks and two of the three pre-training admissibility
tests. If any test fails, nothing downstream is meaningful.

`ModuleNotFoundError: No module named 'mgxrl'` means an empty `__init__.py` is
missing from `src\mgxrl` or one of its subfolders. Create the missing one:

```powershell
New-Item -ItemType File -Force src\mgxrl\plant\__init__.py
```

### 1.6 Place the data

The IEEE DataPort release (doi:10.21227/bmtr-fr76) is not redistributed with
this code. Download it from IEEE DataPort and arrange it as:

```
Dataset\Normal\*.csv
Dataset\Additive\*.csv
Dataset\Ramp\*.csv
Dataset\SlowRamp\*.csv
Dataset\Sinusoid\*.csv
Dataset\Stealth\*.csv
Dataset\DoS\*.csv
```

The folder names must match exactly; their order defines the multiclass label
indices in `src/mgxrl/detection/dataio.py`. Verify:

```powershell
Get-ChildItem Dataset -Directory | Select-Object Name
```

The synthetic corpus can be regenerated (`scripts\09_generate_detector_dataset.py`)
or downloaded from doi:10.5281/zenodo.24082860 into `data\synthetic`.

### 1.7 Load an archived agent

```powershell
python -c "from stable_baselines3 import PPO; m=PPO.load(r'runs\ppo3_s0\agent_final.zip'); print(m.observation_space.shape)"
```

Expected: `(320,)`. An error here usually means a different
Stable-Baselines3 or torch version from the lock file.

### 1.8 Ollama (explanation ablation only)

Install from ollama.com, then pull the four models the paper compares:

```powershell
ollama pull llama3:8b
```
```powershell
ollama pull llama3.1:8b
```
```powershell
ollama pull llama3.2:3b
```
```powershell
ollama pull mistral-nemo
```

Leave `ollama serve` running in its own terminal. Verify:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

The models run locally; no account or API key is needed and no data leaves the
machine.

### 1.9 ONNX export (optional)

```powershell
pip install onnxscript
```

Newer torch routes `torch.onnx.export` through an exporter that imports
`onnxscript` unconditionally, and fails with a bare `ModuleNotFoundError`
without it.

---

## Part 2 — MATLAB

### 2.1 What is needed

Only **base MATLAB** is needed for the results in the paper: Tier A fidelity
(`validate_fidelity.m`) and the inner-loop split (`check_inner_loops.m`) use no
toolboxes. `validate_fidelity.m` uses `corrcoef`, not the Statistics Toolbox
`corr`, for this reason.

Simulink, Reinforcement Learning Toolbox and Deep Learning Toolbox (with the
free *Deep Learning Toolbox Converter for ONNX Model Format* add-on, R2023b or
newer) are needed only by `build_reference_model.m` and `deploy_policy.m`.
Neither is used by any reported result; see Known issues in `README.md`.

### 2.2 Verify

In MATLAB:

```matlab
cd('C:\research\mgxrl\matlab')
p = mg_params();
disp(p.n)
```

Expected: `10`. `Undefined function 'mg_params'` means the current folder is
wrong; check with `pwd`.

### 2.3 How Python and MATLAB connect

They never call each other. They exchange CSV files through `results\`:

```
scripts\05_export_reference_traj.py  ->  results\reduced_<Attack>.csv  ->  validate_fidelity.m  ->  results\fidelity_tierA.csv
```

Always run the Python export before the MATLAB step that reads it.

### 2.4 One parameter set, two languages

Plant parameters live in the `plant:` section of `configs\microgrid.yaml`.
`matlab\mg_params.m` is generated from it; never edit it by hand. After any
change to the YAML:

```powershell
python scripts\08_sync_matlab_params.py
```

If the two drift apart, Tier A measures the parameter mismatch instead of the
modelling assumption.

The `reward:` section of the YAML is documentation only. Training reads
`RewardWeights` in `src\mgxrl\envs\resilient_env.py`. After any change to the
plant, load model or disturbance schedule, re-measure the reward references:

```powershell
python scripts\00_calibrate_reward.py --episodes 30 --sep-seeds 10 --out results
```

### 2.5 If `build_reference_model.m` fails

It has not been run against a live MATLAB install. The likely failure point is
`setChartScript`, which uses the Stateflow `sfroot` API. Fallback: in the model
window open Library Browser → Simulink → User-Defined Functions, drag a
**MATLAB Function** block onto the canvas, double-click it and paste the
`plantFcn` script text from `build_reference_model.m`.

---

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `'python' is not recognized` | PATH not set at install | Re-run the installer → Modify → add to PATH; open a new terminal |
| `running scripts is disabled` | execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `No module named 'stable_baselines3'` | wrong or inactive environment | `.\.venv\Scripts\Activate.ps1` in the project folder |
| `No module named 'mgxrl'` | missing `__init__.py` | see 1.5 |
| `ModuleNotFoundError: onnxscript` | optional exporter dependency | `pip install onnxscript` |
| `No CSVs under ...\SlowRamp` | dataset folder names differ | rename to the names in 1.6 |
| `overflow encountered in reduce` on release loads | known solver divergence in the release | expected; see `FINDINGS.md`, finding 1 |
| agent fails to load | package versions differ from the lock file | `pip install -r requirements-lock.txt` |
| VecNormalize dimension error in `03_evaluate.py` | statistics from an older observation layout | use the `vecnormalize.pkl` beside the same agent |
| `Undefined function 'mg_params'` (MATLAB) | wrong current folder | `cd` to `matlab`, check with `pwd` |
| Tier A rows show REVIEW | stale MATLAB parameters | run `scripts\08_sync_matlab_params.py`, then rerun |

Every script accepts `--help`, and every MATLAB file documents its reasoning in
its header comments.
