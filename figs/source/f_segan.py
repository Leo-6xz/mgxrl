"""Data figures for the SEGAN manuscript (Figs. 4-8). Run from figs/source."""
import json, os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, '..', '..', 'results'))
os.chdir(HERE)
os.makedirs('out', exist_ok=True)
_need = ['baselines/summary.csv', 'baselines/trajectory_Ramp_s1.npz', 'robustness/comms_paired.csv',
         'robustness/graphs_paired.csv', 'robustness/adaptive.csv', 'robustness/latency_threshold.csv',
         'robustness/latency_oracle.csv', 'matlab_validation/python_metrics.csv',
         'matlab_validation/matlab_metrics.csv']
_missing = [p for p in _need if not os.path.exists(os.path.join(R, p))]
if _missing:
    raise SystemExit('Missing input files under ' + R + ':\n  ' + '\n  '.join(_missing))
INK, GREY, LGREY = '#1a1a1a', '#5f5f5f', '#b0b0b0'
# categorical slots 1-4 (validated, light surface); nominal is a neutral grey
C = {'nominal': '#8a8985', 'wmsr': '#eb6834', 'threshold': '#2a78d6', 'oracle': '#1baf7a', 'agent': '#eda100'}
LAB = {'nominal': 'Nominal consensus', 'wmsr': 'W-MSR (F = 1)', 'threshold': 'Innovation threshold',
       'oracle': 'Oracle isolation', 'agent': 'PPO agent (3 training seeds)'}
from matplotlib import font_manager as _fm
_have = {f.name for f in _fm.fontManager.ttflist}
SERIF = next((f for f in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif') if f in _have), 'serif')
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [SERIF], 'mathtext.fontset': 'custom',
    'mathtext.rm': SERIF, 'mathtext.it': f'{SERIF}:italic',
    'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.8, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'svg.fonttype': 'none', 'savefig.dpi': 600})
W = 7.0
FAM = "'Times New Roman', 'Liberation Serif', Times, serif"
ATT = ['Normal', 'Additive', 'Ramp', 'SlowRamp', 'Sinusoid', 'Stealth', 'DoS']


def save(fig, name):
    fig.savefig(f'out/{name}.png'); fig.savefig(f'out/{name}.svg')
    s = open(f'out/{name}.svg', encoding='utf-8').read()
    s = re.sub(r"font-family:\s*'?" + re.escape(SERIF) + r"'?", f"font-family: {FAM}", s)
    open(f'out/{name}.svg', 'w', encoding='utf-8').write(s)
    plt.close(fig)


def panel(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=10, ha='right', va='bottom')


# ------------------------------------------------------------------ Fig 4: main comparison
S = pd.read_csv(f'{R}/baselines/summary.csv')
ag = S[S.controller.str.startswith('agent')]
fig, axs = plt.subplots(2, 1, figsize=(W, 4.9), layout='constrained', sharex=True)
x = np.arange(len(ATT)); arms = ['nominal', 'wmsr', 'threshold', 'oracle', 'agent']; bw = 0.15
for k, arm in enumerate(arms):
    off = (k - 2) * (bw + 0.02)
    for ai, (ax, col, scale) in enumerate(zip(axs, ['return', 'p_sharing_error'], [1e-3, 1])):
        if arm == 'agent':
            g = ag.groupby('attack')[col].agg(['mean', 'min', 'max']).reindex(ATT)
            y = g['mean'].to_numpy() * scale
            err = np.vstack([y - g['min'].to_numpy() * scale, g['max'].to_numpy() * scale - y])
        else:
            y = S[S.controller == arm].set_index('attack')[col].reindex(ATT).to_numpy() * scale; err = None
        if ai == 0:
            ax.bar(x + off, y, bw, color=C[arm], label=LAB[arm], yerr=err, ecolor=INK,
                   error_kw=dict(lw=0.8, capsize=1.6), edgecolor='white', linewidth=0.6)
        else:
            ax.bar(x + off, y, bw, color=C[arm], yerr=err, ecolor=INK,
                   error_kw=dict(lw=0.8, capsize=1.6), edgecolor='white', linewidth=0.6)
axs[0].axhline(0, color=INK, lw=0.8)
axs[0].set_ylabel('Mean episode return (×10³)')
axs[1].set_yscale('log'); axs[1].set_ylabel('Power-sharing error\n(std/mean of P)')
axs[1].set_xticks(x, ['Normal', 'Additive', 'Ramp', 'Slow ramp', 'Sinusoid', 'Coordinated\nstealth', 'Denial of\nservice'])
panel(axs[0], '(a)'); panel(axs[1], '(b)')
axs[0].legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.06), handlelength=1.2, columnspacing=1.2)
save(fig, 'Fig04_main_comparison')

# ------------------------------------------------------------------ Fig 5: trajectory
T = np.load(f'{R}/baselines/trajectory_Ramp_s1.npz')
on = float(T['nominal_onset']); tgt = int(T['nominal_targets'][0])
fig, axs = plt.subplots(2, 3, figsize=(W, 3.9), layout='constrained', sharex=True)
for c, arm in enumerate(['nominal', 'wmsr', 'threshold']):
    t = T[f'{arm}_t']; f = T[f'{arm}_f']; P = T[f'{arm}_P']
    ax = axs[0, c]
    for j in range(f.shape[1]):
        ax.plot(t, (f[:, j] - 50) * 1e3, lw=1.6 if j == tgt else 0.7,
                color=C[arm] if j == tgt else LGREY, zorder=3 if j == tgt else 2)
    ax.set_title({'nominal': 'Nominal consensus', 'wmsr': 'W-MSR (F = 1)', 'threshold': 'Innovation threshold'}[arm],
                 fontsize=9.5)
    ax = axs[1, c]
    for j in range(P.shape[1]):
        ax.plot(t, P[:, j], lw=1.6 if j == tgt else 0.7, color=C[arm] if j == tgt else LGREY,
                zorder=3 if j == tgt else 2)
    for r in range(2):
        axs[r, c].axvline(on, color=INK, lw=0.7, ls=(0, (3, 2)))
    if arm == 'nominal' and len(t) < 1200:
        for r in range(2):
            axs[r, c].axvline(t[-1], color=INK, lw=0.8)
            axs[r, c].text(t[-1], 0.96, ' trip', transform=axs[r, c].get_xaxis_transform(), fontsize=8,
                           ha='right', va='top', color=INK)
axs[0, 0].set_ylabel('Frequency deviation (mHz)')
axs[1, 0].set_ylabel('Active power (pu)')
for c in range(3):
    axs[1, c].set_xlabel('Time (s)')
for c in (1, 2):
    lo, hi = np.inf, -np.inf
    for r in range(1):
        pass
axs[0, 0].set_ylim(-600, 600); axs[0, 1].set_ylim(-3, 3); axs[0, 2].set_ylim(-3, 3)
axs[1, 0].set_ylim(-0.2, 1.4); axs[1, 1].set_ylim(0.55, 0.75); axs[1, 2].set_ylim(0.55, 0.75)
axs[0, 0].text(on, 0.04, ' onset', transform=axs[0, 0].get_xaxis_transform(), fontsize=8, color=INK)
for c in range(3):
    panel(axs[0, c], f'({"abc"[c]})'); panel(axs[1, c], f'({"def"[c]})')
save(fig, 'Fig05_trajectory_ramp')

# ------------------------------------------------------------------ Fig 6: trust and edges
fig, ax = plt.subplots(figsize=(W, 1.9), layout='constrained')
t = T['threshold_t']; w = T['threshold_w_xi']
for j in range(w.shape[1]):
    if j != tgt:
        ax.plot(t, w[:, j], color=LGREY, lw=0.7)
ax.plot(t, w[:, tgt], color=C['threshold'], lw=1.6, label=f'Trust in attacked source DG{tgt + 1} (threshold rule)')
ax.plot(T['wmsr_t'], T['wmsr_kept_xi'], color=C['wmsr'], lw=1.2, label='Fraction of ξ edges kept by W-MSR')
ax.axvline(on, color=INK, lw=0.7, ls=(0, (3, 2)))
ax.set_ylim(-0.05, 1.08); ax.set_xlabel('Time (s)'); ax.set_ylabel('Weight / fraction')
ax.legend(loc='center right', handlelength=1.4)
save(fig, 'Fig06_trust_edges')

# ------------------------------------------------------------------ Fig 7: robustness panel
fig, axs = plt.subplots(2, 2, figsize=(W, 4.9), layout='constrained')
# (a) communications
cp = pd.read_csv(f'{R}/robustness/comms_paired.csv'); cp = cp[cp.arm == 'threshold_matched']
chans = ['d10ms_l0', 'd50ms_l0', 'd0ms_l0.2', 'd0ms_l0.5', 'd20ms_l0.2']
cl = {'d10ms_l0': '10 ms delay', 'd50ms_l0': '50 ms delay', 'd0ms_l0.2': '20 % loss',
      'd0ms_l0.5': '50 % loss', 'd20ms_l0.2': '20 ms + 20 % loss'}
ax = axs[0, 0]; att = ['Additive', 'Ramp', 'SlowRamp', 'Sinusoid', 'Stealth']
greys = ['#c9c8c4', '#a9a8a4', '#86b6ef', '#2a78d6', '#184f95']
for k, ch in enumerate(chans):
    v = cp[cp.channel == ch].set_index('attack').reindex(att)['delta_vs_nominal'] / 1e3
    ax.bar(np.arange(5) + (k - 2) * 0.16, v, 0.15, color=greys[k], label=cl[ch], edgecolor='white', linewidth=0.5)
ax.set_xticks(range(5), ['Additive', 'Ramp', 'Slow\nramp', 'Sinusoid', 'Stealth'])
ax.set_ylabel('Return gain over nominal (×10³)'); ax.legend(fontsize=7.5, ncol=1, loc='upper right')
panel(ax, '(a)')
# (b) graph density: W-MSR vs threshold, Normal cost and Stealth return
gp = pd.read_csv(f'{R}/robustness/graphs_paired.csv'); base = S.set_index(['controller', 'attack'])['return']
ax = axs[0, 1]
labs = ['Ring\n(1-robust)', 'Degree 4\n(2-robust)', 'Degree 6\n(3-robust)']
wn = [base['wmsr', 'Normal'] - base['nominal', 'Normal']] + \
     [gp[(gp.graph == g) & (gp.arm == 'wmsr') & (gp.attack == 'Normal')].delta_vs_nominal.item() for g in ('circ2', 'circ3')]
ws = [base['wmsr', 'Stealth']] + [gp[(gp.graph == g) & (gp.arm == 'wmsr') & (gp.attack == 'Stealth')].mean_return.item() for g in ('circ2', 'circ3')]
ts = [base['threshold', 'Stealth']] + [gp[(gp.graph == g) & (gp.arm == 'threshold') & (gp.attack == 'Stealth')].mean_return.item() for g in ('circ2', 'circ3')]
xx = np.arange(3)
ax.bar(xx - 0.2, np.array(ws) / 1e3, 0.36, color=C['wmsr'], label='W-MSR, coordinated stealth', edgecolor='white')
ax.bar(xx + 0.2, np.array(ts) / 1e3, 0.36, color=C['threshold'], label='Threshold, coordinated stealth', edgecolor='white')
labs = ['Ring (1-robust)', 'Degree 4 (2-robust)', 'Degree 6 (3-robust)']
labs = [f'{l}\nW-MSR attack-free\ncost ' + f'{v:,.0f}'.replace('-', '−') for l, v in zip(labs, wn)]
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(xx, labs, fontsize=7.8); ax.set_ylabel('Mean return (×10³)')
ax.legend(fontsize=7.5, loc='lower right'); ax.set_ylim(-13.5, 8.5)
panel(ax, '(b)')
# (c) sub-threshold attacker
ad = pd.read_csv(f'{R}/robustness/adaptive.csv')
g = ad.groupby(['alpha', 'arm'])['p_sharing_error'].mean().unstack()
ax = axs[1, 0]; al = g.index.to_numpy()
ax.plot(al, g['threshold'], 's-', color=C['threshold'], lw=1.6, ms=5, label='Innovation threshold', zorder=3)
ax.plot(al, g['nominal'], 'o--', color=C['nominal'], lw=1.4, ms=10, mfc='none', mew=1.6, label='Nominal consensus', zorder=4)
ax.axhline(0.0030, color=INK, lw=0.7, ls=(0, (3, 2)))
ax.text(0.47, 0.00305, 'attack-free level', fontsize=7.5, va='bottom', color=GREY)
ax.axvline(1.0, color=LGREY, lw=0.8)
ax.text(1.0, 0.0098, ' threshold τ', fontsize=7.5, color=GREY, va='top')
ax.set_xlabel('Attacker innovation as a multiple of τ (α)'); ax.set_ylabel('Power-sharing error')
ax.set_xticks([0.5, 0.9, 1.2]); ax.legend(fontsize=7.5, loc='upper left'); ax.set_ylim(0, 0.0105)
panel(ax, '(c)')
# (d) isolation latency
lt = pd.read_csv(f'{R}/robustness/latency_threshold.csv'); lo = pd.read_csv(f'{R}/robustness/latency_oracle.csv')
ax = axs[1, 1]; la = ['Additive', 'Ramp', 'Sinusoid', 'Stealth']
for i, a in enumerate(la):
    v = lt[(lt.attack == a) & lt.isolated].latency_s * 1e3
    jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.28
    ax.scatter(np.full(len(v), i) + jit, v, s=14, color=C['threshold'], edgecolor='white', linewidth=0.5, zorder=3,
               label='Threshold rule, per episode' if i == 0 else None)
    ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color=INK, lw=1.4, zorder=4,
            label='Median' if i == 0 else None)
om = lo.latency_s.median() * 1e3
ax.axhline(om, color=C['oracle'], lw=1.2, ls=(0, (4, 2)), label=f'Oracle (rate-limit floor, {om:.0f} ms)')
ax.set_yscale('log'); ax.set_xticks(range(4), ['Additive', 'Ramp', 'Sinusoid', 'Stealth'])
ax.set_ylabel('Time to full isolation (ms)'); ax.legend(fontsize=7.5, loc='upper left')
ax.set_ylim(25, 500); ax.set_yticks([30, 50, 100, 200, 400], ['30', '50', '100', '200', '400']); ax.minorticks_off()
panel(ax, '(d)')
save(fig, 'Fig07_robustness')

# ------------------------------------------------------------------ Fig 8: MATLAB vs Python
M = pd.read_csv(f'{R}/matlab_validation/python_metrics.csv').merge(
    pd.read_csv(f'{R}/matlab_validation/matlab_metrics.csv'), on=['attack', 'seed', 'arm'], suffixes=(' py', ' ml'))
M = M[M.attack != 'DoS']
fig, axs = plt.subplots(1, 2, figsize=(W, 2.9), layout='constrained')
ax = axs[0]
for arm, mk in (('nominal', 'o'), ('threshold', 's')):
    d = M[M.arm == arm]
    ax.scatter(d['f_rms_mHz py'], d['f_rms_mHz ml'], s=18, marker=mk, color=C[arm], edgecolor='white', linewidth=0.5,
               label={'nominal': 'Nominal consensus', 'threshold': 'Innovation threshold'}[arm], zorder=3)
lim = [0.05, 1000]; ax.plot(lim, lim, color=INK, lw=0.7)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Python plant, RMS frequency error (mHz)'); ax.set_ylabel('MATLAB reference (mHz)')
ax.legend(fontsize=7.5, loc='upper left'); panel(ax, '(a)')
ax = axs[1]
d = M[(M.arm == 'threshold') & M['latency_ms py'].notna() & M['latency_ms ml'].notna()]
ax.scatter(d['latency_ms py'], d['latency_ms ml'], s=18, color=C['threshold'], edgecolor='white', linewidth=0.5, zorder=3)
lim2 = [20, 400]; ax.plot(lim2, lim2, color=INK, lw=0.7)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim2); ax.set_ylim(lim2)
tk = [30, 50, 100, 200, 400]; ax.minorticks_off()
ax.set_xticks(tk, [str(v) for v in tk]); ax.set_yticks(tk, [str(v) for v in tk])
ax.set_xlabel('Python plant, isolation latency (ms)'); ax.set_ylabel('MATLAB reference (ms)')
panel(ax, '(b)')
save(fig, 'Fig08_matlab_validation')
print('done')
