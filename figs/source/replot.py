import json, os, re, numpy as np, matplotlib
os.makedirs('out', exist_ok=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = json.load(open('data.json'))
INK, GREY, LGREY, RED, LIGHT, DARK = '#1a1a1a', '#5f5f5f', '#b0b0b0', '#8c1515', '#d9d9d9', '#5a5a5a'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Liberation Serif'], 'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Serif', 'mathtext.it': 'Liberation Serif:italic', 'mathtext.bf': 'Liberation Serif:bold',
    'font.size': 9, 'axes.labelsize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.8, 'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'svg.fonttype': 'none', 'savefig.dpi': 600})
plt.rcParams['figure.constrained_layout.w_pad'] = 0.04; plt.rcParams['figure.constrained_layout.wspace'] = 0.06
W = 6.3
FAM = "'Times New Roman', 'Liberation Serif', Times, serif"

def save(fig, name, title):
    fig.savefig(f'out/{name}.svg')
    fig.savefig(f'out/{name}.png')
    s = open(f'out/{name}.svg', encoding='utf-8').read()
    s = re.sub(r"font-family:\s*'?Liberation Serif'?", f"font-family: {FAM}", s)
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.S)
    s = s.replace('<defs>', f'<title>{title}</title>\n <defs>', 1)
    open(f'out/{name}.svg', 'w', encoding='utf-8').write(s)
    plt.close(fig)

# ---------------- Fig 12
d = D['fig12']
fig, ax = plt.subplots(layout='constrained', figsize=(W, 2.85))
x = np.arange(6); w = 0.34
ax.bar(x - w / 2, d['obs'], w, color=LIGHT, edgecolor=INK, lw=0.8, label='observation only')
ax.bar(x + w / 2, d['inn'], w, color=DARK, edgecolor=INK, lw=0.8, label='+ consensus innovation')
for k, r in enumerate(d['ref']):
    ax.plot([k - 0.47, k + 0.47], [r, r], color=RED, lw=1.8, solid_capstyle='butt',
            label='reference (null / trivial)' if k == 0 else None)
for k in (4, 5):
    ax.text(k, max(d['obs'][k], d['inn'][k], d['ref'][k]) + 0.035, 'not\nmeasurable', color=RED, fontsize=8.5,
            ha='center', va='bottom', linespacing=1.0)
ax.set_xticks(x, d['classes']); ax.set_ylim(0, 1.05); ax.set_xlim(-0.6, 5.6)
ax.set_ylabel('attacked-source\nidentification accuracy')
h, l = ax.get_legend_handles_labels()
order = [0, 1, 2]
ax.legend([h[i] for i in order], [l[i] for i in order], ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          handlelength=1.8, columnspacing=1.6, borderaxespad=0.2)
save(fig, 'Fig12_observability', 'Fig. 12 - Attacked-source identification accuracy')

# ---------------- Fig 13
d = D['fig13']
fig, ax = plt.subplots(layout='constrained', figsize=(W, 3.9))
mk = ['o', 's', '^']
for s in d['series']:
    c = d['classes'].index(s['cls'])
    yc = c + (s['seed'] - 1) * 0.25
    col = s['color']
    ax.plot([s['lo'], s['hi']], [yc, yc], color=col, alpha=0.6, lw=1.0, zorder=2)
    jy = yc + np.array(s['dy']) / d['pitch']
    ax.scatter(s['x'], jy, s=9, facecolors='none', edgecolors=col, linewidths=0.6, alpha=0.8, zorder=3)
    ax.plot(s['mean'], yc, marker=mk[s['seed']], ms=6.5 if s['seed'] else 7, color=col, mec=col, zorder=4, ls='none')
for k in range(1, 7): ax.axhline(k - 0.5, color=LGREY, lw=0.5, zorder=0)
ax.axvline(0, color=INK, lw=0.9, zorder=1)
ax.set_yticks(range(7), d['classes']); ax.set_ylim(6.55, -0.55)
ax.set_xlim(-22000, 88000)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f'{v:,.0f}'.replace('-', '−')))
ax.set_xlabel('paired return difference, agent minus nominal consensus controller')
ax.set_title('Each point is one of twenty evaluation seeds; bars span the 5th to 95th percentile; filled markers '
             'are class means.', fontsize=8.5, color=GREY, loc='left', pad=6)
hs = [Line2D([], [], marker=m, color=INK, ls='none', ms=6.5) for m in mk] + \
     [Line2D([], [], marker='o', mfc='none', mec=RED, color=RED, ls='none', ms=6.5)]
ax.legend(hs, ['training seed 0', 'training seed 1', 'training seed 2', 'mean degraded'], loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=1,
          handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)
save(fig, 'Fig13_paired_returns', 'Fig. 13 - Paired per-seed return differences')

# ---------------- Fig 15
d = D['fig15']
fig, (a1, a2) = plt.subplots(1, 2, layout='constrained', figsize=(W, 2.95), gridspec_kw=dict(width_ratios=[1.45, 1]))
labs = ['seed 0', 'seed 1', 'seed 2', r'$p_{\mathrm{normal}}$ = 0.5', r'$w_{\mathrm{action}}$ = 3', r'$w_{\mathrm{action}}$ = 10']
ys = np.arange(6)
a1.axvspan(*d['band'], color='#6e6e6e', alpha=0.16, lw=0)
for k, (lab, v) in enumerate(d['left']):
    a1.plot([0.14, v], [k, k], color=LGREY, lw=0.8)
    if k < 3: a1.plot(v, k, 'o', color=INK, ms=7)
    else: a1.plot(v, k, 'D', color=RED, ms=6.5)
a1.axhline(2.5, color=INK, lw=0.8, ls=(0, (3, 1.5)))
a1.set_yticks(ys, labs); a1.set_ylim(5.5, -1.0); a1.set_xlim(0.14, 0.28)
a1.set_xticks(np.arange(0.14, 0.281, 0.02))
a1.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
a1.set_xlabel('mean connectivity loss during healthy operation')
a1.text(sum(d['band']) / 2, -0.62, 'seed-variance band at fixed settings', ha='center', va='center', fontsize=8.5, color=GREY)
a1.set_title('every intervention lies inside the band', fontsize=8.5, color=RED, pad=4)
a1.tick_params(axis='y', length=0)
mks = dict(Normal='o', DoS='s', Stealth='^', Sinusoid='o')
for lab, p in d['right'].items():
    p = np.array(p)
    a2.plot(p[:, 0], p[:, 1], color='#6e6e6e', lw=0.9, marker=mks[lab], mfc=INK, mec=INK, ms=5.5)
    a2.text(p[-1, 0] + 0.005, p[-1, 1], lab, va='center', fontsize=8.5)
a2.set_xlim(0.145, 0.30); a2.set_xticks([0.15, 0.20, 0.25])
a2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
a2.set_ylim(20, 190); a2.set_yticks(range(25, 176, 25))
a2.set_xlabel('mean connectivity loss')
a2.set_ylabel('p-sharing error,\n% worse than nominal')
a2.set_title('the cost, measured physically', fontsize=8.5, color=GREY, loc='left', pad=4)
save(fig, 'Fig15_structural_cost', 'Fig. 15 - Invariance of standing distrust')

# ---------------- Fig 16
d = D['fig16']
fig, (a1, a2) = plt.subplots(1, 2, layout='constrained', figsize=(W, 2.95), gridspec_kw=dict(width_ratios=[1, 1.3]))
x = np.arange(3); w = 0.36
a1.bar(x - w / 2, d['binary'], w, color=LIGHT, edgecolor=INK, lw=0.8, label='binary')
e = np.array(d['multi_err'])
a1.bar(x + w / 2, d['multi'], w, color=DARK, edgecolor=INK, lw=0.8, label='multiclass',
       yerr=[np.array(d['multi']) - e[:, 0], e[:, 1] - np.array(d['multi'])], capsize=3,
       error_kw=dict(lw=0.8, capthick=0.8, ecolor=INK))
a1.set_xticks(x, d['protocols']); a1.set_ylim(0, 1.12); a1.set_ylabel('macro-F1')
a1.set_title('(a) partitioning protocol', fontsize=9.5)
xa = 2 + w / 2 + 0.02
a1.annotate('', xy=(xa, d['multi'][2] + 0.02), xytext=(xa, d['binary'][2]),
            arrowprops=dict(arrowstyle='<->', color=RED, lw=0.9, shrinkA=0, shrinkB=0))
a1.text(xa + 0.06, (d['multi'][2] + d['binary'][2]) / 2 + 0.04, '−24 pts', color=RED, fontsize=8.5, va='center', ha='left')
a1.legend(loc='upper left', ncol=2, handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)
a1.set_xlim(-0.6, 3.0); a1.set_ylim(0, 1.2); a1.set_yticks(np.arange(0, 1.01, 0.2))
xb = np.arange(5)
ce = np.array([[c, c] for c in d['cov'][:1]] + d['cov_err'])
cols = [DARK] + [LIGHT] * 4
a2.bar(xb, d['cov'], 0.55, color=cols, edgecolor=INK, lw=0.8)
a2.errorbar(xb[1:], d['cov'][1:], yerr=[np.array(d['cov'][1:]) - ce[1:, 0], ce[1:, 1] - np.array(d['cov'][1:])],
            fmt='none', ecolor=INK, elinewidth=0.8, capsize=3, capthick=0.8)
a2.axhline(d['baseline'], color=RED, lw=0.9, ls=(0, (4, 2)))
a2.text(4.35, d['baseline'] + 0.03, 'deterministic baseline', color=RED, fontsize=8.5, ha='right', va='bottom')
a2.set_xticks(xb, ['template', 'Llama\n3 8B', 'Llama\n3.1 8B', 'Llama\n3.2 3B', 'Mistral\nNeMo']); a2.tick_params(axis='x', labelsize=8.5)
a2.set_ylim(0, 1.2); a2.set_yticks(np.arange(0, 1.01, 0.2)); a2.set_ylabel('weighted coverage')
a2.set_title('(b) operator-explanation ablation', fontsize=9.5)
save(fig, 'Fig16_benchmarks', 'Fig. 16 - Detector benchmark and explanation coverage')
print('done')
