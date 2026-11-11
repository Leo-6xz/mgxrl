from fl import *
OUT = 'out'
N = dict(size=9)
def S(**k): d = dict(size=9); d.update(k); return d
H9 = S(b=1, size=9.5)

# ---------------- Fig 5 scaling law
f = Fig('Fig. 5 - Measured scaling law for a consensus bias')
f.text(227, 15, 'Steady-state displacement from a consensus bias: measured scaling law', 10.5, b=1, anchor='middle')
f.text(227, 30, 'spread = ⌊(n−2)/2⌋ · ⌈(n−2)/2⌉ / n · bias / m_{droop}', 9.5, b=1, c=RED, anchor='middle')
f.text(227, 43, '— contains no secondary-control gain', 9.5, b=1, c=RED, anchor='middle')
# original panel geometry: x 70..260 (190 wide), y 80..210 (130 tall)
PW, PH, PT = 112, 100, 72
panels = [
    (34, '(a) linear in amplitude', 'injected bias_{ξ} (rad/s)', 'spread (pu)',
     'log–log slope 1.0000, ratio constant to zero drift'),
    (190, '(b) inverse in droop', 'm_{droop} (rad/s per pu)', 'mean |ΔP| (pu)',
     'product constant to 0.6% across a 4× range'),
    (334, '(c) quadratic in ring size', 'ring size n', 'spread · m_{droop} / bias',
     'filled = even n, open = odd n; closed form exact at all nine sizes')]
def mp(px, x, y, ox): return px + (x - ox) / 190 * PW, PT + (y - 80) / 130 * PH
bottoms = []
for k, (px, t, xl, yl, note) in enumerate(panels):
    f.text(px - 20, PT - 10, t, 9.5, b=1)
    f.line(px, PT + PH, px + PW, PT + PH); f.line(px, PT, px, PT + PH)
    f.text(px + PW / 2, PT + PH + 13, xl, 9, anchor='middle')
    f.text(px - 9, PT + PH / 2, yl, 9, anchor='middle', rot=-90)
    bottoms.append(f.para(px + PW / 2, PT + PH + 18, note, 132, 8.5, anchor='middle'))
ox = [70, 330, 590]
pa = [(79.0, 203.8), (88.1, 197.6), (106.2, 185.2), (124.3, 172.9), (178.6, 135.7), (251.0, 86.2)]
P = [mp(34, x, y, 70) for x, y in pa]
f.path('M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in P), c=GREY, sw=0.9)
for x, y in P: f.circle(x, y, 2.2, fill=INK, stroke=INK, sw=0.5)
pb = [(373.9, 91.6, '0.20033'), (417.8, 150.9, '0.10002'), (505.6, 180.6, '0.04977')]
P = [mp(190, x, y, 330) for x, y, _ in pb]
f.path('M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in P), c=GREY, sw=0.9, dash='3,2')
for (x, y), (_, _, l), dx, dy, an in zip(P, pb, [5, 4, 0], [3, -5, 13], ['start', 'start', 'middle']):
    f.circle(x, y, 2.2, fill=INK, stroke=INK, sw=0.5)
    f.text(x + dx, y + dy, l, 8.5, anchor=an)
pc = [(615.0, 179.0), (640.0, 170.2), (665.0, 157.8), (690.0, 148.1), (715.0, 135.7), (740.0, 125.6),
      (765.0, 113.3), (790.0, 102.9), (815.0, 90.6)]
for k, (x, y) in enumerate(pc):
    X, Y = mp(334, x, y, 590 - 0) ; X = 334 + (x - 590) / 250 * PW
    f.circle(X, Y, 2.3, fill=INK if k % 2 == 0 else '#ffffff', stroke=INK, sw=0.8)
yb = max(bottoms) + 10
f.box(6, yb, 442, [('No secondary-control gain attenuates it', S(b=1, size=10, c=RED)),
                   ('consensus gain c_{f} over 30× (10 → 300):  steady-state effect 0.0965 → 0.10003 pu — unchanged',
                    S(gap=4)),
                   ('restoration gain k_{p} over 8× (5 → 40):  0.10003 → 0.09957 pu unchanged, while settling rises '
                    '0.83 → 3.11 s', N),
                   ('Integral action drives the error to zero against a corrupted reference. Higher gain reaches the '
                    'wrong equilibrium faster, which is the analytic case for acting on the signal before it enters the '
                    'consensus update.', S(c=GREY, gap=5))],
      fill=F3, stroke=RED, sw=1.0, dash='4,2.5', pad=7, padx=10)
f.save('Fig05_scaling_law', OUT)

# ---------------- Fig 17 fidelity tiers
f = Fig('Fig. 17 - Two-tier fidelity validation')
f.text(227, 15, 'Two-tier fidelity validation program', 10.5, b=1, anchor='middle')
y = f.para(227, 20, 'Switching-level comparison is not part of the program: the averaging assumption is adopted from '
           'established practice and deferred to hardware validation.', 440, 8.5, c=GREY, anchor='middle') + 6
tiers = [
    ('TIER A', ['reduced-order plant vs stiff-solver reference', 'Python fwd-Euler 1 ms  vs  MATLAB ode23tb with inner '
                'voltage loops'],
     [('Tests: discretization and the algebraic-voltage assumption', N),
      ('f-RMSE  4e−8 … 5e−5 Hz; error ordered by dynamic content', S(gap=5)),
      ('inner loops contribute 18% of the gap', S(b=1, gap=5)), ('(Sinusoid: 9.7e−6 of 5.3e−5 Hz)', N)], 'PASS'),
    ('TIER B', ['vs IEEE Dataport measurements', 'differential: attacked file minus Normal, matched windows, per timestep'],
     [('Tests: the physics itself — the only non-self-referential comparison', N),
      ('primary layer, 50 ms window:', S(gap=5)), ('direction agrees on P, corr = 0.9368', S(b=1)),
      ('5 of 6 classes pass the pre-onset gate', S(gap=5)), ('(ratios 3.2e6 … 9.7e7)', N)], 'PASS — PRIMARY LAYER')]
tw = 206; xs = [6, 242]
hs = [f.block_h([(s, S(i=1)) for s in t[1]], tw - 14, gap=1) for t in tiers]
hb = [f.block_h(t[2], tw - 14) for t in tiers]
H = 8 + 14 + max(hs) + 10 + max(hb) + 10 + 22 + 8
for k, (name, sub, body, verdict) in enumerate(tiers):
    x = xs[k]
    f.rect(x, y, tw, H, fill=PANEL, stroke=INK, sw=1.2)
    f.text(x + tw / 2, y + 17, name, 11, b=1, anchor='middle')
    f.block(x + tw / 2, y + 22, [(s, S(i=1)) for s in sub], tw - 14, gap=1)
    yl = y + 22 + max(hs) + 5
    f.line(x + 8, yl, x + tw - 8, yl, c=LGREY, sw=0.6)
    f.block(x + tw / 2, yl + 5, body, tw - 14)
    bw = 50 if k == 0 else 132
    f.box(x + tw / 2 - bw / 2, y + H - 28, bw, [(verdict, S(b=1, size=9.5))], h=20, fill=F3, sw=1.1)
f.line(212, y + H / 2, 242, y + H / 2, c=GREY, arrow=True)
ys = y + H + 10
f.box(6, ys, 442, [('Scope limits, and both are properties of the reference rather than of the plant', S(b=1, size=9.5, c=RED)),
                   ('Secondary layer not validated: the release’s restoration runs beyond one second (confirmed by an '
                    'author) while the record is 1.00 s, so the attack’s effect on secondary control never develops '
                    'inside the data. Denial of service excluded: that file does not share a startup with Normal (gate '
                    'ratio 0.92 against a floor of 10), so it is not the same simulation and no differential against it '
                    'is defensible.', S(align='start', gap=4))],
      fill=F3, stroke=RED, sw=1.0, dash='4,2.5', pad=7, padx=10)
f.save('Fig17_fidelity_tiers', OUT)

# ---------------- Fig 18 differential method
f = Fig('Fig. 18 - Differential fidelity comparison and pre-onset gate')
f.text(227, 15, 'Differential fidelity comparison and the pre-onset admissibility gate', 10.5, b=1, anchor='middle')
f.text(6, 36, '(a) why a single-file baseline cannot work', 9.5, b=1)
# schematic: x 50..226
X0, X1 = 52, 226
sx = lambda x: X0 + (x - 60) / 360 * (X1 - X0)
for yb, lab in [(78, 'attacked run'), (122, 'Normal run')]:
    f.line(X0, yb, X1, yb, sw=0.9)
    f.text(X0 - 3, yb + 3, lab, 9, anchor='end') if False else None
    f.path(f'M {sx(60)} {yb} Q {sx(82)} {yb - 18} {sx(104)} {yb - 4} T {sx(156)} {yb - 1} L {X1} {yb}', c=GREY, sw=0.9)
f.text(X0 - 3, 81, 'attacked', 9, anchor='end'); f.text(X0 - 3, 92, 'run', 9, anchor='end')
f.text(X0 - 3, 125, 'Normal', 9, anchor='end'); f.text(X0 - 3, 136, 'run', 9, anchor='end')
f.rect(sx(246), 62, sx(354) - sx(246), 22, fill='#f3e3e3', stroke=RED, sw=0.9, dash='3,2', rx=0)
f.text((sx(246) + sx(354)) / 2 + 6, 51, 'attack window 0.5005–0.801 s', 8.5, c=RED, anchor='middle')
f.line(sx(104), 52, sx(104), 140, c=GREY, sw=0.7, dash='2.5,2')
f.text(sx(104), 151, 'trim 0.15 s', 8.5, c=GREY, anchor='middle')
f.line(sx(246), 58, sx(246), 140, sw=0.7, dash='2.5,2')
f.text(sx(246), 151, 'onset', 8.5, anchor='middle')
ya = f.block(120, 158, [('both runs are the SAME SOLVE before onset: pre-onset difference 1e−6 … 1e−13 on every P, Q '
                         'and f channel', N)], 222)
# (b) table
bx = 250
f.text(bx, 36, '(b) gate: post-onset / pre-onset', 9.5, b=1)
f.text(bx + 15, 48, 'differential RMS', 9.5, b=1)
c1, c2, c3 = bx + 6, bx + 90, bx + 104
f.text(c1, 66, 'class', 9, b=1); f.text(c2, 66, 'ratio', 9, b=1, anchor='end'); f.text(c3, 66, 'verdict', 9, b=1)
f.line(c1, 70, 446, 70, c=LGREY, sw=0.6)
rows = [('Additive', '9.7e7'), ('Ramp', '6.2e7'), ('Sinusoid', '3.9e7'), ('SlowRamp', '4.1e6'), ('Stealth', '3.2e6')]
yy = 70
for n_, r in rows:
    yy += 13.5
    f.text(c1, yy, n_, 9); f.text(c2, yy, r, 9, anchor='end'); f.text(c3, yy, 'same solve', 9)
f.line(c1, yy + 4, 446, yy + 4, c=LGREY, sw=0.6)
yy += 15
f.text(c1, yy, 'DoS', 9, c=RED); f.text(c2, yy, '0.92', 9, c=RED, anchor='end')
f.text(c3, yy, 'NOT COMPARABLE', 9, b=1, c=RED)
f.text(c1, yy + 16, 'floor = 10', 8.5, c=GREY)
# (c)
yc = max(ya, yy + 20) + 12
f.text(6, yc + 9, '(c) what is actually compared', 9.5, b=1)
y1 = yc + 18
h1 = f.box(20, y1, 150, [('Δ_{release}(t) =', H9), ('attacked file − Normal file', N)], fill=PANEL, sw=1.2, h=38)
y2 = y1 + 48
h2 = f.box(20, y2, 150, [('Δ_{plant}(t) =', H9), ('attacked run − unattacked run', N), ('same seed, same settle', S(size=8.5, c=GREY))],
           fill=PANEL, sw=1.2, h=50)
cy = (y1 + y2 + h2) / 2
hc = f.box(222, cy - 28, 210, [('compare on the 50 ms droop-dominated window', N), ('direction and shape only', S(i=1, c=GREY))],
           fill=F3, sw=1.2, h=56)
f.line(170, y1 + 19, 222, cy - 8, c=GREY, arrow=True)
f.line(170, y2 + 25, 222, cy + 8, c=GREY, arrow=True)
f.para(227, y2 + h2 + 8, 'The energization transient is common to both sides and cancels exactly. No baseline window is '
       'chosen anywhere.', 440, 9, c=GREY, anchor='middle')
f.save('Fig18_differential_method', OUT)
