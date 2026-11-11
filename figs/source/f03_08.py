from fl import *
OUT = 'out'
B = dict(b=1, size=9.5); N = dict(size=9); I = dict(i=1, size=9)
def S(**k): d = dict(size=9); d.update(k); return d

# ---------------- Fig 3 observation
f = Fig('Fig. 3 - Observation construction')
f.text(54, 15, '8 quantities per DG', 9.5, b=1, anchor='middle')
q = ['V − V_{nom}', '(f − f_{nom})·10', 'P − P_{set}', 'Q − Q_{set}', 'ξ / ξ_{lim}', 'ζ / ζ_{lim}',
     'inn_{ξ} / ξ_{lim}', 'inn_{ζ} / ζ_{lim}']
y = 22; hq = 17
for k, s in enumerate(q):
    sh = k >= 6
    f.rect(6, y, 96, hq - 2, fill=F2 if sh else F1, stroke=INK if sh else GREY, sw=1.3 if sh else 0.7)
    f.text(54, y + 11, s, 9, b=sh, anchor='middle'); y += hq
f.text(54, y + 10, 'shaded = added after Test 3', 8.5, i=1, anchor='middle')
mid = 22 + 4 * hq - 1
f.line(102, mid, 122, mid, arrow=True)
# x10 DGs
f.text(158, 34, '× 10 DGs', 9.5, b=1, anchor='middle')
f.rect(122, 40, 72, 104, fill=F1)
for k in range(1, 6): f.line(130, 40 + k * 17.3, 186, 40 + k * 17.3, c=LGREY, sw=0.6)
f.rect(127, mid - 9, 62, 14, fill=F1, stroke='none', sw=0)
f.text(158, mid + 2, '80 per frame', 9, i=1, anchor='middle')
f.line(194, mid, 214, mid, arrow=True)
# frames
f.text(257, 34, '× 4 frames (history)', 9.5, b=1, anchor='middle')
f.rect(214, 40, 86, 104, fill=F1)
for k, s in enumerate(['t−3', 't−2', 't−1', 't']):
    f.rect(222, 47 + k * 23.5, 70, 18, fill=F3, stroke=GREY, sw=0.7)
    f.text(257, 47 + k * 23.5 + 12.6, s, 9, anchor='middle')
f.line(300, mid, 320, mid, arrow=True)
f.box(320, mid - 27, 128, [('Observation vector', B), ('320 elements', N), ('(detector off in all runs)', N)],
      h=54, fill=F2)
y = 22 + 8 * hq + 22
h = f.box(6, y, 442, [('Memory, not prediction.', S(b=1, size=9.5, align='start')),
                      ('The history stack conditions the policy on the recent past; it contains no forecast, no '
                       'horizon and no learned transition model. The controller is reactive.', S(align='start'))],
          fill=F3, stroke=LGREY, pad=6, padx=9, anchor='start')
f.para(227, y + h + 9, 'Exactly 320 in every reported run. Enabling the detector posterior would make it 327 and '
       'invalidate all existing checkpoints.', 440, 8.5, anchor='middle')
f.save('Fig03_observation', OUT)

# ---------------- Fig 4 trust map
f = Fig('Fig. 4 - Trust map and connectivity loss')
X0, X1, Y0, Y1 = 46, 186, 168, 46      # axes box: a in [-1,1], w in [0,1]
xa = lambda a: X0 + (a + 1) / 2 * (X1 - X0)
yw = lambda w: Y0 - w * (Y0 - Y1 - 8)
f.rect(xa(0), yw(1) - 4, X1 - xa(0), Y0 - yw(1) + 4, fill=F2, stroke='none', sw=0, rx=0)
f.text((xa(0) + X1) / 2, yw(1) - 18, 'reward-flat orthant', 9, i=1, anchor='middle')
f.text((xa(0) + X1) / 2, yw(1) + 18, 'every a ≥ 0', 9, anchor='middle'); f.text((xa(0) + X1) / 2, yw(1) + 30, '⇒ w = 1', 9, anchor='middle')
f.line(X0, Y0, X1, Y0); f.line(X0, Y0, X0, Y1 - 8)
for a, l in [(-1, '−1'), (0, '0'), (1, '+1')]:
    f.line(xa(a), Y0, xa(a), Y0 + 3); f.text(xa(a), Y0 + 13, l, 9, anchor='middle')
for w_, l in [(0, '0'), (0.5, '0.5'), (1, '1')]:
    f.line(X0 - 3, yw(w_), X0, yw(w_)); f.text(X0 - 5, yw(w_) + 3, l, 9, anchor='end')
f.path(f'M {xa(-1)} {yw(0)} L {xa(0)} {yw(1)} L {xa(1)} {yw(1)}', sw=1.6)
f.text(xa(-0.45) + 5, yw(0.45) + 4, 'w = 1 + a', 9, i=1)
f.text((X0 + X1) / 2, Y0 + 27, 'action a_{j}', 9.5, i=1, anchor='middle')
f.text(14, (Y0 + Y1) / 2, 'trust w_{j}', 9.5, i=1, anchor='middle', rot=-90)
# consequences panel
px, pw = 204, 244
items = [('Consequences for reward design', S(b=1, size=10)),
         ('Control cost must be connectivity loss, not |a|.', S(b=1, align='start', gap=5)),
         ('L = (1/2n) Σ (1 − w_{j})    (12)', S(align='start')),
         ('Penalizing the action norm taxes a behaviorally inert half of the space while under-pricing genuine '
          'isolation.', S(align='start')),
         ('The flat orthant has zero reward gradient.', S(b=1, align='start', gap=6)),
         ('Only the smoothness term acts there, and it penalises movement — a ratchet. Measured drift: 50–63% of '
          'action components parked at a ≥ 0 after 5M steps.', S(align='start')),
         ('Remedy: a small quadratic tie-break on a ≥ 0', S(b=1, align='start', gap=6)),
         ('(w_{flat} = 0.05). Its derivative vanishes at the origin, so it breaks the degeneracy without shifting '
          'the optimum.', S(align='start'))]
h = f.box(px, 6, pw, items, fill=PANEL, stroke=LGREY, pad=7, padx=8, valign='top')
y = max(6 + h, Y0 + 34) + 10
f.box(6, y, 442, [('Diagnostic note:', S(b=1, size=9.5, align='start')),
                  ('reporting max|a| as a false-trigger metric scores a policy at a = +0.87 — the published '
                   'controller — as 0.87. Connectivity loss is sign-resolved and does not.', S(align='start'))],
      fill=F3, stroke=LGREY, pad=6, padx=9)
f.save('Fig04_trust_map', OUT)

# ---------------- Fig 6 action spaces
f = Fig('Fig. 6 - Three candidate action spaces')
cols = [
    ('A1 — residual on dξ/dt', 'u added to the consensus derivative  (13)',
     [('Expressive?  NO', S(b=1, align='start')),
      ('At equilibrium k_{p}(ω−ω_{nom})~=~−c_{f}~cons_{ξ}~+~u, so ω~=~ω_{nom} only if c_{f}~cons_{ξ}~=~u. '
       'A persistent residual holds frequency off nominal permanently. Measured: action 0.06 → 0.0029 Hz offset, '
       '≈10,600 reward lost.', S(align='start'))], 'REJECTED', False),
    ('A2 — offset on received signal', 'ξ_{rx} ← ξ_{rx} + u',
     [('Expressive?  YES (oracle cancels attack)', S(b=1, align='start')), ('Affordable?  NO', S(b=1, align='start')),
      ('c_{f} amplifies the action 10×; action 0.005 already costs 7,500 reward. Useful precision ≈1/300 of range '
       'against exploration σ = 0.368 — roughly 70× too coarse to sample.', S(align='start'))], 'REJECTED', False),
    ('A3 — per-source trust weight', 'w_{j} = clip(1 + a_{j}, 0, 1)  (9)',
     [('Expressive?  YES (96% recovered)', S(b=1, align='start')), ('Affordable?  YES (≈100× cheaper)', S(b=1, align='start')),
      ('At consensus the neighbors already agree, so down-weighting a source costs almost nothing when nothing is '
       'wrong. Nominal, σ = 0.368:  +1,306 vs −44,883.', S(align='start'))], 'ADOPTED', True)]
cw, gx = 144, 5
tw = cw - 14
hh = max(f.block_h([(c[0], S(b=1, size=10)), (c[1], I)], tw) for c in cols)
hb = max(f.block_h(c[2], tw, gap=3) for c in cols)
H = 8 + hh + 8 + hb + 12 + 20 + 8
for k, (t, sub, body, verdict, ok) in enumerate(cols):
    x = 6 + k * (cw + gx)
    f.rect(x, 6, cw, H, fill='#f0f0f0' if ok else PANEL, stroke=INK if ok else LGREY, sw=1.4 if ok else 0.8)
    f.block(x + cw / 2, 14, [(t, S(b=1, size=10)), (sub, I)], tw)
    f.line(x + 8, 14 + hh + 4, x + cw - 8, 14 + hh + 4, c=LGREY, sw=0.6)
    f.block(x + cw / 2, 14 + hh + 10, body, tw, gap=3)
    f.box(x + cw / 2 - 40, 6 + H - 28, 80, [(verdict, S(b=1, size=9.5, c=INK if ok else RED))], h=20,
          fill=F3, stroke=INK if ok else RED, sw=1.2)
f.box(6, 6 + H + 8, 442, [('Both rejected spaces were trained to convergence before the failure was diagnosed; '
                           'neither failure was visible in the training curves.', S(i=1, align='start'))],
      fill=F3, stroke=LGREY, pad=6, padx=9)
f.save('Fig06_action_spaces', OUT)

# ---------------- Fig 7 safety projection
f = Fig('Fig. 7 - Safety projection')
h1 = 62
f.box(6, 6, 88, [('Raw policy output', S(b=1, size=9.5)), ('a ∈ [−1,1]^{2n}', N)], h=h1)
f.line(94, 6 + h1 / 2, 112, 6 + h1 / 2, arrow=True)
f.box(112, 6, 112, [('Step 1 — rate limit', S(b=1, size=9.5)), ('clip to a_{t−1} ± 0.25 per control step', N)], h=h1)
f.line(224, 6 + h1 / 2, 242, 6 + h1 / 2, arrow=True)
f.box(242, 6, 206, [('Step 2 — connectivity floor   (14)', S(b=1, size=9.5)),
                    ('count trusted channels per block: w_{j} ≥ τ = 0.5; if count < k_{min} = 3, promote the '
                     'least-distrusted channels to exactly τ (minimal intervention)', N)],
      h=h1, fill=F2, sw=1.4)
y = 6 + h1 + 14
# table panel
tx, tw_ = 6, 226
rows = [('a = 0  (nominal)', '10 of 10', '0.000'), ('a = −1 everywhere', '3 of 10', '0.850'),
        ('graded distrust', '3 of 10', '0.750'), ('a = +0.87 (flat orthant)', '10 of 10', '0.000')]
right = [('What was removed, and why', S(b=1, size=10)),
         ('A directional constraint inherited from the offset action space forbade actions pushing a variable '
          'further outside its envelope.', S(align='start', gap=6)),
         ('Under trust semantics the sign of an action means TRUST, not direction. That constraint therefore '
          'forced full trust in a possibly compromised source at precisely the moment local voltage left its '
          'envelope.', S(align='start', gap=6)),
         ('It was removed, not repaired: there is no admissible-direction test on a trust weight without knowing '
          'which source is anomalous — the detector\'s job.', S(align='start', gap=6))]
rx_, rw = 240, 208
hr = f.block_h(right, rw - 16) + 14
ht = 22 + 18 + 4 * 19 + 34
H = max(hr, ht)
f.rect(tx, y, tw_, H, fill=PANEL, stroke=LGREY)
f.text(tx + tw_ / 2, y + 15, 'Behaviour under commanded isolation', 10, b=1, anchor='middle')
c1, c2, c3 = tx + 8, tx + 150, tx + tw_ - 8
yy = y + 24
f.line(tx + 6, yy, tx + tw_ - 6, yy, c=LGREY, sw=0.6)
f.text(c1, yy + 12, 'commanded a', 9, b=1); f.text(c2, yy + 12, 'trusted after', 9, b=1, anchor='middle')
f.text(c3, yy + 12, 'mean loss', 9, b=1, anchor='end')
f.text(c2, yy + 22, 'projection', 9, b=1, anchor='middle')
yy += 27; f.line(tx + 6, yy, tx + tw_ - 6, yy, c=LGREY, sw=0.6)
for r in rows:
    yy += 15.5
    f.text(c1, yy, r[0], 9); f.text(c2, yy, r[1], 9, anchor='middle'); f.text(c3, yy, r[2], 9, anchor='end')
f.para(tx + tw_ / 2, yy + 9, 'the floor binds only under commanded isolation; at convergence 8–9 of 10 sources '
       'remain trusted', tw_ - 16, 9, i=1, anchor='middle')
f.box(rx_, y, rw, right, h=H, fill=PANEL, stroke=LGREY, pad=7, padx=8, valign='top')
f.save('Fig07_safety_projection', OUT)

# ---------------- Fig 8 admissibility tests
f = Fig('Fig. 8 - Pre-training admissibility tests')
h0 = f.box(167, 6, 120, [('Candidate action space', S(b=1, size=9.5)), ('+ observation design', N)], pad=5)
y = 6 + h0 + 16
tests = [('Test 1 — Expressiveness', 'Can an ORACLE action cancel the attack?',
          'oracle isolation recovers 96% of the damage', 'PASS', INK),
         ('Test 2 — Affordability', 'Can the policy afford to explore?',
          'σ = 0.368 costs ≈470 on nominal (offset: 46,000)', 'PASS', INK),
         ('Test 3 — Observability', 'Is the required evidence in the observation?',
          'source ID vs permutation null + trivial predictor', 'PARTIAL', RED)]
cw, g = 142, 8
hb = max(f.block_h([(t[0], S(b=1, size=10)), (t[1], N), (t[2], I)], cw - 14, gap=4) for t in tests)
HT = hb + 14 + 26
for k, t in enumerate(tests):
    x = 6 + k * (cw + g)
    f.rect(x, y, cw, HT, fill=PANEL, stroke=INK, sw=1.0)
    f.block(x + cw / 2, y + 7, [(t[0], S(b=1, size=10)), (t[1], N), (t[2], I)], cw - 14, gap=4)
    f.box(x + cw / 2 - 32, y + HT - 26, 64, [(t[3], S(b=1, size=9.5, c=t[4]))], h=19, fill=F3, stroke=t[4], sw=1.1)
    f.line(x + cw / 2, y + HT, x + cw / 2, y + HT + 7)
f.path(f'M 227 {6 + h0} L 227 {y - 6} L 77 {y - 6} L 77 {y}', end=(77, y, 77, y - 6))
f.line(227, y - 6, 227, y, arrow=True); f.line(227, y - 6, 377, y - 6); f.line(377, y - 6, 377, y, arrow=True)
y2 = y + HT + 20
f.line(77, y + HT + 7, 377, y + HT + 7); f.line(227, y + HT + 7, 227, y2, arrow=True)
h2 = f.box(97, y2, 260, [('1 and 2 pass, 3 partial ⇒ commit compute', S(b=1, size=9.5)),
                         ('cost: minutes.   training cost: ≈48 min per seed', N)], fill=F2, pad=5)
y3 = y2 + h2 + 12
f.box(6, y3, 442, [('Why these belong before training, not after', S(b=1, size=10)),
                   ('Each of the three failure modes produces a training curve indistinguishable from healthy '
                    'learning. In this work two action spaces were trained to convergence and evaluated before either '
                    'failure was diagnosed. Test 3 additionally revealed that the observation, built from true internal '
                    'states, omitted the received consensus values entirely — so the policy was being asked to assign '
                    'per-source trust with no per-source evidence. Adding the innovation of (8) raised SlowRamp source '
                    'identification from 0.456 to 0.892, and left Coordinated Stealth and denial of service at their '
                    'reference levels, which bounds the achievable scope.', S(align='start'))],
      fill=PANEL, stroke=LGREY, pad=7, padx=9, gap=4)
f.para(227, f.H + 6, 'Test 3 returns PARTIAL by design: four of six classes are identifiable, and the two that are not '
       '(denial of service, Coordinated Stealth) are the attacks behaving as specified. The test bounds the claim '
       'rather than blocking the run.', 440, 8.5, anchor='middle')
f.save('Fig08_admissibility_tests', OUT)
