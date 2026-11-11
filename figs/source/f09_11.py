from fl import *
OUT = 'out'
N = dict(size=9)
def S(**k): d = dict(size=9); d.update(k); return d
H9 = S(b=1, size=9.5)

# ---------------- Fig 9 framework
f = Fig('Fig. 9 - Trust-weighted control framework')
cw, gap = 134, 20
xs = [6 + k * (cw + gap) for k in range(3)]       # 6, 160, 314  -> 448
top = 22
heads = ['Plant (10 ms sampling)', 'Learned supervisory policy', 'Constraint layer']
cols = [
    [([('Microgrid', H9), ('10 DG, ring topology, droop primary', N)], F1, 0.8),
     ([('Measurement', H9), ('V, f, P, Q per DG', N)], F1, 0.8),
     ([('Consensus signals', H9), ('transmitted ξ, ζ', N), ('received ξ_{rx}, ζ_{rx}', N)], F1, 0.8),
     ([('Consensus innovation   (8)', H9), ('inn_{j} = (Σ~A_{ij}~x_{i} − d_{j}~x_{rx,j})/d_{j}', N)], F2, 0.8)],
    [([('Observation stack', H9), ('4 frames × 8 quantities × 10 DG = 320 elements', N)], F1, 0.8),
     ([('Normalisation', H9), ('running mean / variance', N)], F1, 0.8),
     ([('Policy π(a|s)', H9), ('256–256 MLP, Gaussian', N)], F1, 0.8),
     ([('Trust offsets', H9), ('a ∈ [−1,1]^{2n},  n = 10', N)], F2, 0.8)],
    [([('Rate limit', H9), ('|a_{t} − a_{t−1}| ≤ 0.25', N)], F1, 0.8),
     ([('Connectivity floor   (14)', H9), ('≥ 3 sources with w ≥ 0.5, per channel, minimal promotion', N)], F2, 1.4),
     ([('Trust map   (9)', H9), ('w_{j} = clip(1 + a_{j}, 0, 1)', N)], F1, 0.8),
     ([('Weighted consensus   (10)', H9), ('cons = (ΣA w)x − ΣA w x_{rx}', N)], F2, 0.8)]]
notes = [[('the only per-source evidence of which source is lying', S(i=1, c=GREY)),
          ('→ see admissibility Test 3', S(i=1))],
         [('a = 0 ⇒ nominal controller exactly recovered', S(i=1))],
         [('hard guarantee, not learned: the policy cannot dismantle the layer it protects', S(i=1))]]
iw = cw - 12
# common row heights
rowh = [max(f.block_h(cols[c][r][0], iw - 10) + 10 for c in range(3)) for r in range(4)]
pos = {}
for c in range(3):
    x = xs[c]
    y = top + 7
    for r in range(4):
        items, fill, sw = cols[c][r]
        f.box(x + 6, y, iw, items, h=rowh[r], fill=fill, sw=sw, padx=5)
        pos[(c, r)] = (x + 6, y, iw, rowh[r]); y += rowh[r] + 7
    pos[('n', c)] = y
nh = max(f.block_h(notes[c], iw) for c in range(3))
ph = (pos[('n', 0)] - top) + nh + 8
for c in range(3):
    f.rect(xs[c], top, cw, ph, fill='none', stroke=LGREY, dash='3,3')
    f.text(xs[c] + cw / 2, top - 6, heads[c], 10, b=1, anchor='middle')
    f.block(xs[c] + cw / 2, pos[('n', c)] + 1, notes[c], iw, gap=3)
# move panels behind: draw again fill — simpler: panels drawn with fill none
# arrows plant -> observation stack
ox, oy, ow, oh = pos[(1, 0)]
for r in (1, 2, 3):
    x, y, w, h = pos[(0, r)]
    f.line(x + w, y + h / 2, ox, oy + oh * (0.35 + 0.15 * r), arrow=True, head=3.8)
# trust offsets -> rate limit
x, y, w, h = pos[(1, 3)]; rx, ry, rw, rh = pos[(2, 0)]
f.line(x + w, y + h / 2, rx, ry + rh / 2, arrow=True, head=3.8)
# closed loop from weighted consensus back to plant column
x, y, w, h = pos[(2, 3)]
yb = top + ph + 16
px, py, pw, phh = pos[(0, 2)]
f.path(f'M {x + w} {y + h / 2} L {x + w + 9} {y + h / 2} L {x + w + 9} {yb} L {xs[0] + cw / 2} {yb} L {xs[0] + cw / 2} {top + ph + 1.5}',
       sw=1.1, end=(xs[0] + cw / 2, top + ph + 1.5, xs[0] + cw / 2, yb))
f.text(300, yb + 13, 'weighted consensus returned to the secondary integrators — closed loop', 9, i=1, anchor='middle')
# bottom row
yr = yb + 24
ha = f.box(6, yr, 134, [('Adversary', S(b=1, size=10, c=RED)), ('corrupts transmitted ξ, ζ', S(c=RED)),
                        ('before they are received', S(i=1, c=RED))], fill=F3, stroke=RED, sw=1.1)
hd = f.box(160, yr, 288, [('Design invariant:', S(b=1, size=9.5, align='start')),
                          ('the published consensus controller is never removed from the loop. The learned layer only '
                           're-weights its inputs, so zero action reproduces the baseline controller bit-for-bit.',
                           S(align='start'))], fill=F3, stroke=LGREY, padx=8)
# adversary -> consensus signals (dashed red, along left margin)
cx_, cy_, cw_, ch_ = pos[(0, 2)]
f.path(f'M 6 {yr + ha / 2} L 2.5 {yr + ha / 2} L 2.5 {cy_ + ch_ / 2} L {cx_} {cy_ + ch_ / 2}', c=RED, sw=1.0,
       dash='4,3', end=(cx_, cy_ + ch_ / 2, 2.5, cy_ + ch_ / 2), head=3.8)
f.save('Fig09_framework', OUT)

# ---------------- Fig 10 explanation pipeline
f = Fig('Fig. 10 - Explanation and audit pipeline')
y = 6; h = 74
f.box(6, y, 86, [('Policy decision', H9), ('state s, action a, at operator request', N)], h=h)
f.line(92, y + h / 2, 104, y + h / 2, arrow=True, head=3.6)
f.box(104, y, 108, [('Integrated Gradients', H9), ('64 path steps, midpoint rule; target: total trust reduction; '
                                                 'completeness asserted', N)], h=h)
f.line(212, y + h / 2, 224, y + h / 2, arrow=True, head=3.6)
f.box(224, y, 100, [('Grounded prompt', H9), ('every number system-supplied; limit verdicts precomputed', N)],
      h=h, fill=F2)
f.line(324, y + h / 2, 336, y + h / 2 - 17, arrow=True, head=3.6)
f.line(324, y + h / 2, 336, y + h / 2 + 17, arrow=True, head=3.6)
f.box(336, y, 112, [('Language model', H9), ('asynchronous, outside loop', N)], h=34)
f.box(336, y + 40, 112, [('Template baseline', H9), ('deterministic', N)], h=34)
y2 = y + h + 16
f.line(392, y + h, 392, y2, arrow=True)
ha = f.box(228, y2, 220, [('Automated audit', H9), ('numeric consistency vs supplied values', N),
                         ('relational check on stated comparisons', N), ('weighted coverage of operational facts', N),
                         ('chance floor from random decoys: 5.6%', N)], fill=F2, sw=1.3, gap=1)
f.line(228, y2 + ha / 2, 204, y2 + ha / 2, arrow=True)
f.box(84, y2 + ha / 2 - 30, 120, [('Operator display', H9), ('explanation + audit flags', N), ('no actuation path', N)],
      h=60)
y3 = y2 + ha + 12
bl = [('Failure modes the numeric audit cannot detect (all observed)', S(b=1, size=10)),
      ('•  False comparison: "50.0034 Hz … is above the upper limit of 50.05 Hz" — both figures correct and supplied, '
       'the relation false.', S(align='start', gap=4)),
      ('•  Sign inversion: "attributed a significant positive value (−0.2078)" — self-contradictory within one clause.',
       S(align='start')),
      ('•  Causal inversion: the agent\'s own trust output cited as a measurement that drove the action.', S(align='start')),
      ('•  Category error: an attribution value added to a voltage as though it were a control delta.', S(align='start')),
      ('Numeric grounding is necessary but not sufficient; a human-subject study would be required to claim operator '
       'benefit.', S(b=1, align='start', gap=5))]
f.box(6, y3, 442, bl, fill=PANEL, stroke=LGREY, pad=7, padx=9, gap=2.5)
f.save('Fig10_explanation_pipeline', OUT)

# ---------------- Fig 11 curriculum
f = Fig('Fig. 11 - Training curriculum and evaluation protocol')
f.text(227, 15, 'Curriculum (running normalization statistics carried across stages)', 10, b=1, anchor='middle')
x0, tot = 6, 442
st = [(0.15, 'Stage 1', ['nominal only', 'p_{normal} = 1.0', '750k steps'], '15%'),
      (0.35, 'Stage 2', ['overt attacks', 'Additive, Ramp,', 'Sinusoid, DoS', '1.75M steps'], '35%'),
      (0.50, 'Stage 3', ['full mixture', '+ SlowRamp, Stealth', 'p_{normal} = 1/7', '2.5M steps'], '50%')]
x = x0; y = 22; h = 78
for frac, t, ls, pc in st:
    w = frac * tot
    f.box(x, y, w, [(t, S(b=1, size=10))] + [(l, S(gap=1)) for l in ls], h=h, sw=1.0, padx=3)
    f.text(x + w / 2, y + h + 13, pc, 9, i=1, anchor='middle')
    x += w
ya = y + h + 22
f.line(x0, ya, x0 + tot, ya, arrow=True)
f.text(227, ya + 14, '5,000,000 environment steps  ·  8 parallel workers  ·  ≈48 min per seed', 9, i=1, anchor='middle')
yp = ya + 26
items = [[('Nominal controller arm', H9), ('a = 0 throughout', N), ('20 seeds × 7 classes', N)],
         [('Trained agent arm', H9), ('identical seeds,', N), ('identical injectors', N)],
         [('Paired per-seed differences', H9), ('mean, std, t statistic,', N), ('seeds improved / tied', N)]]
bw = [124, 124, 140]; bx = [16, 160, 298]
bh = max(f.block_h(it, w - 12) for it, w in zip(items, bw)) + 14
f.rect(6, yp, 442, bh + 38, fill=PANEL, stroke=LGREY)
f.text(227, yp + 15, 'Evaluation protocol', 10, b=1, anchor='middle')
for k in range(3):
    f.box(bx[k], yp + 24, bw[k], items[k], h=bh, fill=F2 if k == 2 else F1)
f.line(140, yp + 24 + bh / 2, 160, yp + 24 + bh / 2, c=GREY)
f.text(150, yp + 24 + bh / 2 - 4, '=', 11, b=1, anchor='middle')
f.line(284, yp + 24 + bh / 2, 298, yp + 24 + bh / 2, arrow=True)
f.save('Fig11_curriculum_protocol', OUT)
