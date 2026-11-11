from fl import *
OUT = 'out'
B = dict(b=1, size=9.5); N = dict(size=9); I = dict(i=1, size=9)

f = Fig('Fig. 2 - Attack injection point in the consensus loop')
y0 = 6
# row 1
h1 = 54
f.box(6, y0, 106, [('DG j — local state', B), ('true ξ_{j}, ζ_{j}', N), ('(uncompromised)', N)], h=h1)
f.line(112, y0 + h1/2, 150, y0 + h1/2, arrow=True)
f.text(131, y0 + h1/2 - 4, 'transmit', 8.5, i=1, anchor='middle')
f.box(150, y0, 128, [('Communication link', B), ('ξ_{rx,j} = ξ_{j} + δ_{j}(t)', N), ('ATTACK SURFACE', dict(size=9, c=RED, b=1))],
      h=h1, fill=F3, stroke=RED, sw=1.3)
f.line(278, y0 + h1/2, 306, y0 + h1/2, c=RED, sw=1.1, arrow=True)
f.text(292, y0 + h1/2 - 4, 'receive', 8.5, i=1, c=RED, anchor='middle')
f.box(306, y0, 142, [('Every neighbor i of j', B), ('cons_{ξ,i} = d_{i} ξ_{i} − Σ A_{ij} ξ_{rx,j}', N),
                     ('corruption enters ALL integrators', N)], h=h1, fill=F2, sw=1.0)
# row 2: delta
y2 = y0 + h1 + 22
f.box(160, y2, 108, [('δ_{j}(t) — six classes', dict(b=1, size=9.5, c=RED)), ('Table 3', dict(i=1, size=9, c=RED))],
      h=36, fill=F3, stroke=RED, sw=1.0)
f.line(214, y2, 214, y0 + h1 + 1, c=RED, sw=1.0, arrow=True)
# row 3: trust weight + why box
y3 = y2 + 36 + 22
h3 = f.box(160, y3, 108, [('Trust weight w_{j}', B), ('applied HERE', N), ('before (5) is evaluated', N)],
           fill=F2, sw=1.5, pad=7)
f.line(214, y3, 214, y2 + 37, sw=1.0, arrow=True)
why = [('Why standing distrust is required', dict(b=1, size=9.5)),
       ('The weight acts BEFORE the consensus update. A source trusted at onset has already injected a '
        'corrupted value into every neighbor\'s integrator; no later reduction in trust retracts it.', N)]
hw = f.block_h(why, 160 - 12) + 12
yw = y2 + 6
f.box(288, yw, 160, why, fill=PANEL, stroke=LGREY, sw=0.8, pad=6)
f.line(288, y3 + h3/2, 269, y3 + h3/2, c=GREY, sw=0.8, arrow=True)
# consequence
yc = max(y3 + h3, yw + hw) + 14
cons = [('Consequence (Section 8.3):', dict(b=1, size=9.5, align='start')),
        ('the converged policy holds 2.7–4.7 sources below full trust during healthy operation, '
         'invariant to attack prevalence and to the price charged for it.', dict(size=9, align='start'))]
hc = f.box(6, yc, 442, cons, fill=F3, stroke=LGREY, pad=6, padx=9, anchor='start')
# modeling note
yn = yc + hc + 10
y = f.para(227, yn, 'Modeling note. This is the communication-path formulation of Section III-C of the source study, '
           'and it is this paper\'s threat model.', 440, 8.5, anchor='middle')
f.para(227, y + 1, 'The public release\'s own differential is node-local (the attacked DG sheds 36.60 W while the other '
       'nine gain 35.19 W in near-equal shares, and its ring neighbors are not preferentially disturbed), so the two '
       'formulations differ. Stated, not elided.', 440, 8.5, c=RED, anchor='middle')
f.save('Fig02_injection_point', OUT)
