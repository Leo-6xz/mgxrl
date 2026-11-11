from fl import *
import math
OUT = 'out'
N = dict(size=9)
def S(**k): d = dict(size=9); d.update(k); return d
H9 = S(b=1, size=9.5)

f = Fig('Fig. 1 - Microgrid and control architecture')
CX, CY, RE, RC, RN = 118, 188, 88, 57, 15
ang = [math.radians(-90 + 36 * k) for k in range(10)]
E = [(CX + RE * math.cos(a), CY + RE * math.sin(a)) for a in ang]
C = [(CX + RC * math.cos(a), CY + RC * math.sin(a)) for a in ang]
for k in range(10):   # electrical ring
    f.line(*E[k], *E[(k + 1) % 10], sw=1.8)
for k in range(10):   # communication ring
    f.line(*C[k], *C[(k + 1) % 10], c=GREY, sw=1.0, dash='3.5,2.5')
for k in range(10):
    f.line(*E[k], *C[k], c=LGREY, sw=0.7, dash='1.5,2')
    f.circle(*C[k], 2.6, fill='#ffffff', stroke=GREY, sw=0.8)
for k in range(10):
    x, y = E[k]
    f.circle(x, y, RN, fill=F1, stroke=INK, sw=1.1)
    f.text(x, y + 1, f'DG{k + 1}', 8.5, b=1, anchor='middle')
    f.text(x, y + 9.5, 'VSC', 7.5, c=GREY, anchor='middle')
f.text(CX, CY - 22, 'ξ, ζ  exchanged', 9, c=GREY, anchor='middle')
f.text(CX, CY - 11, 'consensus signals', 9, i=1, c=GREY, anchor='middle')
f.text(CX, CY + 9, 'Communication graph', 9, c=GREY, anchor='middle')
f.text(CX, CY + 20, '(ring, adjacency A)', 9, i=1, c=GREY, anchor='middle')
# adversary
f.box(4, 4, 150, [('Adversary', S(b=1, size=10, c=RED)), ('corrupts the TRANSMITTED consensus value  ξ → ξ_{rx}', S(c=RED)),
                  ('propagates via (5) to all neighbors', S(i=1, size=8.5, c=RED))], fill=F3, stroke=RED, sw=1.1, pad=4, padx=4, gap=1)
tx, ty = C[9]
f.path(f'M 24 62 C 22 {ty - 30}, {tx - 30} {ty - 4}, {tx - 3} {ty - 1}', c=RED, sw=1.4,
       end=(tx - 3, ty - 1, tx - 15, ty - 3))
# electrical line callout
ex, ey = (E[6][0] + E[7][0]) / 2, (E[6][1] + E[7][1]) / 2
f.box(4, CY + 104, 92, [('Electrical line', N), ('r = 0.05, x = 0.15 pu', S(i=1))], fill=F3, stroke=LGREY, sw=0.7, pad=3, gap=0.5)
f.line(30, CY + 104, ex - 2, ey + 2, c=LGREY, sw=0.7)

# right column
RX, RW = 244, 204
BX, BW = RX + 8, RW - 16
y = 16
f.text(RX + RW / 2, y - 4, 'Control hierarchy at each DG', 10, b=1, anchor='middle')
p1 = y; y += 8
bx = [[('Primary — droop control', H9), ('ω = ω_{nom} − m(P − P_{set}) + ξ    (1)', N), ('V = V_{nom} − n(Q − Q_{set}) + ζ    (2)', N)],
      [('Secondary — distributed consensus', H9), ('dξ/dt = −k_{p}(ω − ω_{nom}) − c_{f}·cons_{ξ}   (6)', N), ('cons_{ξ} = d ξ − Σ A ξ_{rx}    (5)', N)],
      [('PROPOSED — trust weighting layer', H9), ('w_{j} = clip(1 + a_{j}, 0, 1)    (9)', N), ('acts between transmit and receive', N)]]
for k, it in enumerate(bx):
    h = f.box(BX, y, BW, it, fill=F2 if k == 2 else F1, sw=1.4 if k == 2 else 0.8, pad=4, padx=4, gap=1)
    y += h
    if k < 2: f.line(RX + RW / 2, y, RX + RW / 2, y + 10, arrow=True, head=3.6); y += 10
y += 4
y = f.block(RX + RW / 2, y, [('a = 0 ⇒ w = 1 ⇒ reduces exactly to (5)', S(i=1)),
                              ('nominal-case stability argument', S(i=1, c=GREY))], BW, gap=1) + 5
f.rect(RX, p1, RW, y - p1, fill='none', stroke=LGREY, dash='3,3')
y += 18
f.text(RX + RW / 2, y - 5, 'Supervisory layer (10 ms)', 10, b=1, anchor='middle')
p2 = y; y += 8
sb = [[('Observation', H9), ('V, f, P, Q, ξ, ζ + consensus innovation (8)', N)],
      [('RL policy  π(a | s)', H9), ('256–256 MLP → a ∈ [−1,1]^{2n}', N)],
      [('Safety projection    (14)', H9), ('connectivity floor k_{min} = 3, rate limit', N)]]
pos = []
for k, it in enumerate(sb):
    h = f.box(BX, y, BW, it, fill=F2 if k == 2 else F1, sw=1.3 if k == 2 else 0.8, pad=4, padx=4, gap=1)
    pos.append((y, h)); y += h
    f.line(RX + RW / 2, y, RX + RW / 2, y + 10, arrow=True, head=3.6); y += 10
hw = 30
f.box(BX, y, 80, [('Trust weights', H9), ('w ∈ [0,1]^{2n}', N)], h=hw, fill=F2, pad=3, gap=1)
f.box(BX + BW - 104, y, 104, [('Detector', H9), ('posterior (not in loop)', N)], h=hw, fill=F3, pad=3, gap=1)
twy = y + hw / 2
y += hw + 8
f.rect(RX, p2, RW, y - p2, fill='none', stroke=LGREY, dash='3,3')
# measurements: DG3 -> observation
ox, oy = BX, pos[0][0] + pos[0][1] / 2
sx_, sy_ = E[2][0] + RN, E[2][1]
f.path(f'M {sx_} {sy_} L 238 {sy_} L 238 {oy} L {ox - 1} {oy}', c=GREY, sw=1.0, dash='3,2.5',
       end=(ox - 1, oy, ox - 10, oy))
f.text(233, (sy_ + oy) / 2, 'measurements', 8.5, i=1, c=GREY, anchor='middle', rot=-90)
# trust weights -> communication graph (at DG6 comm node)
mx, my = (C[4][0] + C[5][0]) / 2, (C[4][1] + C[5][1]) / 2
f.path(f'M {BX} {twy} C 170 {twy}, {mx + 22} {my + 60}, {mx + 1.5} {my + 3}', sw=1.3, end=(mx + 1.5, my + 3, mx + 8, my + 18))
f.para(104, twy - 20, 'trust weights scale the communication graph  (10)', 120, 9, i=1, anchor='middle')
# legend
ly = y + 10
f.rect(6, ly, 442, 40, fill=F3, stroke=LGREY, sw=0.7)
f.text(14, ly + 13, 'Legend', 9.5, b=1)
r1, r2 = ly + 14, ly + 30
f.line(66, r1 - 3, 90, r1 - 3, sw=1.8); f.text(95, r1, 'electrical line', 9)
f.line(176, r1 - 3, 200, r1 - 3, c=GREY, sw=1.0, dash='3.5,2.5'); f.text(205, r1, 'communication edge', 9)
f.line(310, r1 - 3, 334, r1 - 3, c=RED, sw=1.4); f.text(339, r1, 'adversarial injection', 9, c=RED)
f.line(66, r2 - 3, 90, r2 - 3, c=LGREY, sw=0.7, dash='1.5,2'); f.text(95, r2, 'DG ↔ communication node', 9)
f.text(205, r2, 'Numbers in parentheses refer to manuscript equations.', 9, i=1, c=GREY)
f.para(227, ly + 48, 'The detector is architecturally available but is NOT part of the observation in the reported runs: '
       'the policy acts on physical measurements and the consensus innovation alone, so a detector false positive cannot '
       'propagate into a control action.', 440, 8.5, anchor='middle')
f.save('Fig01_architecture', OUT)
