"""Recover plotted data from the original matplotlib SVGs (exact to the 6-decimal coordinates written)."""
import re, json
from xml.etree import ElementTree as ET
ns = '{http://www.w3.org/2000/svg}'; XL = '{http://www.w3.org/1999/xlink}href'

def load(n):
    root = ET.parse(f'orig/{n}.svg').getroot()
    el = {}
    def walk(e, gid):
        if e.tag == ns + 'g' and e.get('id'): gid = e.get('id')
        if e.tag in (ns + 'path', ns + 'use'):
            el.setdefault(gid, []).append(e)
        for c in e: walk(c, gid)
    walk(root, None)
    return el

def pts(d):
    return [(float(a), float(b)) for a, b in re.findall(r'[ML] ?([-\d.]+) ([-\d.]+)', d)]

def color(e):
    m = re.search(r'stroke: (#\w+)', e.get('style', '')); return m.group(1) if m else None

out = {}
# ---------- Fig 12
el = load('Fig12_observability')
y0, y1 = 189.92, 23.068148                       # 0.0 and 1.0 tick positions
v = lambda y: (y0 - y) / (y0 - y1)
bars = [pts(el[f'patch_{k}'][0].get('d')) for k in range(3, 15)]
top = [v(min(p[1] for p in b)) for b in bars]
ref = [v(pts(el[f'line2d_{k}'][0].get('d'))[0][1]) for k in range(13, 19)]
out['fig12'] = dict(classes=['Additive', 'Ramp', 'SlowRamp', 'Sinusoid', 'Stealth', 'DoS'],
                    obs=top[:6], inn=top[6:], ref=ref)

# ---------- Fig 13
el = load('Fig13_paired_returns')
x0, xs = 147.602834, (226.97338 - 147.602834) / 20000
X = lambda x: (x - x0) / xs
rows_y = [26.75419, 33.211519, 39.668848]       # first class lines
classes = ['Ramp', 'Additive', 'Sinusoid', 'Stealth', 'SlowRamp', 'DoS', 'Normal']
centers = [36.440855 - 3.2296, 63.346393 - 3.2296]
pitch = 26.905538
pts13, bars13, means13 = [], [], []
for c in range(7):
    for s in range(3):
        k = c * 3 + s + 1
        uses = el[f'PathCollection_{k}']
        px = [X(float(u.get('x'))) for u in uses if u.tag == ns + 'use']
        py = [float(u.get('y')) for u in uses if u.tag == ns + 'use']
        col = color([u for u in uses if u.tag == ns + 'use'][0])
        ln = pts(el[f'line2d_{20 + k}'][0].get('d'))
        mu = [u for u in el[f'line2d_{41 + k}'] if u.tag == ns + 'use'][0]
        yc = ln[0][1]
        pts13.append(dict(cls=classes[c], seed=s, x=px, dy=[y - yc for y in py], color=col,
                          lo=X(ln[0][0]), hi=X(ln[1][0]), mean=X(float(mu.get('x'))), yrow=yc))
out['fig13'] = dict(classes=classes, series=pts13, pitch=pitch)

# ---------- Fig 15
el = load('Fig15_structural_cost')
lx0, lxs = 80.43625, (286.26242 - 80.43625) / 0.14
L = lambda x: 0.14 + (x - lx0) / lxs
left = []
for k, lab in zip(range(22, 28), ['seed 0', 'seed 1', 'seed 2', 'p_normal = 0.5', 'w_action = 3', 'w_action = 10']):
    u = [e for e in el[f'line2d_{k}'] if e.tag == ns + 'use'][0]
    left.append((lab, L(float(u.get('x')))))
band = pts(el['patch_3'][0].get('d'))
band = (L(min(p[0] for p in band)), L(max(p[0] for p in band)))
rx0, rxs = 354.146758, (457.863649 - 354.146758) / 0.10
R = lambda x: 0.15 + (x - rx0) / rxs
off = 8.5 * 0.37993
ry = lambda y: 25 + (179.996884 - off - y) / ((179.996884 - 46.812537) / 150)
right = {}
for k, lab in zip(range(28, 32), ['Normal', 'DoS', 'Stealth', 'Sinusoid']):
    p = pts(el[f'line2d_{k}'][0].get('d'))
    right[lab] = [(R(a), ry(b)) for a, b in p]
out['fig15'] = dict(left=left, band=band, right=right)

# ---------- Fig 16
el = load('Fig16_benchmarks')
ya0 = 178.112; ya1 = 42.967011 - 9 * 0.37993
A = lambda y: (ya0 - y) / (ya0 - ya1)
bin_ = [A(min(p[1] for p in pts(el[f'patch_{k}'][0].get('d')))) for k in (3, 4, 5)]
mul = [A(min(p[1] for p in pts(el[f'patch_{k}'][0].get('d')))) for k in (6, 7, 8)]
lc = [pts(e.get('d')) for e in el['LineCollection_1']]
mul_err = [(A(p[0][1]), A(p[1][1])) for p in lc]
yb1 = 33.729392 - 9 * 0.37993
B = lambda y: (ya0 - y) / (ya0 - yb1)
cov = [B(min(p[1] for p in pts(el[f'patch_{k}'][0].get('d')))) for k in range(15, 20)]
lc2 = [pts(e.get('d')) for e in el['LineCollection_2']]
cov_err = [(B(p[0][1]), B(p[1][1])) for p in lc2]
base = B(pts(el['line2d_25'][0].get('d'))[0][1])
out['fig16'] = dict(protocols=['random', 'block', 'run-level'], binary=bin_, multi=mul, multi_err=mul_err,
                    gens=['template', 'Llama 3\n8B', 'Llama 3.1\n8B', 'Llama 3.2\n3B', 'Mistral\nNeMo'],
                    cov=cov, cov_err=cov_err, baseline=base)
json.dump(out, open('data.json', 'w'), indent=1)

r = lambda l: [round(a, 4) for a in l]
print('12 obs', r(out['fig12']['obs'])); print('12 inn', r(out['fig12']['inn'])); print('12 ref', r(out['fig12']['ref']))
for s in pts13: print('13', s['cls'], s['seed'], len(s['x']), round(s['mean']), round(sum(s['x']) / 20), round(s['lo']), round(s['hi']), s['color'])
print('15 left', [(a, round(b, 4)) for a, b in left], 'band', r(band))
for k, v_ in right.items(): print('15', k, [(round(a, 4), round(b, 1)) for a, b in v_])
print('16 bin', r(bin_), 'mul', r(mul), [r(e) for e in mul_err])
print('16 cov', r(cov), [r(e) for e in cov_err], 'base', round(base, 4))
