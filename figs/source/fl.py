"""Tiny figure-layout helper: canvas in points (1 unit = 1 pt at 6.3 in width)."""
import re, math, subprocess, os
from PIL import ImageFont

W = 454.0                      # 6.3 in
INK, GREY, LGREY, RED = '#1a1a1a', '#5f5f5f', '#b0b0b0', '#8c1515'
F1, F2, F3, PANEL = '#f2f2f2', '#e6e6e6', '#ffffff', '#fafafa'
FAM = "Times New Roman, Liberation Serif, Times, serif"
SYMFAM = "Cambria Math, DejaVu Serif, serif"
SYMS = set('∈⇒⌊⌋⌈⌉⟹')
LD = '/usr/share/fonts/truetype/liberation/LiberationSerif-'
_fonts = {}
def _font(b, i, sym=False):
    k = (b, i, sym)
    if k not in _fonts:
        if sym:
            _fonts[k] = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 100)
        else:
            n = {(0,0):'Regular',(1,0):'Bold',(0,1):'Italic',(1,1):'BoldItalic'}[(int(b),int(i))]
            _fonts[k] = ImageFont.truetype(LD + n + '.ttf', 100)
    return _fonts[k]

SUBK, SUBDY, SUPDY = 0.72, 0.22, 0.36

def parse(s):
    s = s.replace('~', '\u00a0')
    """-> list of (text, kind) kind in '', 'sub', 'sup'. Markup: _{..} ^{..}"""
    out, i = [], 0
    for m in re.finditer(r'([_^])\{([^}]*)\}', s):
        if m.start() > i: out.append((s[i:m.start()], ''))
        out.append((m.group(2), 'sub' if m.group(1) == '_' else 'sup'))
        i = m.end()
    if i < len(s): out.append((s[i:], ''))
    return out

def _runs(txt):
    """split into symbol / non-symbol runs"""
    out, cur, cs = [], '', None
    for ch in txt:
        s = ch in SYMS
        if cs is None or s == cs: cur += ch
        else: out.append((cur, cs)); cur = ch
        cs = s
    if cur: out.append((cur, cs))
    return out

def width(s, size=9, b=False, i=False):
    w = 0
    for t, k in parse(s):
        sz = size * (SUBK if k else 1)
        for r, sym in _runs(t):
            w += _font(b, i, sym).getlength(r) * sz / 100
    return w

def esc(t):
    return t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def wrap(s, maxw, size=9, b=False, i=False):
    lines = []
    for para in s.split('\n'):
        words, cur = para.split(' '), ''
        for wd in words:
            t = wd if not cur else cur + ' ' + wd
            if width(t, size, b, i) <= maxw or not cur: cur = t
            else: lines.append(cur); cur = wd
        lines.append(cur)
    return lines

class Fig:
    def __init__(self, title, W=W):
        self.W, self.els, self.title, self.H = W, [], title, 0
    def _y(self, y): self.H = max(self.H, y)

    # ---- primitives
    def text(self, x, y, s, size=9, b=False, i=False, c=INK, anchor='start', rot=None):
        w = width(s, size, b, i)
        x0 = x - (w / 2 if anchor == 'middle' else w if anchor == 'end' else 0)
        parts, cur = [], 0.0
        for t, k in parse(s):
            sz = size * (SUBK if k else 1)
            off = {'': 0, 'sub': SUBDY * size, 'sup': -SUPDY * size}[k]
            dy = off - cur; cur = off
            for j, (r, sym) in enumerate(_runs(t)):
                a = []
                if j == 0 and abs(dy) > 1e-6: a.append(f'dy="{dy:.2f}"')
                if k: a.append(f'font-size="{sz:.2f}"')
                if sym: a.append(f'font-family="{SYMFAM}"')
                parts.append(f'<tspan {" ".join(a)}>{esc(r)}</tspan>' if a else esc(r))
        tr = f' transform="rotate({rot} {x:.2f} {y:.2f})"' if rot else ''
        if rot:  # rotated text: keep anchor semantic with x0 computed along rotated axis
            x0 = x - (w / 2 if anchor == 'middle' else w if anchor == 'end' else 0)
        self.els.append(f'<text x="{x0:.2f}" y="{y:.2f}" font-size="{size}"'
                        f'{" font-weight=\"bold\"" if b else ""}{" font-style=\"italic\"" if i else ""}'
                        f' fill="{c}"{tr}>{"".join(parts)}</text>')
        if not rot: self._y(y + 0.3 * size)
        return w

    def para(self, x, y, s, maxw, size=9, b=False, i=False, c=INK, anchor='start', lead=1.22):
        """y = top of block; returns bottom y"""
        ls = wrap(s, maxw, size, b, i)
        yy = y + 0.78 * size
        for ln in ls:
            self.text(x, yy, ln, size, b, i, c, anchor); yy += lead * size
        return y + lead * size * len(ls) - (lead - 1) * size + 0.22 * size

    def block_h(self, items, maxw, lead=1.22, gap=2.0):
        h = 0
        for k, (s, st) in enumerate(items):
            n = len(wrap(s, maxw, st.get('size', 9), st.get('b', 0), st.get('i', 0)))
            sz = st.get('size', 9)
            h += lead * sz * n - (lead - 1) * sz + 0.22 * sz + (st.get('gap', gap) if k else 0)
        return h

    def block(self, x, y, items, maxw, lead=1.22, gap=2.0, anchor='middle'):
        for k, (s, st) in enumerate(items):
            if k: y += st.get('gap', gap)
            al = st.get('align', anchor)
            xx = x if al == 'middle' else x - maxw / 2 if al == 'start' else x + maxw / 2
            y = self.para(xx, y, s, maxw, st.get('size', 9), st.get('b', 0), st.get('i', 0),
                          st.get('c', INK), al, lead)
        return y

    def rect(self, x, y, w, h, fill=F1, stroke=INK, sw=0.8, dash=None, rx=2):
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.els.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" '
                        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
        self._y(y + h + sw)

    def box(self, x, y, w, items, h=None, fill=F1, stroke=INK, sw=0.8, dash=None, pad=5, padx=6,
            anchor='middle', valign='middle', lead=1.22, gap=2.0):
        bh = self.block_h(items, w - 2 * padx, lead, gap)
        H = h if h is not None else bh + 2 * pad
        self.rect(x, y, w, H, fill, stroke, sw, dash)
        ty = y + (H - bh) / 2 if valign == 'middle' else y + pad
        self.block(x + w / 2, ty, items, w - 2 * padx, lead, gap, anchor)
        return H

    def line(self, x1, y1, x2, y2, c=INK, sw=0.8, dash=None, arrow=False, head=4.2):
        if arrow:
            ang = math.atan2(y2 - y1, x2 - x1)
            ex, ey = x2 - head * 0.9 * math.cos(ang), y2 - head * 0.9 * math.sin(ang)
        else: ex, ey = x2, y2
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.els.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{ex:.2f}" y2="{ey:.2f}" stroke="{c}" '
                        f'stroke-width="{sw}" stroke-linecap="butt"{d}/>')
        if arrow: self.head(x2, y2, math.atan2(y2 - y1, x2 - x1), c, head)
        self._y(max(y1, y2))

    def head(self, x, y, ang, c=INK, L=4.2):
        hw = L * 0.45
        p = [(x, y), (x - L * math.cos(ang) + hw * math.sin(ang), y - L * math.sin(ang) - hw * math.cos(ang)),
             (x - L * math.cos(ang) - hw * math.sin(ang), y - L * math.sin(ang) + hw * math.cos(ang))]
        self.els.append(f'<polygon points="{" ".join(f"{a:.2f},{b:.2f}" for a, b in p)}" fill="{c}"/>')

    def path(self, d, c=INK, sw=0.8, dash=None, fill='none', end=None, head=4.2):
        """end = (x, y, from_x, from_y) to draw arrowhead at x,y pointing from from_*"""
        dd = f' stroke-dasharray="{dash}"' if dash else ''
        self.els.append(f'<path d="{d}" fill="{fill}" stroke="{c}" stroke-width="{sw}"{dd}/>')
        if end:
            x, y, fx, fy = end
            self.head(x, y, math.atan2(y - fy, x - fx), c, head)

    def circle(self, x, y, r, fill=F1, stroke=INK, sw=0.8):
        self.els.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        self._y(y + r)

    def raw(self, s): self.els.append(s)

    def save(self, name, outdir, H=None, bottom=6):
        H = H or math.ceil(self.H + bottom)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.W/72:.4f}in" height="{H/72:.4f}in" '
               f'viewBox="0 0 {self.W:g} {H:g}" xml:space="preserve" font-family="{FAM}">\n<title>{esc(self.title)}</title>\n'
               f'<rect width="{self.W:g}" height="{H:g}" fill="#ffffff"/>\n' + '\n'.join(self.els) + '\n</svg>\n')
        os.makedirs(outdir, exist_ok=True)
        p = os.path.join(outdir, name + '.svg')
        open(p, 'w', encoding='utf-8').write(svg)
        subprocess.run(['rsvg-convert', '-d', '600', '-p', '600', p, '-o', os.path.join(outdir, name + '.png')], check=True)
        return p
