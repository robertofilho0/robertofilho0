#!/usr/bin/env python3
"""Gera o banner animado do perfil (dark e light).

Uso, a partir da raiz do repositório:
    pip install -r scripts/requirements.txt
    python scripts/gen_banner.py

Entrada : assets/source/portrait.jpg   (foto com fundo liso funciona melhor)
          scripts/logos/*.png          (formas em branco; ordem em LOGO_ORDER)
Saída   : assets/banner-dark.svg  e  assets/banner-light.svg

O retrato vira um pontilhado 1-bit embutido no SVG; a foto original não é
publicada. Em loop, uma parte dos pontos se solta do retrato, forma cada logo
de scripts/logos/ e volta a ser o retrato. Para trocar ou adicionar logos,
coloque um PNG (forma branca / fundo transparente) na pasta e inclua o nome em
LOGO_ORDER. Para ajustar o enquadramento da foto, mude CROP.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/source/portrait.jpg"
LOGO_DIR = ROOT / "scripts/logos"
OUT = ROOT / "assets"

# (esquerda, topo, direita, base) em pixels da foto original
CROP = (60, 205, 840, 1035)
GRID_W, GRID_H = 300, 319          # resolução do pontilhado
W, H = 1180, 610

# área do retrato dentro do banner
PX0, PY0, PW, PH = 49, 124, 390, 414
SX, SY = PW / GRID_W, PH / GRID_H

# transformação em logos
LOGO_ORDER = ("linux", "kubernetes")
PARTICLES = 2600                    # pontos que viajam entre as formas
SEED = 314159
CYCLE_DELAY = 3.2                   # s: o ciclo começa quando o retrato termina de aparecer
HOLD_PORTRAIT = 2.2                 # s parado como retrato
MORPH = 1.8                         # s de cada transformação
HOLD_LOGO = 2.6                     # s parado como logo
LOGO_SCALE = 1.5                    # os pontos crescem um pouco ao formar o logo

MONO = "ui-monospace,SFMono-Regular,Consolas,monospace"

THEMES = {
    "dark": {
        "bg": "#07070d", "panel": "#0d0b16", "panel2": "#120f1f", "line": "#2a2140",
        "muted": "#7d7596", "text": "#e9e2f7", "dots": (181, 123, 238),
        "accent": "#b57bee", "key": "#d8b4fe", "section": "#f3d29b", "ok": "#9de4d1",
        "pink": "#f3a0b5", "min": 0.16, "invert": False,
    },
    "light": {
        "bg": "#f5f0ff", "panel": "#ffffff", "panel2": "#f0e8ff", "line": "#d9cdf0",
        "muted": "#7a6f96", "text": "#2b1b4a", "dots": (126, 58, 206),
        "accent": "#7e3ace", "key": "#6d28d9", "section": "#b45309", "ok": "#0f766e",
        "pink": "#be185d", "min": 0.10, "invert": True,
    },
}

YAML_ROWS = [
    (0, "profile", ""),
    (1, "subject", "Roberto Filho"),
    (1, "role", "Back-End Developer"),
    (1, "education", "ADS @ IFPB"),
    (1, "focus", "Back-End · REST APIs · Infra · Mobile"),
    (0, "stack", ""),
    (1, "languages", "TypeScript · JavaScript · PHP · Python · Dart"),
    (1, "frameworks", "Node.js · Fastify · NestJS · Laravel · Django"),
    (1, "front_mobile", "Vue.js · Flutter"),
    (1, "databases", "Postgres · MySQL · MongoDB · SQL Server · Redis"),
    (1, "infra", "Docker · Docker Swarm · Kubernetes · AWS · Linux"),
    (1, "tools", "Git · Postman · Swagger · RabbitMQ · Figma"),
    (0, "contact", ""),
    (1, "linkedin", "/in/roberto-filho-393b372b2"),
    (1, "github", "robertofilho0"),
    (0, "status", "always learning, always shipping"),
]


# --------------------------------------------------------------------------
# retrato
# --------------------------------------------------------------------------
def subject_mask(rgb: np.ndarray) -> np.ndarray:
    """Separa a pessoa do fundo liso comparando com a cor da parede."""
    h, w, _ = rgb.shape
    samples = np.concatenate([
        rgb[:30, :40].reshape(-1, 3), rgb[:30, -40:].reshape(-1, 3),
        rgb[h // 3: h // 3 + 60, :25].reshape(-1, 3), rgb[h // 3: h // 3 + 60, -25:].reshape(-1, 3),
    ])
    wall = np.median(samples, axis=0)
    dist = np.linalg.norm(rgb - wall, axis=2)
    dist = ndimage.gaussian_filter(dist, 1.5)

    # pele é "quente" (R > B); parede é azulada. Cabelo e camisa são escuros.
    warm = ndimage.gaussian_filter(rgb[..., 0] - rgb[..., 2], 1.5)
    lum = ndimage.gaussian_filter(rgb @ np.array([0.299, 0.587, 0.114]), 1.5)
    wall_warm = float(wall[0] - wall[2])
    is_skin = (warm > wall_warm + 14) & (dist > 28)
    is_dark = (lum < 128) & (dist > 40)
    mask = is_skin | is_dark
    mask = ndimage.binary_closing(mask, iterations=5)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=4)
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labels, range(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    return mask


def make_portrait(theme: dict) -> tuple[str, np.ndarray]:
    """Devolve (PNG em base64, matriz booleana dos pontos acesos)."""
    img = Image.open(SOURCE).convert("RGB").crop(CROP).resize((GRID_W, GRID_H), Image.LANCZOS)
    rgb = np.asarray(img).astype(float)
    mask = subject_mask(rgb)

    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1).filter(
        ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    g = np.asarray(gray).astype(float)
    lo, hi = np.percentile(g[mask], [3, 97])
    norm = np.clip((g - lo) / max(hi - lo, 1), 0, 1) ** 0.9

    if theme["invert"]:
        dens = 0.10 + 0.90 * (1 - norm)
    else:
        dens = theme["min"] + (1 - theme["min"]) * norm

    # ombros se dissolvem para baixo
    fade = np.ones(GRID_H)
    start = int(GRID_H * 0.80)
    fade[start:] = np.linspace(1, 0.12, GRID_H - start)
    dens = dens * mask * fade[:, None]

    bits = np.asarray(Image.fromarray((dens * 255).astype("uint8")).convert("1"))
    dot = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    alpha = np.kron(bits.astype(np.uint8), dot) * 255
    rgba = np.zeros((GRID_H * 3, GRID_W * 3, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = theme["dots"]
    rgba[..., 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), bits


# --------------------------------------------------------------------------
# transformação retrato -> logos -> retrato
# --------------------------------------------------------------------------
_logo_cache: dict[str, np.ndarray] = {}


def logo_targets(name: str, n: int) -> np.ndarray:
    """n pontos distribuídos de forma uniforme dentro da forma do logo (coords da grade)."""
    if name in _logo_cache:
        return _logo_cache[name]
    path = next((p for p in sorted(LOGO_DIR.glob(f"{name}.*"))), None)
    if path is None:
        raise SystemExit(f"logo não encontrado: {LOGO_DIR / name}.png")
    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3]
    if a.min() == 255:                       # imagem opaca: branco = forma
        a = np.asarray(im.convert("L"))
    shape = a > 127
    ys, xs = np.nonzero(shape)
    shape = shape[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = shape.shape
    scale = min(GRID_W * 0.80 / w, GRID_H * 0.76 / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    small = np.asarray(Image.fromarray((shape * 255).astype("uint8")).resize((nw, nh), Image.LANCZOS)) > 127
    ox, oy = (GRID_W - nw) / 2, GRID_H * 0.47 - nh / 2
    pts = np.argwhere(small)[:, ::-1].astype(float) + [ox + 0.5, oy + 0.5]
    rng = np.random.default_rng(SEED)
    if len(pts) > n:
        cent = stipple(pts, n, rng)
    else:
        cent = pts[rng.integers(0, len(pts), size=n)]
    _logo_cache[name] = cent
    return cent


def stipple(pts: np.ndarray, n: int, rng: np.random.Generator, iters: int = 14) -> np.ndarray:
    """Relaxamento de Lloyd: n pontos espalhados de forma uniforme sobre pts."""
    cent = pts[rng.choice(len(pts), size=n, replace=False)].astype(float)
    for _ in range(iters):
        _, idx = cKDTree(cent).query(pts)
        cnt = np.bincount(idx, minlength=n)
        ok = cnt > 0
        for axis in (0, 1):
            sums = np.bincount(idx, weights=pts[:, axis], minlength=n)
            cent[ok, axis] = sums[ok] / cnt[ok]
    return cent


def assign(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Para cada ponto de src, o destino em dst que minimiza o deslocamento total."""
    rows, cols = linear_sum_assignment(cdist(src, dst, "sqeuclidean"))
    out = np.empty_like(dst)
    out[rows] = dst[cols]
    return out


def compute_stages(bits: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """P0 (retrato) e as posições de cada etapa: logo 1..L e retorno (conjunto = P0)."""
    dots = np.argwhere(bits)[:, ::-1].astype(float) + 0.5
    rng = np.random.default_rng(SEED)
    n = min(PARTICLES, len(dots))
    p0 = dots[rng.choice(len(dots), size=n, replace=False)]
    stages, cur = [], p0
    for name in LOGO_ORDER:
        cur = assign(cur, logo_targets(name, n))
        stages.append(cur)
    stages.append(assign(cur, p0))           # volta a ocupar as posições do retrato
    return p0, stages


# --------------------------------------------------------------------------
# SVG
# --------------------------------------------------------------------------
def row_timings() -> tuple[list[tuple[float, float, int]], float]:
    out, start = [], 1.0
    for lvl, key, val in YAML_ROWS:
        chars = lvl * 2 + len(key) + 1 + (1 + len(val) if val else 0)
        dur = max(0.2, chars * 0.015)
        out.append((start, dur, chars))
        start += dur + 0.08
    return out, start


def keyed(attr: str, final: str | float, start: float, dur: float, total: float) -> str:
    """Anima attr de 0 até final; sem suporte a SMIL o valor final aparece."""
    a = min(start / total, 0.999)
    b = min((start + dur) / total, 1.0)
    return (f'<animate attributeName="{attr}" values="0;0;{final}" keyTimes="0;{a:.4f};{b:.4f}" '
            f'dur="{total:.2f}s" fill="freeze"/>')


def pct(t: float, total: float) -> str:
    return f"{t / total * 100:.2f}%"


def cycle_css(color: str, total: float, n_logos: int) -> str:
    """CSS do ciclo: pontos (m), retrato (pf) e legendas (c0..cL)."""
    t_ret = HOLD_PORTRAIT + n_logos * (MORPH + HOLD_LOGO)      # início do retorno ao retrato
    anim = f"{total:.2f}s {CYCLE_DELAY}s infinite"
    states = ["translate(0,0) scale(1)"]
    for k in range(1, n_logos + 1):
        states.append(f"translate(calc(var(--x{k})*1px),calc(var(--y{k})*1px)) scale({LOGO_SCALE})")
    end = n_logos + 1
    end_state = f"translate(calc(var(--x{end})*1px),calc(var(--y{end})*1px)) scale(1)"

    kf = [(0.0, states[0]), (HOLD_PORTRAIT, states[0])]
    t = HOLD_PORTRAIT
    for k in range(1, n_logos + 1):
        kf.append((t + MORPH, states[k]))
        kf.append((t + MORPH + HOLD_LOGO, states[k]))
        t += MORPH + HOLD_LOGO
    kf.append((total, end_state))
    m = "".join(f"{pct(tt, total)}{{transform:{s}}}" for tt, s in kf)

    pf = [(0.0, 1), (HOLD_PORTRAIT, 1), (HOLD_PORTRAIT + 0.35, 0),
          (t_ret + MORPH * 0.55, 0), (t_ret + MORPH, 1), (total, 1)]
    pfk = "".join(f"{pct(tt, total)}{{opacity:{v}}}" for tt, v in pf)

    caps = [f".c0{{animation:c0 {anim} linear}}"]
    kfs = [f"@keyframes c0{{" + "".join(
        f"{pct(tt, total)}{{opacity:{v}}}" for tt, v in
        [(0, 1), (HOLD_PORTRAIT, 1), (HOLD_PORTRAIT + 0.3, 0), (t_ret + MORPH * 0.7, 0), (t_ret + MORPH, 1), (total, 1)]
    ) + "}"]
    for k in range(1, n_logos + 1):
        ms = HOLD_PORTRAIT + (k - 1) * (MORPH + HOLD_LOGO)
        a0, a1 = ms + MORPH * 0.6, ms + MORPH
        b0, b1 = ms + MORPH + HOLD_LOGO, ms + MORPH + HOLD_LOGO + 0.3
        pts = [(0, 0), (a0, 0), (a1, 1), (b0, 1), (b1, 0), (total, 0)]
        caps.append(f".c{k}{{animation:c{k} {anim} linear}}")
        kfs.append(f"@keyframes c{k}{{" + "".join(f"{pct(tt, total)}{{opacity:{v}}}" for tt, v in pts) + "}")

    return (
        f".d{{fill:{color};transform-box:fill-box;transform-origin:center;"
        f"animation:m {anim} cubic-bezier(.7,0,.2,1)}}"
        f".p{{animation:pf {anim} linear}}" + "".join(caps) +
        f"@keyframes m{{{m}}}@keyframes pf{{{pfk}}}" + "".join(kfs)
    )


def build(name: str, theme: dict) -> str:
    t = theme
    b64, bits = make_portrait(t)
    p0, stages = compute_stages(bits)
    n_logos = len(LOGO_ORDER)
    cycle_total = HOLD_PORTRAIT + n_logos * (MORPH + HOLD_LOGO) + MORPH

    timings, intro_end = row_timings()
    intro_total = intro_end + 0.5
    dots_hex = "#%02x%02x%02x" % t["dots"]

    fs, cw, step = 16, 9.7, 26          # 16 linhas cabem no painel
    x_num, x_txt, y0 = 497, 535, 150
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    p.append('<title id="title">Roberto Filho — perfil animado</title>'
             '<desc id="desc">Terminal animado: o retrato pontilhado se transforma nos logos do Linux e do '
             'Kubernetes enquanto o perfil de Roberto Filho é digitado em YAML.</desc>')
    p.append(f"<style>{cycle_css(dots_hex, cycle_total, n_logos)}</style>")
    p.append(f'<defs><clipPath id="portraitClip"><rect x="{PX0}" y="{PY0}" width="{PW}" height="{PH}">'
             f'{keyed("height", PH, 0.3, 2.6, intro_total)}</rect></clipPath></defs>')
    p.append(f'<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>')
    p.append(f'<rect x="13" y="13" width="1154" height="584" rx="13" fill="{t["panel"]}" stroke="{t["line"]}"/>')
    p.append(f'<path d="M13 62H1167" stroke="{t["line"]}"/>')
    for cx, col in ((38, "#ff5f57"), (59, "#febc2e"), (80, "#28c840")):
        p.append(f'<circle cx="{cx}" cy="38" r="6" fill="{col}"/>')
    p.append(f'<text x="590" y="43" text-anchor="middle" fill="{t["muted"]}" font-family="{MONO}" '
             f'font-size="13" letter-spacing=".4">vim profile.yml</text>')

    # painel do retrato
    p.append(f'<rect x="35" y="88" width="418" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>')
    p.append(f'<path d="M35 124H453" stroke="{t["line"]}"/>')
    p.append(f'<text x="49" y="111" fill="{t["accent"]}" font-family="{MONO}" font-size="13" '
             f'font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>')
    p.append(f'<text x="438" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
             f'font-size="11">{GRID_W}×{GRID_H} / 1-BIT</text>')

    # retrato (PNG) + pontos que viajam, ambos recortados pela revelação inicial
    p.append('<g clip-path="url(#portraitClip)">')
    p.append(f'<g class="p"><image x="{PX0}" y="{PY0}" width="{PW}" height="{PH}" '
             f'preserveAspectRatio="xMidYMid slice" href="data:image/png;base64,{b64}" '
             f'xlink:href="data:image/png;base64,{b64}"/></g>')
    p.append('<g>')
    for i in range(len(p0)):
        var = []
        for k, st in enumerate(stages, 1):
            dx, dy = (st[i] - p0[i]) * (SX, SY)
            var.append(f"--x{k}:{dx:.1f};--y{k}:{dy:.1f}")
        p.append(f'<circle class="d" cx="{PX0 + p0[i, 0] * SX:.1f}" cy="{PY0 + p0[i, 1] * SY:.1f}" '
                 f'r=".8" style="{";".join(var)}"/>')
    p.append('</g></g>')

    # legenda do que está sendo desenhado
    labels = ["portrait"] + list(LOGO_ORDER)
    for k, label in enumerate(labels):
        p.append(f'<text class="c{k}" x="{PX0}" y="553" opacity="{1 if k == 0 else 0}" fill="{t["muted"]}" '
                 f'font-family="{MONO}" font-size="11"><tspan fill="{t["accent"]}">›</tspan> '
                 f'visual.map/{escape(label)}</text>')

    # linha de varredura da revelação
    p.append(f'<rect x="{PX0}" y="{PY0}" width="{PW}" height="2" fill="{t["accent"]}" opacity="0">'
             f'<animate attributeName="y" values="{PY0};{PY0 + PH};{PY0 + PH}" '
             f'keyTimes="0;{2.9 / intro_total:.4f};1" dur="{intro_total:.2f}s" fill="freeze"/>'
             f'<animate attributeName="opacity" values="0;0.85;0.85;0;0" '
             f'keyTimes="0;{0.3 / intro_total:.4f};{2.8 / intro_total:.4f};{3.0 / intro_total:.4f};1" '
             f'dur="{intro_total:.2f}s" fill="freeze"/></rect>')

    # painel do YAML
    p.append(f'<rect x="475" y="88" width="670" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>')
    p.append(f'<path d="M475 124H1145" stroke="{t["line"]}"/>')
    p.append(f'<text x="491" y="111" fill="{t["accent"]}" font-family="{MONO}" font-size="13" '
             f'font-weight="700" letter-spacing="1.2">PROFILE.YML</text>')
    p.append(f'<text x="1130" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
             f'font-size="11">UTF-8 · YAML</text>')

    y = y0
    for i, ((lvl, key, val), (start, dur, chars)) in enumerate(zip(YAML_ROWS, timings), 1):
        width = chars * cw + 6
        p.append(f'<clipPath id="r{i}"><rect x="{x_txt - 2}" y="{y - 18}" width="{width:.1f}" height="24">'
                 f'{keyed("width", f"{width:.1f}", start, dur, intro_total)}</rect></clipPath>')
        p.append(f'<text x="{x_num}" y="{y}" fill="{t["muted"]}" opacity=".55" font-family="{MONO}" '
                 f'font-size="13" text-anchor="end" dx="14">{i}</text>')
        indent = "&#160;" * (lvl * 2)
        key_col = t["section"] if lvl == 0 else t["key"]
        if val:
            sep_val = f'<tspan fill="{t["muted"]}">: </tspan><tspan fill="{t["text"]}">{escape(val)}</tspan>'
        else:
            sep_val = f'<tspan fill="{t["muted"]}">:</tspan>'
        p.append(f'<g clip-path="url(#r{i})"><text x="{x_txt}" y="{y}" font-family="{MONO}" font-size="{fs}" '
                 f'xml:space="preserve">{indent}<tspan fill="{key_col}" font-weight="{700 if lvl == 0 else 400}">'
                 f'{escape(key)}</tspan>{sep_val}</text></g>')
        y += step

    # cursor piscando após a última linha
    cx = x_txt + (len(YAML_ROWS[-1][1]) + 2 + len(YAML_ROWS[-1][2])) * cw + 6
    cy = y - step - 15
    p.append(f'<rect x="{cx:.1f}" y="{cy}" width="9" height="19" fill="{t["accent"]}" opacity="0">'
             f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.01;0.55;1" dur="1.1s" '
             f'begin="{intro_end:.2f}s" repeatCount="indefinite"/></rect>')

    # barra de status
    p.append(f'<text x="35" y="582" fill="{t["muted"]}" font-family="{MONO}" font-size="12">'
             f'<tspan fill="{t["ok"]}" font-weight="700">NORMAL</tspan>  profile.yml  ·  '
             f'robertofilho0  ·  status: ready</text>')
    p.append(f'<text x="1145" y="582" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
             f'font-size="12">always learning, always shipping</text>')
    p.append("</svg>")
    return "".join(p)


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"foto não encontrada: {SOURCE}")
    for name, theme in THEMES.items():
        svg = build(name, theme)
        (OUT / f"banner-{name}.svg").write_text(svg, encoding="utf-8")
        print(f"ok: banner-{name}.svg ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
