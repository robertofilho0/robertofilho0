#!/usr/bin/env python3
"""Gera o banner animado do perfil (dark e light).

Uso, a partir da raiz do repositório:
    pip install -r scripts/requirements.txt
    python scripts/gen_banner.py

Entrada : assets/source/portrait.jpg  (foto com fundo liso funciona melhor)
Saída   : assets/banner-dark.svg  e  assets/banner-light.svg

O retrato vira um pontilhado 1-bit embutido no SVG; a foto original não é
publicada. Para ajustar o enquadramento, mude CROP abaixo.
"""

from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets/source/portrait.jpg"
OUT = ROOT / "assets"

# (esquerda, topo, direita, base) em pixels da foto original
CROP = (60, 205, 840, 1035)
GRID_W, GRID_H = 300, 319          # resolução do pontilhado
W, H = 1180, 610
TOTAL = 7.0                          # duração (s) da animação de entrada

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
    (1, "frameworks", "NestJS · Laravel · Django · Flutter"),
    (1, "databases", "PostgreSQL · MySQL · SQL Server · Redis"),
    (1, "infra", "Docker · Kubernetes · AWS · Linux"),
    (1, "tools", "Git · Postman · Swagger · RabbitMQ · Figma"),
    (0, "contact", ""),
    (1, "linkedin", "/in/roberto-filho-393b372b2"),
    (1, "github", "robertofilho0"),
    (0, "status", "always learning, always shipping"),
]


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


def make_portrait(theme: dict) -> str:
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
    return base64.b64encode(buf.getvalue()).decode()


def keyed(attr: str, final: float, start: float, dur: float) -> str:
    """Anima attr de 0 até final; sem suporte a SMIL o valor final aparece."""
    a = min(start / TOTAL, 0.999)
    b = min((start + dur) / TOTAL, 1.0)
    return (f'<animate attributeName="{attr}" values="0;0;{final}" keyTimes="0;{a:.4f};{b:.4f}" '
            f'dur="{TOTAL}s" fill="freeze"/>')


def build(name: str, theme: dict) -> str:
    t = theme
    b64 = make_portrait(t)
    fs, cw, step = 16, 9.7, 27
    x_num, x_txt, y0 = 497, 535, 152
    p = []
    p.append(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    p.append('<title id="title">Roberto Filho — perfil animado</title>'
             '<desc id="desc">Terminal animado com retrato pontilhado e o perfil de Roberto Filho em YAML.</desc>')
    p.append(f'<defs><clipPath id="portraitClip"><rect x="49" y="124" width="390" height="414">'
             f'{keyed("height", 414, 0.3, 2.6)}</rect></clipPath></defs>')
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
    p.append(f'<g clip-path="url(#portraitClip)"><image x="49" y="124" width="390" height="414" '
             f'preserveAspectRatio="xMidYMid slice" href="data:image/png;base64,{b64}" '
             f'xlink:href="data:image/png;base64,{b64}"/></g>')
    p.append(f'<rect x="49" y="124" width="390" height="2" fill="{t["accent"]}" opacity="0">'
             f'<animate attributeName="y" values="124;538;538" keyTimes="0;{2.9 / TOTAL:.4f};1" dur="{TOTAL}s" fill="freeze"/>'
             f'<animate attributeName="opacity" values="0;0.85;0.85;0;0" '
             f'keyTimes="0;{0.3 / TOTAL:.4f};{2.8 / TOTAL:.4f};{3.0 / TOTAL:.4f};1" dur="{TOTAL}s" fill="freeze"/></rect>')

    # painel do YAML
    p.append(f'<rect x="475" y="88" width="670" height="472" rx="6" fill="{t["panel2"]}" stroke="{t["line"]}"/>')
    p.append(f'<path d="M475 124H1145" stroke="{t["line"]}"/>')
    p.append(f'<text x="491" y="111" fill="{t["accent"]}" font-family="{MONO}" font-size="13" '
             f'font-weight="700" letter-spacing="1.2">PROFILE.YML</text>')
    p.append(f'<text x="1130" y="111" text-anchor="end" fill="{t["muted"]}" font-family="{MONO}" '
             f'font-size="11">UTF-8 · YAML</text>')

    start = 1.0
    y = y0
    for i, (lvl, key, val) in enumerate(YAML_ROWS, 1):
        chars = lvl * 2 + len(key) + 1 + (1 + len(val) if val else 0)
        width = chars * cw + 6
        dur = max(0.25, chars * 0.022)
        p.append(f'<clipPath id="r{i}"><rect x="{x_txt - 2}" y="{y - 18}" width="{width:.1f}" height="24">'
                 f'{keyed("width", f"{width:.1f}", start, dur)}</rect></clipPath>')
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
        start += dur + 0.12
        y += step

    # cursor piscando após a última linha
    cx = x_txt + (len(YAML_ROWS[-1][1]) + 2 + len(YAML_ROWS[-1][2])) * cw + 6
    cy = y - step - 15
    p.append(f'<rect x="{cx:.1f}" y="{cy}" width="9" height="19" fill="{t["accent"]}" opacity="0">'
             f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.01;0.55;1" dur="1.1s" '
             f'begin="{start:.2f}s" repeatCount="indefinite"/></rect>')

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
