#!/usr/bin/env python3
"""Gera os SVGs do perfil (somente biblioteca padrão).

Uso, a partir da raiz do repositório:
    python scripts/gen_assets.py

Lê assets/skills.json e assets/langmix.json e escreve:
    assets/radar-dark.svg        assets/radar-light.svg
    assets/radar-langs-dark.svg  assets/radar-langs-light.svg
    assets/whoami-card.svg
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

FONT_SANS = "ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
FONT_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "dark": {
        "grid": "#2a2140",
        "spoke": "#1f1832",
        "label": "#d8cdee",
        "value": "#8b83a3",
        "title": "#ffffff",
        "fill": "#7e3ace",
        "stroke": "#b57bee",
    },
    "light": {
        "grid": "#d9cdf0",
        "spoke": "#eae2f8",
        "label": "#2b1b4a",
        "value": "#6b5f8a",
        "title": "#2b1b4a",
        "fill": "#7e3ace",
        "stroke": "#7e3ace",
    },
}


def radar_svg(data: dict, theme: dict) -> str:
    axes = data["axes"]
    n = len(axes)
    w, h = 712, 549
    cx, cy, radius = 356.0, 289.0, 200.0
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{escape(data["title"])}" font-family="{FONT_SANS}">',
        f'<text x="{w / 2}" y="28" text-anchor="middle" font-size="15" font-weight="700" '
        f'fill="{theme["title"]}">{escape(data["title"])}</text>',
        f'<g transform="translate({cx},{cy})">',
    ]

    def point(i: int, r: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return r * math.cos(ang), r * math.sin(ang)

    # grade (4 anéis) e raios
    for ring in range(1, 5):
        r = radius * ring / 4
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, r) for i in range(n)))
        out.append(
            f'<polygon points="{pts}" fill="none" stroke="{theme["grid"]}" stroke-width="1" '
            f'opacity="{0.45 + ring * 0.13:.2f}"/>'
        )
    for i in range(n):
        x, y = point(i, radius)
        out.append(f'<line x1="0" y1="0" x2="{x:.1f}" y2="{y:.1f}" stroke="{theme["spoke"]}" stroke-width="1"/>')

    # polígono de dados, animado
    vals = [max(0, min(100, a["value"])) / 100 for a in axes]
    dpts = [point(i, radius * v) for i, v in enumerate(vals)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in dpts)
    out.append('<g><animateTransform attributeName="transform" type="scale" values="0.04;1" dur="1.1s" '
               'calcMode="spline" keyTimes="0;1" keySplines="0.22 1 0.36 1" fill="freeze"/>')
    out.append(
        f'<polygon points="{poly}" fill="{theme["fill"]}" fill-opacity="0.28" '
        f'stroke="{theme["stroke"]}" stroke-width="2.5" stroke-linejoin="round"/>'
    )
    for x, y in dpts:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.8" fill="{theme["stroke"]}"/>')
    out.append("</g>")

    # rótulos
    for i, a in enumerate(axes):
        x, y = point(i, radius + 26)
        if abs(x) < 8:
            anchor = "middle"
        else:
            anchor = "start" if x > 0 else "end"
        dy = 4 if abs(x) >= 8 else (-4 if y < 0 else 14)
        out.append(
            f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" font-size="13" font-weight="600" '
            f'fill="{theme["label"]}">{escape(a["label"])}</text>'
        )
    out.append("</g></svg>")
    return "".join(out)


def whoami_svg() -> str:
    t = lambda x, y, fill, size, text, extra="": (  # noqa: E731
        f'<text x="{x}" y="{y}" fill="{fill}" font-family="{FONT_MONO}" font-size="{size}" {extra}>'
        f"{escape(text)}</text>"
    )
    rows = [
        ("back-end/", "NestJS · Laravel · Django"),
        ("mobile/", "Flutter · Dart"),
        ("infra/", "Docker · Kubernetes · AWS · Linux"),
        ("data/", "PostgreSQL · MySQL · SQL Server · Redis"),
    ]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 440" width="960" height="440" '
        'role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Roberto Filho — Back-End Developer</title>",
        '<desc id="desc">Cartão de perfil em estilo terminal com formação, foco e áreas de atuação.</desc>',
        "<defs>"
        '<linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">'
        '<stop stop-color="#7e3ace"/><stop offset="1" stop-color="#b57bee"/></linearGradient>'
        '<linearGradient id="sun" x1="0" x2="0" y1="0" y2="1">'
        '<stop stop-color="#d8b4fe"/><stop offset="1" stop-color="#7e3ace"/></linearGradient>'
        '<filter id="glow" x="-20%" y="-30%" width="140%" height="160%">'
        '<feGaussianBlur stdDeviation="2.2" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        "</defs>",
        '<rect width="960" height="440" rx="18" fill="#0d0d12"/>',
        '<rect x="9" y="9" width="942" height="422" rx="13" fill="none" stroke="url(#sky)" stroke-width="2"/>',
        '<circle cx="35" cy="35" r="5" fill="#f3a0b5"/>',
        '<circle cx="54" cy="35" r="5" fill="#f1d09a"/>',
        '<circle cx="73" cy="35" r="5" fill="#9de4d1"/>',
        t(98, 41, "#8b83a3", 15, "robertofilho0  ·  profile shell  ·  online"),
        '<path d="M29 58H931" stroke="#2a2140"/>',
        '<rect x="29" y="78" width="594" height="321" rx="9" fill="#14121d" stroke="#7e3ace" stroke-width="1.5"/>',
        t(48, 111, "#b57bee", 20, "$ whoami", 'font-weight="700"'),
        t(48, 143, "#d8cdee", 18, "roberto_filho"),
        t(206, 143, "#8b83a3", 18, "—"),
        t(231, 143, "#f3d29b", 18, "Back-End Developer"),
        '<path d="M48 166H604" stroke="#2a2140"/>',
        t(48, 196, "#8b83a3", 17, "education:"),
        t(166, 196, "#d8b4fe", 17, "ADS @ IFPB"),
        t(48, 225, "#8b83a3", 17, "focus:"),
        t(166, 225, "#9de4d1", 17, "Back-End · REST APIs · Infra · Mobile"),
        t(48, 267, "#b57bee", 20, "$ ls expertise/", 'font-weight="700"'),
    ]
    y = 299
    for name, desc in rows:
        parts.append(t(48, y, "#f3a0b5", 16, name))
        parts.append(t(190, y, "#d8cdee", 16, desc))
        y += 27

    parts += [
        '<rect x="651" y="78" width="280" height="321" rx="9" fill="#14121d" stroke="#7e3ace" stroke-width="1.5"/>',
        t(674, 111, "#f3d29b", 17, "night-drive runtime"),
        '<circle cx="791" cy="218" r="62" fill="url(#sun)" opacity=".95"/>',
        '<path d="M674 214H908M674 230H908M674 246H908M674 262H908M674 278H908" '
        'stroke="#14121d" stroke-width="5" opacity=".9"/>',
        '<path d="M674 312C711 282 746 338 785 307S861 284 908 316" fill="none" stroke="#9de4d1" '
        'stroke-width="3" filter="url(#glow)"/>',
        '<path d="M674 329C716 299 746 355 789 324S862 305 908 333" fill="none" stroke="#b57bee" stroke-width="2"/>',
        t(674, 378, "#8b83a3", 14, "status: always shipping"),
        "</svg>",
    ]
    return "\n  ".join(parts)


def main() -> None:
    for src, out in (("skills.json", "radar"), ("langmix.json", "radar-langs")):
        data = json.loads((ASSETS / src).read_text(encoding="utf-8"))
        for name, theme in THEMES.items():
            (ASSETS / f"{out}-{name}.svg").write_text(radar_svg(data, theme), encoding="utf-8")
    (ASSETS / "whoami-card.svg").write_text(whoami_svg(), encoding="utf-8")
    print("ok: assets gerados em", ASSETS)


if __name__ == "__main__":
    main()
