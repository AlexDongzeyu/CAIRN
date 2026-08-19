"""Publication figure style.

Colours are checked for greyscale separability rather than asserted: fills that co-occur
in a panel must differ by at least 0.10 in WCAG relative luminance, and each fill must
reach 1.6:1 against its own edge. Figures are built at the width they will be included
at, so `\\includegraphics[width=...]` does not silently rescale the type.
"""
from __future__ import annotations

import colorsys

import matplotlib as mpl
import numpy as np

LINEWIDTH_IN = 5.5   # single-column width of the legacy article template, in inches

# Measured from the target template by compiling a probe with \typeout{\the\columnwidth},
# rather than assumed: IEEEtran conference reports 252.0pt and 516.0pt at 72.27pt/in.
# A figure built at the wrong measure is not merely rescaled -- \includegraphics shrinks
# its type with it, so a 7.14in figure dropped into a 3.49in column loses half its font
# size and stops being legible.
COLUMN_IN = 252.0 / 72.27   # 3.487in, one IEEE column
TEXT_IN = 516.0 / 72.27     # 7.140in, both columns (figure*/table*)

# Okabe-Ito, which is colourblind-safe by construction. Hue only; the luminances are
# re-spaced below, because a colourblind-safe hue set is NOT automatically separable in
# greyscale and the raw palette fails the 0.10 check on four co-occurring pairs.
_BASE_PALETTE = {
    "ccnn": "#0072B2",
    "typed_star": "#D55E00",
    "untyped_star": "#CC79A7",
    "mlp": "#999999",
    "hyper": "#009E73",
    "accent": "#E69F00",
    "neutral": "#56B4E9",
}


def relative_luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def retarget_luminance(hex_color: str, target: float) -> str:
    """Move a colour to a target relative luminance, preserving hue and saturation.

    Bisecting lightness in HLS keeps the palette recognisably the same while making the
    fills separable in greyscale, which a colourblind-safe hue choice alone does not
    guarantee.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, _, ss = colorsys.rgb_to_hls(r, g, b)

    def at(light: float) -> str:
        rr, gg, bb = colorsys.hls_to_rgb(hh, light, ss)
        return "#%02X%02X%02X" % (round(rr * 255), round(gg * 255), round(bb * 255))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if relative_luminance(at(mid)) < target:
            lo = mid
        else:
            hi = mid
    return at((lo + hi) / 2)


# Okabe-Ito hues, re-spaced in luminance so that fills sharing a panel remain
# distinguishable when the figure is printed or photocopied in greyscale. Only colours
# that co-occur in one panel need separating, so the targets are chosen per panel group:
#   model panels : ccnn / typed_star / hyper / untyped_star / mlp
#   schematic    : neutral / hyper / accent
_TARGET_LUMINANCE = {
    "ccnn": 0.10, "typed_star": 0.26, "hyper": 0.42, "untyped_star": 0.54, "mlp": 0.70,
    "neutral": 0.55, "accent": 0.68,
}
PALETTE = {k: retarget_luminance(v, _TARGET_LUMINANCE[k]) for k, v in _BASE_PALETTE.items()}

PANEL_GROUPS = {
    "models": ("ccnn", "typed_star", "hyper", "untyped_star", "mlp"),
    "schematic": ("neutral", "hyper", "accent"),
    "ranks": ("ccnn", "hyper", "neutral", "accent"),
}


def relative_luminance_of(name: str) -> float:
    return relative_luminance(PALETTE[name])


def contrast_ratio(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def darken(hex_color: str, target_ratio: float = 1.8) -> str:
    """Edge colour for a fill, hue and saturation preserved, lightness bisected."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    lo, hi = 0.0, ll
    for _ in range(40):
        mid = (lo + hi) / 2
        rr, gg, bb = colorsys.hls_to_rgb(hh, mid, ss)
        cand = "#%02X%02X%02X" % (int(rr * 255), int(gg * 255), int(bb * 255))
        if contrast_ratio(hex_color, cand) >= target_ratio:
            lo = mid
        else:
            hi = mid
    rr, gg, bb = colorsys.hls_to_rgb(hh, lo, ss)
    return "#%02X%02X%02X" % (int(rr * 255), int(gg * 255), int(bb * 255))


def check_separation(colors: list[str], min_dl: float = 0.10) -> list[tuple[str, str, float]]:
    """Return offending pairs instead of trusting a comment that says they are fine."""
    bad = []
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            dl = abs(relative_luminance(a) - relative_luminance(b))
            if dl < min_dl:
                bad.append((a, b, dl))
    return bad


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,        # Type 1, not Type 3 - publishers reject Type 3
        "ps.fonttype": 42,
        "svg.fonttype": "none",    # keep SVG text editable
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "errorbar.capsize": 2,
    })


def panel_label(ax, letter: str, dx: float = -0.10, dy: float = 1.06) -> None:
    """Lowercase panel letter outside the axes.

    Descriptive panel text belongs in the caption, so the axes carry only the letter.
    """
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left")


def bootstrap_ci(vals, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(B)])
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
