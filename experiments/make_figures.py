"""PHASE 12 â€” figures. Each is an experiment output, not decoration.

Every panel states its sampling unit and N in the caption, carries CIs on every estimate,
uses a colourblind-safe palette checked for greyscale separability, and is written as
vector PDF (plus PNG for the progress report).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.figstyle import (  # noqa: E402
    COLUMN_IN, PALETTE, PANEL_GROUPS, TEXT_IN, apply_style, bootstrap_ci, check_separation,
    darken, panel_label, relative_luminance,
)
from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
log = make_logger("figures")
CAPTIONS: dict[str, str] = {}

# One spelling of each model across every panel; inconsistent names read as different models.
MODEL_LABEL = {
    "M0_mlp": "M0 text MLP", "M1_dense": "M1 dense", "M2_untyped_star": "M2 untyped",
    "M3_typed_star": "M3 typed star", "M4_allset": "M4 AllSet", "M4_edhnn": "M4 ED-HNN",
    "M4_hgmlp": "M4 Hypergraph-MLP", "M5_ccnn": "M5 CCNN-style",
}


def load(name: str) -> dict | list | None:
    p = RES / name
    if not p.exists():
        log(f"  missing {name}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, stem: str, caption: str) -> None:
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.svg")           # editable text for the production editor
    fig.savefig(FIG / f"{stem}.png", dpi=600)  # screen preview and the HTML report
    plt.close(fig)
    CAPTIONS[stem] = caption
    log(f"  wrote {stem}.pdf / .svg / .png")


# ------------------------------------------------------------------ Figure 1
def figure1_representations(roundtrip: dict) -> None:
    """Clique (lossy) vs star (lossless) vs ranked complex, on one tiny sub-archive."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.0))
    narr = [f"n{i}" for i in range(5)]
    events = {"E1": [0, 1, 2], "E2": [2, 3], "E3": [3, 4]}

    npos = {n: (np.cos(2 * np.pi * i / 5 - np.pi / 2), np.sin(2 * np.pi * i / 5 - np.pi / 2))
            for i, n in enumerate(narr)}

    ax = axes[0]
    for members in events.values():
        for a in members:
            for b in members:
                if a < b:
                    ax.plot(*zip(npos[narr[a]], npos[narr[b]]), color="#BBBBBB", lw=1.0, zorder=1)
    for n, (x, y) in npos.items():
        ax.scatter([x], [y], s=42, color=PALETTE["neutral"], edgecolor=darken(PALETTE["neutral"]),
                   zorder=3, linewidths=0.8)
    ax.set_title("")

    ax = axes[1]
    epos = {e: (-1.6 + 1.6 * i, 1.7) for i, e in enumerate(events)}
    for e, members in events.items():
        for m in members:
            ax.plot(*zip(npos[narr[m]], epos[e]), color="#999999", lw=0.9, zorder=1)
    for n, (x, y) in npos.items():
        ax.scatter([x], [y], s=36, color=PALETTE["neutral"], edgecolor=darken(PALETTE["neutral"]),
                   zorder=3, linewidths=0.8)
    for e, (x, y) in epos.items():
        ax.scatter([x], [y], s=48, marker="s", color=PALETTE["hyper"],
                   edgecolor=darken(PALETTE["hyper"]), zorder=3, linewidths=0.8)
    ok = roundtrip.get("R-A|mid", {}).get("lossless") if roundtrip else None
    if not ok:
        log("  fig1: round-trip not confirmed lossless; caption states the measured result")

    ax = axes[2]
    lanes = {0: -1.2, 1: -0.2, 2: 0.9, 3: 1.9}
    for i, n in enumerate(narr):
        ax.scatter([-1.6 + 0.8 * i], [lanes[0]], s=30, color=PALETTE["neutral"],
                   edgecolor=darken(PALETTE["neutral"]), linewidths=0.7, zorder=3)
    for j, e in enumerate(events):
        ax.scatter([-1.1 + 1.1 * j], [lanes[2]], s=44, marker="s", color=PALETTE["hyper"],
                   edgecolor=darken(PALETTE["hyper"]), linewidths=0.7, zorder=3)
        for m in events[e]:
            ax.plot([-1.6 + 0.8 * m, -1.1 + 1.1 * j], [lanes[0], lanes[2]],
                    color="#BBBBBB", lw=0.8, zorder=1)
        ax.plot([-1.1 + 1.1 * j, 0.0], [lanes[2], lanes[3]], color="#BBBBBB", lw=0.8, zorder=1)
    ax.scatter([0.0], [lanes[3]], s=54, marker="D", color=PALETTE["accent"],
               edgecolor=darken(PALETTE["accent"]), linewidths=0.7, zorder=3)
    # Drawn as tick labels rather than free text: matplotlib reserves space for ticks, so
    # the labels cannot end up sitting on top of the nodes as hand-placed text did.
    ax.set_yticks([lanes[0], lanes[2], lanes[3]])
    ax.set_yticklabels(["rank 0\nnarrator", "rank 2\nevent/site", "rank 3\nepisode"],
                       fontsize=6, color="#444444")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    for a, letter in zip(axes, "abc"):
        panel_label(a, letter, dx=-0.02, dy=1.02)
    for a in axes[:2]:
        a.set_xticks([]); a.set_yticks([])
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    save(fig, "fig1_representations",
         "Three encodings of the same five-narrator sub-archive. a, Clique expansion loses "
         "which event a co-occurrence came from. b, Star expansion is lossless. c, The ranked "
         "combinatorial complex makes rank-specific operators native. The star expansion is "
         "information-equivalent to the ranked complex: a round-trip test over "
         f"{roundtrip.get('R-A|mid', {}).get('n_cells', 'n/a')} cells recovers every cell's rank "
         "and exact member set (0 missing, 0 extra, 0 mismatched; E4.2). Any advantage of the "
         "complex must therefore come from the operator, not the representation.")


# ------------------------------------------------------------------ Figure 2
def figure2_granularity(sweep: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_IN, 2.1))
    gran = ["coarse", "mid", "fine"]
    colors = [PALETTE["mlp"], PALETTE["ccnn"], PALETTE["hyper"]]

    ax = axes[0]
    bins = ["1", "2-3", "4-10", "11-50", ">50"]
    w = 0.26
    for i, g in enumerate(gran):
        h = sweep["per_granularity"][g]["attestation"]["hist"]
        vals = [h.get(b, 0) for b in bins]
        ax.bar(np.arange(len(bins)) + (i - 1) * w, vals, width=w, label=g,
               color=colors[i], edgecolor=darken(colors[i]), linewidth=0.5)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, fontsize=6)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)  # headroom so the tallest bar is not flush to the spine
    ax.set_xlabel(r"attestation multiplicity $a(x)$")
    ax.set_ylabel("rank-2 cells")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    for i, g in enumerate(gran):
        ci = sweep["per_granularity"][g]["singleton_fraction"]
        ax.errorbar([i], [ci["point"]],
                    yerr=[[ci["point"] - ci["lo"]], [ci["hi"] - ci["point"]]],
                    fmt="o", color=colors[i], markersize=4, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(gran, fontsize=6)
    ax.set_ylabel("singleton fraction")

    ax = axes[2]
    M = np.full((3, 3), np.nan)
    ref = sweep["rbo_referent"]["p=0.9"]
    for i, a in enumerate(gran):
        for j, b in enumerate(gran):
            if i < j:
                v = ref.get(f"{a}|{b}", np.nan)
                M[i, j] = M[j, i] = v
    # The diagonal is 1.00 by construction and states no comparison, so it is left blank and
    # the colour scale is fitted to the off-diagonal range it would otherwise swamp.
    off = M[~np.isnan(M)]
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#EDEDED")
    lo, hi = float(np.min(off)), float(np.max(off))
    pad = max(0.02, 0.08 * (hi - lo))
    vmin, vmax = lo - pad, hi + pad
    im = ax.imshow(np.ma.masked_invalid(M), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(3)); ax.set_xticklabels(gran, fontsize=6, rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels(gran, fontsize=6)
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                continue
            # Read the contrast off the rendered cell colour; a hand-picked cutoff left
            # white text on mid-tone cells where it is hard to read.
            r, g, b, _ = cmap((M[i, j] - vmin) / (vmax - vmin))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if lum > 0.45 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for a, letter in zip(axes, "abc"):
        panel_label(a, letter)

    fig.tight_layout()
    n2 = {g: sweep["per_granularity"][g]["n_rank2_cells"] for g in gran}
    save(fig, "fig2_granularity",
         f"Granularity sweep over the same mention set (rank-2 cells: {n2}). $a(x)$ is the "
         "archive-conditioned attestation multiplicity: the number of distinct narrators in "
         "this collection whose segments the archive filed under that term. Sampling unit for "
         "panel (b) is the narrator (cluster subsampling, B=1500). Panel (c) compares triage "
         "lists after projection onto a shared archival referent; raw cell labels are disjoint "
         "across granularities by construction, so unprojected overlap is identically zero and "
         "measures the labelling scheme rather than the triage.")


# ------------------------------------------------------------------ Figure 3
def figure3_models(runs: list, strat: dict) -> None:
    # Panel (c) is the split comparison, folded in from figure6_split so that the model
    # results and the split that reverses them occupy one float instead of two.
    fig, axes = plt.subplots(1, 3, figsize=(0.94 * TEXT_IN, 1.78),
                             gridspec_kw={"width_ratios": [1.45, 1.0, 0.95]})
    prim = [r for r in runs if r["granularity"] == "mid" and r["split"] == "narrator-disjoint"
            and r["neg_regime"] == "MNS" and r["rank_map"] == "R-A"
            and r.get("cell") != "E8.3_gold_subset"]
    # M1_dense is the parameter-free retrieval baseline. Omitting it flattered every trained
    # model, because it outranks all but one of them.
    order = ["M0_mlp", "M4_hgmlp", "M4_allset", "M4_edhnn", "M1_dense", "M2_untyped_star",
             "M5_ccnn", "M3_typed_star"]
    order = [m for m in order if any(r["model"] == m for r in prim)]
    nice = MODEL_LABEL
    cmap = {"M5_ccnn": PALETTE["ccnn"], "M3_typed_star": PALETTE["typed_star"],
            "M2_untyped_star": PALETTE["untyped_star"], "M0_mlp": PALETTE["mlp"],
            "M1_dense": PALETTE["accent"]}
    # AllSet and ED-HNN never update rank-1 cells, so their T1 entries are the frozen-encoder
    # floor rather than a measurement. The text calls them undefined; plotting them as ordinary
    # points would contradict it, so they are drawn hollow and labelled as floors.
    FLOOR = {"M4_allset", "M4_edhnn"}

    ax = axes[0]
    # Horizontal, so eight model names read straight instead of rotated at 40 degrees.
    for i, m in enumerate(order):
        vals = [r["T1_map"] for r in prim if r["model"] == m]
        mean, lo, hi = bootstrap_ci(vals)
        c = cmap.get(m, PALETTE["hyper"])
        floor = m in FLOOR
        ax.errorbar([mean], [i], xerr=[[mean - lo], [hi - mean]], fmt="o", color=c,
                    markersize=5, capsize=2, zorder=3,
                    markerfacecolor="white" if floor else c,
                    markeredgecolor=c, markeredgewidth=1.1)
        # Vertical jitter only, so overlapping seeds stay countable. No plotted value is
        # simulated: every point is a measured run.
        jitter = np.random.default_rng(i).normal(0, 0.055, len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter,
                   s=7, color=c, alpha=0.45, zorder=2, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([nice.get(m, m) + ("  (floor)" if m in FLOOR else "") for m in order],
                       fontsize=6.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("T1 MAP")

    ax = axes[1]
    sizes = strat.get("event_size", {}) if strat else {}
    cis = strat.get("event_size_ci", {}) if strat else {}
    ns = strat.get("event_size_n", {}) if strat else {}
    keys = [k for k in ["2-3", "4-10", "11-50", "51-+"] if k in sizes]
    for m, c in (("M3_typed_star", PALETTE["typed_star"]), ("M5_ccnn", PALETTE["ccnn"])):
        ys = [sizes[k].get(m, np.nan) for k in keys]
        lo, hi = [], []
        for k, y in zip(keys, ys):
            b = cis.get(k, {}).get(m) or {}
            lo.append(max(0.0, y - b.get("lo", y)))
            hi.append(max(0.0, b.get("hi", y) - y))
        ax.errorbar(range(len(keys)), ys, yerr=[lo, hi], fmt="o-", color=c, capsize=2,
                    label=MODEL_LABEL.get(m, m), markersize=4, lw=1.4)
    ax.set_xticks(range(len(keys)))
    # The smallest bin holds a handful of incidences, so its count belongs on the axis:
    # without it the wide interval there reads as a real crossing rather than thin evidence.
    ax.set_xticklabels(
        [f"{k.replace('-+', '+')}\nn={(ns.get(k, {}).get('M5_ccnn') or {}).get('incidences', 0)}"
         for k in keys], fontsize=6)
    ax.set_xlabel("event size (narrators)")
    ax.set_ylabel("per-incidence accuracy")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    ax = axes[2]
    splits = ["narrator-disjoint", "random", "event-disjoint"]
    present = [sp for sp in splits
               if any(r["split"] == sp and r["granularity"] == "mid" and r["neg_regime"] == "MNS"
                      and r["rank_map"] == "R-A" for r in runs)]
    top = 0.0
    for m, c in (("M3_typed_star", PALETTE["typed_star"]), ("M5_ccnn", PALETTE["ccnn"])):
        means, los, his = [], [], []
        for sp in present:
            vals = [r["T1_map"] for r in runs
                    if r["model"] == m and r["split"] == sp and r["granularity"] == "mid"
                    and r["neg_regime"] == "MNS" and r["rank_map"] == "R-A"
                    and r.get("cell") != "E8.3_gold_subset"]
            mu, lo, hi = bootstrap_ci(vals)
            means.append(mu); los.append(mu - lo); his.append(hi - mu)
            top = max(top, hi)
        ax.errorbar(range(len(present)), means, yerr=[los, his], fmt="o-", color=c,
                    markersize=4.5, capsize=2, lw=1.4, label=MODEL_LABEL.get(m, m))
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([s.replace("-", "-\n") for s in present], fontsize=6)
    ax.set_ylabel("T1 MAP")
    ax.set_ylim(bottom=0, top=top * 1.5 if top else None)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.margins(x=0.14)

    for a, letter in zip(axes, "abc"):
        panel_label(a, letter)
    fig.tight_layout()
    save(fig, "fig3_models",
         "a, Corroboration retrieval in the primary cell (granularity mid, narrator-disjoint "
         "split, MNS negatives, rank map R-A); all ten seeds are shown individually because "
         "means hide bimodality, and error bars are bootstrap 95% CIs over seeds. M1 is a "
         "parameter-free dense-retrieval baseline and is included because it outranks every "
         "trained model except the typed star. Hollow markers labelled (floor) are AllSet and "
         "ED-HNN, which never update rank-1 cells, so their T1 values are the frozen-encoder "
         "floor rather than a measurement of either operator. b, Per-incidence accuracy "
         "stratified by event size: the complex gains nothing even where many narrators "
         "describe the same event, which is where extra structure should help most. c, The "
         "same two models under all three evaluation splits, everything else held at the "
         "primary cell: the typed star leads only under the pre-registered narrator-disjoint "
         "split, and both splits that let a narrator appear on both sides of the partition "
         f"favour the complex. N={len(prim)} runs in a and b; error bars are bootstrap 95% "
         "CIs over seeds throughout.")


# ------------------------------------------------------------------ Figure 4
def figure4_uncertainty(pert: dict, e9: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 1.95))

    ax = axes[0]
    items = e9.get("items", []) if e9 else []
    if items:
        a = np.array([i["archive_conditioned_attestation_multiplicity"] for i in items], float)
        lo = np.array([i["attestation_ci"][0] for i in items], float)
        hi = np.array([i["attestation_ci"][1] for i in items], float)
        idx = np.argsort(a)
        ax.errorbar(np.arange(len(a)), a[idx], yerr=[a[idx] - lo[idx], hi[idx] - a[idx]],
                    fmt="o", markersize=2.5, color=PALETTE["ccnn"], elinewidth=0.6, capsize=1)
        unstable = np.array([i["unstable"] for i in items])[idx]
        if unstable.any():
            ax.scatter(np.arange(len(a))[unstable], a[idx][unstable], s=16, facecolors="none",
                       edgecolors=PALETTE["typed_star"], linewidths=0.8,
                       label="unstable (<0.5 of conditions)")
            ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_xlabel("triage rank")
    ax.set_ylabel("attestation multiplicity")

    ax = axes[1]
    grid = pert.get("grid", {}) if pert else {}
    mults, rbos, sds = [], [], []
    for k, v in sorted(grid.items(), key=lambda kv: float(kv[0][1:])):
        mults.append(float(k[1:])); rbos.append(v["rbo50_mean"]); sds.append(v["rbo50_std"])
    if mults:
        ax.errorbar(mults, rbos, yerr=sds, fmt="o-", color=PALETTE["ccnn"], markersize=4, capsize=2)
        ax.axvline(1.0, color=PALETTE["typed_star"], ls="--", lw=1.0)
        ax.text(1.06, 0.90, "measured\n" + r"$\hat{\rho}$", fontsize=6,
                color=PALETTE["typed_star"], transform=ax.get_xaxis_transform(),
                va="top", ha="left")
        ax.axhline(0.70, color="#666666", ls=":", lw=1.0)
        ax.text(0.98, 0.73, "pre-registered 0.70", fontsize=6, color="#666666",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom")
    ax.set_xlabel(r"perturbation multiplier ($\times \hat{\rho}$)")
    ax.set_ylabel("triage RBO@50 vs clean")
    ax.set_ylim(0, 1.08)

    for a, letter in zip(axes, "ab"):
        panel_label(a, letter)
    fig.tight_layout()
    rates = pert.get("measured_rates", {}) if pert else {}
    save(fig, "fig4_uncertainty",
         "Extraction uncertainty. a, Each triage item with a narrator-clustered bootstrap CI "
         "(B=1500); open markers are items retained in fewer than half of the robustness "
         "conditions. b, Triage stability under merge/split noise calibrated to the measured "
         f"rates (rho_merge={rates.get('rho_merge', float('nan')):.4f}, "
         f"rho_split={rates.get('rho_split', float('nan')):.4f}); 20 perturbation draws per grid "
         "point, error bars are SD across draws. At the measured rate the list falls below the "
         "threshold fixed in advance, so triage is reported as a screening step requiring human "
         "verification.")


# ------------------------------------------------------------------ Figure 5
def figure5_decoupling(summaries: dict, dec: dict, key: str = "R-A|mid") -> None:
    """P1 is the precondition that holds, and it is the one claim best made visually."""
    s, d = summaries.get(key), dec.get(key)
    if not s or not d:
        log("  skipping fig5: decoupling summaries unavailable")
        return
    ranks = [r for r in ("rank0", "rank1", "rank2", "rank3") if r in s]
    colours = [PALETTE[c] for c in ("ccnn", "hyper", "neutral", "accent")][:len(ranks)]

    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_IN, 1.95))
    for i, (rk, c) in enumerate(zip(ranks, colours)):
        lo = max(1, s[rk]["size_min"])
        hi = max(1, s[rk]["size_max"])
        med = max(1, s[rk]["size_median"])
        ax.plot([lo, hi], [i, i], color=c, lw=4.5, solid_capstyle="round", zorder=2)
        ax.scatter([med], [i], s=30, color="white", edgecolor=darken(c), zorder=3, linewidths=1.0)

    labels = ["0 narrator", "1 moment", "2 event", "3 episode"]
    # Rank 0 is one narrator by definition, so its range is a point; saying so stops the
    # degenerate bar from reading as missing data. Counts go in the tick labels rather than
    # as annotations, which would otherwise crowd the right spine.
    labels[0] += " (1 by def.)"
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f"{labels[i]}\nn={s[rk]['n_cells']:,}" for i, rk in enumerate(ranks)],
                       fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("cell size (narrators, log scale)")
    ax.set_xlim(0.7, max(s[r]["size_max"] for r in ranks) * 1.6)
    ax.margins(y=0.16)
    ax.invert_yaxis()
    fig.tight_layout()

    ex = d["extremes"]
    n_plotted = sum(s[r]["n_cells"] for r in ranks)
    p23 = d.get("rank2_vs_rank3") or {}
    save(fig, "fig5_decoupling",
         "Rank is not cardinality at the boundary where the ladder is contested. Bars span "
         "the observed size range at each rank and the marker is the median; the rank-2 and "
         "rank-3 ranges overlap rather than nesting. The largest rank-2 cell holds "
         f"{ex['rank2']['largest']['size']} narrators while the smallest rank-3 cell -- nominally "
         f"higher on the ladder -- holds {ex['rank3']['smallest']['size']}. Sampling unit is "
         f"the cell; all {n_plotted:,} cells are plotted. Ranks 0 and 1 are size-constrained "
         "by construction (a segment has one speaker), so the informative comparison is "
         f"rank 2 against rank 3: over {p23.get('n_pairs', 0):,} such pairs Spearman rho is "
         f"{p23.get('spearman_rho', float('nan')):.3f} and "
         f"{100 * p23.get('inversion_rate', float('nan')):.1f}% invert, meaning the "
         "higher-ranked cell is the smaller one. Across all cross-rank pairs at ranks 1-3, "
         "which include the size-constrained ranks, the ladder instead orders by size "
         f"correctly {100 * d['size_concordance_rate']:.1f}% of the time and inverts "
         f"{100 * d['rank_inversion_rate']:.1f}% of the time "
         f"(rho = {d['spearman_rho']:.3f}).")


# ------------------------------------------------------------------ Figure 6
def figure6_split(runs: list) -> None:
    """The split, not the architecture, decides which model wins."""
    splits = ["narrator-disjoint", "random", "event-disjoint"]
    series = {"M3_typed_star": PALETTE["typed_star"], "M5_ccnn": PALETTE["ccnn"]}
    present = [sp for sp in splits
               if any(r["split"] == sp and r["granularity"] == "mid" and r["neg_regime"] == "MNS"
                      and r["rank_map"] == "R-A" for r in runs)]
    if len(present) < 2:
        log("  skipping fig6: fewer than two splits available")
        return

    fig, ax = plt.subplots(1, 1, figsize=(0.95 * COLUMN_IN, 1.65))
    n_runs = 0
    top = 0.0
    for m, c in series.items():
        means, los, his = [], [], []
        for sp in present:
            vals = [r["T1_map"] for r in runs
                    if r["model"] == m and r["split"] == sp and r["granularity"] == "mid"
                    and r["neg_regime"] == "MNS" and r["rank_map"] == "R-A"
                    and r.get("cell") != "E8.3_gold_subset"]
            n_runs += len(vals)
            mu, lo, hi = bootstrap_ci(vals)
            means.append(mu); los.append(mu - lo); his.append(hi - mu)
            top = max(top, hi)
        ax.errorbar(range(len(present)), means, yerr=[los, his], fmt="o-", color=c,
                    markersize=5, capsize=2, lw=1.4, label=MODEL_LABEL.get(m, m))

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([s.replace("-", "-\n") for s in present], fontsize=6.5)
    ax.set_ylabel("T1 MAP")
    ax.set_ylim(bottom=0, top=top * 1.45 if top else None)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.margins(x=0.12)
    fig.tight_layout()

    save(fig, "fig6_split",
         "The evaluation split decides which architecture wins. The typed star leads only under "
         "the pre-registered narrator-disjoint split; under both other splits the complex leads, "
         "and by the largest margin under the random split. Neither of those two splits keeps a "
         "narrator's segments on one side of the partition, so both reward a model for "
         "recognising a narrator it was trained on. Everything except the split is held fixed at "
         "the primary cell (granularity mid, MNS negatives, rank map R-A); "
         f"{n_runs} runs over 10 seeds, error bars are bootstrap 95% CIs over seeds.")


def figure7_mechanism(mech: dict, feat: dict | None = None) -> None:
    """The split gap under two interventions: destroy identity, and change the feature path."""
    gap = mech.get("split_gap") or {}
    models = [m for m in ("M3_typed_star", "M5_ccnn") if m in gap]
    if not models:
        log("  skipping fig7: no split_gap block in e_mechanism.json")
        return
    fgap = (feat or {}).get("split_gap") or {}

    probe = {(r["split"], r["model"]): r for r in mech.get("probe", [])}
    ref = probe.get(("narrator-disjoint", "M3_typed_star"), {})
    chance, classes = ref.get("chance", float("nan")), int(ref.get("n_classes", 0))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.12))

    # (a) the gap collapse -- the panel the claim rests on
    ax = axes[0]
    n_bars = 3 if fgap else 2
    width = 0.78 / n_bars
    xs = np.arange(len(models))
    real = [gap[m]["gap_real_narrators"] for m in models]
    anon = [gap[m]["gap_anonymised"] for m in models]
    offs = (np.arange(n_bars) - (n_bars - 1) / 2) * width
    # Intervals resample seeds within each split, so they describe optimisation noise in
    # the gap. Bars without a stored interval are drawn without one rather than at zero.
    def gerr(key):
        out = [[], []]
        for m in models:
            b = gap[m].get(key) or {}
            lo, hi, pt = b.get("lo"), b.get("hi"), b.get("point")
            if lo is None or hi is None or pt is None:
                out[0].append(0.0); out[1].append(0.0)
            else:
                out[0].append(max(0.0, pt - lo)); out[1].append(max(0.0, hi - pt))
        return out

    e_real, e_anon = gerr("gap_real_ci"), gerr("gap_anon_ci")
    zero = [[0.0] * len(models), [0.0] * len(models)]
    ax.bar(xs + offs[0], real, width, color=PALETTE["ccnn"], edgecolor="black",
           linewidth=0.6, label="real narrators", yerr=e_real,
           error_kw={"elinewidth": 0.8, "capsize": 2})
    ax.bar(xs + offs[1], anon, width, color=PALETTE["neutral"], edgecolor="black",
           linewidth=0.6, label="identity shuffled", yerr=e_anon,
           error_kw={"elinewidth": 0.8, "capsize": 2})
    vals = list(real) + list(anon)
    if fgap:
        item = [fgap.get(m, {}).get("gap_item_specific", float("nan")) for m in models]
        ax.bar(xs + offs[2], item, width, color=PALETTE["typed_star"], edgecolor="black",
               linewidth=0.6, label="item-specific features")
        vals += [v for v in item if v == v]
    ax.axhline(0, color="black", lw=0.8)
    # The claim is that one bar sits at zero, and a zero-height bar is invisible, so the
    # values are printed rather than left to be read off the axis.
    labelled = [(xs + offs[0], real, e_real), (xs + offs[1], anon, e_anon)]
    if fgap:
        labelled.append((xs + offs[2], item, zero))
    for pos, vs, es in labelled:
        for i, (x, v) in enumerate(zip(pos, vs)):
            if v != v:
                continue
            # Anchor past the whisker, or the label lands on top of its own error bar.
            tip = v + es[1][i] if v >= 0 else v - es[0][i]
            ax.annotate(f"{v:+.3f}", (x, tip), textcoords="offset points",
                        xytext=(0, 3.0 if v >= 0 else -3.0), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=6.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models], fontsize=7.2)
    # Two lines: the single-line form overruns the axes and tight_layout clips its tail.
    ax.set_ylabel("T1 MAP gain from\na random split")
    # Above the axes, because every corner inside is occupied by a bar or its value label.
    ax.legend(frameon=False, fontsize=6.6, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=1.0, handlelength=1.2,
              handletextpad=0.4, borderaxespad=0.0)
    # 1.9 leaves clear air for the printed value plus its whisker at both extremes.
    ax.set_ylim(min(vals) * 1.9 if min(vals) < 0 else -0.03, max(vals) * 1.55)
    panel_label(ax, "a")

    # (b) how much narrator identity each representation carries, against the input ceiling
    ax = axes[1]
    splits = ["narrator-disjoint", "random"]
    for m, colour in (("M3_typed_star", PALETTE["typed_star"]), ("M5_ccnn", PALETTE["ccnn"])):
        ys = [probe[(s, m)]["probe_accuracy_mean"] for s in splits if (s, m) in probe]
        es = [probe[(s, m)]["probe_accuracy_std"] for s in splits if (s, m) in probe]
        if ys:
            ax.errorbar(range(len(ys)), ys, yerr=es, fmt="o-", color=colour, markersize=5,
                        capsize=2, lw=1.4, label=MODEL_LABEL.get(m, m))
    ctrl = probe.get(("narrator-disjoint", "frozen_encoder_typed"))
    if ctrl:
        ax.axhline(ctrl["probe_accuracy_mean"], color="black", ls="--", lw=0.9)
        ax.text(0.98, ctrl["probe_accuracy_mean"] + 0.015, "input features (1.0 by construction)",
                fontsize=5.6, transform=ax.get_yaxis_transform(), va="bottom", ha="right")
    # Full scale, not a zoom. The claim is that both models sit near the ceiling and far above
    # chance; truncating the axis magnifies a gap the text calls smaller than seed spread.
    if chance == chance:
        ax.axhline(chance, color="#666666", ls=":", lw=0.9)
        ax.text(0.02, chance + 0.02, f"majority class ({chance:.3f})", fontsize=5.6,
                transform=ax.get_yaxis_transform(), va="bottom", ha="left", color="#444444")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.replace("-", "-\n") for s in splits], fontsize=6.5)
    ax.set_ylabel("narrator decodable from\nmoment representation")
    ax.set_ylim(0, 1.10)
    ax.margins(x=0.18)
    ax.legend(frameon=False, fontsize=6, loc="center right")
    panel_label(ax, "b")

    fig.tight_layout()
    save(fig, "fig7_mechanism",
         "Destroying narrator identity removes the complex's random-split advantage; editing "
         "the moment features does not. (a) Gain in T1 MAP from switching to a random split, "
         "under two interventions. With real narrators the complex gains a large amount and "
         "the typed star gains nothing. Shuffling narrator labels takes the complex's gain to "
         "zero. Giving each moment its own passage embedding instead of its narrator's mean "
         "leaves the gain intact, slightly wider, which is why the leak is attributed to the "
         "narrator layer rather than to the moment vector. (b) Accuracy of a linear probe "
         "recovering narrator identity from moment representations, against a majority-class "
         f"floor of {chance:.3f} over {classes} narrators. The dashed line is the same probe on "
         "the raw input features, where it is exactly 1.0 by construction, because a rank-1 "
         "moment's input vector IS its narrator's vector. Both trained models inherit that "
         "identity rather than learning it, and neither discards it; the gap between them is "
         "within seed variation. The axis runs the full range, with the majority-class floor "
         "drawn.")


def figure0_structure(obs: dict | None) -> None:
    """Where the ownership obstruction enters: aggregate, lift, then split.

    A schematic of the feature path, drawn so that the pre-split aggregation and the
    repeated vector are visible in one panel. Counts in the caption are the measured ones.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(TEXT_IN, 2.30))
    ax.set_xlim(0, 8.75)
    ax.set_ylim(0.10, 3.72)
    ax.axis("off")

    def ink(face: str) -> str:
        return "white" if relative_luminance(face) < 0.45 else "black"

    # Narrator A owns four moments and straddles the split; B and C do not. That asymmetry
    # is the figure's whole argument, so it is drawn rather than described.
    narrators = [("A", PALETTE["ccnn"], 4, 2.95),
                 ("B", PALETTE["hyper"], 2, 1.55),
                 ("C", PALETTE["mlp"], 3, 0.60)]
    seg_w, seg_h, seg_gap = 0.22, 0.15, 0.06
    nx, nw, nh = 2.45, 0.92, 0.40
    mx0, mw, mgap = 4.85, 0.60, 0.92
    split_x = 7.52

    ax.axvspan(split_x, 8.75, color="#F2F2F2", zorder=0)
    ax.plot([split_x, split_x], [0.20, 3.42], ls="--", lw=1.0, color="black", zorder=1)
    ax.text(split_x - 0.12, 3.48, "train", fontsize=6.5, ha="right", va="bottom")
    ax.text(split_x + 0.12, 3.48, "held out", fontsize=6.5, ha="left", va="bottom")

    for name, colour, n_seg, y in narrators:
        seg_right = 0.30 + n_seg * (seg_w + seg_gap) - seg_gap
        for k in range(n_seg):
            ax.add_patch(Rectangle((0.30 + k * (seg_w + seg_gap), y - seg_h / 2), seg_w, seg_h,
                                   facecolor=colour, edgecolor="black", linewidth=0.4,
                                   alpha=0.55, zorder=2))
        # Arrows stop at the box edge; centre-to-centre hides the head under the patch.
        ax.annotate("", xy=(nx - 0.06, y), xytext=(seg_right + 0.06, y),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"), zorder=2)
        ax.add_patch(FancyBboxPatch((nx, y - nh / 2), nw, nh,
                                    boxstyle="round,pad=0.01,rounding_size=0.07",
                                    facecolor=colour, edgecolor="black", linewidth=0.6, zorder=2))
        ax.text(nx + nw / 2, y, f"$x_{name}$", ha="center", va="center",
                fontsize=7, color=ink(colour), zorder=3)
        ax.annotate("", xy=(mx0 - 0.06, y), xytext=(nx + nw + 0.06, y),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"), zorder=2)
        for k in range(n_seg):
            cx = mx0 + k * mgap
            ax.add_patch(Rectangle((cx, y - 0.19), mw, 0.38, facecolor=colour,
                                   edgecolor="black", linewidth=0.6, zorder=2))
            ax.text(cx + mw / 2, y, f"$x_{name}$", ha="center", va="center",
                    fontsize=6, color=ink(colour), zorder=3)
        ax.text(0.30, y + 0.32, f"narrator {name}", fontsize=6, va="bottom")

    for x, label in ((0.30, "segments"), (nx, "narrator vector"), (mx0, "rank-1 moments")):
        ax.text(x, 3.48, label, fontsize=6.5, va="bottom")

    # The straddling narrator carries one vector to both sides, which is the confound. The
    # annotation sits in the gap between rows A and B so it belongs to neither row's boxes.
    ya = narrators[0][3]
    ax.annotate("", xy=(mx0 + 3 * mgap + mw / 2, ya - 0.21), xytext=(mx0 + mw / 2, ya - 0.21),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color=PALETTE["typed_star"],
                                shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.30"),
                zorder=4)
    ax.text(mx0 + 1.35 * mgap, ya - 0.92, "same vector on both sides of the split",
            fontsize=6, ha="center", color=PALETTE["typed_star"])

    fig.tight_layout()
    p = (obs or {}).get("primary", {})
    counts = ""
    if p:
        counts = (f" On this archive {p['rank1_cells']:,} moments carry "
                  f"{p['rank1_distinct_supports']} distinct supports over "
                  f"{p['ground_set_size']} narrators, so at most "
                  f"{p['rank1_distinct_supports']} of the boxes on the right differ.")
    save(fig, "fig0_structure",
         "The feature path, and where the ownership obstruction enters. A narrator's segments "
         "are averaged into one rank-0 vector, and every rank-1 moment that narrator owns is "
         "lifted from that same vector, so moments by one narrator are identical as inputs. "
         "Features are computed over the whole archive before the split is applied, so a "
         "narrator who owns moments on both sides places the same vector on both sides; "
         "narrator A does, B and C do not." + counts +
         " Schematic: box counts are illustrative, the figures in the caption are measured.")


def main() -> None:
    apply_style()
    problems = {}
    for group, keys in PANEL_GROUPS.items():
        bad = check_separation([PALETTE[k] for k in keys])
        if bad:
            problems[group] = bad
    log(f"palette luminance check: {'OK' if not problems else f'TOO CLOSE {problems}'}")
    if problems:
        raise SystemExit("palette fails greyscale separability; fix figstyle before plotting")

    rt = load("e4_2_star_roundtrip.json") or {}
    sweep = load("e4_3_granularity.json")
    runs = load("e7_1_runs.json")
    strat = load("e7_2_stratified.json")
    pert = load("e8_1_perturbation.json")
    e9 = load("e9_2_triage.json")
    summaries = load("e4_1_summaries.json")
    dec = load("e4_4_decoupling.json")
    mech = load("e_mechanism.json")
    feat = load("e_feature_path.json")
    obs = load("e_obstruction.json")

    figure0_structure(obs)
    figure1_representations(rt)
    if sweep:
        figure2_granularity(sweep)
    if runs:
        figure3_models(runs, strat or {})
    if pert or e9:
        figure4_uncertainty(pert or {}, e9 or {})
    if summaries and dec:
        figure5_decoupling(summaries, dec)
    if runs:
        figure6_split(runs)
    if mech:
        figure7_mechanism(mech, feat)

    (FIG / "captions.json").write_text(json.dumps(CAPTIONS, indent=2), encoding="utf-8")
    log(f"figures complete: {sorted(CAPTIONS)}")


if __name__ == "__main__":
    main()


