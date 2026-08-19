"""Recompute E7.5 on the primary cell alone.

The original fit filtered `all_items` on model only, but `all_items` accumulates per-item
records from every cell in the grid: coarse and fine granularities, the random and
event-disjoint splits, four alternative negative regimes, two alternative rank maps, and
the E8.3 gold subset. That is why it reported 343 events against a complex that has 141
rank-2 cells at the primary granularity.

The pooling is not merely a labelling problem. The random split is in the pool, and that is
the one split where the ordering this coefficient describes is reversed, so the estimate
mixed two populations that disagree about its sign.

Existing records carry no cell tags, so they cannot be filtered after the fact. This re-runs
the primary cell for the two models the coefficient compares and refits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.metrics import cluster_bootstrap  # noqa: E402
from src.models import match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.stats import interaction_model, stratify  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("e7_5_recompute")
MODELS = ("M5_ccnn", "M3_typed_star")
SEEDS = tuple(range(10))


def stratified_with_uncertainty(items: list[dict]) -> dict:
    """Per-bin accuracy with narrator-clustered CIs and the bin count behind each point.

    The point estimate keeps its original key so existing consumers still read a float;
    counts and intervals sit alongside it. Narrators are the resampling unit here for the
    same reason as E7.4: incidences by one narrator are not independent.
    """
    strat: dict = {}
    for key in ("event_size", "rank"):
        buckets = stratify(items, key)
        strat[key] = {}
        strat[f"{key}_ci"] = {}
        strat[f"{key}_n"] = {}
        for b, rows in buckets.items():
            strat[key][b], strat[f"{key}_ci"][b], strat[f"{key}_n"][b] = {}, {}, {}
            for m in MODELS:
                rs = [x for x in rows if x["model"] == m]
                if not rs:
                    continue
                by_narr: dict[str, list[float]] = {}
                for r in rs:
                    by_narr.setdefault(r["narrator_id"], []).append(float(r["correct"]))
                boot = cluster_bootstrap(
                    lambda us: float(np.mean([v for u in us for v in by_narr[u]])) if us else 0.0,
                    list(by_narr), B=2000, seed=0, bca=False)
                strat[key][b][m] = float(np.mean([x["correct"] for x in rs]))
                strat[f"{key}_ci"][b][m] = {"lo": boot["lo"], "hi": boot["hi"]}
                strat[f"{key}_n"][b][m] = {
                    "incidences": len(rs) // len(SEEDS),
                    "narrators": len(by_narr),
                    "rows": len(rs)}
    return strat


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segments = load_corpus()[1]
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    log(f"device={device} refitting E7.5 on the primary cell only: {PRIMARY}")

    cx, bp, be, data, _sp, _dense = prepare_cell(
        segments, rank_maps, PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
        PRIMARY["rank_map"], encoder, device, {})
    log(f"primary complex: {len(cx.by_rank(2))} rank-2 cells, {len(cx.narrators)} narrators")

    items: list[dict] = []
    for m in MODELS:
        bundle = be if m == "M3_typed_star" else bp
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(m, dims, bundle, PARAM_BUDGET)
        for s in SEEDS:
            set_all_seeds(s)
            cfg = RunConfig(model=m, granularity=PRIMARY["granularity"],
                            rank_map=PRIMARY["rank_map"], split=PRIMARY["split"],
                            neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
            out = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
            pit = out.pop("per_item", [])
            for r in pit:
                r["model"] = m
                r["seed"] = s
                r["granularity"] = PRIMARY["granularity"]
                r["split"] = PRIMARY["split"]
                r["neg_regime"] = PRIMARY["neg"]
                r["rank_map"] = PRIMARY["rank_map"]
            items.extend(pit)
        log(f"  {m:16s} {len(SEEDS)} seeds done, {len(items)} cumulative per-item rows")

    inter = interaction_model(items, "M5_ccnn", "M3_typed_star")
    prev = json.loads((RES / "e7_5_interaction.json").read_text(encoding="utf-8"))
    inter["superseded"] = {
        "reason": ("the earlier fit filtered on model only and pooled every cell in the "
                   "grid, including the random split where this ordering reverses"),
        "n_events": prev.get("n_events"),
        "n_narrators": prev.get("n_narrators"),
        "n_obs": prev.get("n_obs"),
        "interaction_coef": prev.get("interaction_coef"),
        "interaction_ci": prev.get("interaction_ci"),
        "interaction_p": prev.get("interaction_p"),
    }
    inter["scope"] = ("primary cell only: "
                      f"granularity={PRIMARY['granularity']}, split={PRIMARY['split']}, "
                      f"negatives={PRIMARY['neg']}, rank_map={PRIMARY['rank_map']}")
    (RES / "e7_5_interaction.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")

    # Persist the records so the stratified bands never need another retrain to reproduce.
    (RES / "e7_per_item_primary.json").write_text(json.dumps(items), encoding="utf-8")
    strat = stratified_with_uncertainty(items)
    (RES / "e7_2_stratified.json").write_text(json.dumps(strat, indent=2), encoding="utf-8")
    for b, d in strat["event_size_n"].items():
        for m, n in d.items():
            lo = strat["event_size_ci"][b][m]["lo"]
            hi = strat["event_size_ci"][b][m]["hi"]
            log(f"  STRAT {b:>8s} {m:16s} acc={strat['event_size'][b][m]:.3f} "
                f"CI=[{lo:.3f},{hi:.3f}] n={n['incidences']} inc / {n['narrators']} narr")

    log(f"E7.5 refit: n_events={inter.get('n_events')} (was {prev.get('n_events')}), "
        f"n_narrators={inter.get('n_narrators')}, n_obs={inter.get('n_obs')}")
    log(f"  coef={inter.get('interaction_coef')} CI={inter.get('interaction_ci')} "
        f"p={inter.get('interaction_p')} crosses_zero={inter.get('ci_crosses_zero')}")
    log(f"  was  coef={prev.get('interaction_coef')} CI={prev.get('interaction_ci')}")


if __name__ == "__main__":
    main()
