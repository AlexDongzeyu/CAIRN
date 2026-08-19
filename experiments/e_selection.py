"""E5: does the model ordering survive reselecting checkpoints by validation MAP?

Checkpoints in the published runs are chosen by validation AUC while the headline task is
retrieval. For the complex that criterion sits at chance, so the selected epoch is close to
arbitrary. Each run here tracks both criteria simultaneously, which makes the comparison
paired: same seed, same trajectory, two checkpoints.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import match_hidden_to_budget  # noqa: E402
from src.seeds import SEEDS  # noqa: E402
from src.tasks import build_t1_queries  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

from experiments.run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell

RES = ROOT / "data" / "results"
MODELS = ("M3_typed_star", "M5_ccnn")
# The primary cell is where the registered claim lives; the random arm's T1 pool makes the
# per-epoch MAP evaluation prohibitive.
SPLITS = ("narrator-disjoint",)
SEEDS_USED = SEEDS[:5]
VAL_QUERY_LIMIT = 150

log = make_logger("e_selection")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, segments = load_corpus()
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    cache: dict = {}

    out: dict = {"seeds": list(SEEDS_USED), "val_query_limit": VAL_QUERY_LIMIT, "cells": {}}
    for split in SPLITS:
        cx, bp, be, data, sp, dense = prepare_cell(
            segments, rank_maps, PRIMARY["granularity"], split, PRIMARY["neg"],
            PRIMARY["rank_map"], encoder, device, cache)
        vq = build_t1_queries(cx, segments, sp, bucket="val", limit=VAL_QUERY_LIMIT)
        log(f"{split}: {len(vq)} validation queries")
        for model in MODELS:
            bundle = be if model == "M3_typed_star" else bp
            dims = {k: bundle.X[k].shape[1] for k in bundle.X}
            hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
            acc: dict[str, list] = {k: [] for k in
                                    ("auc_sel", "map_sel", "ep_auc", "ep_map")}
            for s in SEEDS_USED:
                cfg = RunConfig(model=model, granularity=PRIMARY["granularity"],
                                rank_map=PRIMARY["rank_map"], split=split,
                                neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, val_queries=vq)
                acc["auc_sel"].append(r["T1_map"])
                acc["map_sel"].append(r.get("T1_map_mapsel", float("nan")))
                acc["ep_auc"].append(r["auc_selected_epoch"])
                acc["ep_map"].append(r["map_selected_epoch"])
            key = f"{model}|{split}"
            out["cells"][key] = {k: float(np.nanmean(v)) for k, v in acc.items()}
            log(f"  {key:34s} MAP(auc-sel)={np.nanmean(acc['auc_sel']):.4f}  "
                f"MAP(map-sel)={np.nanmean(acc['map_sel']):.4f}  "
                f"epoch {np.mean(acc['ep_auc']):.0f} vs {np.mean(acc['ep_map']):.0f}")

    for split in SPLITS:
        for crit in ("auc_sel", "map_sel"):
            d = (out["cells"][f"M5_ccnn|{split}"][crit]
                 - out["cells"][f"M3_typed_star|{split}"][crit])
            out.setdefault("ordering", {})[f"{crit}|{split}"] = float(d)
            log(f"  M5-M3 under {crit:8s} / {split:18s}: {d:+.4f}")

    (RES / "e_selection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote e_selection.json")


if __name__ == "__main__":
    main()
