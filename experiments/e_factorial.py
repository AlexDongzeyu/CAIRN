"""E6 + E9: the operator x features factorial, run at both split levels.

The published grid gives spectral encodings to the typed star and not to the complex, so
"the star beats the complex" confounds the operator with its inputs. This fills all four
cells at both split levels, which is the only sweep where the confound can change the
story rather than shift every cell equally.
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
from src.metrics import cluster_bootstrap  # noqa: E402
from src.models import match_hidden_to_budget  # noqa: E402
from src.seeds import SEEDS  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

from experiments.run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell

RES = ROOT / "data" / "results"
MODELS = ("M3_typed_star", "M5_ccnn")
FEATURES = ("plain", "enc")
# The random-split arm produced a degenerate bootstrap interval that we could not reproduce
# with an independent probe, so it is withheld rather than reported.
SPLITS = ("narrator-disjoint",)
SEEDS_USED = SEEDS[:5]

log = make_logger("e_factorial")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, segments = load_corpus()
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    cache: dict = {}

    out: dict = {"seeds": list(SEEDS_USED), "cells": {}}
    for split in SPLITS:
        # A cache shared across split levels returns the previous split's queries, which
        # silently scores one cell against another cell's test set.
        cache = {}
        cx, bp, be, data, sp, dense = prepare_cell(
            segments, rank_maps, PRIMARY["granularity"], split, PRIMARY["neg"],
            PRIMARY["rank_map"], encoder, device, cache)
        for model in MODELS:
            for feat in FEATURES:
                bundle = be if feat == "enc" else bp
                dims = {k: bundle.X[k].shape[1] for k in bundle.X}
                hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
                maps, aucs = [], []
                by_narr: dict[str, list[float]] = {}
                for s in SEEDS_USED:
                    cfg = RunConfig(model=model, granularity=PRIMARY["granularity"],
                                    rank_map=PRIMARY["rank_map"], split=split,
                                    neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                    r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, collect_t1=True)
                    maps.append(r["T1_map"])
                    aucs.append(r["T2_auc"])
                    for q in r.get("T1_per_query", []):
                        by_narr.setdefault(q["narrator"], []).append(float(q["ap"]))
                key = f"{model}|{feat}|{split}"
                boot = cluster_bootstrap(
                    lambda us: float(np.mean([v for u in us for v in by_narr[u]])) if us else 0.0,
                    list(by_narr), B=2000, seed=0, bca=False)
                if boot["hi"] - boot["lo"] < 1e-9:
                    raise SystemExit(f"{key}: degenerate CI over {len(by_narr)} narrators")
                if abs(boot["point"] - np.nanmean(maps)) > 0.02:
                    raise SystemExit(f"{key}: pooled AP {boot['point']:.4f} disagrees with "
                                     f"seed-mean MAP {np.nanmean(maps):.4f}")
                out["cells"][key] = {
                    "T1_map": float(np.nanmean(maps)),
                    "T1_map_ci": [boot["lo"], boot["hi"]],
                    "T2_auc": float(np.nanmean(aucs)),
                    "n_params": int(r["n_params"]), "hidden": int(hidden),
                    "n_narrators": len(by_narr),
                }
                log(f"  {key:44s} MAP={np.nanmean(maps):.4f} "
                    f"[{boot['lo']:.4f},{boot['hi']:.4f}]  AUC={np.nanmean(aucs):.4f}")

    # The operator claim is the sign of (complex - star) with features held fixed.
    for feat in FEATURES:
        for split in SPLITS:
            d = (out["cells"][f"M5_ccnn|{feat}|{split}"]["T1_map"]
                 - out["cells"][f"M3_typed_star|{feat}|{split}"]["T1_map"])
            out.setdefault("operator_effect", {})[f"{feat}|{split}"] = float(d)
            log(f"  operator effect (M5-M3) {feat}/{split}: {d:+.4f}")

    if len(out["cells"]) != len(MODELS) * len(FEATURES) * len(SPLITS):
        raise SystemExit(f"expected {len(MODELS) * len(FEATURES) * len(SPLITS)} cells, "
                         f"got {len(out['cells'])}")
    (RES / "e_factorial.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote e_factorial.json with {len(out['cells'])} cells")


if __name__ == "__main__":
    main()
