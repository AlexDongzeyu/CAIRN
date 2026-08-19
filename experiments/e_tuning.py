"""E7: equal-budget tuning for the two load-bearing models.

The registered protocol ran no hyperparameter search, which leaves the ordering open to
the objection that the complex was simply mis-tuned. The sweep is deliberately confined to
the pair the argument rests on, with an identical grid, budget and selection criterion for
both, so that whatever the complex gains the star had the same chance to gain.
"""
from __future__ import annotations

import itertools
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
from src.train import RunConfig, train_eval  # noqa: E402

from experiments.run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell

RES = ROOT / "data" / "results"
MODELS = ("M3_typed_star", "M5_ccnn")
LRS = (1e-3, 5e-3, 1e-2, 2e-2)
DROPOUTS = (0.0, 0.1)
SEEDS_USED = SEEDS[:3]

log = make_logger("e_tuning")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, segments = load_corpus()
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    cache: dict = {}

    cx, bp, be, data, sp, dense = prepare_cell(
        segments, rank_maps, PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
        PRIMARY["rank_map"], encoder, device, cache)

    out: dict = {"grid": {"lr": list(LRS), "dropout": list(DROPOUTS)},
                 "seeds": list(SEEDS_USED), "trials": {}, "best": {}}
    for model in MODELS:
        bundle = be if model == "M3_typed_star" else bp
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
        best = None
        for lr, dr in itertools.product(LRS, DROPOUTS):
            vals, maps = [], []
            for s in SEEDS_USED:
                cfg = RunConfig(model=model, granularity=PRIMARY["granularity"],
                                rank_map=PRIMARY["rank_map"], split=PRIMARY["split"],
                                neg_regime=PRIMARY["neg"], seed=s, hidden=hidden,
                                lr=lr, dropout=dr)
                r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
                vals.append(r["val_auc"])
                maps.append(r["T1_map"])
            key = f"{model}|lr={lr}|do={dr}"
            rec = {"val_auc": float(np.nanmean(vals)), "T1_map": float(np.nanmean(maps))}
            out["trials"][key] = rec
            log(f"  {key:36s} val_auc={rec['val_auc']:.4f}  T1_map={rec['T1_map']:.4f}")
            # Selection uses the registered criterion, not the test metric.
            if best is None or rec["val_auc"] > best[1]["val_auc"]:
                best = (key, rec, lr, dr)
        out["best"][model] = {"config": best[0], **best[1], "lr": best[2], "dropout": best[3]}
        log(f"  BEST {model}: {best[0]}  T1_map={best[1]['T1_map']:.4f}")

    d = out["best"]["M5_ccnn"]["T1_map"] - out["best"]["M3_typed_star"]["T1_map"]
    out["tuned_gap_M5_minus_M3"] = float(d)
    log(f"  tuned M5-M3 T1 MAP: {d:+.4f}")
    (RES / "e_tuning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote e_tuning.json")


if __name__ == "__main__":
    main()
