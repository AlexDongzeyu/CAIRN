"""Measured cost of the three structural operations, for the scalable-computing claim.

The archive is not large, and the paper does not argue that it is. What the session's
scalable-computing remit does ask is what the machinery costs, so this measures the three
operations a reader would have to run on their own collection: auditing support injectivity,
expanding the complex to its typed star, and training the rank-aware network for one seed.

Peak memory is process resident-set growth around each operation, which is an upper bound on
the operation itself and is reported as such.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences, support_injectivity  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import build_features, get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.tasks import NegativeSampler, make_split, partition_incidences  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
PRIMARY = {"granularity": "mid", "neg": "MNS", "rank_map": "R-A", "split": "narrator-disjoint"}
PARAM_BUDGET = 130_000
log = make_logger("cost")


def timed(label: str, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    log(f"{label:22s} {dt:8.2f}s  peak {peak / 2**20:7.1f} MiB")
    return out, {"seconds": round(dt, 2), "peak_mib": round(peak / 2**20, 1)}


def main() -> None:
    set_all_seeds(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, segments = load_corpus()
    rm = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))["R-A_consensus"]
    encoder = get_encoder()

    cx, t_build = timed("build complex", lambda: build_complex(
        segments, rm, granularity=PRIMARY["granularity"], rank_map_name=PRIMARY["rank_map"]))
    _, t_audit = timed("support audit (r1)", lambda: support_injectivity(cx, 1))
    feats, t_feat = timed("feature construction", lambda: build_features(cx, segments, encoder))

    sp = make_split(cx, PRIMARY["split"], seed=0)
    parts = partition_incidences(cx, sp)
    train_cx = mask_incidences(cx, set(parts["train"]))
    _, t_star = timed("star expansion", lambda: make_bundle(train_cx, feats, device))

    bundle = make_bundle(train_cx, feats, device)
    ns = NegativeSampler(cx, seed=0)
    data = {p: ns.build(parts[p], PRIMARY["neg"], ratio=10) for p in ("train", "val", "test")}
    dims = {k: bundle.X[k].shape[1] for k in bundle.X}
    hidden = match_hidden_to_budget("M5_ccnn", dims, bundle, PARAM_BUDGET)
    cfg = RunConfig(model="M5_ccnn", granularity=PRIMARY["granularity"],
                    rank_map=PRIMARY["rank_map"], split=PRIMARY["split"],
                    neg_regime=PRIMARY["neg"], seed=0, hidden=hidden)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _, t_train = timed("M5 training (1 seed)",
                       lambda: train_eval(cfg, cx, bundle, data, device, eval_cx=cx))
    gpu_mib = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
               if device.type == "cuda" else None)

    payload = {
        "device": str(device),
        "n_segments": len(segments),
        "n_narrators": len(cx.narrators),
        "build_complex": t_build,
        "support_audit_rank1": t_audit,
        "feature_construction": t_feat,
        "star_expansion": t_star,
        "m5_training_one_seed": t_train,
        "m5_peak_gpu_mib": gpu_mib,
        "note": ("peak_mib is Python allocation tracked by tracemalloc around each operation, "
                 "an upper bound on that operation"),
    }
    (RES / "e_cost.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote e_cost.json  (M5 peak GPU {gpu_mib} MiB)")


if __name__ == "__main__":
    main()
