"""E10: partition replication. Resample the split itself, not just the optimiser seed.

Every result elsewhere in this paper is conditional on one realised partition per split type,
because `prepare_cell` fixes `make_split(..., seed=0)`. Ten optimiser seeds measure
optimisation noise and the narrator bootstrap measures query sampling, but neither answers
whether a different partition of this archive would reproduce the ordering. The headline
claim is that the partition decides the conclusion, so the partition is the thing to resample.

Matched (plain) inputs only: that is the causal operator comparison, with the feature
confound held out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import build_features, get_encoder, segment_embeddings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.seeds import SEEDS  # noqa: E402
from src.tasks import (  # noqa: E402
    NegativeSampler, build_t1_queries, make_split, partition_incidences,
)
from src.train import RunConfig, train_eval  # noqa: E402

from experiments.run_phase5_7 import (  # noqa: E402
    MAX_T1_QUERIES, PARAM_BUDGET, PRIMARY, load_rank_maps,
)

RES = ROOT / "data" / "results"
MODELS = ("M3_typed_star", "M5_ccnn")
SPLITS = ("narrator-disjoint", "random")
N_PARTITIONS = 10
SEEDS_USED = SEEDS[:3]

log = make_logger("e_partition")


def build_for_split(cx, feats, segments, split_kind, part_seed, device, sid2i):
    """Everything downstream of the partition, rebuilt for one realised split."""
    sp = make_split(cx, split_kind, seed=part_seed)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=0)
    data = {p: ns.build(parts[p], PRIMARY["neg"], ratio=10) for p in ("train", "val", "test")}

    # Message passing may only see training incidences.
    train_cx = mask_incidences(cx, set(parts["train"]))
    bundle = make_bundle(train_cx, feats, device)
    data["t1_queries"] = build_t1_queries(cx, segments, sp, bucket="test",
                                          limit=MAX_T1_QUERIES)
    data["seg_index"] = sid2i
    return bundle, data, len(data["t1_queries"])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, segments = load_corpus()
    rank_maps = load_rank_maps()
    encoder = get_encoder()

    cx = build_complex(segments, rank_maps[PRIMARY["rank_map"]],
                       granularity=PRIMARY["granularity"], rank_map_name=PRIMARY["rank_map"])
    feats = build_features(cx, segments, encoder)
    _, sid2i = segment_embeddings(segments, encoder)
    log(f"complex built once: {len(cx.cells)} cells, reused across all partitions")

    out: dict = {"n_partitions": N_PARTITIONS, "seeds": list(SEEDS_USED),
                 "features": "plain", "partitions": {}}

    for split in SPLITS:
        rows = []
        for p in range(N_PARTITIONS):
            bundle, data, n_q = build_for_split(cx, feats, segments, split, p, device, sid2i)
            dims = {k: bundle.X[k].shape[1] for k in bundle.X}
            per_model = {}
            for model in MODELS:
                hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
                maps = []
                for s in SEEDS_USED:
                    cfg = RunConfig(model=model, granularity=PRIMARY["granularity"],
                                    rank_map=PRIMARY["rank_map"], split=split,
                                    neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                    r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, collect_t1=True)
                    maps.append(float(r["T1_map"]))
                per_model[model] = maps
            delta = float(np.mean(per_model["M5_ccnn"]) - np.mean(per_model["M3_typed_star"]))
            rows.append({"partition_seed": p, "n_queries": n_q,
                         "M3_map_per_seed": per_model["M3_typed_star"],
                         "M5_map_per_seed": per_model["M5_ccnn"],
                         "delta": delta})
            log(f"  {split:18s} partition {p}: M3={np.mean(per_model['M3_typed_star']):.4f} "
                f"M5={np.mean(per_model['M5_ccnn']):.4f}  delta={delta:+.4f}  (n_q={n_q})")
        out["partitions"][split] = rows

    nd = [r["delta"] for r in out["partitions"]["narrator-disjoint"]]
    rd = [r["delta"] for r in out["partitions"]["random"]]

    # The claim is that the two split types disagree about the sign of the operator effect.
    sign_flips = sum(1 for a, b in zip(nd, rd) if (a < 0) != (b < 0))

    def decompose(rows):
        within = float(np.mean([np.var(r["M5_map_per_seed"], ddof=1)
                                + np.var(r["M3_map_per_seed"], ddof=1) for r in rows]))
        between = float(np.var([r["delta"] for r in rows], ddof=1))
        return {"between_partition_var": between, "within_partition_seed_var": within,
                "ratio_between_over_within": between / within if within else None}

    out["summary"] = {
        "narrator_disjoint": {"delta_mean": float(np.mean(nd)), "delta_sd": float(np.std(nd, ddof=1)),
                              "delta_min": float(min(nd)), "delta_max": float(max(nd)),
                              "n_negative": int(sum(1 for d in nd if d < 0)),
                              **decompose(out["partitions"]["narrator-disjoint"])},
        "random": {"delta_mean": float(np.mean(rd)), "delta_sd": float(np.std(rd, ddof=1)),
                   "delta_min": float(min(rd)), "delta_max": float(max(rd)),
                   "n_positive": int(sum(1 for d in rd if d > 0)),
                   **decompose(out["partitions"]["random"])},
        "sign_differs_pairs": sign_flips,
        "sign_differs_fraction": sign_flips / N_PARTITIONS,
    }

    s = out["summary"]
    log(f"  narrator-disjoint delta {s['narrator_disjoint']['delta_mean']:+.4f} "
        f"sd {s['narrator_disjoint']['delta_sd']:.4f} "
        f"[{s['narrator_disjoint']['delta_min']:+.4f},{s['narrator_disjoint']['delta_max']:+.4f}]")
    log(f"  random            delta {s['random']['delta_mean']:+.4f} "
        f"sd {s['random']['delta_sd']:.4f} "
        f"[{s['random']['delta_min']:+.4f},{s['random']['delta_max']:+.4f}]")
    log(f"  sign of delta differs between split types on {sign_flips}/{N_PARTITIONS} partitions")
    log(f"  between/within variance ratio: nd "
        f"{s['narrator_disjoint']['ratio_between_over_within']:.2f}, "
        f"random {s['random']['ratio_between_over_within']:.2f}")

    (RES / "e_partition.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {RES / 'e_partition.json'}")


if __name__ == "__main__":
    main()
