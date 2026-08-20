"""E10b: partition replication of the *registered system* contrast.

`e_partition.py` resamples the partition for the matched-input operator comparison and finds
the complex ahead on every draw under both splits, so the sign never differs. That is the
right answer to the operator question and the wrong instrument for the headline claim: the
reversal the paper reports is between the system as registered -- the typed star carrying
spectral encodings -- and the complex, which never receives them. This script resamples the
partition for that contrast.

Delta here is MAP(M5, plain) - MAP(M3, enc). It is negative under a narrator-disjoint split
(the star leads, as registered) and positive under a random split (the complex leads), so the
count that matters is how often the sign differs between the two split types.

Both arms run in one process on purpose. `make_split` draws narrator-disjoint units from
`sorted(cx.narrators)` but random units from `cx.by_rank(2)`, which is dict insertion order,
so a random partition is only guaranteed to match within a single interpreter. Reusing the
M5 numbers from the earlier run would have silently compared two different partitions.
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
from src.hypergraph_encodings import hypergraph_encodings  # noqa: E402
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
# The registered system: the star as published, the complex as published.
ARMS = (("M3_typed_star", "enc"), ("M5_ccnn", "plain"))
SPLITS = ("narrator-disjoint", "random")
N_PARTITIONS = 10
SEEDS_USED = SEEDS[:3]

log = make_logger("e_partition_registered")


def build_for_split(cx, feats, segments, split_kind, part_seed, device, sid2i):
    sp = make_split(cx, split_kind, seed=part_seed)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=0)
    data = {p: ns.build(parts[p], PRIMARY["neg"], ratio=10) for p in ("train", "val", "test")}

    # Message passing may only see training incidences, and the encodings are built from the
    # same masked complex so they cannot carry held-out structure either.
    train_cx = mask_incidences(cx, set(parts["train"]))
    H = train_cx.incidence_matrix(0, 2)
    enc0 = (hypergraph_encodings(H, k_spectral=16) if H.shape[1]
            else np.zeros((H.shape[0], 21), np.float32))
    bundles = {"plain": make_bundle(train_cx, feats, device),
               "enc": make_bundle(train_cx, feats, device, extra={0: enc0})}
    data["t1_queries"] = build_t1_queries(cx, segments, sp, bucket="test",
                                          limit=MAX_T1_QUERIES)
    data["seg_index"] = sid2i
    return bundles, data, len(data["t1_queries"])


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
                 "contrast": "M5_ccnn/plain minus M3_typed_star/enc", "partitions": {}}

    for split in SPLITS:
        rows = []
        for p in range(N_PARTITIONS):
            bundles, data, n_q = build_for_split(cx, feats, segments, split, p, device, sid2i)
            per_arm: dict[str, list[float]] = {}
            for model, feat in ARMS:
                bundle = bundles[feat]
                dims = {k: bundle.X[k].shape[1] for k in bundle.X}
                hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
                maps = []
                for s in SEEDS_USED:
                    cfg = RunConfig(model=model, granularity=PRIMARY["granularity"],
                                    rank_map=PRIMARY["rank_map"], split=split,
                                    neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                    r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, collect_t1=True)
                    maps.append(float(r["T1_map"]))
                per_arm[f"{model}|{feat}"] = maps
            m3 = float(np.mean(per_arm["M3_typed_star|enc"]))
            m5 = float(np.mean(per_arm["M5_ccnn|plain"]))
            rows.append({"partition_seed": p, "n_queries": n_q,
                         "M3_enc_map_per_seed": per_arm["M3_typed_star|enc"],
                         "M5_plain_map_per_seed": per_arm["M5_ccnn|plain"],
                         "delta": m5 - m3})
            log(f"  {split:18s} partition {p}: M3_enc={m3:.4f} M5_plain={m5:.4f}  "
                f"delta={m5 - m3:+.4f}  (n_q={n_q})")
        out["partitions"][split] = rows

    nd = [r["delta"] for r in out["partitions"]["narrator-disjoint"]]
    rd = [r["delta"] for r in out["partitions"]["random"]]
    sign_flips = sum(1 for a, b in zip(nd, rd) if (a < 0) != (b < 0))

    def summarise(rows, deltas):
        # `between` is the variance of a seed-averaged delta, so it already carries
        # seed noise; the partition component is what is left after removing it.
        within = float(np.mean([np.var(r["M5_plain_map_per_seed"], ddof=1)
                                + np.var(r["M3_enc_map_per_seed"], ddof=1) for r in rows]))
        observed = float(np.var(deltas, ddof=1))
        part = observed - within / len(SEEDS_USED)
        return {"delta_mean": float(np.mean(deltas)), "delta_sd": float(np.std(deltas, ddof=1)),
                "delta_min": float(min(deltas)), "delta_max": float(max(deltas)),
                "n_negative": int(sum(1 for d in deltas if d < 0)),
                "n_positive": int(sum(1 for d in deltas if d > 0)),
                "observed_var_of_seed_mean": observed,
                "within_partition_seed_var": within,
                "partition_component_var": part,
                "partition_component_share": (part / observed) if observed else None}

    out["summary"] = {
        "narrator_disjoint": summarise(out["partitions"]["narrator-disjoint"], nd),
        "random": summarise(out["partitions"]["random"], rd),
        "sign_differs_pairs": sign_flips,
        "sign_differs_fraction": sign_flips / N_PARTITIONS,
        "star_leads_narrator_disjoint": int(sum(1 for d in nd if d < 0)),
        "complex_leads_random": int(sum(1 for d in rd if d > 0)),
    }

    s = out["summary"]
    log(f"  narrator-disjoint delta {s['narrator_disjoint']['delta_mean']:+.4f} "
        f"sd {s['narrator_disjoint']['delta_sd']:.4f} "
        f"[{s['narrator_disjoint']['delta_min']:+.4f},{s['narrator_disjoint']['delta_max']:+.4f}]")
    log(f"  random            delta {s['random']['delta_mean']:+.4f} "
        f"sd {s['random']['delta_sd']:.4f} "
        f"[{s['random']['delta_min']:+.4f},{s['random']['delta_max']:+.4f}]")
    log(f"  star leads under narrator-disjoint on {s['star_leads_narrator_disjoint']}/"
        f"{N_PARTITIONS}; complex leads under random on {s['complex_leads_random']}/"
        f"{N_PARTITIONS}")
    log(f"  sign of delta differs between split types on {sign_flips}/{N_PARTITIONS} partitions")

    (RES / "e_partition_registered.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("  wrote e_partition_registered.json")


if __name__ == "__main__":
    main()
