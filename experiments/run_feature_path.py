"""E-NEW-5: is the leak caused by the feature construction itself?

The mechanism section establishes that a rank-1 moment's input vector IS its narrator's
vector, because `build_features` builds a rank-k cell as the mean of its constituent rank-0
narrator features and a moment has exactly one narrator. That is a property of the
construction, not of this archive -- which is why re-running the whole study on a second
archive with the same code would reproduce it trivially and prove nothing.

The decisive test intervenes on the hypothesised cause. Here rank-1 cells are given their
OWN segment embedding instead of their narrator's mean, holding everything else fixed:
same complex, same splits, same seeds, same budget, same rank indicator.

Prediction, stated before running: if aggregating features up the ladder is what the random
split leaks, then under item-specific rank-1 features the complex's random-split advantage
should shrink markedly. If it survives intact, the mechanism as stated is wrong and the
paper must say so.

Reported on the split GAP, for the same reason as the anonymisation ablation: the two
conditions change what the representation contains, so absolute scores are not comparable,
but the gap between splits is transformed the same way in both.
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

from run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps  # noqa: E402
from src.complexes import build_complex, mask_incidences  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import (  # noqa: E402
    build_features, cell_text_embeddings, get_encoder, narrator_text_embeddings,
    segment_embeddings,
)
from src.hypergraph_encodings import hypergraph_encodings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.tasks import (  # noqa: E402
    NegativeSampler, build_t1_queries, make_split, partition_incidences,
)
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("feature_path")
MODELS = ("M5_ccnn", "M3_typed_star")
SEEDS = tuple(range(10))
SPLITS = ("random", "narrator-disjoint")
MAX_T1 = 400


def item_specific_rank1(feats: dict, cx, segments, seg_emb, sid2i) -> dict:
    """Replace each rank-1 row with that segment's own embedding, keeping the rank flag."""
    out = {k: v.copy() for k, v in feats.items()}
    m = out[1]
    d = seg_emb.shape[1]
    cells = sorted(cx.by_rank(1), key=lambda c: c.cid)
    if m.shape[0] != len(cells):
        raise RuntimeError(f"rank-1 matrix has {m.shape[0]} rows, complex has {len(cells)}")
    replaced = 0
    for i, c in enumerate(cells):
        sid = c.cid.split(":", 1)[1] if ":" in c.cid else c.cid
        j = sid2i.get(sid)
        if j is not None:
            m[i, :d] = seg_emb[j]
            replaced += 1
    log(f"    replaced {replaced}/{len(cells)} rank-1 rows with segment-specific embeddings")
    return out


def within_narrator_spread(feats: dict, cx) -> float:
    """Max elementwise spread between two moments of one narrator -- 0 means identical."""
    cells = sorted(cx.by_rank(1), key=lambda c: c.cid)
    by_narr: dict[str, list[int]] = {}
    for i, c in enumerate(cells):
        if len(c.members) == 1:
            by_narr.setdefault(next(iter(c.members)), []).append(i)
    worst = 0.0
    for idx in (v for v in by_narr.values() if len(v) > 1):
        block = feats[1][idx]
        worst = max(worst, float(np.abs(block - block[0]).max()))
    return worst


def prepare(segments, rank_maps, split_kind, encoder, device, item_specific: bool):
    cx = build_complex(segments, rank_maps[PRIMARY["rank_map"]],
                       granularity=PRIMARY["granularity"], rank_map_name=PRIMARY["rank_map"])
    feats = build_features(cx, segments, encoder)
    seg_emb, sid2i = segment_embeddings(segments, encoder)
    if item_specific:
        feats = item_specific_rank1(feats, cx, segments, seg_emb, sid2i)
    log(f"    within-narrator rank-1 spread = {within_narrator_spread(feats, cx):.3e}")

    sp = make_split(cx, split_kind, seed=0)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=0)
    data = {p: ns.build(parts[p], PRIMARY["neg"], ratio=10) for p in ("train", "val", "test")}

    train_cx = mask_incidences(cx, set(parts["train"]))
    H = train_cx.incidence_matrix(0, 2)
    enc0 = (hypergraph_encodings(H, k_spectral=16) if H.shape[1]
            else np.zeros((H.shape[0], 21), np.float32))
    bp = make_bundle(train_cx, feats, device)
    be = make_bundle(train_cx, feats, device, extra={0: enc0})
    data["t1_queries"] = build_t1_queries(cx, segments, sp, bucket="test", limit=MAX_T1)
    data["seg_index"] = sid2i
    return cx, bp, be, data


def run_condition(segments, rank_maps, encoder, device, item_specific: bool) -> dict:
    scores: dict[str, dict[str, float]] = {}
    for split_kind in SPLITS:
        log(f"  {'item-specific' if item_specific else 'narrator-mean'} / {split_kind}")
        cx, bp, be, data = prepare(segments, rank_maps, split_kind, encoder, device,
                                   item_specific)
        for m in MODELS:
            bundle = be if m == "M3_typed_star" else bp
            dims = {k: bundle.X[k].shape[1] for k in bundle.X}
            hidden = match_hidden_to_budget(m, dims, bundle, PARAM_BUDGET)
            vals = []
            for s in SEEDS:
                set_all_seeds(s)
                cfg = RunConfig(model=m, granularity=PRIMARY["granularity"],
                                rank_map=PRIMARY["rank_map"], split=split_kind,
                                neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                vals.append(train_eval(cfg, cx, bundle, data, device, eval_cx=cx)["T1_map"])
            scores.setdefault(m, {})[split_kind] = float(np.mean(vals))
            log(f"    {m:16s} {split_kind:18s} T1_MAP={np.mean(vals):.4f}"
                f"+-{np.std(vals):.4f}")
    return scores


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segments = load_corpus()[1]
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    log(f"device={device} segments={len(segments)}")

    out: dict = {
        "labelled": "EXPLORATORY - intervention on the hypothesised cause",
        "prediction": ("if aggregating features up the ladder is what the random split "
                       "leaks, the complex's random-split gap shrinks markedly under "
                       "item-specific rank-1 features"),
    }
    for tag, item_specific in (("narrator_mean", False), ("item_specific", True)):
        out[tag] = run_condition(segments, rank_maps, encoder, device, item_specific)
        (RES / "e_feature_path.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    gaps = {}
    for m in MODELS:
        a = out["narrator_mean"].get(m, {})
        b = out["item_specific"].get(m, {})
        if {"random", "narrator-disjoint"} <= set(a) and {"random", "narrator-disjoint"} <= set(b):
            g0 = a["random"] - a["narrator-disjoint"]
            g1 = b["random"] - b["narrator-disjoint"]
            gaps[m] = {"gap_narrator_mean": g0, "gap_item_specific": g1,
                       "gap_reduction": g0 - g1,
                       "gap_reduction_frac": (g0 - g1) / g0 if g0 else float("nan")}
            log(f"  GAP {m:16s} narrator-mean={g0:+.4f} item-specific={g1:+.4f} "
                f"reduction={g0 - g1:+.4f}")
    out["split_gap"] = gaps
    five = gaps.get("M5_ccnn", {})
    out["prediction_held"] = bool(
        five and five["gap_narrator_mean"] > 0
        and five["gap_item_specific"] < 0.5 * five["gap_narrator_mean"])
    log(f"  prediction_held = {out['prediction_held']}")
    (RES / "e_feature_path.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("feature-path intervention complete")


if __name__ == "__main__":
    main()
