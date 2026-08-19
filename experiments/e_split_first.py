"""E1, E2 and E4 in one pass over the primary cell.

E1 asks which of two mechanisms produces the split reversal. Global feature construction
confounds them: under a random split a straddling narrator both repeats an identity across
the partition AND carries test-side text into a training-side vector. Building a narrator's
vector from that narrator's training-side segments only removes the second while leaving the
first intact, so the reversal either survives (identity repetition is the mechanism) or it
does not (cross-partition aggregation is).

E2 separates the two jobs the M3 baseline was doing at once. M3 received a 21-dimensional
spectral block that M5 never saw, so the published comparison supports "a strong simpler
baseline wins" but not "rank as a type beats rank as an operator". M3-matched drops the
block and answers the second question; M3-strong keeps it and remains the deliberately
advantaged baseline.

E4 persists per-query T1 average precision with the querying narrator, which is the unit the
paired bootstrap has to resample. Seeds measure optimisation noise; narrators are the
sampling unit the claim is about.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
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
from src.seeds import SEEDS, set_all_seeds  # noqa: E402
from src.tasks import (  # noqa: E402
    NegativeSampler, build_t1_queries, make_split, partition_incidences,
)
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
PRIMARY = {"granularity": "mid", "neg": "MNS", "rank_map": "R-A"}
SPLITS = ["narrator-disjoint", "random", "event-disjoint"]
FEATURE_MODES = ["global", "split-first"]
MAX_T1_QUERIES = 400
PARAM_BUDGET = 130_000

# (label, architecture, spectral block) - M3_matched and M3_strong share an architecture and
# differ only in whether the 21-dim encoding block is concatenated at rank 0.
MODELS = [("M3_matched", "M3_typed_star", False),
          ("M3_strong", "M3_typed_star", True),
          ("M5_ccnn", "M5_ccnn", False)]

log = make_logger("split_first")


def load_rank_maps() -> dict[str, dict[str, int]]:
    d = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    return {"R-A": d["R-A_consensus"], "R-B": d["R-B_archive_native"], "R-C": d["R-C_adversarial"]}


def training_side_by_narrator(cx, segments, sp) -> dict[str, set[str]]:
    """For each narrator, that narrator's segments carrying no held-out incidence.

    Membership is a property of the (narrator, segment) pair. One segment can be
    training-side for a narrator whose incidence through it sits in the training partition
    and held out for a co-narrator whose incidence does not, so a single global segment set
    would drop a narrator's own evidence because of somebody else's split assignment.

    A segment supporting no rank-2 cell carries no incidence, is therefore never held out,
    and stays for every narrator who speaks it.
    """
    cell_of_seg: dict[str, list[str]] = defaultdict(list)
    for c in cx.by_rank(2):
        for sid in c.segments:
            cell_of_seg[sid].append(c.cid)
    held = sp.val | sp.test
    out: dict[str, set[str]] = defaultdict(set)
    for s in segments:
        cids = cell_of_seg.get(s.segment_id, [])
        for n in s.narrators:
            blocked = any(
                (n if sp.unit == "narrator" else cid if sp.unit == "event" else f"{cid}||{n}")
                in held for cid in cids
            )
            if not blocked:
                out[n].add(s.segment_id)
    return out


def main(quick: bool = False) -> None:
    seeds = SEEDS[:2] if quick else SEEDS
    set_all_seeds(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  seeds={seeds}")

    _, segments = load_corpus()
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    cx = build_complex(segments, rank_maps[PRIMARY["rank_map"]],
                       granularity=PRIMARY["granularity"], rank_map_name=PRIMARY["rank_map"])
    log(f"complex: {len(cx.narrators)} narrators, {len(cx.by_rank(2))} rank-2 cells")

    feats_global = build_features(cx, segments, encoder)
    seg_emb, sid2i = segment_embeddings(segments, encoder)

    runs: list[dict] = []
    per_query: list[dict] = []
    exposure: dict[str, dict] = {}
    t0 = time.time()

    for split_kind in SPLITS:
        sp = make_split(cx, split_kind, seed=0)
        parts = partition_incidences(cx, sp)
        allowed = training_side_by_narrator(cx, segments, sp)
        feats_sf = build_features(cx, segments, encoder, allowed_by_narrator=allowed)
        n_fb, n_tot = build_features.last_fallback
        own: dict[str, int] = defaultdict(int)
        for s in segments:
            for n in s.narrators:
                own[n] += 1
        removed = {n: 1.0 - len(allowed.get(n, ())) / c for n, c in own.items() if c}
        touched = [v for v in removed.values() if v > 0]
        kept = sum(len(v) for v in allowed.values())
        total_pairs = sum(len(s.narrators) for s in segments)
        exposure[split_kind] = {
            "training_side_pairs": kept,
            "total_narrator_segment_pairs": total_pairs,
            "retained_fraction": round(kept / total_pairs, 4) if total_pairs else float("nan"),
            "narrators_losing_text": len(touched),
            "narrators_total": n_tot,
            "mean_removed_fraction_among_affected": round(float(np.mean(touched)), 4) if touched else 0.0,
            "max_removed_fraction": round(float(max(removed.values())), 4) if removed else 0.0,
            "narrators_on_fallback": n_fb,
            "fallback_fraction": round(n_fb / n_tot, 4) if n_tot else float("nan"),
        }
        log(f"{split_kind}: kept {kept}/{total_pairs} pairs; {len(touched)} narrators lost text "
            f"(mean {exposure[split_kind]['mean_removed_fraction_among_affected']:.1%} of theirs); "
            f"{n_fb} on fallback")

        ns = NegativeSampler(cx, seed=0)
        data = {p: ns.build(parts[p], PRIMARY["neg"], ratio=10) for p in ("train", "val", "test")}
        data["t1_queries"] = build_t1_queries(cx, segments, sp, bucket="test",
                                              limit=MAX_T1_QUERIES)
        data["seg_index"] = sid2i

        train_cx = mask_incidences(cx, set(parts["train"]))
        H = train_cx.incidence_matrix(0, 2)
        enc0 = (hypergraph_encodings(H, k_spectral=16) if H.shape[1]
                else np.zeros((H.shape[0], 21), np.float32))

        for fmode in FEATURE_MODES:
            feats = feats_global if fmode == "global" else feats_sf
            bundles = {False: make_bundle(train_cx, feats, device),
                       True: make_bundle(train_cx, feats, device, extra={0: enc0})}
            for label, base, spectral in MODELS:
                bundle = bundles[spectral]
                dims = {k: bundle.X[k].shape[1] for k in bundle.X}
                hidden = match_hidden_to_budget(base, dims, bundle, PARAM_BUDGET)
                for s in seeds:
                    cfg = RunConfig(model=base, granularity=PRIMARY["granularity"],
                                    rank_map=PRIMARY["rank_map"], split=split_kind,
                                    neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
                    out = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, collect_t1=True)
                    pq = out.pop("T1_per_query", [])
                    out.pop("per_item", None)
                    out.update({"model": label, "feature_mode": fmode, "spectral": spectral})
                    runs.append(out)
                    for r in pq:
                        per_query.append({"model": label, "feature_mode": fmode,
                                          "split": split_kind, "seed": s, **r})
                mine = [r for r in runs
                        if r["model"] == label and r["feature_mode"] == fmode
                        and r["split"] == split_kind]
                log(f"  {fmode:11s} {label:11s} T1_MAP={np.nanmean([r['T1_map'] for r in mine]):.4f}"
                    f"  T2_AUC={np.nanmean([r['T2_auc'] for r in mine]):.4f}  hidden={hidden}")

    payload = {
        "primary": PRIMARY,
        "splits": SPLITS,
        "feature_modes": FEATURE_MODES,
        "seeds": seeds,
        "exposure": exposure,
        "runs": runs,
        "per_query": per_query,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (RES / "e_split_first.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    log(f"wrote e_split_first.json  ({len(runs)} runs, {len(per_query)} per-query records, "
        f"{payload['wall_clock_s']}s)")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
