"""Run PHASE 5 (tasks/splits/negatives), PHASE 6 (models) and PHASE 7 (inference).

Grid reduction is declared, not hidden: the full model set runs at the pre-registered
primary cell (mid x narrator-disjoint x MNS x R-A) and every other axis is a
one-at-a-time variation from that cell.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import (  # noqa: E402
    build_features, cell_text_embeddings, get_encoder, narrator_text_embeddings,
    segment_embeddings,
)
from src.hypergraph_encodings import hypergraph_encodings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.metrics import cluster_bootstrap, design_effect  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.seeds import SEEDS, set_all_seeds  # noqa: E402
from src.stats import aso_matrix, interaction_model, stratify  # noqa: E402
from src.tasks import (  # noqa: E402
    NegativeSampler, build_t1_queries, find_near_duplicates, make_split, partition_incidences,
)
from src.train import RunConfig, evaluate_m1_dense, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
RES.mkdir(parents=True, exist_ok=True)

PRIMARY = {"granularity": "mid", "split": "narrator-disjoint", "neg": "MNS", "rank_map": "R-A"}
MODELS = ["M0_mlp", "M1_dense", "M2_untyped_star", "M3_typed_star", "M4_allset",
          "M4_edhnn", "M4_hgmlp", "M5_ccnn"]
ABLATIONS = {
    "A1_shared_weights": {"share_weights": True},
    "A4_no_down": {"use_down": False},
    "A5_no_moments": {"skip_moments": True},
    "A6_no_within": {"use_within": False},
}
MAX_T1_QUERIES = 400
PARAM_BUDGET = 130_000          # shared target; E6.0 requires matching within +-15%

log = make_logger("phase5_7")


def load_rank_maps() -> dict[str, dict[str, int]]:
    d = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    return {"R-A": d["R-A_consensus"], "R-B": d["R-B_archive_native"], "R-C": d["R-C_adversarial"]}


def _drop_terms(seg, flagged: set[str]):
    """Shallow copy of a segment with the flagged topic terms removed."""
    import copy as _copy

    s = _copy.copy(seg)
    s.topics = [t for t in seg.topics if t.get("term") not in flagged]
    return s


def prepare_cell(segments, rank_maps, granularity, split_kind, neg, rank_map, encoder,
                 device, cache: dict):
    key = (rank_map, granularity)
    if key not in cache:
        cx = build_complex(segments, rank_maps[rank_map], granularity=granularity,
                           rank_map_name=rank_map)
        feats = build_features(cx, segments, encoder)
        cache[key] = (cx, feats)
    cx, feats = cache[key]

    sp = make_split(cx, split_kind, seed=0)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=0)
    data = {}
    for part in ("train", "val", "test"):
        data[part] = ns.build(parts[part], neg, ratio=10)

    # Message passing may only see TRAINING incidences, otherwise the model is handed the
    # edges it is asked to predict.
    train_cx = mask_incidences(cx, set(parts["train"]))
    H = train_cx.incidence_matrix(0, 2)
    enc0 = (hypergraph_encodings(H, k_spectral=16) if H.shape[1]
            else np.zeros((H.shape[0], 21), np.float32))
    bundle_plain = make_bundle(train_cx, feats, device)
    bundle_enc = make_bundle(train_cx, feats, device, extra={0: enc0})

    seg_emb, sid2i = segment_embeddings(segments, encoder)
    queries = build_t1_queries(cx, segments, sp, bucket="test", limit=MAX_T1_QUERIES)
    data["t1_queries"] = queries
    data["seg_index"] = sid2i
    dense = {
        "segments": segments,
        "seg_emb": seg_emb,
        "cell_emb": cell_text_embeddings(cx, segments, 2, encoder),
        "narr_emb": narrator_text_embeddings(cx, segments, encoder),
    }
    return cx, bundle_plain, bundle_enc, data, sp, dense


def run_models(models, cx, bp, be, data, device, granularity, split_kind, neg, rank_map,
               seeds, extra_kw=None, dense=None):
    rows, per_item = [], []
    # Per-item records are pooled across the whole grid downstream. Without these tags a
    # consumer cannot tell a coarse-granularity row from a random-split one, and the E7.5
    # interaction model silently fitted over all of them.
    cell_tag = {"granularity": granularity, "split": split_kind, "neg_regime": neg,
                "rank_map": rank_map}
    for m in models:
        if m == "M1_dense":
            # Parameter-free: one deterministic evaluation, replicated across the seed list
            # only so that ASO sees the same sample size as every other model.
            cfg = RunConfig(model=m, granularity=granularity, rank_map=rank_map,
                            split=split_kind, neg_regime=neg, seed=seeds[0])
            out = evaluate_m1_dense(cx, dense["segments"], data, dense["seg_emb"],
                                    dense["cell_emb"], dense["narr_emb"], cfg)
            pit = out.pop("per_item", [])
            for s in seeds:
                r = dict(out)
                r["seed"] = s
                rows.append(r)
            for r in pit:
                r["model"] = m
                r["seed"] = seeds[0]
                r.update(cell_tag)
            per_item.extend(pit)
            log(f"    {m:18s} T1_MAP={out.get('T1_map', float('nan')):.4f}  "
                f"T2_AUC={out.get('T2_auc', float('nan')):.4f}  params=0 (deterministic)")
            continue

        bundle = be if m == "M3_typed_star" else bp
        kw = (extra_kw or {}).get(m, {})
        base = m if m in ("M0_mlp", "M2_untyped_star", "M3_typed_star", "M4_allset",
                          "M4_edhnn", "M4_hgmlp", "M5_ccnn") else "M5_ccnn"
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(base, dims, bundle, PARAM_BUDGET, **kw)
        for s in seeds:
            cfg = RunConfig(model=base, granularity=granularity, rank_map=rank_map,
                            split=split_kind, neg_regime=neg, seed=s, hidden=hidden, extra=kw)
            out = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
            out["model"] = m
            pit = out.pop("per_item", [])
            for r in pit:
                r["model"] = m
                r["seed"] = s
                r.update(cell_tag)
            per_item.extend(pit)
            rows.append(out)
        mine = [r for r in rows if r["model"] == m]
        log(f"    {m:18s} T1_MAP={np.nanmean([r['T1_map'] for r in mine]):.4f}  "
            f"T2_AUC={np.nanmean([r['T2_auc'] for r in mine]):.4f}  "
            f"hidden={hidden} params={mine[-1]['n_params']}")
    return rows, per_item


def main(seeds=None, quick=False) -> None:
    seeds = seeds or (SEEDS[:3] if quick else SEEDS)
    set_all_seeds(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  seeds={seeds}")

    interviews, segments = load_corpus()
    log(f"corpus: {len(interviews)} interviews / {len(segments)} segments")
    rank_maps = load_rank_maps()
    encoder = get_encoder()
    cache: dict = {}

    # E5.2 near-duplicate audit (logged, never silently deleted)
    dups = find_near_duplicates(segments, threshold=0.8)
    log(f"E5.2 - cross-narrator near-duplicate pairs: {len(dups)}")

    all_rows, all_items = [], []

    # ---- primary cell -------------------------------------------------------------
    log(f"E7.1 - PRIMARY CELL {PRIMARY}")
    cx, bp, be, data, sp, dense = prepare_cell(
        segments, rank_maps, PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
        PRIMARY["rank_map"], encoder, device, cache)
    log(f"  |r2 cells|={len(cx.by_rank(2))}  |train pairs|={len(data['train'][1])}  "
        f"|T1 queries|={len(data['t1_queries'])}")
    rows, items = run_models(MODELS, cx, bp, be, data, device, PRIMARY["granularity"],
                             PRIMARY["split"], PRIMARY["neg"], PRIMARY["rank_map"], seeds,
                             dense=dense)
    all_rows += rows
    all_items += items

    # ablations of M5 (A2 and A3 are rank-map transforms, handled separately below)
    log("E6.7 - ablations")
    abl_rows, abl_items = run_models(list(ABLATIONS), cx, bp, be, data, device,
                                     PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
                                     PRIMARY["rank_map"], seeds, extra_kw=ABLATIONS, dense=dense)
    all_rows += abl_rows
    all_items += abl_items

    # A2: shuffled rank labels, distribution preserved
    rng = np.random.default_rng(0)
    base = rank_maps["R-A"]
    keys = sorted(base)
    vals = [base[k] for k in keys]
    rng.shuffle(vals)
    rank_maps["A2_shuffled"] = dict(zip(keys, vals))
    # A3: collapse ranks 2 and 3 so the hierarchy has no depth left to exploit
    rank_maps["A3_collapsed"] = {k: 2 for k in base}

    for tag, rm_name in (("A2_shuffled_ranks", "A2_shuffled"),
                         ("A3_collapse_r2r3", "A3_collapsed")):
        cx_s, bp_s, be_s, data_s, _, dense_s = prepare_cell(
            segments, rank_maps, PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
            rm_name, encoder, device, cache)
        r, it = run_models(["M5_ccnn"], cx_s, bp_s, be_s, data_s, device,
                           PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"], rm_name,
                           seeds, dense=dense_s)
        for x in r:
            x["model"] = tag
        for x in it:
            x["model"] = tag
        all_rows += r
        all_items += it

    # ---- one-at-a-time axis variations --------------------------------------------
    core = ["M3_typed_star", "M5_ccnn", "M0_mlp", "M1_dense"]
    for g in ("coarse", "fine"):
        log(f"E7.1 - granularity={g}")
        c2, b2, e2, d2, _, dn = prepare_cell(segments, rank_maps, g, PRIMARY["split"],
                                             PRIMARY["neg"], PRIMARY["rank_map"], encoder,
                                             device, cache)
        r, it = run_models(core, c2, b2, e2, d2, device, g, PRIMARY["split"], PRIMARY["neg"],
                           PRIMARY["rank_map"], seeds, dense=dn)
        all_rows += r; all_items += it
    for skind in ("random", "event-disjoint"):
        log(f"E7.1 - split={skind}")
        c2, b2, e2, d2, _, dn = prepare_cell(segments, rank_maps, PRIMARY["granularity"], skind,
                                             PRIMARY["neg"], PRIMARY["rank_map"], encoder,
                                             device, cache)
        r, it = run_models(core, c2, b2, e2, d2, device, PRIMARY["granularity"], skind,
                           PRIMARY["neg"], PRIMARY["rank_map"], seeds, dense=dn)
        all_rows += r; all_items += it
    for neg in ("UNS", "SNS", "CNS", "hard"):
        log(f"E7.1 - negatives={neg}")
        c2, b2, e2, d2, _, dn = prepare_cell(segments, rank_maps, PRIMARY["granularity"],
                                             PRIMARY["split"], neg, PRIMARY["rank_map"], encoder,
                                             device, cache)
        r, it = run_models(core, c2, b2, e2, d2, device, PRIMARY["granularity"],
                           PRIMARY["split"], neg, PRIMARY["rank_map"], seeds, dense=dn)
        all_rows += r; all_items += it
    for rm in ("R-B", "R-C"):
        log(f"E7.1 - rank_map={rm}")
        c2, b2, e2, d2, _, dn = prepare_cell(segments, rank_maps, PRIMARY["granularity"],
                                             PRIMARY["split"], PRIMARY["neg"], rm, encoder,
                                             device, cache)
        r, it = run_models(core, c2, b2, e2, d2, device, PRIMARY["granularity"],
                           PRIMARY["split"], PRIMARY["neg"], rm, seeds, dense=dn)
        all_rows += r; all_items += it

    # ---- E8.3 gold-subset upper bound ------------------------------------------------
    # The noise-free ceiling. Restricting to terms the E3.6 ambiguity flagger did NOT
    # flag isolates "the method does not work" from "the labelling is contested".
    amb_path = RES / "e3_6_ambiguity.json"
    if amb_path.exists():
        flagged = set(json.loads(amb_path.read_text(encoding="utf-8")).get("ambiguous_terms", []))
        clean_segments = [_drop_terms(s, flagged) for s in segments]
        clean_segments = [s for s in clean_segments if s.topics]
        log(f"E8.3 - gold subset: {len(clean_segments)}/{len(segments)} segments retain "
            f"only unflagged terms ({len(flagged)} terms excluded)")
        cache_gold: dict = {}
        cg, bg, eg, dg, _, dng = prepare_cell(clean_segments, rank_maps, PRIMARY["granularity"],
                                              PRIMARY["split"], PRIMARY["neg"],
                                              PRIMARY["rank_map"], encoder, device, cache_gold)
        rg, itg = run_models(core, cg, bg, eg, dg, device, PRIMARY["granularity"],
                             PRIMARY["split"], PRIMARY["neg"], PRIMARY["rank_map"], seeds,
                             dense=dng)
        for x in rg:
            x["cell"] = "E8.3_gold_subset"
        gold_summary = {
            "n_segments": len(clean_segments),
            "n_segments_full": len(segments),
            "n_terms_excluded": len(flagged),
            "n_rank2_cells": len(cg.by_rank(2)),
            "model_means": {m: {
                "T1_map": float(np.nanmean([r["T1_map"] for r in rg if r["model"] == m])),
                "T2_auc": float(np.nanmean([r["T2_auc"] for r in rg if r["model"] == m])),
            } for m in {r["model"] for r in rg}},
            "interpretation": ("upper bound achievable when the contested labels are removed; "
                               "compare against the primary cell to separate method failure "
                               "from labelling noise"),
        }
        (RES / "e8_3_gold_subset.json").write_text(json.dumps(gold_summary, indent=2),
                                                   encoding="utf-8")
        all_rows += rg
        all_items += itg

    (RES / "e7_1_runs.json").write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    log(f"E7.1 - {len(all_rows)} runs recorded")

    # ---- E7.3 ASO ------------------------------------------------------------------
    def cell_rows(rows, **kw):
        return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]

    prim = cell_rows(all_rows, granularity=PRIMARY["granularity"], split=PRIMARY["split"],
                     neg_regime=PRIMARY["neg"], rank_map=PRIMARY["rank_map"])
    # The gold-subset runs share the primary cell's four keys but are a different corpus.
    # Leaving them in gave four models n=20 (two pooled populations) against n=10 for the
    # rest, so the dominance test compared a bimodal sample with a unimodal one.
    prim = [r for r in prim if r.get("cell") != "E8.3_gold_subset"]
    for metric in ("T1_map", "T2_auc"):
        scores = {}
        for r in prim:
            scores.setdefault(r["model"], []).append(r[metric])
        aso = aso_matrix(scores)
        (RES / f"e7_3_aso_{metric}.json").write_text(
            json.dumps({"scores_summary": {k: {"mean": float(np.nanmean(v)),
                                               "std": float(np.nanstd(v)),
                                               "n": len(v), "per_seed": v}
                                           for k, v in scores.items()}, **aso}, indent=2),
            encoding="utf-8")
        pair = aso["eps_min"].get("M5_ccnn>M3_typed_star")
        log(f"E7.3 - {metric}: eps_min(M5>M3)={pair}")

    # ---- E7.4 clustered bootstrap + DEFF -------------------------------------------
    by_narr: dict[str, list[float]] = {}
    for r in all_items:
        if r.get("model") == "M5_ccnn":
            by_narr.setdefault(r["narrator_id"], []).append(float(r["correct"]))
    groups = list(by_narr.values())
    deff = design_effect([len(g) for g in groups], groups)
    units = list(by_narr)
    boot = cluster_bootstrap(
        lambda us: float(np.mean([v for u in us for v in by_narr[u]])) if us else 0.0,
        units, B=10000, seed=0, bca=False)
    (RES / "e7_4_uncertainty.json").write_text(json.dumps(
        {"design_effect": deff, "clustered_bootstrap_accuracy_M5": boot,
         "note": "sampling unit is the narrator; edge-level CIs would be narrower by ~sqrt(DEFF)"},
        indent=2), encoding="utf-8")
    log(f"E7.4 - DEFF={deff['DEFF']:.2f} (ICC={deff['ICC']:.3f})")

    # ---- E7.5 interaction ----------------------------------------------------------
    # Filter to the PRIMARY CELL, not merely to the two models. `all_items` accumulates
    # every granularity, split, negative regime and rank map in the grid, so filtering on
    # model alone pooled 343 event ids from three granularities -- and included the random
    # split, where the ordering this coefficient describes is reversed.
    prim_items = [r for r in all_items
                  if r.get("model") in ("M5_ccnn", "M3_typed_star")
                  and r.get("granularity") == PRIMARY["granularity"]
                  and r.get("split") == PRIMARY["split"]
                  and r.get("neg_regime") == PRIMARY["neg"]
                  and r.get("rank_map") == PRIMARY["rank_map"]
                  and r.get("cell") != "E8.3_gold_subset"]
    inter = interaction_model(prim_items, "M5_ccnn", "M3_typed_star")
    (RES / "e7_5_interaction.json").write_text(json.dumps(inter, indent=2), encoding="utf-8")
    log(f"E7.5 - interaction coef={inter.get('interaction_coef')} "
        f"CI={inter.get('interaction_ci')} crosses_zero={inter.get('ci_crosses_zero')}")

    # ---- E7.2 stratified reporting --------------------------------------------------
    strat = {}
    for key in ("event_size", "rank"):
        buckets = stratify(prim_items, key)
        strat[key] = {
            b: {m: float(np.mean([x["correct"] for x in rs if x["model"] == m]))
                for m in ("M5_ccnn", "M3_typed_star")
                if any(x["model"] == m for x in rs)}
            for b, rs in buckets.items()
        }
    (RES / "e7_2_stratified.json").write_text(json.dumps(strat, indent=2), encoding="utf-8")
    log("PHASE 5+6+7 complete")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
