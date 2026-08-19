"""Run PHASE 2 (rank ontology) and PHASE 4 (complex construction) end to end.

E2.1 rank manual        -> annotation/RANK_MANUAL.md evidence
E2.2 agreement study    -> data/results/e2_2_agreement.json
E2.3 R-A / R-B / R-C    -> data/results/e2_3_rank_maps.json
E4.1 complex build      -> complexes/*.json  + sanity assertions
E4.2 star round-trip    -> data/results/e4_2_star_roundtrip.json
E4.3 granularity sweep  -> data/results/e4_3_granularity.json
E4.4 rank/cardinality   -> data/results/e4_4_decoupling.json   <-- week-one killer
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import (  # noqa: E402
    GRANULARITIES, build_complex, rank_cardinality_decoupling, referent, sanity_assertions,
    save_complex, verify_star_roundtrip,
)
from src.corpus import corpus_stats, load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.metrics import cluster_subsample_ci, jaccard_at, rbo  # noqa: E402
from src.ontology import (  # noqa: E402
    agreement_diagnostics, confusion, disagreement_taxonomy, run_rank_study,
)
from src.seeds import set_all_seeds  # noqa: E402

RES = ROOT / "data" / "results"
RES.mkdir(parents=True, exist_ok=True)
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

log = make_logger("phase2_4")


def get_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"loading encoder {ENCODER_NAME} on {dev}")
        return SentenceTransformer(ENCODER_NAME, device=dev)
    except Exception as e:  # noqa: BLE001
        log(f"encoder unavailable ({type(e).__name__}: {e}); A3 falls back to depth heuristic")
        return None


def triage_cells(cx, k: int = 50):
    """T_g: rank-2 cells ordered by ascending archive-conditioned attestation multiplicity."""
    return sorted(cx.by_rank(2), key=lambda c: (c.size, c.label))[:k]


def project(cells) -> list[str]:
    """Order-preserving dedup of the triage list onto the shared archival referent."""
    seen, out = set(), []
    for c in cells:
        r = referent(c)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def main() -> None:
    set_all_seeds(0)
    interviews, segments = load_corpus()
    stats = corpus_stats(interviews, segments)
    log(f"corpus: {stats['n_interviews']} interviews / {stats['n_segments']} segments / "
        f"{stats['n_narrators']} narrators")
    (RES / "e1_1_corpus_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---------------- E2.2 agreement -------------------------------------------------
    log("E2.2 - rank agreement study (round 1 -> adjudication -> round 2)")
    enc = get_encoder()
    study = run_rank_study(segments, encoder=enc)
    a2 = study.alpha_round2
    diag1 = agreement_diagnostics(study.labels_round1, study.round1_terms)
    diag2 = agreement_diagnostics(study.labels, study.round2_terms)
    agreement = {
        "n_terms": len(study.terms),
        "n_round1_terms": len(study.round1_terms),
        "n_round2_terms": len(study.round2_terms),
        "alpha_round1": study.alpha_round1,
        "alpha_round2_REPORTED": a2,
        "diagnostics_round1": diag1,
        "diagnostics_round2": diag2,
        "adjudication_revision_log": study.revision_log,
        "n_disputed": len(study.disputed),
        "disputed_top20": study.disputed[:20],
        "confusion_A1_A2": confusion(study, "A1_structural", "A2_lexical"),
        "confusion_A1_A3": confusion(study, "A1_structural", "A3_distributional"),
        "confusion_A2_A3": confusion(study, "A2_lexical", "A3_distributional"),
        "disagreement_taxonomy": {k: {"n": len(v), "examples": v[:5]}
                                  for k, v in disagreement_taxonomy(study).items()},
        "stratum_sizes": {k: len(v) for k, v in study.strata.items()},
        "label_distribution": {a: {str(r): sum(1 for t in study.terms if lab[t] == r) for r in (2, 3)}
                               for a, lab in study.labels.items()},
        "kill_criterion": {
            "threshold": 0.55,
            "observed_round2_alpha": a2.get("overall"),
            "passes": bool((a2.get("overall") or -1) >= 0.55),
            "H1_target": 0.67,
            "meets_H1": bool((a2.get("overall") or -1) >= 0.67),
            "raw_agreement_round2": diag2.get("mean_pairwise_raw_agreement"),
            "gwet_ac1_round2": diag2.get("gwet_ac1"),
            "paradox_suspected": diag2.get("paradox_suspected"),
        },
    }
    (RES / "e2_2_agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    log(f"  alpha R1={study.alpha_round1.get('overall'):.3f} -> R2={a2.get('overall'):.3f}  "
        f"raw_agree R2={diag2.get('mean_pairwise_raw_agreement'):.3f}  "
        f"AC1={diag2.get('gwet_ac1'):.3f}  disputed={len(study.disputed)}/{len(study.terms)}")

    # ---------------- E2.3 rank maps -------------------------------------------------
    rank_maps = {"R-A": study.consensus, "R-B": study.archive_native, "R-C": study.adversarial}
    (RES / "e2_3_rank_maps.json").write_text(json.dumps({
        "R-A_consensus": study.consensus,
        "R-B_archive_native": study.archive_native,
        "R-C_adversarial": study.adversarial,
        "n_flipped_RA_to_RC": sum(1 for t in study.terms if study.consensus[t] != study.adversarial[t]),
        "n_differ_RA_to_RB": sum(1 for t in study.terms if study.consensus[t] != study.archive_native[t]),
    }, indent=2), encoding="utf-8")

    # ---------------- E4.1/E4.2/E4.4 -------------------------------------------------
    complexes: dict[tuple[str, str], object] = {}
    sanity, roundtrip, decoupling, summaries = {}, {}, {}, {}
    for rm_name, rm in rank_maps.items():
        for g in GRANULARITIES:
            key = f"{rm_name}|{g}"
            log(f"E4.1 - building complex {key}")
            cx = build_complex(segments, rm, granularity=g, rank_map_name=rm_name)
            complexes[(rm_name, g)] = cx
            summaries[key] = cx.summary()
            sanity[key] = sanity_assertions(cx)
            decoupling[key] = rank_cardinality_decoupling(cx)
            if g == "mid":
                roundtrip[key] = verify_star_roundtrip(cx)
            save_complex(cx, ROOT / "complexes" / f"densho_{rm_name}_{g}.json")

    (RES / "e4_1_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (RES / "e4_1_sanity.json").write_text(json.dumps(sanity, indent=2, default=str), encoding="utf-8")
    (RES / "e4_2_star_roundtrip.json").write_text(json.dumps(roundtrip, indent=2), encoding="utf-8")
    (RES / "e4_4_decoupling.json").write_text(json.dumps(decoupling, indent=2), encoding="utf-8")

    prim = decoupling["R-A|mid"]
    log(f"E4.4 - rho(rank,|x|)={prim['spearman_rho']:.3f}  premise_holds={prim['premise_holds']}  "
        f"inversion_rate={prim['rank_inversion_rate']:.3f}")
    log(f"E4.2 - star round-trip lossless={roundtrip['R-A|mid']['lossless']}")

    # ---------------- E4.3 granularity sweep -----------------------------------------
    log("E4.3 - granularity sweep + RBO")
    sweep: dict = {"per_granularity": {}, "rbo_raw_labels": {}, "rbo_referent": {},
                   "jaccard_referent": {}}
    tri_raw: dict[str, list[str]] = {}
    tri_ref: dict[str, list[str]] = {}
    for g in GRANULARITIES:
        cx = complexes[("R-A", g)]
        r2 = cx.by_rank(2)
        sizes = np.array([c.size for c in r2]) if r2 else np.array([0])

        def singleton_fraction(units: list[str], _g=g) -> float:
            """Recomputed from a resampled narrator set. Units must stay distinct - see
            cluster_subsample_ci for why replacement corrupts this particular statistic."""
            keep = set(units)
            vals = [len(c.members & keep) for c in complexes[("R-A", _g)].by_rank(2)]
            vals = [v for v in vals if v > 0]
            return float(np.mean([v == 1 for v in vals])) if vals else 0.0

        ci = cluster_subsample_ci(singleton_fraction, cx.narrators, B=1500, seed=0)
        cells = triage_cells(cx, 50)
        tri_raw[g] = [c.label for c in cells]
        tri_ref[g] = project(cells)
        sweep["per_granularity"][g] = {
            "n_rank2_cells": len(r2),
            "attestation": {
                "mean": float(sizes.mean()), "median": float(np.median(sizes)),
                "max": int(sizes.max()), "p90": float(np.percentile(sizes, 90)),
                "hist": {b: int(((sizes >= lo) & (sizes < hi)).sum())
                         for b, (lo, hi) in {"1": (1, 2), "2-3": (2, 4), "4-10": (4, 11),
                                             "11-50": (11, 51), ">50": (51, 10**9)}.items()},
            },
            "singleton_fraction": ci,
            "cells_per_rank": {f"rank{k}": len(cx.by_rank(k)) for k in range(4)},
        }

    pairs = [(a, b) for i, a in enumerate(GRANULARITIES) for b in GRANULARITIES[i + 1:]]
    for p in (0.8, 0.9, 0.95):
        sweep["rbo_raw_labels"][f"p={p}"] = {f"{a}|{b}": rbo(tri_raw[a], tri_raw[b], p=p) for a, b in pairs}
        sweep["rbo_referent"][f"p={p}"] = {f"{a}|{b}": rbo(tri_ref[a], tri_ref[b], p=p) for a, b in pairs}
    for k in (10, 25, 50):
        sweep["jaccard_referent"][f"@{k}"] = {f"{a}|{b}": jaccard_at(tri_ref[a], tri_ref[b], k)
                                              for a, b in pairs}
    sweep["triage_lists_raw"] = tri_raw
    sweep["triage_lists_referent"] = tri_ref
    sweep["why_projection"] = (
        "Raw cell labels are disjoint across granularities by construction (coarse keys are "
        "parent paths, fine keys carry a place suffix), so raw-label RBO is identically zero "
        "and measures the labelling scheme rather than the triage. The pre-registered "
        "criterion is therefore evaluated on lists projected onto the shared archival "
        "referent (depth-2 topic path). Both are reported."
    )
    ref90 = sweep["rbo_referent"]["p=0.9"]
    sweep["kill_criterion"] = {
        "threshold": 0.60,
        "rbo_referent_p0.9": ref90,
        "rbo_raw_p0.9": sweep["rbo_raw_labels"]["p=0.9"],
        "min_pair": min(ref90.values()) if ref90 else None,
        "all_pairs_below_threshold": all(v < 0.60 for v in ref90.values()) if ref90 else None,
        "H3_granularity_leg_met": all(v >= 0.60 for v in ref90.values()) if ref90 else None,
    }
    (RES / "e4_3_granularity.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    log(f"E4.3 - RBO(referent) p=0.9: {json.dumps({k: round(v, 3) for k, v in ref90.items()})}")
    log("PHASE 2+4 complete")


if __name__ == "__main__":
    main()
