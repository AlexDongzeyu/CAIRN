"""Run PHASE 8 (perturbation), PHASE 9 (triage), PHASE 11 (topology), PHASE 12 (figures).

E8.1 calibrated perturbation      -> data/results/e8_1_perturbation.json
E8.2 extractor ablation           -> data/results/e8_2_extractor.json
E8.4 interviewer-turn ablation    -> data/results/e8_4_interviewer.json
E9.2 triage list + stability      -> data/results/e9_2_triage.json
E9.3 frequency-baseline contrast  -> data/results/e9_3_frequency_baseline.json
E9.4 cost of error                -> data/results/e9_4_cost_of_error.json
E11  topology + the kill test     -> data/results/e11_topology.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import GRANULARITIES, build_complex, referent  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import cell_text_embeddings, get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.metrics import cluster_subsample_ci, rbo  # noqa: E402
from src.perturb import degree_ks, estimate_error_rates, perturb, singleton_fraction, triage  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.topology import permutation_test, persistence_diagram, simpler_explanation_check  # noqa: E402

RES = ROOT / "data" / "results"
RES.mkdir(parents=True, exist_ok=True)
log = make_logger("phase8_11")
TRIAGE_K = 50


def proj(c):
    return referent(c)


def load_rank_maps() -> dict[str, dict[str, int]]:
    d = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    return {"R-A": d["R-A_consensus"], "R-B": d["R-B_archive_native"], "R-C": d["R-C_adversarial"]}


def load_e3() -> dict | None:
    p = RES / "e3_extraction.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    set_all_seeds(0)
    interviews, segments = load_corpus()
    rank_maps = load_rank_maps()
    log(f"corpus: {len(interviews)} interviews / {len(segments)} segments")

    encoder = get_encoder()
    cx = build_complex(segments, rank_maps["R-A"], granularity="mid", rank_map_name="R-A")
    cell_emb = cell_text_embeddings(cx, segments, 2, encoder)
    base_triage = triage(cx, TRIAGE_K, project=proj)
    log(f"rank-2 cells={len(cx.by_rank(2))}  triage len={len(base_triage)}")

    # ---------------- E8.1 calibrated perturbation ----------------------------------
    # Two calibrations, because they answer different questions:
    #   archive-internal - how ambiguous is the archive's OWN labelling? The complex is
    #                      built from that labelling, so this is the error already present.
    #   E3.5-measured    - what would an automatic extractor cost? This is the protocol's
    #                      intended calibration and the rate anyone applying the method to
    #                      an unstructured archive would actually face.
    internal = estimate_error_rates(cx)
    e3 = load_e3()
    auto = (e3 or {}).get("E3_5_error_decomposition", {})
    calibrations = {
        "archive_internal": (internal["rho_merge"], internal["rho_split"]),
        "E3_5_automatic_extractor": (auto.get("rho_merge", 0.0), auto.get("rho_split", 0.0)),
    }
    log(f"E8.1 - calibrations {calibrations}")

    grids: dict[str, dict] = {}
    MULTS = (0.0, 0.5, 1.0, 2.0, 4.0)
    for cal_name, (rm, rs) in calibrations.items():
        grid = {}
        # Protocol E8.1: rho_m and rho_s are swept independently -> 5 x 5 = 25 grid points.
        for mm in MULTS:
            for ms in MULTS:
                rbos, sing, ks = [], [], []
                for draw in range(20):
                    rng = np.random.default_rng(9000 + 137 * draw + int(mm * 10) * 31 + int(ms * 10))
                    p = perturb(cx, min(1.0, rm * mm), min(1.0, rs * ms), rng, cell_emb=cell_emb)
                    rbos.append(rbo(base_triage, triage(p, TRIAGE_K, project=proj), p=0.9))
                    sing.append(singleton_fraction(p))
                    ks.append(degree_ks(cx, p))
                grid[f"m{mm}_s{ms}"] = {
                    "mult_merge": mm, "mult_split": ms,
                    "rho_merge": min(1.0, rm * mm), "rho_split": min(1.0, rs * ms),
                    "rbo50_mean": float(np.mean(rbos)), "rbo50_std": float(np.std(rbos)),
                    "singleton_fraction_mean": float(np.mean(sing)),
                    "degree_ks_mean": float(np.nanmean(ks)),
                }
            log(f"  [{cal_name}] merge x{mm}: "
                + " ".join(f"s{ms}={grid[f'm{mm}_s{ms}']['rbo50_mean']:.2f}" for ms in MULTS))
        grids[cal_name] = grid

    rm, rs = calibrations["E3_5_automatic_extractor"]
    primary_grid = grids["E3_5_automatic_extractor"]
    # the diagonal is the "both errors at k x the measured rate" curve used in Figure 4B
    diagonal = {f"x{m}": primary_grid[f"m{m}_s{m}"] for m in MULTS}
    at_measured = diagonal["x1.0"]["rbo50_mean"]
    (RES / "e8_1_perturbation.json").write_text(json.dumps({
        "calibrations": {k: {"rho_merge": v[0], "rho_split": v[1]} for k, v in calibrations.items()},
        "measured_rates": {"archive_internal": internal, "E3_5_automatic_extractor": auto},
        "grids": grids,
        "grid": diagonal,
        "n_grid_points_per_calibration": len(primary_grid),
        "primary_calibration": "E3_5_automatic_extractor",
        "kill_criterion": {
            "threshold": 0.70, "rbo50_at_measured_rate": at_measured,
            "passes": bool(at_measured >= 0.70),
            "consequence_if_failed": "triage is a screening step requiring human verification, "
                                     "not an output",
        },
    }, indent=2), encoding="utf-8")

    # ---------------- E8.2 extractor ablation ---------------------------------------
    # Three defensible ways of deciding which cells exist, standing in for S1/S2/S3.
    variants = {
        "S1_leaf_terms": {"granularity": "mid", "min_cell_size": 1},
        "S2_merged_parents": {"granularity": "coarse", "min_cell_size": 1},
        "S3_place_split": {"granularity": "fine", "min_cell_size": 1},
    }
    tri = {}
    for name, kw in variants.items():
        c = build_complex(segments, rank_maps["R-A"], rank_map_name="R-A", **kw)
        tri[name] = triage(c, TRIAGE_K, project=proj)
    ext = {f"{a}|{b}": rbo(tri[a], tri[b], p=0.9)
           for i, a in enumerate(tri) for b in list(tri)[i + 1:]}
    (RES / "e8_2_extractor.json").write_text(json.dumps(
        {"pairwise_rbo50": ext, "min": min(ext.values()) if ext else None,
         "note": "if three defensible extractors give unrelated triage lists that is a "
                 "first-order finding and belongs in the abstract"}, indent=2), encoding="utf-8")
    log(f"E8.2 - extractor RBO: { {k: round(v,3) for k,v in ext.items()} }")

    # ---------------- E8.4 interviewer-turn ablation --------------------------------
    cx_iv = build_complex(segments, rank_maps["R-A"], granularity="mid", rank_map_name="R-A",
                          include_interviewers=True)
    iv = {
        "singleton_fraction_narrators_only": singleton_fraction(cx),
        "singleton_fraction_with_interviewers": singleton_fraction(cx_iv),
        "triage_rbo50": rbo(base_triage, triage(cx_iv, TRIAGE_K, project=proj), p=0.9),
        "n_rank2_narrators_only": len(cx.by_rank(2)),
        "n_rank2_with_interviewers": len(cx_iv.by_rank(2)),
        "note": "interviewer prompts are archive artifacts, not witness attestations; "
                "including them quantifies the archival-protocol confound directly",
    }
    (RES / "e8_4_interviewer.json").write_text(json.dumps(iv, indent=2), encoding="utf-8")
    log(f"E8.4 - singleton {iv['singleton_fraction_narrators_only']:.3f} -> "
        f"{iv['singleton_fraction_with_interviewers']:.3f}, RBO={iv['triage_rbo50']:.3f}")

    # ---------------- E9.2 triage + stability ---------------------------------------
    conditions: dict[str, list[str]] = {}
    for g in GRANULARITIES:
        conditions[f"granularity={g}"] = triage(
            build_complex(segments, rank_maps["R-A"], granularity=g, rank_map_name="R-A"),
            TRIAGE_K, project=proj)
    for rmname in ("R-B", "R-C"):
        conditions[f"rank_map={rmname}"] = triage(
            build_complex(segments, rank_maps[rmname], granularity="mid", rank_map_name=rmname),
            TRIAGE_K, project=proj)
    conditions["interviewers_included"] = triage(cx_iv, TRIAGE_K, project=proj)
    for mult in (1.0, 2.0):
        rng = np.random.default_rng(7)
        conditions[f"perturbed_x{mult}"] = triage(
            perturb(cx, rm * mult, rs * mult, rng, cell_emb=cell_emb), TRIAGE_K, project=proj)
    for k, v in variants.items():
        conditions[f"extractor={k}"] = tri[k]

    stability = {}
    for item in base_triage:
        keeps = sum(1 for lst in conditions.values() if item in lst)
        stability[item] = keeps / len(conditions)

    narrators = cx.narrators
    cells_sorted = sorted(cx.by_rank(2), key=lambda c: (c.size, c.label))[:TRIAGE_K]
    per_item = []
    for c in cells_sorted:
        def att(units, _c=c):
            """Distinct-unit resampling; duplicated narrators would inflate attestation."""
            keep = set(units)
            return float(len(_c.members & keep))
        ci = cluster_subsample_ci(att, narrators, B=800, seed=0)
        per_item.append({
            "label": c.label, "referent": referent(c),
            "archive_conditioned_attestation_multiplicity": c.size,
            "attestation_ci": [ci["lo"], ci["hi"]],
            "stability_score": stability.get(referent(c), 0.0),
            "unstable": bool(stability.get(referent(c), 0.0) < 0.5),
            "n_supporting_segments": len(c.segments),
        })
    (RES / "e9_2_triage.json").write_text(json.dumps({
        "granularity": "mid", "rank_map": "R-A", "k": TRIAGE_K,
        "conditions_tested": list(conditions),
        "items": per_item,
        "n_unstable": sum(1 for x in per_item if x["unstable"]),
        "scope_statement": ("under-attested within the analysed collection, at granularity mid, "
                            "under rank map R-A; this is a property of what this archive "
                            "records, not of history"),
    }, indent=2), encoding="utf-8")
    log(f"E9.2 - triage items={len(per_item)} unstable={sum(1 for x in per_item if x['unstable'])}")

    # ---------------- E9.3 frequency baseline (mandatory contrast) -------------------
    from scipy.stats import spearmanr

    cells_all = sorted(cx.by_rank(2), key=lambda c: c.cid)
    att = np.array([c.size for c in cells_all], dtype=float)
    freq = np.array([len(c.segments) for c in cells_all], dtype=float)
    rho, p = spearmanr(att, freq)
    tri_att = [referent(c) for c in sorted(cells_all, key=lambda c: (c.size, c.label))[:TRIAGE_K]]
    tri_freq = [referent(c) for c in sorted(cells_all, key=lambda c: (len(c.segments), c.label))[:TRIAGE_K]]
    (RES / "e9_3_frequency_baseline.json").write_text(json.dumps({
        "spearman_attestation_vs_mention_frequency": float(rho),
        "p_value": float(p),
        "rbo50_triage_attestation_vs_frequency": rbo(tri_att, tri_freq, p=0.9),
        "jaccard_top10": len(set(tri_att[:10]) & set(tri_freq[:10])) / max(1, len(set(tri_att[:10]) | set(tri_freq[:10]))),
        "interpretation": ("if a bare mention-frequency ranking reproduces the structure-derived "
                           "ranking, the complex adds nothing to the application and we say so"),
        "expert_panel": "NOT RUN - no domain experts available (see PREREGISTRATION "
                        "substitution_ledger E9_3_expert_panel); the application claim is "
                        "downgraded to 'computationally stable', NOT 'expert-validated'",
    }, indent=2), encoding="utf-8")
    log(f"E9.3 - attestation vs frequency: rho={rho:.3f}, triage RBO={rbo(tri_att, tri_freq, 0.9):.3f}")

    # ---------------- E9.4 cost of error --------------------------------------------
    rng = np.random.default_rng(11)
    false_singleton = {}
    draws = 40
    for _ in range(draws):
        p = perturb(cx, rm, rs, rng, cell_emb=cell_emb)
        got = {referent(c) for c in p.by_rank(2) if c.size == 1}
        for c in cells_sorted:
            r = referent(c)
            false_singleton[r] = false_singleton.get(r, 0) + (1 if r in got else 0)
    (RES / "e9_4_cost_of_error.json").write_text(json.dumps({
        "per_item_singleton_under_perturbation": {k: v / draws for k, v in false_singleton.items()},
        "measured_false_singleton_rate_E3_5": auto.get("false_singleton_rate"),
        "measured_false_rescue_rate_E3_5": auto.get("false_rescue_rate"),
        "E3_5_kill_criterion_passes": auto.get("kill_criterion", {}).get("passes"),
        "note": ("each triage item is published with the probability it is a singleton only "
                 "because of pipeline error, not as a bare ranking; the E3.5 rates describe an "
                 "automatic extractor, whereas the complex reported here is built directly from "
                 "archive curation"),
    }, indent=2), encoding="utf-8")

    # ---------------- E11 topology ---------------------------------------------------
    log("E11 - attestation filtration")
    diag = persistence_diagram(cx)
    perm = permutation_test(cx, n_perm=200, seed=0)
    simpler = simpler_explanation_check(cx, n_boot=60, seed=0)
    stability_topo = []
    for mult in (1.0, 2.0):
        rng = np.random.default_rng(5)
        p = perturb(cx, rm * mult, rs * mult, rng, cell_emb=cell_emb)
        stability_topo.append({"mult": mult, "betti0_auc": persistence_diagram(p)["betti0_auc"]})
    # beta_0 is constant on this complex, so its permutation test cannot reject anything and
    # quoting its p-value would be quoting arithmetic. The decision uses whichever Betti
    # summary actually varies.
    if perm.get("betti0_is_degenerate") and not perm.get("betti1", {}).get("degenerate", True):
        decisive_p, decisive_stat = perm["betti1"]["p_value"], "betti1_auc"
    else:
        decisive_p, decisive_stat = perm["p_value"], "betti0_auc"
    keep = (not simpler.get("topology_is_redundant", True)) and decisive_p < 0.05
    (RES / "e11_topology.json").write_text(json.dumps({
        "persistence": diag, "permutation_test": perm, "simpler_explanation_check": simpler,
        "perturbation_stability": stability_topo,
        "kill_criterion": {
            "requires": ["not recoverable from event-size distribution (R2 <= 0.9)",
                         "stable under E8.1 perturbation",
                         "changes an archival decision"],
            "recoverable_from_size_distribution": simpler.get("topology_is_redundant"),
            "permutation_p": perm["p_value"],
            "decisive_permutation_p": decisive_p,
            "decisive_statistic": decisive_stat,
            "verdict_keep_section": bool(keep),
            "action": ("KEEP - and still only if it changes an archival decision" if keep
                       else "DELETE the topology section; it adds no contribution and "
                            "unmotivated persistent homology costs credibility"),
        },
    }, indent=2), encoding="utf-8")
    log(f"E11 - simpler-explanation R2(b0)={simpler.get('r2_size_and_connectivity_only')} "
        f"R2(b1)={simpler.get('r2_betti1_size_and_connectivity_only')} "
        f"decisive_p={decisive_p} ({decisive_stat}) keep={keep}")
    log("PHASE 8+9+11 complete")


if __name__ == "__main__":
    main()
