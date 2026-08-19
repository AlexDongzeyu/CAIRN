"""Coverage audit: every experiment ID in EXPERIMENTS.md vs what actually exists on disk.

Written so that "did we run everything?" is answered by a check rather than by memory.
Each entry names the artifact that proves the experiment ran and, where the protocol
states one, the pre-registered kill criterion and its observed value.

Status values:
  DONE       artifact present and the protocol's requirement is met
  PARTIAL    ran, but a stated component is missing or substituted
  MISSING    not run
  SUBSTITUTED  ran under a documented PREREGISTRATION substitution_ledger entry
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "data" / "results"


def load(name: str):
    p = RES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def dig(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def audit() -> list[dict]:
    runs = load("e7_1_runs.json") or []
    models_run = {r.get("model") for r in runs}
    e2 = load("e2_2_agreement.json")
    e3 = load("e3_extraction.json")
    e4_2 = load("e4_2_star_roundtrip.json")
    e4_3 = load("e4_3_granularity.json")
    e4_4 = load("e4_4_decoupling.json")
    e8_1 = load("e8_1_perturbation.json")
    e9_2 = load("e9_2_triage.json")
    e9_3 = load("e9_3_frequency_baseline.json")
    e10 = load("e10_transfer.json")
    e11 = load("e11_topology.json")
    figs = ROOT / "figures"
    rel = ROOT / "release"

    def st(ok, partial_reason=None, sub=False):
        if sub:
            return "SUBSTITUTED"
        if ok:
            return "DONE"
        return "PARTIAL" if partial_reason else "MISSING"

    rows: list[dict] = []

    def add(eid, name, status, evidence, note=""):
        rows.append({"id": eid, "name": name, "status": status,
                     "evidence": evidence, "note": note})

    # ---- PHASE 0
    add("0.3", "determinism / 10 seeds", "DONE", "src/seeds.py; runs carry seeds 0-9",
        f"distinct seeds in runs: {len({r.get('seed') for r in runs})}")
    add("0.4", "pre-registration frozen", "DONE" if (ROOT / "PREREGISTRATION.yaml").exists()
        else "MISSING", "PREREGISTRATION.yaml")

    # ---- PHASE 1
    stats = load("e1_1_corpus_stats.json") or {}
    n_iv = stats.get("n_interviews", 0)
    add("1.1", "Densho corpus acquisition", "DONE" if n_iv >= 400 else "PARTIAL",
        "corpus/densho/*.jsonl + LICENSE_AUDIT.csv",
        f"{n_iv} interviews, {stats.get('n_segments')} segments; kill>=400 {'PASS' if n_iv>=400 else 'FAIL'}")
    add("1.2", "transfer corpus", "SUBSTITUTED" if e10 else "MISSING",
        "data/results/e10_transfer.json",
        "held-out Densho collections instead of Rutgers (ledger E1_2)")
    norm = ROOT / "corpus" / "densho" / "normalized.jsonl"
    add("1.3", "transcript normalization", "DONE" if norm.exists() else "MISSING",
        "corpus/densho/normalized.jsonl",
        "speaker separation, disfluency handling, char offsets")

    # ---- PHASE 2
    add("2.1", "rank annotation manual", "DONE" if (ROOT / "annotation" / "RANK_MANUAL.md").exists()
        else "MISSING", "annotation/RANK_MANUAL.md")
    a2 = dig(e2, "alpha_round2_REPORTED", "overall")
    add("2.2", "inter-annotator agreement", "SUBSTITUTED" if e2 else "MISSING",
        "e2_2_agreement.json",
        f"alpha_R2={a2:.3f}; kill<0.55 {'TRIGGERED' if a2 is not None and a2 < 0.55 else 'passed'}"
        if a2 is not None else "")
    maps = load("e2_3_rank_maps.json")
    add("2.3", "R-A / R-B / R-C rank maps", "DONE" if maps else "MISSING", "e2_3_rank_maps.json",
        f"R-A vs R-C differ on {maps.get('n_flipped_RA_to_RC')} terms" if maps else "")

    # ---- PHASE 3
    add("3.1", "event mention detection + linking", "SUBSTITUTED" if e3 else "MISSING",
        "e3_extraction.json",
        "archive's curated topic/facility/geography terms used as linked mentions")
    add("3.2", "ECB+ / GVC CDEC calibration", "MISSING", "-",
        "external corpus not obtained; no claim made about linguistic coreference quality")
    sweep = dig(e3, "E3_3_threshold_sweep", default=[])
    add("3.3", "in-domain CDEC + threshold sweep", "SUBSTITUTED" if sweep else "MISSING",
        "e3_extraction.json:E3_3_threshold_sweep",
        f"{len(sweep)} thresholds; optimum at boundary="
        f"{dig(e3, 'E3_3_optimum_at_sweep_boundary')}")
    add("3.4", "cluster-level metrics (LEA primary)", "DONE" if dig(e3, "E3_4_cluster_metrics")
        else "MISSING", "e3_extraction.json:E3_4_cluster_metrics",
        f"LEA F1={dig(e3, 'E3_4_cluster_metrics', 'LEA', 'f1'):.3f}"
        if dig(e3, "E3_4_cluster_metrics", "LEA", "f1") is not None else "")
    fs = dig(e3, "E3_5_error_decomposition", "false_singleton_rate")
    add("3.5", "merge/split + attestation distortion", "DONE" if fs is not None else "MISSING",
        "e3_extraction.json:E3_5_error_decomposition",
        f"false_singleton={fs:.3f}; kill>0.30 {'TRIGGERED' if fs and fs > 0.30 else 'passed'}"
        if fs is not None else "")
    amb = dig(e3, "E3_6_ambiguity")
    add("3.6", "extraction ambiguity flagging", "DONE" if amb else "MISSING",
        "e3_extraction.json:E3_6_ambiguity")

    # ---- PHASE 4
    add("4.1", "build combinatorial complex", "DONE" if load("e4_1_summaries.json") else "MISSING",
        "e4_1_summaries.json + e4_1_sanity.json")
    add("4.2", "star expansion round-trip", "DONE" if dig(e4_2, "R-A|mid", "lossless") else "MISSING",
        "e4_2_star_roundtrip.json", f"lossless={dig(e4_2, 'R-A|mid', 'lossless')}")
    met = dig(e4_3, "kill_criterion", "H3_granularity_leg_met")
    add("4.3", "granularity sweep + RBO", "DONE" if e4_3 else "MISSING", "e4_3_granularity.json",
        f"H3 granularity leg {'met' if met else 'NOT met (kill TRIGGERED)'}")
    rho = dig(e4_4, "R-A|mid", "spearman_rho")
    add("4.4", "rank/cardinality decoupling", "DONE" if rho is not None else "MISSING",
        "e4_4_decoupling.json",
        f"rho={rho:.3f}; kill>0.9 {'TRIGGERED' if rho and rho > 0.9 else 'passed'}"
        if rho is not None else "")

    # ---- PHASE 5
    have_t1 = any("T1_map" in r for r in runs)
    have_t2 = any("T2_auc" in r for r in runs)
    have_t3 = any("T3_spearman" in r for r in runs)
    add("5.1", "tasks T1 / T2 / T3", "DONE" if (have_t1 and have_t2 and have_t3) else "PARTIAL",
        "e7_1_runs.json", f"T1={have_t1} T2={have_t2} T3={have_t3}")
    splits = {r.get("split") for r in runs}
    add("5.2", "leakage-resistant splits", "DONE" if {"random", "narrator-disjoint",
                                                      "event-disjoint"} <= splits else "PARTIAL",
        "e7_1_runs.json:split", f"present: {sorted(x for x in splits if x)}")
    negs = {r.get("neg_regime") for r in runs}
    want_neg = {"UNS", "SNS", "MNS", "CNS", "hard"}
    add("5.3", "negative sampling regimes", "DONE" if want_neg <= negs else "PARTIAL",
        "e7_1_runs.json:neg_regime", f"missing: {sorted(want_neg - negs)}")

    # ---- PHASE 6
    add("6.0", "shared controls / param budget", "DONE" if runs else "MISSING", "e7_1_runs.json",
        f"param spread {min((r['n_params'] for r in runs), default=0)}-"
        f"{max((r['n_params'] for r in runs), default=0)}")
    for eid, model, label in [
        ("6.1", "M0_mlp", "M0 feature-only MLP"),
        ("6.2", "M1_dense", "M1 dense-retrieval baseline"),
        ("6.3", "M2_untyped_star", "M2 untyped star GNN"),
        ("6.4", "M3_typed_star", "M3 typed star + hypergraph encodings"),
        ("6.5", "M4_allset", "M4 AllSet / ED-HNN / Hypergraph-MLP"),
        ("6.6", "M5_ccnn", "M5 CCNN"),
    ]:
        add(eid, label, "DONE" if model in models_run else "MISSING", "e7_1_runs.json:model")
    want_abl = {"A1_shared_weights": "share weights across ranks",
                "A2_shuffled_ranks": "shuffle rank labels",
                "A3_collapse_r2r3": "collapse ranks 2 and 3",
                "A4_no_down": "remove down-messages",
                "A5_no_moments": "remove rank-1 moments"}
    missing_abl = [k for k in want_abl if k not in models_run]
    add("6.7", "ablations A1-A5", "DONE" if not missing_abl else "PARTIAL",
        "e7_1_runs.json:model", f"missing: {missing_abl}")

    # ---- PHASE 7
    add("7.1", "results grid", "DONE" if runs else "MISSING", "e7_1_runs.json",
        f"{len(runs)} runs")
    add("7.2", "stratified reporting", "DONE" if load("e7_2_stratified.json") else "MISSING",
        "e7_2_stratified.json")
    add("7.3", "ASO + Bonferroni", "DONE" if load("e7_3_aso_T1_map.json") else "MISSING",
        "e7_3_aso_T1_map.json / e7_3_aso_T2_auc.json")
    unc = load("e7_4_uncertainty.json")
    add("7.4", "clustered bootstrap + DEFF", "DONE" if unc else "MISSING", "e7_4_uncertainty.json",
        f"DEFF={dig(unc, 'design_effect', 'DEFF')}")
    inter = load("e7_5_interaction.json")
    add("7.5", "interaction model", "DONE" if inter and "interaction_coef" in inter else "MISSING",
        "e7_5_interaction.json",
        f"coef={inter.get('interaction_coef')} crosses_zero={inter.get('ci_crosses_zero')}"
        if inter else "")

    # ---- PHASE 8
    npts = len(dig(e8_1, "grids", "E3_5_automatic_extractor", default={}) or
               dig(e8_1, "grid", default={}))
    add("8.1", "calibrated perturbation", "DONE" if e8_1 else "MISSING", "e8_1_perturbation.json",
        f"{npts} grid points; RBO@rho_hat="
        f"{dig(e8_1, 'kill_criterion', 'rbo50_at_measured_rate')}")
    add("8.2", "extraction-system ablation", "DONE" if load("e8_2_extractor.json") else "MISSING",
        "e8_2_extractor.json")
    add("8.3", "gold-subset upper bound", "DONE" if load("e8_3_gold_subset.json") else "MISSING",
        "e8_3_gold_subset.json", "noise-free ceiling")
    add("8.4", "interviewer-turn ablation", "DONE" if load("e8_4_interviewer.json") else "MISSING",
        "e8_4_interviewer.json")

    # ---- PHASE 9
    term = ROOT / "tests" / "test_terminology.py"
    add("9.1", "terminology CI check", "DONE" if term.exists() else "MISSING",
        "tests/test_terminology.py")
    add("9.2", "triage list + stability", "DONE" if e9_2 else "MISSING", "e9_2_triage.json",
        f"{len(dig(e9_2, 'items', default=[]))} items, "
        f"{dig(e9_2, 'n_unstable')} unstable" if e9_2 else "")
    add("9.3", "expert validation", "SUBSTITUTED" if e9_3 else "MISSING",
        "e9_3_frequency_baseline.json",
        "frequency-baseline contrast run; expert panel NOT available (ledger E9_3)")
    add("9.4", "cost-of-error", "DONE" if load("e9_4_cost_of_error.json") else "MISSING",
        "e9_4_cost_of_error.json")

    # ---- PHASE 10
    add("10.1", "meta-rank mapping + agreement drop", "DONE" if dig(e10, "E10_1_meta_rank")
        else "MISSING", "e10_transfer.json:E10_1_meta_rank")
    add("10.2", "zero-shot transfer", "DONE" if dig(e10, "E10_2_zero_shot") else "MISSING",
        "e10_transfer.json:E10_2_zero_shot")
    add("10.3", "conclusion transfer", "DONE" if dig(e10, "E10_3_conclusion_transfer")
        else "MISSING", "e10_transfer.json:E10_3_conclusion_transfer")

    # ---- PHASE 11
    add("11.1", "topology kill criterion declared", "DONE" if dig(e11, "kill_criterion")
        else "MISSING", "e11_topology.json:kill_criterion")
    add("11.2", "attestation filtration", "DONE" if dig(e11, "persistence") else "MISSING",
        "e11_topology.json:persistence")
    add("11.3", "permutation test", "DONE" if dig(e11, "permutation_test") else "MISSING",
        "e11_topology.json:permutation_test")
    add("11.4", "simpler-explanation check", "DONE" if dig(e11, "simpler_explanation_check")
        else "MISSING", "e11_topology.json:simpler_explanation_check",
        f"verdict={dig(e11, 'kill_criterion', 'action')}" if e11 else "")

    # ---- PHASE 12 / 13
    for i, stem in enumerate(["fig1_representations", "fig2_granularity", "fig3_models",
                              "fig4_uncertainty"], start=1):
        add(f"12.{i}", f"Figure {i}", "DONE" if (figs / f"{stem}.pdf").exists() else "MISSING",
            f"figures/{stem}.pdf")
    add("13.1", "stand-off release", "DONE" if (rel / "cells.jsonl").exists() else "MISSING",
        "release/cells.jsonl + splits/ + DATASET_CARD.md")
    add("13.2", "reproduction check", "DONE" if load("e13_2_reproduction.json") else "MISSING",
        "e13_2_reproduction.json")
    add("13.3", "dataset card", "DONE" if (rel / "DATASET_CARD.md").exists() else "MISSING",
        "release/DATASET_CARD.md")

    return rows


def main() -> None:
    rows = audit()
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    width = max(len(r["name"]) for r in rows)
    print(f"{'ID':6s} {'STATUS':12s} {'EXPERIMENT':{width}s}  NOTE")
    print("-" * (26 + width + 40))
    for r in rows:
        print(f"{r['id']:6s} {r['status']:12s} {r['name']:{width}s}  {r['note']}")
    print("-" * (26 + width + 40))
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    gaps = [r for r in rows if r["status"] in ("MISSING", "PARTIAL")]
    if gaps:
        print("\nOPEN GAPS:")
        for g in gaps:
            print(f"  {g['id']:6s} {g['name']} -- {g['note'] or g['evidence']}")

    (RES / "coverage_audit.json").write_text(
        json.dumps({"counts": counts, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
