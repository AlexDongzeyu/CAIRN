"""Assemble the paper from the results files.

Two rules make number-fabrication structurally difficult rather than merely discouraged:

  1. Every quantity that appears in the prose is defined as a LaTeX macro in
     ``numbers.tex``, generated here by reading the results JSON. Prose cites the macro.
     A macro whose source is missing renders as a loud ??? and is listed in the manifest,
     so an unfinished experiment cannot quietly become a confident sentence.
  2. ``lint_prose`` scans the written sections for bare numerals. Anything that is not a
     macro, a citation year, a section number or a unit is reported. The check is advisory,
     but it makes a hard-coded result visible in review.

The manifest records, for every macro, the file and JSON path it came from, so any number
in the PDF can be traced back to the run that produced it.
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
OUT = ROOT / "paper"
log = make_logger("paper")

MISSING = object()


def load(name: str):
    p = RES / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(f"  {name}.json is not valid JSON")
        return None


def dig(obj, path: str):
    """Resolve 'a/b/0/c' against nested dicts and lists.

    The separator is '/' rather than '.' because result keys legitimately contain dots
    ('p=0.9') and pipes ('coarse|mid'); splitting on '.' silently turned a real path into
    a missing one.
    """
    cur = obj
    for part in path.split("/"):
        if cur is None:
            return MISSING
        if isinstance(cur, list):
            if not part.lstrip("-").isdigit() or int(part) >= len(cur):
                return MISSING
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            if part not in cur:
                return MISSING
            cur = cur[part]
        else:
            return MISSING
    return cur


# --- formatters ---------------------------------------------------------------------

def f3(v):
    return f"{float(v):.3f}"


def f2(v):
    return f"{float(v):.2f}"


def f1(v):
    return f"{float(v):.1f}"


def f0(v):
    return f"{float(v):.0f}"


def sci(v):
    """LaTeX math for a p-value; used inside $...$ so tiny values stay informative.

    Rounding 1.55e-09 to three decimals prints 0.000, which reads as a rounding artefact
    rather than as the strongest evidence in the paper.
    """
    v = float(v)
    if v == 0:
        return r"< 10^{-16}"
    if v >= 1e-3:
        return f"{v:.3f}"
    exp = math.floor(math.log10(v))
    return rf"{v / 10 ** exp:.1f} \times 10^{{{exp}}}"


def pct1(v):
    return f"{100 * float(v):.1f}\\%"


def intc(v):
    return f"{int(v):,}"


def yesno(v):
    return "yes" if bool(v) else "no"


def verbatim(v):
    return str(v).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def count(v):
    return f"{len(v):,}"


# (macro, source file, json path, formatter)
SPEC = [
    # ---- corpus -------------------------------------------------------------------
    ("nInterviews", "e1_1_corpus_stats", "n_interviews", intc),
    ("nSegments", "e1_1_corpus_stats", "n_segments", intc),
    ("nNarrators", "e1_1_corpus_stats", "n_narrators", intc),
    ("nTranscripts", "e1_1_corpus_stats", "n_interviews_with_transcript", intc),
    # ---- E1.3 speaker normalisation -------------------------------------------------
    ("labelCoverage", "e1_3_normalization", "label_coverage", pct1),
    ("nTurns", "e1_3_normalization", "n_turns_total", intc),
    ("turnAccuracy", "e1_3_normalization", "turn_classification_accuracy", f3),
    ("charsRemoved", "e1_3_normalization", "pct_chars_removed_by_cleaning", pct1),
    # ---- E2 rank agreement ----------------------------------------------------------
    ("nRankTerms", "e2_2_agreement", "n_terms", intc),
    ("alphaRankTwo", "e2_2_agreement", "alpha_round2_REPORTED/overall", f3),
    ("alphaRankOne", "e2_2_agreement", "alpha_round1/overall", f3),
    ("alphaHead", "e2_2_agreement", "alpha_round2_REPORTED/head", f3),
    ("alphaTail", "e2_2_agreement", "alpha_round2_REPORTED/tail", f3),
    ("rawAgreement", "e2_2_agreement", "diagnostics_round2/mean_pairwise_raw_agreement", f3),
    ("gwetAC", "e2_2_agreement", "diagnostics_round2/gwet_ac1", f3),
    ("deferLexical", "e2_2_locus", "defer_rate_A2_lexical", pct1),
    ("deferBoth", "e2_2_locus", "defer_rate_both", pct1),
    ("nDisputed", "e2_2_agreement", "n_disputed", intc),
    # ---- E3 extraction ---------------------------------------------------------------
    ("leaFone", "e3_extraction", "E3_3_selected/lea_f1", f3),
    ("leaThreshold", "e3_extraction", "E3_3_selected/threshold", f2),
    ("rhoMerge", "e3_extraction", "E3_5_error_decomposition/rho_merge", pct1),
    ("rhoSplit", "e3_extraction", "E3_5_error_decomposition/rho_split", pct1),
    ("extractorCorr", "e3_extraction",
     "E3_5_error_decomposition/spearman_a_gold_vs_a_system", f3),
    ("falseSingleton", "e3_extraction", "E3_5_error_decomposition/false_singleton_rate", f3),
    ("nAmbiguous", "e3_extraction", "E3_6_ambiguity/ambiguous_terms", count),
    ("nExtractedEvents", "e3_extraction", "E3_3_selected/n_clusters", intc),
    # ---- E4 structure ----------------------------------------------------------------
    ("starLossless", "e4_2_star_roundtrip", "R-A|mid/lossless", yesno),
    ("nCells", "e4_2_star_roundtrip", "R-A|mid/n_cells", intc),
    ("spearmanRho", "e4_4_decoupling", "R-A|mid/spearman_rho", f3),
    ("inversionRate", "e4_4_decoupling", "R-A|mid/rank_inversion_rate", pct1),
    ("concordanceRate", "e4_4_decoupling", "R-A|mid/size_concordance_rate", pct1),
    ("invTwoThree", "e4_4_decoupling", "R-A|mid/rank2_vs_rank3/inversion_rate", pct1),
    ("rhoTwoThree", "e4_4_decoupling", "R-A|mid/rank2_vs_rank3/spearman_rho", f3),
    ("pairsTwoThree", "e4_4_decoupling", "R-A|mid/rank2_vs_rank3/n_pairs", intc),
    ("largestRankTwo", "e4_4_decoupling", "R-A|mid/extremes/rank2/largest/size", intc),
    ("smallestRankThree", "e4_4_decoupling", "R-A|mid/extremes/rank3/smallest/size", intc),
    ("largestRankThree", "e4_4_decoupling", "R-A|mid/extremes/rank3/largest/size", intc),
    # ---- size-only rank predictor (the direct test behind P1) ------------------------
    ("sizeOnlyN", "e_size_only", "R-A|mid/rank2_vs_rank3/n", intc),
    ("sizeOnlyTree", "e_size_only", "R-A|mid/rank2_vs_rank3/tree_depth3/balanced_accuracy", f3),
    ("sizeOnlyLogit", "e_size_only", "R-A|mid/rank2_vs_rank3/logistic/balanced_accuracy", f3),
    ("sizeOnlyPrior", "e_size_only", "R-A|mid/rank2_vs_rank3/prior_only/balanced_accuracy", f3),
    ("sizeOnlyMI", "e_size_only", "R-A|mid/mutual_information_nats_rank2_vs_rank3", f3),
    ("sizeOnlyMax", "e_size_only", "max_balanced_accuracy_rank2_vs_rank3/value", f3),
    # ---- split-first features, input-matched star, and the clustered bootstrap ---------
    ("sfNarrTouched", "e_paired_bootstrap", "exposure/random/narrators_losing_text", intc),
    ("sfMeanRemoved", "e_paired_bootstrap",
     "exposure/random/mean_removed_fraction_among_affected", pct1),
    ("sfLiftFiveGlobal", "e_paired_bootstrap", "split_lift/global/M5_ccnn", f3),
    ("sfLiftFiveSF", "e_paired_bootstrap", "split_lift/split-first/M5_ccnn", f3),
    ("sfLiftThreeGlobal", "e_paired_bootstrap", "split_lift/global/M3_matched", f3),
    ("sfLiftThreeSF", "e_paired_bootstrap", "split_lift/split-first/M3_matched", f3),
    ("sfMthreeMatched", "e_paired_bootstrap", "means/global/narrator-disjoint/M3_matched", f3),
    ("sfMthreeStrong", "e_paired_bootstrap", "means/global/narrator-disjoint/M3_strong", f3),
    ("sfMfive", "e_paired_bootstrap", "means/global/narrator-disjoint/M5_ccnn", f3),
    ("didPoint", "e_paired_bootstrap",
     "did/global|M5-M3_strong/difference_in_differences", f3),
    ("didLo", "e_paired_bootstrap", "did/global|M5-M3_strong/ci95/0", f3),
    ("didHi", "e_paired_bootstrap", "did/global|M5-M3_strong/ci95/1", f3),
    ("didNarrators", "e_paired_bootstrap",
     "did/global|M5-M3_strong/n_narrators_resampled", intc),
    # ---- E4.3 granularity stability --------------------------------------------------
    ("rboCoarseMid", "e4_3_granularity", "rbo_referent/p=0.9/coarse|mid", f3),
    ("rboCoarseFine", "e4_3_granularity", "rbo_referent/p=0.9/coarse|fine", f3),
    ("rboMidFine", "e4_3_granularity", "rbo_referent/p=0.9/mid|fine", f3),
    ("rboThreshold", "e4_3_granularity", "kill_criterion/threshold", f2),
    # ---- E7.5 interaction ------------------------------------------------------------
    ("interactionCoef", "e7_5_interaction", "interaction_coef", f3),
    ("interactionP", "e7_5_interaction", "interaction_p", sci),
    ("interactionCiLo", "e7_5_interaction", "interaction_ci/0", f3),
    ("interactionCiHi", "e7_5_interaction", "interaction_ci/1", f3),
    ("interactionNarrators", "e7_5_interaction", "n_narrators", intc),
    ("interactionEvents", "e7_5_interaction", "n_events", intc),
    ("interactionCoefOld", "e7_5_interaction", "superseded/interaction_coef", f3),
    ("interactionEventsOld", "e7_5_interaction", "superseded/n_events", intc),
    ("designEffect", "e7_4_uncertainty", "design_effect/DEFF", f1),
    ("iccNarrator", "e7_4_uncertainty", "design_effect/ICC", f3),
    ("nClusters", "e7_4_uncertainty", "design_effect/n_clusters", intc),
    ("nIncidences", "e7_4_uncertainty", "design_effect/N", intc),
    ("ccnnAccPoint", "e7_4_uncertainty", "clustered_bootstrap_accuracy_M5/point", f3),
    ("ccnnAccLo", "e7_4_uncertainty", "clustered_bootstrap_accuracy_M5/lo", f3),
    ("ccnnAccHi", "e7_4_uncertainty", "clustered_bootstrap_accuracy_M5/hi", f3),
    ("asoBonferroni", "e7_3_aso_T1_map", "bonferroni_factor", intc),
    ("asoConfidence", "e7_3_aso_T1_map", "confidence_used", f3),
    ("segmentsWithTopics", "e1_1_corpus_stats", "segments_with_topics", intc),
    ("medianSegments", "e1_1_corpus_stats", "segments_per_interview/median", f0),
    # ---- E8 robustness ---------------------------------------------------------------
    ("pertRbo", "e8_1_perturbation", "kill_criterion/rbo50_at_measured_rate", f3),
    ("nRankTwoCells", "e8_1_perturbation", "measured_rates/archive_internal/n_cells", intc),
    ("pertRboArchive", "e8_1_perturbation", "grids/archive_internal/m1.0_s1.0/rbo50_mean", f3),
    ("pertThreshold", "e8_1_perturbation", "kill_criterion/threshold", f2),
    ("pertPasses", "e8_1_perturbation", "kill_criterion/passes", yesno),
    ("pertGridPoints", "e8_1_perturbation", "n_grid_points_per_calibration", intc),
    ("extractorMinRbo", "e8_2_extractor", "min", f3),
    ("goldSubsetSegments", "e8_3_gold_subset", "n_segments", intc),
    ("goldTermsExcluded", "e8_3_gold_subset", "n_terms_excluded", intc),
    ("goldCells", "e8_3_gold_subset", "n_rank2_cells", intc),
    ("goldMthree", "e8_3_gold_subset", "model_means/M3_typed_star/T1_map", f3),
    ("goldMfive", "e8_3_gold_subset", "model_means/M5_ccnn/T1_map", f3),
    ("goldMone", "e8_3_gold_subset", "model_means/M1_dense/T1_map", f3),
    ("singletonNarrators", "e8_4_interviewer", "singleton_fraction_narrators_only", f3),
    ("singletonWithInterviewers", "e8_4_interviewer", "singleton_fraction_with_interviewers", f3),
    ("interviewerRbo", "e8_4_interviewer", "triage_rbo50", f3),
    # ---- E9 application --------------------------------------------------------------
    ("triageK", "e9_2_triage", "k", intc),
    ("triageUnstable", "e9_2_triage", "n_unstable", intc),
    ("freqSpearman", "e9_3_frequency_baseline", "spearman_attestation_vs_mention_frequency", f3),
    ("freqRbo", "e9_3_frequency_baseline", "rbo50_triage_attestation_vs_frequency", f3),
    ("freqJaccard", "e9_3_frequency_baseline", "jaccard_top10", f3),
    # ---- E10 transfer ----------------------------------------------------------------
    ("transferSegments", "e10_transfer", "transfer_corpus_stats/n_segments", intc),
    ("transferInterviews", "e10_transfer", "transfer_corpus_stats/n_interviews", intc),
    ("transferAlpha", "e10_transfer", "E10_1_meta_rank/alpha_transfer", f3),
    ("transferAlphaDrop", "e10_transfer", "E10_1_meta_rank/alpha_drop", f3),
    ("transferRho", "e10_transfer", "E10_2_zero_shot/rank_cardinality/spearman_rho", f3),
    ("transferCells", "e10_transfer", "E10_2_zero_shot/n_rank2_cells", intc),
    ("transferMthree", "e10_transfer", "E10_2_zero_shot/model_means/M3_typed_star/T1_map", f3),
    ("transferMfive", "e10_transfer", "E10_2_zero_shot/model_means/M5_ccnn/T1_map", f3),
    ("conclusionTransfers", "e10_transfer", "E10_3_conclusion_transfer/conclusion_transfers", yesno),
    # ---- E11 topology ----------------------------------------------------------------
    ("topoPermP", "e11_topology", "permutation_test/p_value", f3),
    ("topoBetaOneP", "e11_topology", "permutation_test/betti1/p_value", f3),
    ("topoBetaOneNullStd", "e11_topology", "permutation_test/betti1/null_std", f1),
    ("topoBetaZeroNullStd", "e11_topology", "permutation_test/betti0/null_std", f1),
    ("topoBetaOneRtwo", "e11_topology",
     "simpler_explanation_check/r2_betti1_size_and_connectivity_only", f3),
    ("topoFiltrationLevels", "e11_topology", "persistence/n_levels", intc),
    ("topoKeep", "e11_topology", "kill_criterion/verdict_keep_section", yesno),
    ("topoRedundant", "e11_topology", "kill_criterion/recoverable_from_size_distribution", yesno),
    # ---- convergence control: is M5's link result an optimisation failure? ------------
    ("convSubset", "e_convergence", "subset_size", intc),
    ("convLossFive", "e_convergence", "models/M5_ccnn/train_loss_final", f3),
    ("convValFiveFirst", "e_convergence", "models/M5_ccnn/val_auc_first", f3),
    ("convValFiveLast", "e_convergence", "models/M5_ccnn/val_auc_final", f3),
    ("convValFiveMax", "e_convergence", "models/M5_ccnn/val_auc_max", f3),
    ("convEpochs", "e_convergence", "epochs", intc),
    # ---- the exposure condition is a property of the lifting, not of this archive -----
    ("liftConfigs", "e_lifting", "n_configs", intc),
    ("liftExpLo", "e_lifting", "exposure_min", f3),
    ("liftExpHi", "e_lifting", "exposure_max", f3),
    ("liftSingLo", "e_lifting", "singleton_min", f3),
    ("liftSingHi", "e_lifting", "singleton_max", f3),
    # ---- does the ordering survive the selection rule, and equal-budget tuning? -------
    ("selDisjAuc", "e_selection", "ordering/auc_sel|narrator-disjoint", f3),
    ("selDisjMap", "e_selection", "ordering/map_sel|narrator-disjoint", f3),
    ("selRandAuc", "e_selection", "ordering/auc_sel|random", f3),
    ("selRandMap", "e_selection", "ordering/map_sel|random", f3),
    ("selThreeMap", "e_selection", "cells/M3_typed_star|narrator-disjoint/map_sel", f3),
    ("selFiveMap", "e_selection", "cells/M5_ccnn|narrator-disjoint/map_sel", f3),
    ("selThreeEpAuc", "e_selection", "cells/M3_typed_star|narrator-disjoint/ep_auc", f1),
    ("selThreeEpMap", "e_selection", "cells/M3_typed_star|narrator-disjoint/ep_map", f0),
    ("tuneFiveAuc", "e_tuning", "best/M5_ccnn/val_auc", f3),
    ("tuneGap", "e_tuning", "tuned_gap_M5_minus_M3", f3),
    ("tuneThreeLr", "e_tuning", "best/M3_typed_star/lr", f3),
    ("tuneThreeDo", "e_tuning", "best/M3_typed_star/dropout", f1),
    # ---- the operator x features factorial, all four cells from one run ---------------
    ("facThreePlain", "e_factorial", "cells/M3_typed_star|plain|narrator-disjoint/T1_map", f3),
    ("facThreeEnc", "e_factorial", "cells/M3_typed_star|enc|narrator-disjoint/T1_map", f3),
    ("facFivePlain", "e_factorial", "cells/M5_ccnn|plain|narrator-disjoint/T1_map", f3),
    ("facFiveEnc", "e_factorial", "cells/M5_ccnn|enc|narrator-disjoint/T1_map", f3),
    ("facFiveEncAuc", "e_factorial", "cells/M5_ccnn|enc|narrator-disjoint/T2_auc", f3),
    ("facFivePlainAuc", "e_factorial", "cells/M5_ccnn|plain|narrator-disjoint/T2_auc", f3),
    ("facOpPlainDisj", "e_factorial", "operator_effect/plain|narrator-disjoint", f3),
    ("facOpPlainDisjLo", "e_factorial", "operator_effect_ci/plain|narrator-disjoint/0", f3),
    ("facOpPlainDisjHi", "e_factorial", "operator_effect_ci/plain|narrator-disjoint/1", f3),
    ("facOpEncDisjLo", "e_factorial", "operator_effect_ci/enc|narrator-disjoint/0", f3),
    ("facOpEncDisjHi", "e_factorial", "operator_effect_ci/enc|narrator-disjoint/1", f3),
    ("facNarrDisj", "e_factorial",
     "cells/M3_typed_star|plain|narrator-disjoint/n_narrators", intc),
    ("facOpPlainRand", "e_factorial", "operator_effect/plain|random", f3),
    ("facOpEncDisj", "e_factorial", "operator_effect/enc|narrator-disjoint", f3),
    ("facOpEncRand", "e_factorial", "operator_effect/enc|random", f3),
    ("facThreePlainLo", "e_factorial",
     "cells/M3_typed_star|plain|narrator-disjoint/T1_map_ci/0", f3),
    ("facThreePlainHi", "e_factorial",
     "cells/M3_typed_star|plain|narrator-disjoint/T1_map_ci/1", f3),
    ("facFivePlainLo", "e_factorial", "cells/M5_ccnn|plain|narrator-disjoint/T1_map_ci/0", f3),
    ("facFivePlainHi", "e_factorial", "cells/M5_ccnn|plain|narrator-disjoint/T1_map_ci/1", f3),
    # ---- measured cost of the structural operations (scalable-computing remit) --------
    ("costBuild", "e_cost", "build_complex/seconds", f2),
    ("costStar", "e_cost", "star_expansion/seconds", f2),
    ("costFeat", "e_cost", "feature_construction/seconds", f2),
    ("costTrain", "e_cost", "m5_training_one_seed/seconds", f1),
    ("costGpu", "e_cost", "m5_peak_gpu_mib", f0),
    # ---- mechanism (exploratory, run after the split result) --------------------------
    ("gapRealFive", "e_mechanism", "split_gap/M5_ccnn/gap_real_narrators", f3),
    ("gapAnonFive", "e_mechanism", "split_gap/M5_ccnn/gap_anonymised", f3),
    ("gapReductionFive", "e_mechanism", "split_gap/M5_ccnn/gap_reduction", f3),
    ("gapRealThree", "e_mechanism", "split_gap/M3_typed_star/gap_real_narrators", f3),
    ("gapAnonThree", "e_mechanism", "split_gap/M3_typed_star/gap_anonymised", f3),
    ("anonFiveRandom", "e_mechanism", "split_gap/M5_ccnn/anonymised/random", f3),
    ("anonFiveNarr", "e_mechanism", "split_gap/M5_ccnn/anonymised/narrator-disjoint", f3),
    ("doseSlope", "e_dose_response", "random/slope", f3),
    ("doseSlopeLo", "e_dose_response", "random/slope_ci/0", f3),
    ("doseSlopeHi", "e_dose_response", "random/slope_ci/1", f3),
    ("doseSlopeP", "e_dose_response", "random/slope_p", f2),
    ("doseNarrators", "e_dose_response", "random/n_narrators", intc),
    ("doseSlopeControl", "e_dose_response", "narrator-disjoint/slope", f3),
    ("featGapMean", "e_feature_path", "split_gap/M5_ccnn/gap_narrator_mean", f3),
    ("featGapItem", "e_feature_path", "split_gap/M5_ccnn/gap_item_specific", f3),
    ("featGapMeanThree", "e_feature_path", "split_gap/M3_typed_star/gap_narrator_mean", f3),
    ("featGapItemThree", "e_feature_path", "split_gap/M3_typed_star/gap_item_specific", f3),
    # ---- E13 release -----------------------------------------------------------------    ("releaseReproduces", "e13_2_reproduction", "all_match", yesno),
    ("exposurePredicted", "e_apriori_exposure", "expected_three_fold", pct1),
    ("exposureHyperGap", "e_apriori_exposure", "hypergeometric_gap_pp", f2),
    ("exposureSingletonN", "e_apriori_exposure", "n_k_equals_one", intc),
    # ---- ownership obstruction (Proposition 0) -----------------------------------------
    ("obsGround", "e_obstruction", "primary/ground_set_size", intc),
    ("obsCells", "e_obstruction", "primary/rank1_cells", intc),
    ("obsDistinct", "e_obstruction", "primary/rank1_distinct_supports", intc),
    ("obsSingletonSupports", "e_obstruction", "primary/rank1_distinct_singleton_supports", intc),
    ("obsMultiSupports", "e_obstruction", "primary/rank1_multi_narrator_supports", intc),
    ("obsCollapse", "e_obstruction", "primary/rank1_collapse_factor", f1),
    ("obsMaxMult", "e_obstruction", "primary/rank1_max_multiplicity", intc),
    ("obsDesignCells", "e_obstruction", "primary/n_design_cells_checked", intc),
    ("straddleRandom", "e_split_straddle", "random/frac_straddling", pct1),
    ("straddleRandomN", "e_split_straddle", "random/n_straddling", intc),
    ("straddleEvent", "e_split_straddle", "event_disjoint/frac_straddling", pct1),
    ("straddleEventN", "e_split_straddle", "event_disjoint/n_straddling", intc),
    ("straddleTotal", "e_split_straddle", "random/n_narrators", intc),
    ("citationsVerified", "citation_verification", "n_verified", intc),
    ("citationsTotal", "citation_verification", "n_total", intc),
    # ---- E10 partition replication: resample the split itself, not just the seed --------
    ("partN", "e_partition", "n_partitions", intc),
    ("partNdMean", "e_partition", "summary/narrator_disjoint/delta_mean", f3),
    ("partNdMin", "e_partition", "summary/narrator_disjoint/delta_min", f3),
    ("partNdMax", "e_partition", "summary/narrator_disjoint/delta_max", f3),
    ("partRandMean", "e_partition", "summary/random/delta_mean", f3),
    ("partSignFlips", "e_partition", "summary/sign_differs_pairs", intc),
    ("partRegNdMean", "e_partition_registered", "summary/narrator_disjoint/delta_mean", f3),
    ("partRegRandMean", "e_partition_registered", "summary/random/delta_mean", f3),
    ("partRegFlips", "e_partition_registered", "summary/sign_differs_pairs", intc),
]


# --- model results are computed, not transcribed --------------------------------------

PRIMARY = {"granularity": "mid", "split": "narrator-disjoint",
           "neg_regime": "MNS", "rank_map": "R-A"}
MODEL_LABEL = {
    "M0_mlp": "M0 text-only MLP", "M1_dense": "M1 dense retrieval",
    "M2_untyped_star": "M2 untyped star", "M3_typed_star": "M3 typed star",
    "M4_allset": "M4 AllSet", "M4_edhnn": "M4 ED-HNN", "M4_hgmlp": "M4 Hypergraph-MLP",
    "M5_ccnn": "M5 CCNN-style (ours)",
}

# T1 is scored over rank-1 (moment) representations. These two variants pass messages only
# between ranks 0 and 2 and never write h[1], so their T1 value is the frozen-encoder floor
# -- identical across both variants across all seeds -- and not a measurement of either
# operator. Reported as undefined rather than as a number a reader would compare.
NO_RANK1_PATHWAY = {"M4_allset", "M4_edhnn"}
ABLATION_LABEL = {
    "A1_shared_weights": "share weights across rank pairs",
    "A2_shuffled_ranks": "shuffle the rank assignment",
    "A3_collapse_r2r3": "collapse ranks 2 and 3",
    "A4_no_down": "remove down-messages",
    "A5_no_moments": "remove the moment rank",
    "A6_no_within": "remove within-rank exchange",
}


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def cell(runs, **kw):
    return [r for r in runs if all(r.get(k) == v for k, v in kw.items())]


def model_table(runs) -> tuple[str, dict]:
    prim = [r for r in cell(runs, **PRIMARY) if r.get("cell") != "E8.3_gold_subset"]
    prim = [r for r in prim if r.get("model") in MODEL_LABEL]
    rows, extra = [], {}
    for m, label in MODEL_LABEL.items():
        rs = [r for r in prim if r["model"] == m]
        if not rs:
            continue
        t1, t2 = mean([r["T1_map"] for r in rs]), mean([r["T2_auc"] for r in rs])
        nd = mean([r.get("T1_ndcg@10", float("nan")) for r in rs])
        pars = int(mean([r.get("n_params", 0) for r in rs]))
        best = "\\bf " if m == "M3_typed_star" else ""
        t1_cell = "\\textsc{n/a}$^{\\dagger}$" if m in NO_RANK1_PATHWAY else f"{t1:.3f}"
        nd_cell = "\\textsc{n/a}$^{\\dagger}$" if m in NO_RANK1_PATHWAY else f"{nd:.3f}"
        rows.append(f"{best}{label} & {pars:,} & {t1_cell} & {nd_cell} & {t2:.3f} \\\\")
        extra[m] = {"T1": t1, "T2": t2, "nDCG": nd, "params": pars, "n_seeds": len(rs)}
    # \footnotesize keeps the table inside one IEEE column (252pt). At body size the
    # natural width overruns by ~18pt, and spanning both columns costs a whole page.
    # IEEEtran already sets 8pt inside a float (probed: body f@size=10, in-table f@size=8),
    # so a \footnotesize here is a no-op. Narrowing \tabcolsep is what actually brings the
    # table inside the 252pt column, and it does so without rescaling the type.
    tex = ("\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrr}\n\\toprule\nmodel & parameters & T1 MAP & T1 nDCG@10 & "
           "T2 AUC \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
           "\\end{tabular}\n\n\\vspace{2pt}\n{\\footnotesize $\\dagger$ Frozen-encoder floor, "
           "identical across seeds, not a measurement of either operator; see "
           "Section~\\ref{sec:p3}.}")
    return tex, extra


def ablation_table(runs) -> str:
    abl = [r for r in runs if r.get("model") in ABLATION_LABEL
           or r.get("ablation") in ABLATION_LABEL]
    rows = []
    for key, label in ABLATION_LABEL.items():
        rs = [r for r in abl if r.get("model") == key or r.get("ablation") == key]
        if not rs:
            continue
        rows.append(f"{label} & {mean([r['T1_map'] for r in rs]):.3f} & "
                    f"{mean([r['T2_auc'] for r in rs]):.3f} \\\\")
    if not rows:
        return "\\emph{ablation rows unavailable}"
    return ("\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrr}\n\\toprule\nablation & T1 MAP & T2 AUC \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def split_table(runs) -> tuple[str, dict]:
    """The split comparison is a result, so it gets its own table."""
    rows, got = [], {}
    for skind in ("narrator-disjoint", "random", "event-disjoint"):
        rs = cell(runs, granularity="mid", split=skind, neg_regime="MNS", rank_map="R-A")
        rs = [r for r in rs if r.get("cell") != "E8.3_gold_subset"]
        vals = {}
        for m in ("M3_typed_star", "M5_ccnn"):
            v = mean([r["T1_map"] for r in rs if r["model"] == m])
            vals[m] = v
        if any(v == v for v in vals.values()):
            got[skind] = vals
            rows.append(f"{skind} & {vals['M3_typed_star']:.3f} & {vals['M5_ccnn']:.3f} \\\\")
    tex = ("\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrr}\n\\toprule\nsplit & M3 typed star & M5 CCNN-style \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    return tex, got


def stratified_table(strat) -> str:
    """T2 accuracy by event size: the complex should help most where events are large."""
    if not strat or "event_size" not in strat:
        return "\\emph{stratification unavailable}"

    def keyfn(b):
        head = b.split("-")[0]
        return int(head) if head.isdigit() else 0

    rows = []
    for b in sorted(strat["event_size"], key=keyfn):
        d = strat["event_size"][b]
        rows.append(f"{b} & {d.get('M3_typed_star', float('nan')):.3f} & "
                    f"{d.get('M5_ccnn', float('nan')):.3f} \\\\")
    return ("\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrr}\n\\toprule\nevent size (narrators) & M3 typed star & "
            "M5 CCNN-style \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def invariance_table(runs) -> tuple[str, tuple[int, int]]:
    """Every design axis the field argues about, and the one that changes the answer.

    Counting is over DISTINCT cells. Each axis includes the primary cell as one of its
    levels, so summing per-axis counts scores the primary cell once per axis and inflates
    both the numerator and the denominator.
    """
    P = PRIMARY
    axes = [
        ("granularity", "granularity", ["coarse", "mid", "fine"]),
        ("negative sampling", "neg_regime", ["MNS", "UNS", "SNS", "CNS", "hard"]),
        ("rank map", "rank_map", ["R-A", "R-B", "R-C"]),
        ("split", "split", ["narrator-disjoint", "random", "event-disjoint"]),
    ]
    seen: dict[tuple, bool] = {}
    rows = []
    for label, field, values in axes:
        got = []
        for v in values:
            kw = dict(granularity=P["granularity"], split=P["split"],
                      neg_regime=P["neg_regime"], rank_map=P["rank_map"])
            kw[field] = v
            rs = [r for r in cell(runs, **kw) if r.get("cell") != "E8.3_gold_subset"]
            a = mean([r["T1_map"] for r in rs if r["model"] == "M3_typed_star"])
            b = mean([r["T1_map"] for r in rs if r["model"] == "M5_ccnn"])
            if a == a and b == b:
                got.append((v, a > b))
                seen[(kw["granularity"], kw["split"], kw["neg_regime"], kw["rank_map"])] = a > b
        if not got:
            continue
        w = sum(1 for _, ok in got if ok)
        rows.append(f"{label} & {', '.join(v for v, _ in got)} & {w} / {len(got)} \\\\")
    wins, total = sum(seen.values()), len(seen)
    # The levels column carries comma lists, so this is the widest table in the paper;
    # 2pt of column padding is what brings it inside the column.
    tex = ("\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{lll}\n\\toprule\ndesign axis varied & levels & typed star leads "
           "\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
           "\\vspace{2pt}\n{\\footnotesize Per-axis counts share the primary cell, so they do "
           f"not sum: over the {total} distinct cells the typed star leads in {wins}.}}")
    return tex, (wins, total)


def build_numbers() -> tuple[str, list[dict], dict]:
    lines, manifest = [], []
    cache: dict[str, object] = {}
    for macro, src, path, fmt in SPEC:
        if src not in cache:
            cache[src] = load(src)
        val = dig(cache[src], path) if cache[src] is not None else MISSING
        if val is MISSING or val is None:
            lines.append(f"\\newcommand{{\\{macro}}}{{\\textbf{{???}}}}")
            manifest.append({"macro": macro, "source": f"{src}.json", "path": path,
                             "status": "MISSING", "value": None})
            continue
        try:
            rendered = fmt(val)
        except (TypeError, ValueError):
            # Falling back to str() here once put a whole Python dict into a sentence --
            # "the design effect is {'DEFF': 37.38..., 'ICC': ...}" -- and LaTeX compiled it
            # without complaint. A macro pointed at a container is a broken path, so it is
            # reported as MISSING rather than rendered.
            lines.append(f"\\newcommand{{\\{macro}}}{{\\textbf{{???}}}}")
            manifest.append({"macro": macro, "source": f"{src}.json", "path": path,
                             "status": "MISSING", "value": None,
                             "reason": f"path resolves to {type(val).__name__}, not a scalar"})
            continue
        lines.append(f"\\newcommand{{\\{macro}}}{{{rendered}}}")
        manifest.append({"macro": macro, "source": f"{src}.json", "path": path,
                         "status": "ok", "value": val, "rendered": rendered})

    # Quantities that are differences or extrema across cells rather than stored fields.
    # Deriving them here keeps every one traceable to the run it came from.
    def derive(macro, src, note, fn, fmt=None):
        obj = cache.get(src) if src in cache else load(src)
        cache[src] = obj
        try:
            value = fn(obj)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError, ZeroDivisionError):
            value = None
        if value is None or value != value:
            lines.append(f"\\newcommand{{\\{macro}}}{{\\textbf{{???}}}}")
            manifest.append({"macro": macro, "source": f"{src}.json", "path": note,
                             "status": "MISSING", "value": None})
            return
        if fmt is not None:
            rendered = fmt(value)
        else:
            rendered = f"{value:.3f}" if isinstance(value, float) else f"{value}"
        lines.append(f"\\newcommand{{\\{macro}}}{{{rendered}}}")
        manifest.append({"macro": macro, "source": f"{src}.json", "path": note,
                         "status": "ok", "value": value, "rendered": rendered})

    def exposure(model, feats):
        def go(o):
            c = o["cells"]
            return (c[f"{model}|{feats}|random"]["T1_map"]
                    - c[f"{model}|{feats}|narrator-disjoint"]["T1_map"])
        return go

    for macro, model, feats in (("facExpFivePlain", "M5_ccnn", "plain"),
                                ("facExpFiveEnc", "M5_ccnn", "enc"),
                                ("facExpThreePlain", "M3_typed_star", "plain"),
                                ("facExpThreeEnc", "M3_typed_star", "enc")):
        derive(macro, "e_factorial", f"random minus narrator-disjoint T1_map, {model}|{feats}",
               exposure(model, feats))

    derive("facOpPlainGap", "e_factorial", "operator_effect plain|random minus plain|narrator-disjoint",
           lambda o: o["operator_effect"]["plain|random"]
           - o["operator_effect"]["plain|narrator-disjoint"])

    derive("tuneFiveBest", "e_tuning", "max T1_map over M5_ccnn trials",
           lambda o: max(v["T1_map"] for k, v in o["trials"].items() if k.startswith("M5_ccnn")))
    derive("tuneThreeWorst", "e_tuning", "min T1_map over M3_typed_star trials",
           lambda o: min(v["T1_map"] for k, v in o["trials"].items()
                         if k.startswith("M3_typed_star")))
    derive("tuneTrials", "e_tuning", "trials per model",
           lambda o: len(o["trials"]) // 2)
    for macro, src in (("facSeeds", "e_factorial"), ("selSeeds", "e_selection"),
                       ("tuneSeeds", "e_tuning")):
        derive(macro, src, "len(seeds)", lambda o: len(o["seeds"]))

    # Partition replication. `between_partition_var` is the variance of a seed-averaged
    # delta, so it already carries seed noise; the partition component is what survives
    # subtracting that noise, and under the random split nothing does.
    derive("partSeeds", "e_partition", "len(seeds)", lambda o: len(o["seeds"]))
    derive("partAmp", "e_partition",
           "random delta_mean divided by narrator-disjoint delta_mean",
           lambda o: o["summary"]["random"]["delta_mean"]
           / o["summary"]["narrator_disjoint"]["delta_mean"], f1)
    derive("partSeedVar", "e_partition",
           "within-partition seed variance, random over narrator-disjoint",
           lambda o: round(o["summary"]["random"]["within_partition_seed_var"]
                           / o["summary"]["narrator_disjoint"]["within_partition_seed_var"]))
    derive("partNdShare", "e_partition",
           "partition variance component as a share of observed spread, narrator-disjoint",
           lambda o: (o["summary"]["narrator_disjoint"]["between_partition_var"]
                      - o["summary"]["narrator_disjoint"]["within_partition_seed_var"]
                      / len(o["seeds"]))
           / o["summary"]["narrator_disjoint"]["between_partition_var"], pct1)

    runs = load("e7_1_runs") or []
    tbl, per_model = model_table(runs)
    abl = ablation_table(runs)
    stbl, splits = split_table(runs)

    # Cited in the ablation paragraph, which now reads the two task columns differently.
    a5 = [r for r in runs if r.get("model") == "A5_no_moments"]
    if a5:
        v = mean([r["T2_auc"] for r in a5])
        lines.append(f"\\newcommand{{\\ablationNoMomentsTwo}}{{{v:.3f}}}")
        manifest.append({"macro": "ablationNoMomentsTwo", "source": "e7_1_runs.json",
                         "path": "mean T2_auc, A5_no_moments", "status": "ok", "value": v,
                         "rendered": f"{v:.3f}"})

    def add(macro, value, rendered, note):
        lines.append(f"\\newcommand{{\\{macro}}}{{{rendered}}}")
        manifest.append({"macro": macro, "source": "e7_1_runs.json", "path": note,
                         "status": "ok" if value == value else "MISSING", "value": value,
                         "rendered": rendered})

    for m, key in (("M3_typed_star", "Mthree"), ("M5_ccnn", "Mfive"),
                   ("M1_dense", "Mone"), ("M0_mlp", "Mzero"), ("M4_edhnn", "Mfour")):
        d = per_model.get(m)
        if d:
            add(f"{key}Tone", d["T1"], f"{d['T1']:.3f}", f"mean T1_map, model={m}, primary cell")
            add(f"{key}Ttwo", d["T2"], f"{d['T2']:.3f}", f"mean T2_auc, model={m}, primary cell")
        else:
            # Define it anyway. A macro that simply disappears takes the LaTeX build down
            # with an undefined-control-sequence error far from the cause; a loud ??? in
            # the PDF and a MISSING row in the manifest points straight at the empty run.
            for suffix in ("Tone", "Ttwo"):
                lines.append(f"\\newcommand{{\\{key}{suffix}}}{{\\textbf{{???}}}}")
                manifest.append({"macro": f"{key}{suffix}", "source": "e7_1_runs.json",
                                 "path": f"model={m} absent from runs", "status": "MISSING",
                                 "value": None})
    if per_model:
        ps = [d["params"] for d in per_model.values() if d["params"] > 0]
        if ps:
            spread = (max(ps) - min(ps)) / max(ps)
            add("paramSpread", spread, f"{100 * spread:.1f}\\%",
                "max-min parameter spread across trained models")
        seeds = max((d["n_seeds"] for d in per_model.values()), default=0)
        add("nSeeds", seeds, str(seeds), "distinct seeds per model in the primary cell")

    # The task section states its own scale rather than asking the reader to trust it.
    # `cell` is None for the main grid -- the primary cell is identified by its factor
    # levels, exactly as model_table() does it.
    prim = [r for r in cell(runs, **PRIMARY) if r.get("cell") != "E8.3_gold_subset"]
    nq = [r["T1_n_queries"] for r in prim
          if isinstance(r.get("T1_n_queries"), (int, float)) and r["T1_n_queries"] > 0]
    if nq:
        add("nTOneQueries", nq[0], f"{int(nq[0]):,}".replace(",", "{,}"),
            "T1 test queries in the primary cell")
    npar = [r["n_params"] for r in prim
            if isinstance(r.get("n_params"), (int, float)) and r["n_params"] > 0]
    if npar:
        add("paramBudget", max(npar), f"{int(max(npar)):,}".replace(",", "{,}"),
            "largest trained-model parameter count in the primary cell")
    for skind, key in (("random", "Random"), ("event-disjoint", "Event")):
        if skind in splits:
            v3, v5 = splits[skind]["M3_typed_star"], splits[skind]["M5_ccnn"]
            add(f"MthreeTone{key}", v3, f"{v3:.3f}", f"mean T1_map M3, split={skind}")
            add(f"MfiveTone{key}", v5, f"{v5:.3f}", f"mean T1_map M5, split={skind}")

    # ---- narrator-identity probe (list-shaped, so extracted rather than dug) ----------
    mech = load("e_mechanism") or {}
    probe_key = {("narrator-disjoint", "M3_typed_star"): "probeNarrThree",
                 ("narrator-disjoint", "M5_ccnn"): "probeNarrFive",
                 ("random", "M3_typed_star"): "probeRandThree",
                 ("random", "M5_ccnn"): "probeRandFive",
                 ("narrator-disjoint", "frozen_encoder_typed"): "probeInputControl"}
    for rec in mech.get("probe", []):
        key = probe_key.get((rec.get("split"), rec.get("model")))
        if key:
            add(key, rec["probe_accuracy_mean"], f"{rec['probe_accuracy_mean']:.3f}",
                f"probe accuracy, {rec['model']}, split={rec['split']}")
        if key == "probeNarrThree":
            add("probeChance", rec["chance"], f"{rec['chance']:.3f}",
                "majority-class floor for the narrator probe")
            add("probeClasses", rec["n_classes"], str(int(rec["n_classes"])),
                "narrators appearing in the probe")
            add("probeN", rec["n"], f"{int(rec['n']):,}".replace(",", "{,}"),
                "single-narrator rank-1 cells entering the probe")

    # CI width scales with sqrt(DEFF), not DEFF; stating the wrong one understated the effect.
    deff = dig(load("e7_4_uncertainty"), "design_effect/DEFF")
    if isinstance(deff, (int, float)):
        add("deffSqrt", deff, f"{math.sqrt(deff):.0f}", "sqrt(DEFF), the CI-width inflation")

    # The pre-registered primary test. eps_min = 0 means the first model stochastically
    # dominates the second; 1 means no dominance at all.
    aso = load("e7_3_aso_T1_map") or {}
    for pair, macro in (("M3_typed_star>M5_ccnn", "asoThreeOverFive"),
                        ("M5_ccnn>M3_typed_star", "asoFiveOverThree")):
        v = dig(aso, f"eps_min/{pair}")
        if isinstance(v, (int, float)):
            add(macro, v, f"{v:.2f}", f"ASO eps_min for {pair}")

    invariance, inv_rows = invariance_table(runs)
    (OUT / "table_invariance.tex").write_text(invariance, encoding="utf-8")
    add("invarianceWins", inv_rows[0], str(inv_rows[0]), "conditions where M3 leads M5")
    add("invarianceTotal", inv_rows[1], str(inv_rows[1]), "conditions compared")

    (OUT / "table_models.tex").write_text(tbl, encoding="utf-8")
    (OUT / "table_ablations.tex").write_text(abl, encoding="utf-8")
    (OUT / "table_splits.tex").write_text(stbl, encoding="utf-8")
    (OUT / "table_stratified.tex").write_text(stratified_table(load("e7_2_stratified")),
                                              encoding="utf-8")
    lines.append(f"\\newcommand{{\\buildDate}}{{{date.today().isoformat()}}}")
    return "\n".join(lines) + "\n", manifest, {"per_model": per_model, "splits": splits}


# --- prose linting --------------------------------------------------------------------

ALLOWED_BARE = re.compile(
    r"(?:\\cite\w*\{[^}]*\})|(?:\\ref\{[^}]*\})|(?:\\label\{[^}]*\})|"
    r"(?:\\documentclass\[[^\]]*\])|(?:\\usepackage(?:\[[^\]]*\])?\{[^}]*\})|"
    r"(?:\\(?:section|subsection|newcommand|input|includegraphics|begin|end)\b[^\n]*)|"
    r"(?:\$[^$]*\$)|(?:%.*)")

# Invoked as \name{} but supplied by LaTeX, not by numbers.tex.
LATEX_EMPTY_BRACE = {"author"}


def lint_prose(tex: str) -> list[str]:
    """Report bare numerals in prose. Macros, math and citations are exempt."""
    stripped = ALLOWED_BARE.sub(" ", tex)
    # A metric cutoff is part of the metric's NAME -- nDCG@10, Recall@50, RBO@50 -- not a
    # measured quantity, so it is removed before scanning rather than whitelisted as a value.
    stripped = re.sub(r"@\d+", "@", stripped)
    # Conventional constants, not measurements: a 95% interval is named, not observed.
    conventional = {"95", "99", "50"}
    offenders = []
    for i, line in enumerate(stripped.splitlines(), 1):
        for m in re.finditer(r"(?<![\\A-Za-z0-9.])\d[\d,.]*", line):
            tok = m.group(0).rstrip(".,")  # a sentence-final period is not part of the number
            if not tok or tok in {"1", "2", "3", "0"} or tok in conventional:
                continue
            offenders.append(f"line {i}: bare numeral '{tok}'")
    return offenders


PAPER = r"""\documentclass[conference]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}  % IEEE house face; also keeps Type 1 outlines, not Type 3
\let\Bbbk\relax  % newtxmath already defines it; without this amssymb raises a LaTeX error
\usepackage{booktabs,graphicx,amsmath,amssymb,microtype,xcolor}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
% IEEEtranN.bst is the natbib-aware IEEE style, so \citep survives the template change.
\usepackage[numbers,sort&compress]{natbib}
\newtheorem{proposition}{Proposition}
\setlength{\textfloatsep}{3pt plus 1pt minus 1pt}
\setlength{\dblfloatsep}{3pt plus 1pt minus 1pt}
\setlength{\dbltextfloatsep}{3pt plus 1pt minus 1pt}
\setlength{\floatsep}{3pt plus 1pt minus 1pt}
\setlength{\intextsep}{3pt plus 1pt minus 1pt}
\setlength{\abovecaptionskip}{2pt}
\setlength{\belowcaptionskip}{0pt}
\input{numbers}

\title{CAIRN: Ownership Obstruction and Evaluation Design
in Higher-Order Models of Oral History}
\author{\IEEEauthorblockN{Zeyu Dong and Roy Choi}
\IEEEauthorblockA{\textit{Independent Researcher}, Toronto, Canada \\
Correspondence: dongzeyu123@outlook.com}}
\newcommand{\repoURL}{https://github.com/AlexDongzeyu/CAIRN}

\begin{document}
\maketitle

\begin{abstract}
A rank ladder is an information granulation, and combinatorial complexes make one
computable~\citep{hajij2023tdl,papillon2023architectures}. Networks over those complexes act on
cells, each a subset of a ground set, so where that set holds entities and the items above
them are singly owned, two items with one owner are one cell. Singly owned items are what an
oral-history archive holds: across \nInterviews{}
interviews, \obsCells{} described moments realise \obsDistinct{} supports over
\obsGround{} narrators, so no combinatorial complex over those narrators holds them apart, a
collapse we call the \emph{ownership obstruction}. Lifting carries the obstruction into the
features, leaving at most \obsDistinct{} distinct rank-1 inputs, a \obsCollapse{}-fold
collapse, and
the split decides whether that collapse is visible: \straddleRandom{} of narrators straddle a
random split, a rate the entity-degree distribution predicts at \exposurePredicted{} before
any model exists. Given that exposure, our CCNN-style ranked-incidence model appears to beat a
lossless typed-star expansion \MfiveToneRandom{} to \MthreeToneRandom{} in T1 MAP, and trails
it \MfiveTone{} to \MthreeTone{} once narrators are held out, while destroying entity identity
removes the complex's gain. Gains of that kind track the inputs rather than the operator: the
star leads as registered, carrying spectral encodings the complex never receives, the ordering
reverses once both read the same features, and on those matched inputs both leak,
\facExpFivePlain{} against \facExpThreePlain{}.
Assigning the middle rank is itself underdetermined, three programmatic operationalisations
of one written manual agreeing at $\alpha = \alphaRankTwo{}$. What is new is the combination:
an injectivity test that runs before training, an exposure rate predicted in closed form, and
a controlled ranking reversal inside an owner-grounded
hierarchy. The magnitude of the reversal is a finding about this archive; the obstruction
itself is structural, and reaches any collection whose items are singly owned.\footnote{Data,
code and audit record: \url{\repoURL}.}
\end{abstract}

\begin{IEEEkeywords}
information granulation, granular computing, topological deep learning, higher-order
networks, evaluation design, data leakage, oral history archives
\end{IEEEkeywords}

\section{Introduction}

An oral-history archive is described at several levels of resolution at once: a narrator, a
moment inside an interview, an event that several narrators describe, a longer episode that
contains those events. Those levels are an information granulation, and an unusual one, since
archivists built it rather than an algorithm inducing it from data. Granular computing asks
what makes such a description a good one, and topological deep learning offers to compute on
it: a combinatorial complex assigns each cell a rank, holds a coarser granule at each rank
than the one below, and lets messages travel between ranks as well as within
them~\citep{hajij2023tdl,papillon2023architectures}. Across those ranks the operator family is
rank-aware in a way graph and hypergraph message passing is not, so a curated ladder is
tempting to read as a rank ladder. Reading the ladder that way meets a
counting problem (Figure~\ref{fig:overview}): a combinatorial complex identifies a
cell with the ground elements it contains, so two moments told by one narrator are one cell,
and this archive's \obsCells{} moments carry \obsDistinct{} distinct supports over
\obsGround{} narrators. Those collapsed supports are what we call the \emph{ownership
obstruction}.

Before a rank-structured model can be said to help, the granulation itself has to pass three
tests, one for each of the qualities a granulation is asked for.

\begin{description}[leftmargin=0em,itemsep=1pt,topsep=2pt,parsep=0pt]
\item[P1. Distinguishability.] Rank carries information beyond size. Coarsening a partition
ordinarily produces
larger blocks, so rank and size are expected to move together in any granulation. What a
curated granulation has to add is the residual: if rank is recoverable from
cell size alone, the
ladder is a sorting of set sizes and nothing is gained by calling it structure.
\item[P2. Reproducible assignment.] The specification is stable. If independently written
implementations of the
same manual, each consistent with its rules, assign a term to different ranks, the ladder
cannot be operationalised
from the manual alone, and what fills the gap is the judgement of whoever built it.
\item[P3. Computational utility.] The structure pays. Two questions sit under this and the registration answers only
the first: which end-to-end system ranks better at equal parameter count, and which operator
ranks better once inputs are matched as well. An operator claim does not follow from a system
margin carried by features.
\end{description}

We fixed thresholds and failure outcomes for all three before looking at any model result.
Against them, on an archive of \nInterviews{} interviews and \nSegments{} described segments,
\textbf{rank carries information beyond size, the specification moves when it is applied
twice, and the typed star as registered beats the complex at equal parameter count}. The induced
partition moves under a defensible change of granularity as well, and the star's margin comes
from its inputs: on matched features the ordering reverses. That ladder is real; the
machinery built on it has still to earn its place.

A fourth result, undeclared in advance, matters most and concerns the train/test split. Under
a random split the ordering of architectures reverses and the complex's advantage is large,
while it leads under no other design axis we varied, placing that advantage in the split
rather than the operator.

The reason lies in the feature construction these models are normally given. Nearly every
moment has a single narrator, so a moment's input vector \emph{is} its narrator's vector, and
two such moments land on both sides of a random split as the same vector. Shuffling narrator
identity removes that vector's privilege and with it the complex's entire random-split
advantage, while replacing the moment features with passage-specific text does not, which
locates the leak in the narrator layer. Those extra ranks buy a shorter path to who is
speaking.

Three things are new here: an exact support-injectivity test decidable on the encoding before
a model exists, where the surrounding literature offers diagnostics that correlate with
downstream behaviour; an entity-exposure rate the degree distribution predicts in closed form,
so the leak is quantifiable from the data alone; and a controlled reversal of the architecture
ranking inside an owner-grounded hierarchy, with the mechanism named and intervened on.

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{../figures/fig1_overview.jpg}
\caption{The ownership obstruction and what it does to evaluation. \textbf{a}, The archive's
four-level descriptive ladder; node counts are illustrative, tier counts exact.
\textbf{b}, Rank-1 moments are almost all singly owned, so as subsets they are
indistinguishable: \obsCells{} moments realise only \obsDistinct{} supports over \obsGround{}
narrators, which caps the distinct rank-1 inputs after lifting.
\textbf{c}, Whether the collapse is visible depends on the partition. Straddle rate is the
fraction of narrators on both sides, and the predicted rate follows from the entity-degree
distribution alone, with $p_f$ the proportion of fold $f$ and $k$ the cells an entity owns.
T1 MAP reverses under a random split, and shuffling narrator identity removes the
complex's gain while the star's, never positive, falls further. \textbf{d}, The released
diagnostics; the first two require no trained model.}
\label{fig:overview}\end{figure*}

\section{Archive and rank ladder}

Our corpus is the Densho Digital Repository's Japanese American oral-history collection:
\nInterviews{} interviews covering \nNarrators{} narrators, \nSegments{} segments carrying
archival description, and \nTranscripts{} transcripts, at a median of \medianSegments{}
segments per interview. A segment enters the event ranks only if it carries a topic term,
which \segmentsWithTopics{} of them do. Those ranks come from the archive's own descriptive
practice: rank 0 a narrator, rank 1 a described moment within an interview, rank 2 an event or
site, rank 3 the episode containing it.

That ladder cannot be a combinatorial complex over the narrators, and the reason is
counting.

\begin{proposition}[Ownership obstruction]
\label{prop:obstruction}
Let $S$ be a ground set and let a collection $C_1$ carry a support map
$\mathrm{supp}: C_1 \to \mathcal{P}(S)\setminus\{\emptyset\}$. The cells of a combinatorial
complex over $S$ are subsets of $S$, so any two cells with the same support are the same
cell. If $C_1$ realises fewer than $|C_1|$ distinct supports then $\mathrm{supp}$ is not
injective, and no combinatorial complex over $S$ carries $C_1$ as distinct cells. Where every
cell has a single owner in $S$, that happens as soon as $|C_1| > |S|$.
\end{proposition}

Broader higher-order formalisms admit duplicate supports by carrying edge identifiers in the
incidence records~\citep{coll2025hif}, so Proposition~\ref{prop:obstruction} is a statement
about set-valued encodings: identifiers solve the representation, at the cost of the
narrator-grounded ladder that motivated the model.

The archive meets the condition at rank 1 (Figure~\ref{fig:overview}b). Over \obsGround{}
narrators at that rank, \obsCells{} described
moments realise \obsDistinct{} distinct supports, one of them shared by \obsMaxMult{} moments,
and the counts repeat in all \obsDesignCells{} design cells of the sweep because rank-1 cells
are the archive's segments. Among those supports, \obsSingletonSupports{} name single
narrators,
which is what holds the cell count below \obsGround{}, and the remaining \obsMultiSupports{}
are
the sets of two to four speakers who share a segment. Ranks 2 and 3 are injective, so the
obstruction sits at the rank
where moments live. What we build at that rank is a \emph{named complex}: cells carry
identifiers that index the incidence matrices, so the operators are well defined on them.

Escaping the obstruction means leaving the single-complex setting: giving moments their own
object type works, and treating the ladder as a general ranked incidence structure is the
route taken here.

Only the rank-2 equivalence relation is varied. Three granularities of that relation are
defined in
advance: \emph{coarse} merges a topic leaf into its parent path, \emph{mid} uses the leaf as
given, and \emph{fine} splits a leaf by the geography attached to the segment. Mid, the leaf
as given, is the
registered primary, and no archival principle makes it so, so a conclusion that depends on the
choice is a conclusion about us.

The registration defined the two finer levels with a \emph{chronology} decade bin and the
implementation splits by geography instead: Densho segments carry no date field, so the decade
bin needed an undeclared date-inference step with its own error rate. Auditing the
registration against the code surfaced that step only after the results were in, so those
results
describe sensitivity to a topical and geographic equivalence relation rather than to temporal
binning, and no claim of temporal invariance appears below.

Transcript speaker labels are recovered before any of this, because attributing testimony to
the wrong speaker would corrupt every rank-0 incidence. Archive labels name the speaker for
\labelCoverage{} of \nTurns{} turns, a rule-based classifier reproduces them with accuracy
\turnAccuracy{}, and cleaning removes \charsRemoved{} of characters.

\subsection{Tasks and models}
\label{sec:tasks}

Two tasks are scored, chosen because they ask opposite things of a representation.

\textbf{T1, corroboration retrieval.} Given a held-out passage by narrator $n$, rank all
other passages in the archive so that passages by \emph{different} narrators describing the
same rank-2 event come first. This is an archive-motivated retrieval task: find me
another account of this. Every model trains on the incidence objective, so T1 measures how
representations learned under a common objective transfer to retrieval, not the best each
architecture could reach under a task-aligned loss. Two design choices keep that retrieval non-trivial. Both concern the
query: its own cell
membership is masked, so the answer cannot be read off the structure the model is handed, and
any candidate sharing a narrator with it is excluded from the positives, so a model
cannot score by recognising the speaker instead of the event. We score MAP over
\nTOneQueries{} held-out queries, with nDCG@10, Recall@50 and MRR alongside.
\textbf{T2, incidence prediction.} Given a narrator and an event cell, predict whether that
narrator is incident to that event, a link task over the bipartite incidence relation, scored
by AUC. In the primary regime negatives come from cells at a short random-walk distance, so
they are plausible.

By design, the two tasks are not aligned. T1 is scored over moment representations and T2
over narrator--event pairs, so a representation can be good at one and poor at the other, an
asymmetry the ablations in Section~\ref{sec:p3} turn into the paper's sharpest result.

\textbf{Models.} All learned models share the same frozen sentence
encoder~\citep{reimers2019sbert} and a common
budget of at most \paramBudget{} parameters, with hidden width selected per model to meet it.

\begin{itemize}[leftmargin=1.2em,itemsep=1pt,topsep=2pt]
\item \textbf{M0, MLP.} Per-cell feature transform with no message passing at all, the floor
for ``does any structure help''.
\item \textbf{M1, dense retrieval.} Cosine similarity over the frozen embeddings, with
\emph{zero} learned parameters, scored by identical metric code; if competitive, it bounds
what any learned model can claim.
\item \textbf{M2, untyped star.} Expansion of the complex to a bipartite graph, with a single
weight matrix applied to every relation, discarding rank.
\item \textbf{M3, typed star.} Same expansion with relation-specific weights, so rank is
available as a \emph{type} but not as an operator. This model additionally receives spectral
hypergraph encodings~\citep{zhou2006hypergraph} that the complex does not, a handicap fixed
in advance that Section~\ref{sec:p3} then matches away.
\item \textbf{M4, hypergraph baselines.} Rank-agnostic hypergraph models that treat events
as hyperedges over narrators: AllSet-style attention~\citep{chien2022allset}, ED-HNN
equivariant diffusion~\citep{wang2023edhnn}, and Hypergraph-MLP~\citep{tang2024hypergraphmlp},
which removes message passing entirely and is included as a skeptical control.
\item \textbf{M5, combinatorial complex network.} Rank-aware message passing: per layer, up
the ladder $r_0 \to r_1 \to r_2 \to r_3$, back down it, and within-rank exchange through
shared cofaces, with rank-specific weights throughout. This is the model the study is about,
a CCNN-style network over the named complex of Proposition~\ref{prop:obstruction} rather than
over a standard combinatorial complex the archive does not admit; the operator family is that
of~\citet{hajij2023tdl}, indexed by cell identifiers.
\end{itemize}

M2 and M3 are the load-bearing comparisons. Both see the same incidence structure as M5, and
M2 differs from it only in whether rank is an operator or a label, so it isolates that
question; M3 additionally receives the spectral block, so a margin it wins answers whether a
strong simpler baseline beats the complex. Both appear in Section~\ref{sec:p3}, with an
input-matched M3 that isolates the operator.

\textbf{Implementation.} Every learned model runs two message-passing layers with ReLU and
dropout $0.1$ over one objective, binary cross-entropy on incidences against ten sampled
negatives per positive, optimised with AdamW at learning rate $5\times10^{-3}$, weight
decay $10^{-4}$, for at most $800$ epochs, stopping once validation AUC has not improved for
$120$; the best validation checkpoint is restored before anything is scored. Neighbourhood
operators are the incidence matrices between consecutive ranks together with their
transposes, plus within-rank exchange through shared cofaces. With those operators fixed in
advance, no hyperparameter search enters the primary grid: depth, learning rate, dropout and
the negative ratio are fixed for every model, and Section~\ref{sec:p3} reports a separate
equal-budget sweep over the two models the argument rests on. Only hidden width varies,
binary-searched over $[8,256]$ so that each lands within $15\%$ of the shared
\paramBudget{}-parameter budget. Without that budget, relation-specific models carry roughly
twice the parameters of shared-weight models at equal width, which would leave any difference
between them unattributable.

\section{Related work}
\label{sec:related}

Granular computing studies description at multiple levels of
resolution~\citep{zadeh1997granulation,yao2005perspectives}. Its standard constructions form
granules from the data: partition hierarchies induced by indiscernibility
relations~\citep{pawlak1982rough}, and granule boundaries chosen to trade coverage against
specificity~\citep{pedrycz2013justifiable}. Against those constructions, the ladder examined
here is one an institution already maintains, which settles how a granule is formed and raises
the question this paper answers: what has to hold before a curated granulation can be realised
and evaluated as a higher-order learning structure.
Topological and higher-order deep learning models multi-way structure on complexes rather
than graphs, and combinatorial complex neural networks~\citep{hajij2023tdl} generalise the
graph, simplicial and hypergraph cases under one operator family. The hypergraph line is
older and better characterised: spectral clustering~\citep{zhou2006hypergraph}, then neural
formulations~\citep{chien2022allset,wang2023edhnn} treating a hyperedge as a multiset.

For this paper, the skeptical evidence matters more than the enthusiastic.
\citet{tang2024hypergraphmlp} show that removing message passing from a
hypergraph network entirely costs little on standard benchmarks.
\citet{papillon2024topotune} report that generalised complex networks match or outperform
their predecessors once architectures are matched, building them on Hasse-graph
expansions, which makes our typed star a stripped-down member of the family they found
strong. On a manifold benchmark, \citet{schmidt2026triangulation} find outcomes turn on
representation and feature assignment rather than the operator. Concurrently,
\citet{papillon2026lookbefore} argue that the lifting step deserves inspection before
training, and supply diagnostics whose metrics correlate with downstream performance. The
diagnostic offered here is exact rather than correlational, and
Proposition~\ref{prop:obstruction} says what its failure costs. None of that work names a
mechanism; we name one and show it reverses an architecture ranking through the
\emph{evaluation design} alone.
\citet{rieck2026havegraph} argues the field should source natively higher-order data rather
than keep lifting graph datasets; this archive is such a source, its ladder recorded by
archivists, so the obstruction is a fact about the data rather than a preprocessing choice.

A longer methodological literature stands behind that: retrieval and recommendation have
repeatedly found that weak baselines manufacture apparent
progress~\citep{dacrema2019worrying}, and that train/test leakage inflates
results in ways architecture ablations cannot detect~\citep{kapoor2023leakage}. For graph
networks the split already reorders architectures~\citep{shchur2018pitfalls}; we
add the structure that makes the reordering available and the mechanism
Section~\ref{sec:mechanism} measures. The corroboration retrieval task of
Section~\ref{sec:tasks} is a computational form of a question historians already ask, and
our labelling apparatus, chance-corrected agreement~\citep{krippendorff2004content} and its
known paradoxes~\citep{gwet2008ac1}, comes from content analysis.

\section{Event layer and its extractor}
\label{sec:extractor}

Our primary analysis takes rank-2 cells from the archive's controlled vocabulary. For an
archive without one we built an automatic extractor and measured it: at threshold
\leaThreshold{} its LEA $F_1$~\citep{moosavi2016lea} is \leaFone{} over \nExtractedEvents{}
clusters, it wrongly unifies \rhoMerge{} of comparable gold pairs, splits \rhoSplit{} of gold
clusters, and leaves a false singleton rate of \falseSingleton{}, above the ceiling fixed in
advance. That is a declared kill of narrow scope: these rates bound the pipeline such an
archive would have to build. A flagger marks \nAmbiguous{} contested terms, and the grid is
re-run without them to separate a method that does not work from a labelling that is
disputed.

\section{The pre-registered protocol}

In advance, the registration names the primary cell, the metrics, the seed count and a kill
threshold for each precondition; the analysis is a registered primary with documented
deviations and exploratory follow-ups. Each carries a substitution record, six in
all: most consequentially no second human annotator and no expert panel, so the agreement
study runs three programmatic protocols implementing the written manual.

That substitution changes what P2 can claim. Treating the protocols as a lower bound on human
agreement would assume misreading is the only source of human disagreement, when two people
who both know Minidoka is a camp would agree whatever the manual said. The direction of that
bias is not fixed, so we claim none. What we report is narrower: three independently defined
programmatic interpretations of one written manual, each consistent with its rules, compute
different functions, which is evidence about the specification rather than about people.

In the primary cell the trained models span \paramSpread{} between largest and smallest, and
every result in that grid is averaged over \nSeeds{} seeds; the factorial and checkpoint
sweeps of Section~\ref{sec:p3} run \facSeeds{} seeds, the tuning grid \tuneSeeds{} and the
partition replication \partSeeds{}, stated where each appears. Cost is not the obstacle:
building the complex takes \costBuild{}~s, features \costFeat{}~s and the star expansion
\costStar{}~s, against \costTrain{}~s and \costGpu{}~MiB for one training seed.

\section{Empirical results}

\subsection{Rank against cardinality}

Before any fitting, the criterion was that a Spearman correlation above $0.9$ between a cell's
rank and its size would mean the ladder is cardinality in disguise. Observed size correlation
is $\rho = \spearmanRho{}$ at the primary granularity, and between $0.34$ and $0.48$ across
every granularity and rank map we ran, so the criterion passes everywhere and P1 holds.

That criterion is a proxy for the claim, and it leaves open whether some nonlinear function of
size recovers rank anyway, so we also ran the direct test on the rank-2 and rank-3 boundary,
the one place annotators actually disagree. Predicting rank from log cell size across those
\sizeOnlyN{} boundary cells, with five-fold cross-validation, gives balanced accuracy
\sizeOnlyLogit{} for logistic regression and \sizeOnlyTree{} for a depth-three tree, against a
prior-only baseline of \sizeOnlyPrior{}. The tree's cross-entropy is worse than the prior's,
and a $k$-nearest-neighbour estimate of the mutual information between size and that
distinction returns \sizeOnlyMI{} nats, the floor the estimator clips a non-positive value to.
Over all nine granularity and rank-map cells the best any size-only predictor reaches is
\sizeOnlyMax{}, so P1 no longer rests on a correlation threshold.

That precondition is on \emph{this} construction rather than on granulation in general: rank
here records descriptive abstraction rather than headcount, so a ladder recoverable from size
would carry nothing the cardinality did not.

Two conditions bound it. Ranks 0 and 1 are
size-constrained by construction, since a segment has one speaker, so pairs involving them
re-measure that definition rather than the ladder; over all pairs at ranks 1--3 the ladder
orders by size correctly \concordanceRate{} of the time and inverts \inversionRate{} of the
time. At the rank-2/rank-3 boundary, the one the annotators dispute, the picture depends on
granularity: \invTwoThree{} of those \pairsTwoThree{} pairs invert at mid, rising above half
at coarse and falling to about a tenth at fine. P1 passes, and that boundary is where it is
closest, so how often the ranks invert is not a stable property of the archive.

A star expansion is also lossless, which is what licenses the comparison.

\begin{proposition}[Star expansion is lossless]
\label{prop:lossless}
Let $K$ be a named complex whose cells each carry a rank and a member set. Its typed
star expansion places one node per cell and one typed edge per membership, with the edge type
recording the rank. Rank and membership are recoverable from the expansion, so $K$ is
determined by its typed star.
\end{proposition}

Round-tripping all \nCells{} cells recovers every rank and member set, with no cell missing,
added or altered. The star's extra encodings are a handicap fixed in advance, which makes a
negative result conservative, and being lossless the star carries the same incidence
information arranged more simply, so any advantage the complex shows has to come from the
operator.

\subsection{Latitude left to the implementer}
\label{sec:p2}

Three protocols implementing the same written manual assign ranks to \nRankTerms{} terms.
After a documented adjudication round, their rank assignments agree at $\alpha = \alphaRankTwo{}$, against a
floor of $0.67$ set in advance; $\alpha$ here summarises divergence between program outputs,
not human inter-annotator reliability. \nDisputed{} terms remain disputed. Disagreement is
not confined to rare terms: stratifying by term frequency leaves
$\alpha = \alphaHead{}$ on the
most common terms and $\alpha = \alphaTail{}$ on the rarest, so it is about
the ladder itself rather than thin evidence.

An $\alpha$ near zero can mean genuine disagreement or a skewed-marginal artefact, so we
report the diagnostics that separate the two. Raw agreement is \rawAgreement{} and Gwet's
$AC_1$ is \gwetAC{}~\citep{gwet2008ac1}. Raw agreement far above $\alpha$ with $AC_1$ also
low is the signature of real disagreement on a hard boundary, not a paradox of the
coefficient.

For that case the manual specifies a tie-break: where the evidence is inconclusive, defer to
the archive's own facet depth. Deferral would ground the ladder in archive practice rather
than in the annotator, and it is almost never reached, because two protocols answer the
manual's questions with complementary tests that cannot return inconclusive. Only the lexical
protocol defers at all, on \deferLexical{} of terms, and the three are jointly inconclusive on
\deferBoth{}. The specification appears to settle hard cases by appeal to the archive; in
practice each implementation settles them itself.

Granularity carries the same instability. Changing it changes which cells the model is asked
to rank, so we compare triage lists across the three settings after projecting them onto a
common referent, without which the comparison is vacuous, since coarse and fine label spaces
are disjoint by construction. Rank-biased overlap~\citep{webber2010rbo} between granularities
is \rboCoarseMid{} for coarse against mid, \rboMidFine{} for mid against fine, and
\rboCoarseFine{} for coarse against fine, all below the threshold of \rboThreshold{} we set in
advance.

\subsection{Operator advantage in the primary cell}
\label{sec:p3}

Table~\ref{tab:models} gives the primary cell, where the typed star
expansion is the strongest model on the ranking task (T1 MAP \MthreeTone{}), the
full complex network reaches \MfiveTone{}, and a dense retrieval baseline with no learned
parameters reaches \MoneTone{}. Two hypergraph baselines cannot be scored on T1 at all: AllSet
and ED-HNN pass messages only between narrators and events and never update the moment
representations T1 is scored over, so their entries are the frozen-encoder floor rather than a
measurement, and we report them as undefined. On the link task the
hypergraph diffusion operator~\citep{wang2023edhnn} is strongest (AUC \MfourTtwo{}), and the
complex sits behind every message-passing baseline, on the input-deprived configuration whose
link AUC Section~\ref{sec:p3} raises to \facFiveEncAuc{}.

That comparison gives the star inputs the complex never receives, so the registered verdict
stands: the typed star as specified beats the complex at matched parameter
count, and P3 is settled against the registered system. The matched arm answers the operator
question and answers it the other way. One run supplies all four cells of the factorial at
\facSeeds{} seeds. Carrying the spectral block the star
reaches \facThreeEnc{} against the complex's \facFiveEnc{}; stripped to identical inputs it
reaches \facThreePlain{} against \facFivePlain{}. Those are Table~\ref{tab:models}'s
registered cells at five seeds rather than ten, \facThreeEnc{} against \MthreeTone{} and
\facFivePlain{} against \MfiveTone{}. The encodings are
worth
\facThreeEnc{} minus \facThreePlain{} to the star and almost nothing to the complex, so the
sign of the operator comparison is set by the inputs rather than by the operator. Their
marginal intervals overlap ([\facThreePlainLo{}, \facThreePlainHi{}] against
[\facFivePlainLo{}, \facFivePlainHi{}]), but the paired difference does not: resampling the
\facNarrDisj{} test narrators puts that \facOpPlainDisj{} at [\facOpPlainDisjLo{},
\facOpPlainDisjHi{}], and the encoded cells reverse at [\facOpEncDisjLo{},
\facOpEncDisjHi{}]. On matched inputs under the exposure-free split the rank-aware operator
does keep a small but reliable advantage, and a random split adds a further
\facOpPlainGap{}. Encodings are not wasted on the complex,
since
its link AUC rises from \facFivePlainAuc{} to \facFiveEncAuc{}; it cannot route them to the
rank the retrieval task scores. The gap is 1.4-fold in MAP and 5.1-fold in nDCG@10, so the
complex places relevant passages somewhere in the list but rarely near the top, where an
archivist reads.

Both figures condition on one realised partition, so we drew \partN{} of each kind and
repeated the matched comparison at \partSeeds{} seeds. The operator advantage survives every
draw, averaging \partNdMean{} under the narrator-disjoint split and \partRandMean{} under the
random one, positive on all \partN{} draws of both kinds, with the registered cell's
\facOpPlainDisj{} inside the narrator-disjoint range [\partNdMin{}, \partNdMax{}]. The split
multiplies the gap \partAmp{}-fold and inverts it on \partSignFlips{} of \partN{} pairs; the
inversion reported below is a comparison of systems, where the star also carries the
encodings. Exposure costs precision too: seed variance of the gap runs \partSeedVar{} times
higher under the random split, wide enough that no partition-level component is detectable
there, against \partNdShare{} of the observed spread under the narrator-disjoint one.

\begin{table}[t]\centering
\caption{A lossless star expansion is the stronger ranker at matched parameter count.
Primary cell: mid granularity, narrator-disjoint split, mixed negative sampling, rank map
R-A; means over \nSeeds{} seeds at matched parameter budget.}
\label{tab:models}
{\footnotesize\input{table_models}}\end{table}

In Table~\ref{tab:ablations}, the first ablation removes the moment rank and raises link AUC
from \MfiveTtwo{} to \ablationNoMomentsTwo{}. Ranking for that ablation is not interpretable,
since T1 is scored over the moment representations it removes from the ladder.

Link results stand on their own, and the complex's is not an optimisation failure: with
dropout off it drives training loss to \convLossFive{} on \convSubset{} held-in incidences
while its validation AUC never exceeds \convValFiveMax{} at any of \convEpochs{} epochs, and
its best anywhere on the tuning grid is \tuneFiveAuc{}. The ceiling is representational. Nor
does it move under the selection rule or with tuning: reselecting by validation MAP widens the
narrator-disjoint gap from \selDisjAuc{} to \selDisjMap{} over \selSeeds{} seeds, and an
equal-budget grid of \tuneTrials{} configurations per model over \tuneSeeds{} seeds leaves
\tuneGap{}, with the complex's best T1 MAP, \tuneFiveBest{}, below the star's worst,
\tuneThreeWorst{}. The star's winning configuration there is the registered default,
$\text{lr} = \tuneThreeLr{}$ and dropout \tuneThreeDo{}, so no tuning advantage was withheld
from the baseline the argument runs against. All three architectures that never update moment
representations, this ablation and the two rank-agnostic baselines, sit at the top of the
link range, while the complex that keeps the moment rank sits near chance. That rank is
harmless in itself, since the star expansions carry it and score well above the complex; what
the ablation isolates is narrower, that inside this complex the intermediate rank is where
the rank-aware operator loses the link task.

\begin{table}[t]\centering
\caption{Removing the moment rank collapses ranking and improves linking, so the two tasks
do not want the same structure. Ablations of the complex network, primary cell.}
\label{tab:ablations}
{\footnotesize\input{table_ablations}}\end{table}

Two further checks bound the point estimates. Both start from the unit of dependence,
narrators, whose design effect is \designEffect{} ($\text{ICC} = \iccNarrator{}$ over
\nClusters{} narrators), so intervals computed as though the \nIncidences{} incidences were
independent would be narrower than honest ones by roughly \deffSqrt{}. On the link task the
complex reaches a narrator-clustered per-incidence accuracy of \ccnnAccPoint{} (95\% CI
[\ccnnAccLo{}, \ccnnAccHi{}]), an interval containing chance. Narrator counts differ by
statistic, each naming a population: \obsGround{} in the archive, \doseNarrators{} with a
per-narrator gap, \probeClasses{} probeable, \straddleTotal{} holding rank-2 incidences,
\didNarrators{} bearing test queries, and \facNarrDisj{} in the factorial's
narrator-disjoint split.

Our second check is sharper. Predictions enter it averaged over the \nSeeds{} seeds within
each held-out incidence before fitting, so seeds do not enter as independent observations; the
model then regresses correctness on architecture, log event size and their interaction, with
crossed random intercepts for narrator and event. Fitted over \interactionEvents{} events and
the \interactionNarrators{} narrators appearing in test-set pairs, most of them through
sampled negatives rather than through a held-out incidence, it puts the interaction at
\interactionCoef{} (95\% CI [\interactionCiLo{}, \interactionCiHi{}],
$p = \interactionP{}$). Event count here is smaller than the \nRankTwoCells{} rank-2 cells in
the complex because the model is fitted on held-out test incidences. The interval excludes
zero, and the sign is the one the representation should have ruled out: the complex falls
further behind the typed star as events grow, precisely where the extra ranks were supposed
to earn their place.

This coefficient is fitted on the primary cell alone; the release reports the pooled estimate,
\interactionCoefOld{} over \interactionEventsOld{} events, and why mixing three granularities
with the random split measures nothing the paper claims.

Under the test fixed in advance, almost stochastic order~\citep{dror2019deepdominance}
Bonferroni-corrected across \asoBonferroni{} model pairs and computed with
deep-significance~\citep{ulmer2022deepsig}, the typed star stochastically dominates the
complex on T1 ($\varepsilon_{\min} = \asoThreeOverFive{}$), while the reverse returns
$\varepsilon_{\min} = \asoFiveOverThree{}$, the maximum possible violation. Correcting over
all \asoBonferroni{} ordered pairs rather than the single pair P3 names leaves it
conservative.

One defence of a rank-structured model remains: it should pay off precisely where structure
is richest, on events many narrators describe. Panel (b) of Figure~\ref{fig:models} tests that
directly and finds the opposite, the complex leading only on the two smallest bins, where the
intervals are widest, and falling progressively further behind as events grow, the same
pattern the interaction estimate reports.

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{../figures/fig3_models.pdf}
\caption{The typed star leads at matched budget, its margin widens as events grow, and the
evaluation split decides the ordering. (a) T1 MAP in the primary cell with all ten seeds shown
individually, because means hide bimodality; hollow markers labelled \emph{floor} are AllSet
and ED-HNN, which never update rank-1 cells. (b) Per-incidence accuracy stratified by event
size with narrator-clustered 95\% intervals, where $n$ counts held-out incidences. (c) The
same two models under all three splits, everything else held at the primary cell; both splits
that let a narrator appear on both sides favour the complex. Intervals in (a) and (c) are
bootstrap 95\% CIs over ten seeds, which measure optimisation noise; the narrator-clustered
interval the split claim rests on is in the text.}
\label{fig:models}\label{fig:split}\end{figure*}

\section{Split dependence of the model ordering}

Panel (c) of Figure~\ref{fig:split} was not planned in advance. Under exactly one of the three
splits, the pre-registered narrator-disjoint one, the typed star leads; under a random split
the ordering inverts and the complex leads by a wide margin (\MfiveToneRandom{} against
\MthreeToneRandom{}), and under an event-disjoint split it leads again. What those splits
share is the partition rather than the architecture: only the narrator-disjoint split keeps a
narrator's segments on one side, so the other two reward a model for recognising a narrator it
was trained on.

The quantity carrying the claim is the difference between those two gaps. Seeds measure
optimisation noise, but the sampling unit the claim is about is the query, and queries
cluster by narrator, so we resample \didNarrators{} narrators with replacement over
$10{,}000$ replicates, averaging over seeds within each. Across those replicates the
complex's random-split gain
exceeds the star's by \didPoint{} MAP, with a 95\% interval of [\didLo{}, \didHi{}]. It
therefore does not equal the difference of the cell means above. Nor does the reversal depend
on the selection
rule: reselecting every checkpoint by validation MAP leaves it at \selRandMap{} against
\selRandAuc{} over \selSeeds{} seeds. Nor is it an artefact of one partition: repeating the
registered comparison on \partN{} independently drawn partitions of each kind leaves the star
ahead on every narrator-disjoint draw and the complex ahead on every random one,
\partRegNdMean{} against \partRegRandMean{} in the mean, so the two split types disagree
about the sign on \partRegFlips{} of \partN{} pairs.

Little else matters. Table~\ref{tab:invariance} varies every design axis the field argues
about, granularity, negative-sampling regime and rank map, and the ordering survives
\invarianceWins{} of \invarianceTotal{} conditions. One axis reverses that ordering, and it is
usually a default rather than a decision.

\begin{table}[t]\centering
\caption{The system-level ordering survives every design choice except the split. Each row
varies one
axis with the rest held at the primary cell, and reports how often the typed star leads the
complex on T1 MAP. Every cell uses the star as registered; the input-matched ordering is in
Section~\ref{sec:p3}.}
\label{tab:invariance}
{\footnotesize\input{table_invariance}}\end{table}

A model's score under a random split minus its score under an entity-disjoint one is its
\emph{entity exposure gap}, reported per model in panel (c) of Figure~\ref{fig:split}.

\section{Mechanism: narrator identity in the features}
\label{sec:mechanism}

An explanation is owed, and two experiments run after the split result, labelled exploratory,
supply one more concrete than idiolect. Feature construction is identical for every model: a
rank-0 narrator is the mean of that narrator's segment embeddings, and a rank-$k$ cell the
mean of its constituent rank-0 features, built once from the whole archive and split
afterwards. A narrator's vector therefore aggregates every segment that narrator contributed,
and the narrator-disjoint split closes both routes at once, the repeated vector and the
cross-partition text. Those routes exist because the obstruction of
Proposition~\ref{prop:obstruction} reaches the features, which does not depend on this
archive.

\begin{proposition}[Lifting carries the obstruction into the features]
\label{prop:singleton}
Let rank-$k$ features be lifted from rank 0 by a permutation-invariant multiset reduction
$\varphi$, so that a cell's input is $\varphi(\{x_s : s \in \mathrm{supp}(c)\})$. Cells sharing
a support share an input, and the distinct rank-1 inputs number at most the distinct supports.
Where a cell has the single owner $\pi(c)$, its input is a function of $\pi(c)$ alone, and
separating two such cells requires message passing through a cell of rank at least two.
\end{proposition}

So \obsCells{} moments admit at most \obsDistinct{} distinct rank-1 inputs
(Figure~\ref{fig:overview}b). For the mean, $\varphi(\{x\}) = x$, so a moment's input vector
\emph{is} its narrator's vector; a learnable $\varphi$ changes the value but not the
dependence on $\pi(c)$ alone. We verified the equality: for every narrator holding more than
one such moment, the largest elementwise difference between two of them is exactly zero.

The split decides whether that degeneracy is visible. Of \straddleTotal{} narrators,
\straddleRandomN{} straddle the random split (\straddleRandom{}) and \straddleEventN{} the
event-disjoint one (\straddleEvent{}), since holding out events does not hold out the
narrators who witnessed them; both place a narrator's vector on both sides, and a
narrator-disjoint split places none. Both rates are predictable before any model exists:
the random split partitions incidences, so assigning a narrator's $k$ of them independently
across folds of proportion $p_f$ predicts
$1 - \sum_f p_f^{k}$, or \exposurePredicted{} against \straddleRandom{} observed, with the
\exposureSingletonN{} single-cell narrators contributing zero. The splitter draws fold sizes
exactly, so the exact law is hypergeometric; it differs from the independent form by
\exposureHyperGap{} points here. Exposure is not an artefact of
one lifting: across \liftConfigs{} liftings the singleton fraction ranges
\liftSingLo{} to \liftSingHi{} while predicted exposure stays \liftExpLo{} to \liftExpHi{},
because it follows from how many cells an entity owns rather than from how many it owns
alone. Those liftings are this archive's rank maps and granularities, so the sweep bounds the
choice rather than the collection.

That first experiment is easy to over-read, so its control comes first. A linear probe
recovers narrator identity from the \emph{input} rank-1 features at
\probeInputControl{} over \probeClasses{} narrators and \probeN{} moments, against a
majority-class floor of \probeChance{}, a tautology since the classes are identical vectors.
The probe is a multinomial logistic regression over standardised rank-1 vectors, fitted on a
stratified split of segments rather than narrators, so every class appears on both sides.
What the trained models show is how much inherited identity survives
message passing, and the answer is most of it: \probeNarrFive{} for the complex against
\probeNarrThree{} for the typed star under the narrator-disjoint split, \probeRandFive{}
against \probeRandThree{} under the random one. Complex scores are higher in both and the
seed spreads overlap, so neither architecture discards narrator identity.

A second experiment intervenes on the mechanism: we shuffled narrator labels across segments,
preserving segments per narrator, and re-ran both splits. Anonymisation changes which
passages count as positives, so an anonymised score is not comparable to a real one; what is
comparable is the \emph{gap between splits}, because both are transformed alike. With real
narrators the complex gains \gapRealFive{} MAP moving from the narrator-disjoint split to the
random one. With identity destroyed that difference is \gapAnonFive{}, a reduction of
\gapReductionFive{}, so its whole advantage disappears. The star's gap
falls too, from
\gapRealThree{} to \gapAnonThree{}, by less than half as far and from a value that was never
positive: the shuffle costs the complex its advantage and costs the star a gap it did not
have.

Shuffling also separates the two routes: it preserves how many segments each label holds, so
cross-partition aggregation survives while identity correspondence does not, and a shuffled
aggregate has less to leak, which makes split-first features the direct test.

Each narrator's vector is rebuilt from that narrator's training-side segments, a segment
counting as training-side when none of that narrator's incidences through it are held out,
with a fallback to a narrator's own segments, not a leak because a new interview arrives at
deployment with its transcript. Under the random split the construction strips a
mean \sfMeanRemoved{} of the segments contributed by each of \sfNarrTouched{} affected
narrators.

The reversal survives it, which is what we predicted in writing beforehand. Moving from the
narrator-disjoint split to the random one, the complex gains \sfLiftFiveGlobal{} MAP under
global features and \sfLiftFiveSF{} under split-first ones, and the input-matched star gains
\sfLiftThreeGlobal{} and \sfLiftThreeSF{}. Both arms come from one re-run of the primary
cell, so they are comparable to each other rather than to the grid values above.
Removing every token of test-side text leaves the
advantage where it was, so cross-partition aggregation is not the carrier and what the random
split rewards is repeated narrator identity.

The spectral encodings suppress exposure in both architectures, which is the second lever on
the same mechanism. Across the factorial, the star's exposure gap falls from
\facExpThreePlain{} on plain inputs to \facExpThreeEnc{} with the encodings, and the
complex's from \facExpFivePlain{} to \facExpFiveEnc{}. On matched inputs both leak, the
complex by roughly \facExpFivePlain{} against \facExpThreePlain{}, so the leak is
differential rather than present in one architecture and absent in the other. Features that
carry structure rather than identity reduce what a random split can reward, whichever operator
reads them.

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{../figures/fig7_mechanism.pdf}
\caption{Destroying narrator identity removes the complex's random-split advantage, and
editing the moment features does not. (a) Gain in T1 MAP from switching to a random split,
with 95\% intervals over seeds. The item-specific arm was run separately against its own
baseline, so its bar carries the same quantity but not the same reference as the other two.
(b) A linear probe recovering narrator identity from moment representations, against a
majority-class floor of \probeChance{} over \probeClasses{} narrators. The dashed line is the
same probe on the input features, \probeInputControl{} by construction. The axis runs the
full range.}
\label{fig:mechanism}\end{figure*}

The random split does not merely correlate with the complex's advantage: removing what it
leaks removes it, and the operator that propagates further through the narrator layer is the
one the default split rewards.

A third experiment bounds that account. If the leak were carried by the moment vector alone,
replacing it should close the gap, so we gave every rank-1 cell its own passage embedding
instead of its narrator's mean, holding the complex, the splits, the seeds and the budget
fixed. Two moments by one narrator are then no longer identical at the input, and the gap
still did not close: against that experiment's own baseline it moved from \featGapMean{} to
\featGapItem{} for the complex, slightly wider rather than narrower, and the typed star
stayed where it was (\featGapMeanThree{} against \featGapItemThree{}). We report that
refuted prediction because it bounds the claim. Only the rank-1 features changed, while the
rank-0 narrator layer and every message path through it stayed in place, and that layer is
what anonymisation destroys, which is why splitting on the narrator fixes the leak and editing
features does not.

A dose-response prediction is also on the record: the complex's per-narrator random-split
advantage should rise with that narrator's segment count, against a flat control slope. Over
\doseNarrators{} narrators the random-split slope is \doseSlope{} (95\% CI [\doseSlopeLo{},
\doseSlopeHi{}]) against a control slope of \doseSlopeControl{}. That control is not flat: the
complex falls further behind on prolific narrators exactly where it cannot have seen them, so
the mechanism stands on the intervention rather than on the correlation.

Our dense baseline is scored on each passage's own frozen embedding while every trained model
sees its narrator's mean, so \MoneTone{} reflects that trade; the typed star against the
complex, which is what the paper claims, sits on identical inputs.

\section{Four rival explanations}

A negative result is worth reporting only if it survives the obvious explanations. Four were
tested; the segmentation check settles at once, its three defensible extraction settings being
the granularities of \S\ref{sec:p2} renamed, agreeing no better than \extractorMinRbo{}.

\paragraph{The ordering holds on a cleaned grid.} The event layer is
under-merged (\S\ref{sec:extractor}), so perhaps the complex is being asked to model damage.
Rebuilding the archive with the \goldTermsExcluded{} contested terms removed leaves
\goldSubsetSegments{} segments and \goldCells{} rank-2 cells, against the \nRankTwoCells{}
the primary complex carries. All models rise on the reduced grid because the candidate pool
shrinks, and the star--complex ordering is unchanged, \goldMthree{} against \goldMfive{}, so
the damage is not what the complex is modelling. A parameter-free baseline rises furthest, to
\goldMone{}, and overtakes both, which is a pool-size effect and a reminder of how little of
this ranking the learned structure contributes.

\paragraph{Curated cells survive their own error rates.} We propagated the measured merge and
split error rates through the complex over a \pertGridPoints{}-point grid per calibration,
twenty draws each. Calibrated to the archive's internal disagreement, the triage list retains
$\text{RBO}_{50} = \pertRboArchive{}$ against the unperturbed ranking, clear of the registered
threshold of \pertThreshold{}, and that is the figure governing the triage we report, since
the complex analysed here is built from the curated vocabulary. Substituting the extractor's
own error rate drops retention to \pertRbo{}, below the threshold, whose committed consequence
is that an automatically built triage list needs human verification. Instability here is
inherited from the extractor, not from the curated cells.

\paragraph{Counting interviewers erases the phenomenon.} Counting interviewer turns as
attestations inflates every cell, taking the singleton fraction from \singletonNarrators{} to
\singletonWithInterviewers{}, since an interviewer is present in every segment. The two triage
lists still agree at \interviewerRbo{}, so the decision looks harmless until the quantity that
matters is inspected.

\section{Utility of the representation}

The application claim is archival triage: surfacing events that rest on too few voices. The
quantity ranked is \emph{archive-conditioned} attestation multiplicity, how many narrators
\emph{in this collection} describe an event, so one thinly attested here may be abundantly
documented elsewhere. Of the top \triageK{} triage items \triageUnstable{} are unstable under
the conditions tested, so the list is a screening device; each item ships with the probability
that it is a singleton only through extraction error.

\paragraph{Mention frequency sets the bar the structure has to clear.} Archive-conditioned
attestation multiplicity correlates with bare mention frequency at $\rho = \freqSpearman{}$,
the two triage lists agree at $\text{RBO}_{50} = \freqRbo{}$, and they share a Jaccard overlap
of \freqJaccard{} in their top ten. Counting mentions recovers most of what the ladder, the
encodings and the complex were built to recover, so the structure-derived ranking has to
justify itself where the two disagree, and that region is the minority of the list.

\section{Transfer to a held-out collection}

What transfers is a conclusion rather than a model: every threshold, rank map and
hyperparameter was fixed on the primary archive and applied untuned to held-out collections
of \transferInterviews{} interviews and \transferSegments{} segments. Rank and cardinality:
$\rho = \transferRho{}$ over \transferCells{} rank-2 cells, against \nRankTwoCells{} at home.
Rank agreement: $\alpha = \transferAlpha{}$. Models: typed star \transferMthree{}, complex
\transferMfive{}. Those levels are not comparable with the primary archive's, given the
smaller cell count and the absence of transcripts, so we leave the model comparison unread
there. Our pre-registered test is narrower: whether the interaction estimate keeps its sign
and magnitude. It does not, so the interaction finding is archive-specific, as we committed
in advance.

\section{Release and instruments}

A registered topology analysis is absent: its permutation null tests nothing, since shuffling
which rank-2 cell holds which narrator set is a relabelling and homology ignores labels, so
$p = \topoPermP{}$ was forced by construction. Event-size and connectivity moments predict
$\beta_1$ at $R^2 = \topoBetaOneRtwo{}$, above the registered $0.9$.

Every estimator was asserted against a known answer before being pointed at the archive,
which caught eight defects that would otherwise have been reported as findings. Our release,
\emph{CAIRN} (\url{\repoURL}), is stand-off because the archive's material is not ours to
redistribute; it carries the cell definitions, incidence matrices at all three granularities,
the splits, the annotation manual, a licence audit and a fetch script. A fresh interpreter
recomputes the headline corpus statistics from those files alone; independent human
replication is the one substitution outstanding.

\section{Limitations}

Six substitutions are on the record, each bounding the claim in a stated direction. The
agreement study measures whether the manual determines the assignment rather than whether
people agree, and the transfer collection refutes a strong effect without confirming a weak
one. Four others are registered: no expert panel, no external coreference corpus, no second
archive, and a self-sufficiency check for independent reproduction. No contemporary
generalised-CCNN baseline was run, so the operator comparison is with the star expansion and
the hypergraph family. Our corpus is one archive in one language, and the ladder is its
cataloguers' choice, so a different descriptive tradition could behave differently. What
survives those caveats is the structure of the argument: the three preconditions are testable
anywhere, and the lifting creates the same exposure wherever cells share recurring owners,
though the magnitude and the reversal are findings on this archive.

\section{Conclusion}

The ownership obstruction is what this archive turned up. Where items are singly owned, the
cells above the owners are not subsets a combinatorial complex can hold apart, and here
\obsCells{} moments collapse onto \obsDistinct{} supports, so a lifted rank-1 feature is a
function of who owns it. Rank carries information beyond cell cardinality and the star
expansion is lossless, so the ladder is real; the specification does not yet supply a
reproducible assignment, and our ranked-incidence model has still to earn its extra structure. What the star
wins with, though, is its inputs: on matched features the ordering reverses, so the
registered verdict is about the system rather than the operator.

Evaluation decides whether the collapse is visible. A random split lifts the complex
\didPoint{} MAP more than it lifts the star (95\% CI [\didLo{}, \didHi{}]), and that
difference survives rebuilding every feature split-first, so the carrier is repeated narrator
identity. Shuffling that identity removes the advantage, and giving each moment its own
passage text does not. Which split is correct depends on the deployment population: entity
overlap invalidates unseen-narrator retrieval and requires an entity-disjoint split, while
for within-archive retrieval it is legitimate provided the exposure is reported rather than
absorbed into the headline. Beyond this archive, the obstruction reaches any collection whose
items belong to recurring entities, so for unseen-entity evaluation the defence is to split on
the entity, and otherwise to report the exposure gap beside the score and name the population
estimated.

\bibliographystyle{IEEEtranN}
\bibliography{references}
\end{document}
"""


# --- claims whose DIRECTION the prose asserts ------------------------------------------
# A macro guarantees the paper quotes the right number; it does not stop a sentence from
# saying "the ordering is unchanged" next to a number that reverses it. Each entry names a
# sentence that reads a direction out of the data and the inequality it depends on. A
# violation means the prose must change, not the number.
def _cmp(a, b):
    return None if a is None or b is None or a != a or b != b else a > b


def direction_claims(get):
    def g(src, path):
        v = dig(load(src), path) if load(src) is not None else MISSING
        return None if v is MISSING else v

    return [
        ("P3: typed star leads the complex on T1 in the primary cell",
         _cmp(get("M3_typed_star", "T1"), get("M5_ccnn", "T1"))),
        ("Split: the ordering inverts under a random split",
         _cmp(get("M5_ccnn", "T1_random"), get("M3_typed_star", "T1_random"))),
        ("E8.3: removing contested terms does not reverse the ordering",
         _cmp(g("e8_3_gold_subset", "model_means/M3_typed_star/T1_map"),
              g("e8_3_gold_subset", "model_means/M5_ccnn/T1_map"))),
        ("E8.4: counting interviewers lowers the singleton fraction",
         _cmp(g("e8_4_interviewer", "singleton_fraction_narrators_only"),
              g("e8_4_interviewer", "singleton_fraction_with_interviewers"))),
        ("E10: the typed star again leads on the transfer archive",
         _cmp(g("e10_transfer", "E10_2_zero_shot/model_means/M3_typed_star/T1_map"),
              g("e10_transfer", "E10_2_zero_shot/model_means/M5_ccnn/T1_map"))),
        ("E10: rank and cardinality remain decoupled (rho below 0.8)",
         _cmp(0.8, g("e10_transfer", "E10_2_zero_shot/rank_cardinality/spearman_rho"))),
        ("Mechanism: the complex gains from the random split with real narrators",
         _cmp(g("e_mechanism", "split_gap/M5_ccnn/gap_real_narrators"), 0.0)),
        ("Mechanism: anonymisation removes that gain",
         _cmp(g("e_mechanism", "split_gap/M5_ccnn/gap_real_narrators"),
              g("e_mechanism", "split_gap/M5_ccnn/gap_anonymised"))),
        ("Mechanism: message passing does not discard the inherited narrator identity",
         _cmp(_probe_acc("random", "M5_ccnn"), 0.5)),
        ("Mechanism: the input-feature probe is at the ceiling, so identity is inherited",
         _cmp(_probe_acc("narrator-disjoint", "frozen_encoder_typed") or 0.0,
              _probe_acc("narrator-disjoint", "M5_ccnn"))),
        ("Mechanism: item-specific moment features do NOT close the gap",
         _cmp(g("e_feature_path", "split_gap/M5_ccnn/gap_item_specific"), 0.0)),
    ]


def _probe_acc(split: str, model: str):
    mech = load("e_mechanism") or {}
    for r in mech.get("probe", []):
        if r.get("split") == split and r.get("model") == model:
            return r.get("probe_accuracy_mean")
    return None


def _unresolved_in_text(pdf: Path, logtxt: str) -> bool:
    """True when a reference actually renders as '??' on the page."""
    import subprocess

    if re.search(r"There were undefined (references|citations)", logtxt):
        return True
    try:
        out = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, timeout=120)
        if out.returncode == 0:
            return "??" in out.stdout.decode("utf-8", "replace")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False  # no text extractor available; the log above is the remaining signal


def compile_pdf() -> dict:
    """Build the PDF and check it for defects a reader would notice.

    MiKTeX silently substitutes Type 3 bitmap fonts under some package combinations, and
    most publishers reject them; the check is a raw scan for /Type3 rather than pdffonts,
    which is not on PATH here.
    """
    import subprocess

    steps = [("pdflatex", ["-interaction=nonstopmode", "main.tex"]),
             ("bibtex", ["main"]),
             ("pdflatex", ["-interaction=nonstopmode", "main.tex"]),
             ("pdflatex", ["-interaction=nonstopmode", "main.tex"])]
    for tool, args in steps:
        try:
            subprocess.run([tool, *args], cwd=OUT, capture_output=True, timeout=300, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            log(f"  {tool} unavailable or timed out ({type(e).__name__}); skipping compile")
            return {"compiled": False}

    pdf = OUT / "main.pdf"
    if not pdf.exists():
        log("  compile produced no PDF")
        return {"compiled": False}
    raw = pdf.read_bytes()
    logtxt = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
    # A single pdflatex pass leaves every \ref and \cite as "??" in the output, and a later
    # one-off pass -- a format check, say -- can silently overwrite a good PDF with a broken
    # one. Read the rendered TEXT layer, not the raw bytes: "??" occurs by chance inside
    # compressed streams (13 times here) and a byte scan reports a clean PDF as broken.
    unresolved = _unresolved_in_text(pdf, logtxt)
    if unresolved:
        log("  rendered text shows unresolved '??' references; running two more passes")
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=OUT, capture_output=True, timeout=300, check=False)
        raw = pdf.read_bytes()
        logtxt = (OUT / "main.log").read_text(encoding="utf-8", errors="replace")
        unresolved = _unresolved_in_text(pdf, logtxt)
    info = {
        "compiled": True,
        "bytes": len(raw),
        "pages": logtxt.count("Output written") and
                 int(re.search(r"Output written .*?\((\d+) pages", logtxt).group(1))
                 if re.search(r"Output written .*?\((\d+) pages", logtxt) else None,
        "type3_fonts": b"/Type3" in raw,
        "unresolved_markers": unresolved,
        "undefined_citations": len(re.findall(r"Citation `[^']+' (?:on page \d+ )?undefined", logtxt)),
        "undefined_references": len(re.findall(r"Reference `[^']+' .*? undefined", logtxt)),
        "overfull_boxes": len(re.findall(r"Overfull \\hbox", logtxt)),
    }
    log(f"  PDF: {info['pages']} pages, {info['bytes']:,} bytes, "
        f"type3={info['type3_fonts']}, unresolved={info['unresolved_markers']}, "
        f"undefined citations={info['undefined_citations']}, "
        f"undefined refs={info['undefined_references']}, overfull boxes={info['overfull_boxes']}")
    if info["type3_fonts"]:
        log("  WARNING Type 3 bitmap fonts present; most publishers reject these")
    if info["unresolved_markers"]:
        log("  WARNING PDF still shows '??' for references after extra passes")
    return info


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers, manifest, extras = build_numbers()
    (OUT / "numbers.tex").write_text(numbers, encoding="utf-8")
    (OUT / "main.tex").write_text(PAPER, encoding="utf-8")
    (OUT / "numbers_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    missing = [m for m in manifest if m["status"] == "MISSING"]
    defined = {m["macro"] for m in manifest}
    # Every generated macro is invoked as \name{} in the prose, which makes both
    # directions checkable: cited-but-undefined would fail the LaTeX build, and
    # defined-but-uncited means a number was computed and then quietly dropped.
    cited = set(re.findall(r"\\([A-Za-z]+)\{\}", PAPER)) - LATEX_EMPTY_BRACE
    referenced = sorted(defined & cited)
    undefined = sorted(c for c in cited if c not in defined)
    unused = sorted(defined - cited)

    log(f"macros defined {len(manifest)}, cited in prose {len(referenced)}, "
        f"missing sources {len(missing)}")
    for m in missing:
        flag = "CITED IN PROSE" if m["macro"] in cited else "not cited"
        log(f"  MISSING {m['macro']} <- {m['source']}:{m['path']} ({flag})")
    if unused:
        log(f"  defined but never cited: {', '.join(unused)}")
    if undefined:
        log(f"  CITED BUT UNDEFINED (build will fail): {', '.join(undefined)}")

    offenders = lint_prose(PAPER)
    if offenders:
        log(f"  prose lint: {len(offenders)} bare numeral(s)")
        for o in offenders[:20]:
            log(f"    {o}")
    else:
        log("  prose lint: no bare numerals; every quantity resolves through a macro")

    def getter(model, field):
        pm, sp = extras["per_model"], extras["splits"]
        if field == "T1_random":
            return sp.get("random", {}).get(model)
        return (pm.get(model) or {}).get(field)

    claims = direction_claims(getter)
    violated = [d for d, ok in claims if ok is False]
    pending = [d for d, ok in claims if ok is None]
    log(f"  direction check: {sum(1 for _, ok in claims if ok)} hold, "
        f"{len(violated)} VIOLATED, {len(pending)} pending data")
    for d in violated:
        log(f"    VIOLATED - the prose asserts a direction the data does not support: {d}")
    for d in pending:
        log(f"    pending: {d}")

    blocking = [m for m in missing if m["macro"] in cited]

    # The author block ships with placeholders so the paper compiles at 10 pages before
    # the real names are known. Submitting them would be worse than an empty block.
    placeholders = sorted(set(re.findall(r"REPLACE [^,\\}\n]*", PAPER)))
    if placeholders:
        log(f"  SUBMISSION BLOCKER: {len(placeholders)} placeholder(s) still in the paper")
        for p in placeholders[:6]:
            log(f"    {p.strip()}")

    build = compile_pdf() if "--compile" in sys.argv else {"compiled": False, "skipped": True}
    (OUT / "build_report.json").write_text(json.dumps(
        {"macros": len(manifest), "cited": len(referenced), "missing": len(missing),
         "blocking": [m["macro"] for m in blocking], "prose_lint": offenders,
         "placeholders": placeholders,
         "direction_claims": [{"claim": d, "holds": ok} for d, ok in claims],
         "build": build}, indent=2), encoding="utf-8")
    log(f"paper written to {OUT}; {len(blocking)} cited macro(s) still unresolved")


if __name__ == "__main__":
    main()


