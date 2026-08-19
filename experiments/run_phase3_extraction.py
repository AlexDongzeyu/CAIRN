"""PHASE 3 — extraction validation, and an honest statement of what was substituted.

The protocol's Phase 3 assumes a cross-document event coreference (CDEC) system trained
on hand-annotated testimony. Per PREREGISTRATION.substitution_ledger E3_3, no such
annotation was available, so the archive's own professionally-curated topic assignments
are treated as the gold clustering and an unsupervised text-similarity clusterer plays
the role of the automatic system.

What that buys, and what it does not:
  * It DOES give real, measured false-merge / false-split / false-singleton rates against
    a human-curated reference, which is what E8.1's perturbation calibration and E9.4's
    cost-of-error analysis actually need.
  * It does NOT calibrate a linguistic CDEC system against ECB+/GVC (E3.2). That gap is
    recorded rather than papered over, and no claim about coreference quality is made.

E3.4 reports the full cluster-level metric suite with LEA as primary, because the
downstream quantity is cluster size and MUC/CEAF/B3 are indifferent to where an error
happened.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder, segment_embeddings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("phase3")
SIZE_BINS = [(1, 1), (2, 3), (4, 10), (11, 50), (51, 10**9)]


# ---------------------------------------------------------------- coreference metrics
def muc(gold: list[set], sys_: list[set]) -> tuple[float, float, float]:
    def score(a, b):
        num = den = 0
        for c in a:
            if len(c) < 2:
                continue
            parts = {frozenset(c & s) for s in b}
            parts = {p for p in parts if p}
            num += len(c) - len(parts)
            den += len(c) - 1
        return num / den if den else 0.0

    r, p = score(gold, sys_), score(sys_, gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def b_cubed(gold: list[set], sys_: list[set]) -> tuple[float, float, float]:
    g_of = {m: c for c in gold for m in c}
    s_of = {m: c for c in sys_ for m in c}
    mentions = set(g_of) & set(s_of)
    if not mentions:
        return 0.0, 0.0, 0.0
    p = float(np.mean([len(g_of[m] & s_of[m]) / len(s_of[m]) for m in mentions]))
    r = float(np.mean([len(g_of[m] & s_of[m]) / len(g_of[m]) for m in mentions]))
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def ceaf_e(gold: list[set], sys_: list[set]) -> tuple[float, float, float]:
    from scipy.optimize import linear_sum_assignment

    if not gold or not sys_:
        return 0.0, 0.0, 0.0
    sim = np.zeros((len(gold), len(sys_)))
    for i, g in enumerate(gold):
        for j, s in enumerate(sys_):
            inter = len(g & s)
            if inter:
                sim[i, j] = 2 * inter / (len(g) + len(s))
    r_i, c_i = linear_sum_assignment(-sim)
    total = sim[r_i, c_i].sum()
    p = total / len(sys_)
    r = total / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def lea(gold: list[set], sys_: list[set]) -> tuple[float, float, float]:
    """Moosavi & Strube (2016). Link-based and entity-size aware.

    Primary here because the downstream quantity IS cluster size: a metric indifferent to
    whether an error happened in a large or a small cluster is indifferent to exactly the
    thing this project measures.
    """
    def links(c):
        n = len(c)
        return n * (n - 1) / 2 if n > 1 else 1.0

    def resolution(a, b):
        num = den = 0.0
        for c in a:
            if not c:
                continue
            imp = len(c)
            shared = sum(len(c & s) * (len(c & s) - 1) / 2 if len(c & s) > 1
                         else (1.0 if len(c) == 1 and len(c & s) == 1 and len(s) == 1 else 0.0)
                         for s in b)
            num += imp * (shared / links(c))
            den += imp
        return num / den if den else 0.0

    r, p = resolution(gold, sys_), resolution(sys_, gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- system clustering
def cluster_by_similarity(emb: np.ndarray, threshold: float, seed: int = 0) -> np.ndarray:
    """Agglomerative clustering on cosine distance. The cut threshold is a first-class
    hyperparameter, swept in E3.3 because it directly controls the singleton fraction."""
    from sklearn.cluster import AgglomerativeClustering

    model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                    metric="cosine", linkage="average")
    return model.fit_predict(emb)


def clusters_from_labels(ids: list[str], labels) -> list[set]:
    d: dict[int, set] = defaultdict(set)
    for i, lab in zip(ids, labels):
        d[int(lab)].add(i)
    return list(d.values())


def attestation(clusters: list[set], narrator_of: dict[str, list[str]]) -> dict[frozenset, int]:
    return {frozenset(c): len({n for sid in c for n in narrator_of.get(sid, [])}) for c in clusters}


def main() -> None:
    set_all_seeds(0)
    _, segments = load_corpus()
    encoder = get_encoder()
    emb, sid2i = segment_embeddings(segments, encoder)

    # gold = the archive's own curation: segments sharing a topic term are archive-asserted
    # to be about the same thing.
    gold_map: dict[str, set] = defaultdict(set)
    for s in segments:
        for t in s.topics:
            if t.get("term"):
                gold_map[t["term"]].add(s.segment_id)
    gold = [c for c in gold_map.values() if c]
    covered = sorted({sid for c in gold for sid in c})
    log(f"gold clusters (archive topic terms): {len(gold)} over {len(covered)} segments")

    narrator_of = {s.segment_id: s.narrators for s in segments}

    # Average-linkage agglomerative clustering materialises the full pairwise distance
    # matrix, so the sweep runs on a capped random subsample of the covered segments.
    # The cap is reported; it bounds resolution, not validity.
    MAX_N = 5000
    rng = np.random.default_rng(0)
    if len(covered) > MAX_N:
        keep = set(rng.choice(covered, size=MAX_N, replace=False).tolist())
        covered = [s for s in covered if s in keep]
        gold = [c & keep for c in gold]
        gold = [c for c in gold if c]
        log(f"  subsampled to {len(covered)} segments / {len(gold)} gold clusters for clustering")

    idx = np.array([sid2i[s] for s in covered])
    sub_emb = emb[idx]

    # ---- E3.3 threshold sweep: singleton fraction is directly controlled by this knob.
    # The range must bracket the optimum; a sweep whose best value is its own endpoint has
    # not found one.
    sweep = []
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        labels = cluster_by_similarity(sub_emb, thr)
        sysc = clusters_from_labels(covered, labels)
        a_sys = attestation(sysc, narrator_of)
        sweep.append({
            "threshold": thr,
            "n_clusters": len(sysc),
            "singleton_fraction": float(np.mean([v == 1 for v in a_sys.values()])),
            "lea_f1": lea(gold, sysc)[2],
            "conll_f1": float(np.mean([muc(gold, sysc)[2], b_cubed(gold, sysc)[2],
                                       ceaf_e(gold, sysc)[2]])),
        })
        log(f"  thr={thr:.2f} clusters={len(sysc):5d} LEA-F1={sweep[-1]['lea_f1']:.3f} "
            f"CoNLL-F1={sweep[-1]['conll_f1']:.3f} singleton={sweep[-1]['singleton_fraction']:.3f}")

    best = max(sweep, key=lambda r: r["lea_f1"])          # E3.3 selects on LEA, not CoNLL
    at_boundary = best["threshold"] in (sweep[0]["threshold"], sweep[-1]["threshold"])
    log(f"E3.3 - selected threshold {best['threshold']} by LEA F1 = {best['lea_f1']:.3f}"
        f"{'  [WARNING: optimum at sweep boundary]' if at_boundary else ''}")

    labels = cluster_by_similarity(sub_emb, best["threshold"])
    sysc = clusters_from_labels(covered, labels)

    # ---- E3.4 full metric suite
    metrics = {
        "MUC": muc(gold, sysc), "B3": b_cubed(gold, sysc),
        "CEAFe": ceaf_e(gold, sysc), "LEA": lea(gold, sysc),
    }
    conll = float(np.mean([metrics["MUC"][2], metrics["B3"][2], metrics["CEAFe"][2]]))

    # ---- E3.5 merge/split decomposition + attestation distortion
    from scipy.optimize import linear_sum_assignment

    sim = np.zeros((len(gold), len(sysc)))
    for i, g in enumerate(gold):
        for j, s in enumerate(sysc):
            sim[i, j] = len(g & s)
    r_i, c_i = linear_sum_assignment(-sim)
    aligned = {i: j for i, j in zip(r_i, c_i)}

    split_flags, merged_pairs, comparable_pairs = [], 0, 0
    sys_of_seg = {sid: j for j, s in enumerate(sysc) for sid in s}
    for i, g in enumerate(gold):
        touched = {sys_of_seg[sid] for sid in g if sid in sys_of_seg}
        split_flags.append(len(touched) >= 2)
    gold_of_seg = {sid: i for i, g in enumerate(gold) for sid in g}
    for j, s in enumerate(sysc):
        gs = {gold_of_seg[sid] for sid in s if sid in gold_of_seg}
        if len(gs) >= 2:
            merged_pairs += len(gs) * (len(gs) - 1) / 2
    comparable_pairs = len(gold) * (len(gold) - 1) / 2

    a_gold = {i: len({n for sid in g for n in narrator_of.get(sid, [])})
              for i, g in enumerate(gold)}
    a_sys_aligned = {}
    for i, g in enumerate(gold):
        j = aligned.get(i)
        s = sysc[j] if j is not None and j < len(sysc) else set()
        a_sys_aligned[i] = len({n for sid in s for n in narrator_of.get(sid, [])})

    from scipy.stats import spearmanr

    ga = np.array([a_gold[i] for i in range(len(gold))], float)
    sa = np.array([a_sys_aligned[i] for i in range(len(gold))], float)
    rho, _ = spearmanr(ga, sa)
    gold_single = ga == 1
    sys_single = sa == 1
    false_singleton = float(np.mean(sys_single[~gold_single])) if (~gold_single).any() else 0.0
    false_rescue = float(np.mean(~sys_single[gold_single])) if gold_single.any() else 0.0

    strat = {}
    for lo, hi in SIZE_BINS:
        m = (ga >= lo) & (ga <= hi)
        if m.any():
            strat[f"{lo}-{hi if hi < 10**9 else '+'}"] = {
                "n": int(m.sum()),
                "false_split_rate": float(np.mean([split_flags[i] for i in np.where(m)[0]])),
                "mean_a_gold": float(ga[m].mean()),
                "mean_a_system": float(sa[m].mean()),
            }

    out = {
        "substitution_note": ("archive curation used as the gold clustering "
                              "(PREREGISTRATION substitution_ledger E3_3); ECB+/GVC "
                              "calibration (E3.2) was NOT run and no claim is made about "
                              "linguistic coreference quality"),
        "E3_3_threshold_sweep": sweep,
        "E3_3_selected": best,
        "E3_3_optimum_at_sweep_boundary": at_boundary,
        "E3_4_cluster_metrics": {
            k: {"precision": v[0], "recall": v[1], "f1": v[2]} for k, v in metrics.items()
        } | {"CoNLL_F1": conll, "primary_metric": "LEA"},
        "E3_5_error_decomposition": {
            "rho_merge": float(merged_pairs / comparable_pairs) if comparable_pairs else 0.0,
            "rho_split": float(np.mean(split_flags)) if split_flags else 0.0,
            "spearman_a_gold_vs_a_system": float(rho),
            "false_singleton_rate": false_singleton,
            "false_rescue_rate": false_rescue,
            "stratified_by_gold_size": strat,
            "kill_criterion": {
                "threshold": 0.30,
                "observed_false_singleton_rate": false_singleton,
                "passes": bool(false_singleton <= 0.30),
                "consequence_if_failed": ("triage is a screening step with a stated precision, "
                                          "not an item-level output"),
            },
        },
        "E3_5_scatter": {"a_gold": ga.tolist(), "a_system": sa.tolist()},
    }
    (RES / "e3_extraction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"E3.4 - LEA F1={metrics['LEA'][2]:.3f} CoNLL F1={conll:.3f}")
    log(f"E3.5 - rho_merge={out['E3_5_error_decomposition']['rho_merge']:.4f} "
        f"rho_split={out['E3_5_error_decomposition']['rho_split']:.4f} "
        f"false_singleton={false_singleton:.3f} false_rescue={false_rescue:.3f}")
    log("PHASE 3 complete")


if __name__ == "__main__":
    main()
