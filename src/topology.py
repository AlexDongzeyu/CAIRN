"""E11 — persistent homology, with the kill criterion built in.

The filtration is semantic, not geometric: X_k is the subcomplex induced by cells whose
archive-conditioned attestation multiplicity is at least k. The death time of a component
in this filtration is the attestation level at which that region of the archive loses
support, which is an interpretable archival quantity. Generic node-deletion beta_0 is not.

E11.4 is the actual kill test: if the topological summaries are predictable from the
event-size distribution alone (R^2 > 0.9), the topology is a re-description of the size
distribution and the section is deleted.
"""
from __future__ import annotations

import numpy as np


def cowitness_graph(cx, min_attestation: int = 1) -> tuple[list[str], np.ndarray]:
    """Narrator-narrator co-witness graph restricted to cells with a(x) >= k."""
    narrators = list(cx.narrators)
    idx = {n: i for i, n in enumerate(narrators)}
    n = len(narrators)
    W = np.zeros((n, n), dtype=np.float32)
    for c in cx.by_rank(2):
        if c.size < min_attestation:
            continue
        ms = [idx[m] for m in c.members if m in idx]
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                W[a, b] += 1.0
                W[b, a] += 1.0
    return narrators, W


def attestation_filtration(cx, k_max: int | None = None) -> dict:
    """Descending filtration X_1 superset X_2 superset ... by attestation level."""
    sizes = [c.size for c in cx.by_rank(2)]
    if not sizes:
        return {"levels": [], "betti0": [], "betti1": [], "n_cells": []}
    k_max = k_max or int(min(max(sizes), 25))
    levels, b0, b1, ncells = [], [], [], []
    for k in range(1, k_max + 1):
        _, W = cowitness_graph(cx, min_attestation=k)
        A = (W > 0).astype(np.int8)
        levels.append(k)
        ncells.append(int(sum(1 for c in cx.by_rank(2) if c.size >= k)))
        b0.append(_components(A))
        b1.append(_cycle_rank(A))
    return {"levels": levels, "betti0": b0, "betti1": b1, "n_cells": ncells}


def _components(A: np.ndarray) -> int:
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as sp

    deg = A.sum(axis=1)
    keep = deg > 0
    if keep.sum() == 0:
        return 0
    n, _ = connected_components(sp.csr_matrix(A[np.ix_(keep, keep)]), directed=False)
    return int(n)


def _cycle_rank(A: np.ndarray) -> int:
    """First Betti number of the graph: E - V + C."""
    deg = A.sum(axis=1)
    keep = deg > 0
    if keep.sum() == 0:
        return 0
    S = A[np.ix_(keep, keep)]
    V = int(keep.sum())
    E = int(S.sum() // 2)
    C = _components(A)
    return max(0, E - V + C)


def persistence_diagram(cx, k_max: int | None = None) -> dict:
    """Persistence of H0/H1 over the attestation filtration, via gudhi when available."""
    filt = attestation_filtration(cx, k_max)
    if not filt["levels"]:
        return {"filtration": filt, "total_persistence_h0": 0.0, "total_persistence_h1": 0.0}
    b0 = np.array(filt["betti0"], dtype=float)
    b1 = np.array(filt["betti1"], dtype=float)
    return {
        "filtration": filt,
        "n_levels": len(filt["levels"]),
        "betti0_is_constant": bool(len(set(filt["betti0"])) == 1),
        "betti1_is_constant": bool(len(set(filt["betti1"])) == 1),
        "total_persistence_h0": float(np.abs(np.diff(b0)).sum()),
        "total_persistence_h1": float(np.abs(np.diff(b1)).sum()),
        "max_betti0": float(b0.max()),
        "max_betti1": float(b1.max()),
        "betti0_auc": float(np.trapezoid(b0, filt["levels"])),
        "betti1_auc": float(np.trapezoid(b1, filt["levels"])),
    }


def permutation_test(cx, n_perm: int = 1000, seed: int = 0) -> dict:
    """E11.3 — shuffle attestation while preserving the degree sequence.

    Reassigning which cell holds which attestation count leaves the multiset of cell
    sizes untouched, so anything that survives is structure rather than size.

    Both Betti summaries are tested. beta_0 is constant at 1 on this complex -- every
    filtration level leaves it connected -- so permuting cannot move it and its p-value is
    1.0 by arithmetic rather than by evidence. Reporting that alone would dress a
    degenerate statistic as a result, so beta_1, which does vary, is tested alongside it
    and the degeneracy is flagged in the output.
    """
    from dataclasses import replace

    rng = np.random.default_rng(seed)
    observed = persistence_diagram(cx)
    k_max = max(observed["filtration"]["levels"])

    r2 = sorted(cx.by_rank(2), key=lambda c: c.cid)
    member_sets = [c.members for c in r2]
    null0, null1 = [], []
    import copy as _copy

    for _ in range(n_perm):
        perm = rng.permutation(len(member_sets))
        cells = {cid: c for cid, c in cx.cells.items() if c.rank != 2}
        for c, j in zip(r2, perm):
            cells[c.cid] = replace(c, members=member_sets[j])
        shuffled = _copy.copy(cx)
        shuffled.cells = cells
        d = persistence_diagram(shuffled, k_max=k_max)
        null0.append(d["betti0_auc"])
        null1.append(d["betti1_auc"])

    def _p(null, stat):
        a = np.array(null, dtype=float)
        return {"observed": float(stat), "null_mean": float(a.mean()),
                "null_std": float(a.std()),
                "p_value": float((np.abs(a - a.mean()) >= abs(stat - a.mean())).mean()),
                "degenerate": bool(a.std() == 0.0)}

    b0, b1 = _p(null0, observed["betti0_auc"]), _p(null1, observed["betti1_auc"])
    both_degenerate = bool(b0["degenerate"] and b1["degenerate"])
    return {
        # kept at the top level so existing readers of this file do not silently break
        "observed_betti0_auc": b0["observed"],
        "null_mean": b0["null_mean"],
        "null_std": b0["null_std"],
        "p_value": b0["p_value"],
        "n_perm": n_perm,
        "betti0": b0,
        "betti1": b1,
        "betti0_is_degenerate": b0["degenerate"],
        "null_is_degenerate_by_construction": both_degenerate,
        "note": (
            "This shuffle reassigns member sets across rank-2 cell IDs. Homology does not "
            "depend on how cells are labelled, so the permuted complex is isomorphic to the "
            "observed one and every draw reproduces the observed statistic exactly. Both "
            "nulls have zero variance and p = 1.0 is forced by the construction, not "
            "measured. The test carries no information and the deletion verdict rests on "
            "the simpler-explanation check instead."
            if both_degenerate else
            "beta_0 is constant across the filtration, so its permutation p-value is "
            "arithmetic; beta_1 is the informative statistic."),
    }


def simpler_explanation_check(cx, n_boot: int = 60, seed: int = 0) -> dict:
    """E11.4 — the actual kill test.

    Predict each topological summary from event-size distribution moments and basic
    connectivity statistics only. R^2 > 0.9 means the topology is a re-description of
    the size distribution and adds nothing, so the section goes.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score

    rng = np.random.default_rng(seed)
    narrators = list(cx.narrators)
    X, Y, Y1 = [], [], []
    for _ in range(n_boot):
        keep = set(rng.choice(narrators, size=max(4, int(0.8 * len(narrators))), replace=False))
        sub = _restrict(cx, keep)
        sizes = np.array([c.size for c in sub.by_rank(2)], dtype=float)
        if len(sizes) < 3:
            continue
        _, W = cowitness_graph(sub)
        A = (W > 0).astype(np.int8)
        X.append([
            sizes.mean(), sizes.std(), np.median(sizes), sizes.max(), len(sizes),
            float((sizes == 1).mean()), float(A.sum() / 2), float((A.sum(axis=1) > 0).sum()),
        ])
        d = persistence_diagram(sub)
        Y.append(d["betti0_auc"])
        Y1.append(d["betti1_auc"])
    if len(X) < 8:
        return {"r2": float("nan"), "n": len(X), "verdict": "insufficient data"}

    Xa, Ya = np.array(X), np.array(Y)

    def _r2(target):
        t = np.array(target, dtype=float)
        if t.std() == 0.0:
            # Predicting a constant is not a test. R^2 is undefined against zero variance
            # and sklearn will happily return 1.0, which reads as a strong result.
            return float("nan"), True
        return float(np.mean(cross_val_score(LinearRegression(), Xa, t,
                                             cv=KFold(5, shuffle=True, random_state=0),
                                             scoring="r2"))), False

    r2, r2_degenerate = _r2(Ya)
    r2_b1, r2_b1_degenerate = _r2(Y1)
    # The kill decision rests on the statistic that actually varies.
    decisive = r2_b1 if not r2_b1_degenerate else r2
    redundant = bool(decisive == decisive and decisive > 0.9)
    return {
        "r2_size_and_connectivity_only": r2,
        "betti0_is_constant": r2_degenerate,
        "r2_betti1_size_and_connectivity_only": r2_b1,
        "betti1_is_constant": r2_b1_degenerate,
        "decisive_r2": decisive,
        "n_resamples": len(X),
        "threshold": 0.9,
        "topology_is_redundant": redundant,
        "verdict": ("DELETE the topology section - it is a re-description of the event-size "
                    "distribution" if redundant else
                    "topology carries signal beyond the size distribution; keep only if it "
                    "is also perturbation-stable and changes an archival decision"),
    }


def _restrict(cx, keep: set[str]):
    from dataclasses import replace
    import copy as _copy

    cells = {}
    for cid, c in cx.cells.items():
        m = c.members & keep
        if c.rank == 0 and not m:
            continue
        if c.rank > 0 and not m:
            continue
        cells[cid] = replace(c, members=frozenset(m))
    out = _copy.copy(cx)
    out.cells = cells
    out.narrators = [n for n in cx.narrators if n in keep]
    return out
