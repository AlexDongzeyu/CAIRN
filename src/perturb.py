"""E8 — perturbation and error propagation.

This is what separates "we acknowledged extraction error" from "we quantified it".
Merges are drawn in proportion to embedding similarity because realistic extraction
errors unify *similar* cells; uniform merging understates the damage.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from src.complexes import Cell, Complex


def perturb(cx: Complex, rho_merge: float, rho_split: float, rng: np.random.Generator,
            cell_emb: np.ndarray | None = None) -> Complex:
    """Apply calibrated false-merge and false-split noise to the rank-2 layer."""
    r2 = sorted(cx.by_rank(2), key=lambda c: c.cid)
    if not r2:
        return cx
    idx = {c.cid: i for i, c in enumerate(r2)}
    alive = {c.cid: c for c in r2}

    # ---- merges: pick partners in proportion to similarity, not uniformly
    n_merge = int(round(rho_merge * len(r2)))
    if n_merge > 0 and len(r2) > 1:
        if cell_emb is not None and cell_emb.shape[0] == len(r2):
            sim = cell_emb @ cell_emb.T
            np.fill_diagonal(sim, -np.inf)
        else:
            sim = None
        for _ in range(n_merge):
            ids = [c for c in alive if c in idx]
            if len(ids) < 2:
                break
            a = str(rng.choice(ids))
            if sim is not None:
                row = sim[idx[a]].copy()
                mask = np.array([cid in alive for cid in [c.cid for c in r2]])
                row[~mask] = -np.inf
                row[idx[a]] = -np.inf
                if not np.isfinite(row).any():
                    break
                w = np.exp(row - np.nanmax(row[np.isfinite(row)]))
                w[~np.isfinite(row)] = 0.0
                if w.sum() <= 0:
                    break
                b = r2[int(rng.choice(len(r2), p=w / w.sum()))].cid
            else:
                b = str(rng.choice([c for c in ids if c != a]))
            if a == b or a not in alive or b not in alive:
                continue
            ca, cb = alive.pop(a), alive.pop(b)
            merged = Cell(cid=f"{ca.cid}+M", rank=2, members=ca.members | cb.members,
                          label=f"{ca.label}|{cb.label}", source="topic",
                          segments=ca.segments | cb.segments, base_term=ca.base_term)
            alive[merged.cid] = merged

    # ---- splits: partition members by a random hyperplane in narrator space
    n_split = int(round(rho_split * len(alive)))
    if n_split > 0:
        for cid in list(rng.permutation(sorted(alive)))[:n_split]:
            c = alive.get(str(cid))
            if c is None or c.size < 2:
                continue
            ms = sorted(c.members)
            k = max(1, len(ms) // 2)
            perm = rng.permutation(len(ms))
            left = {ms[i] for i in perm[:k]}
            right = set(ms) - left
            if not left or not right:
                continue
            alive.pop(c.cid)
            for tag, part in (("a", left), ("b", right)):
                alive[f"{c.cid}+S{tag}"] = replace(
                    c, cid=f"{c.cid}+S{tag}", members=frozenset(part),
                    label=f"{c.label}#{tag}")

    cells = {cid: c for cid, c in cx.cells.items() if c.rank != 2}
    cells.update(alive)
    out = copy.copy(cx)
    out.cells = cells
    return out


def triage(cx: Complex, k: int = 50, project=None) -> list[str]:
    cells = sorted(cx.by_rank(2), key=lambda c: (c.size, c.label))[:k]
    if project is None:
        return [c.label for c in cells]
    seen, out = set(), []
    for c in cells:
        r = project(c)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def degree_ks(cx_a: Complex, cx_b: Complex) -> float:
    from scipy.stats import ks_2samp

    # `np.array(xs) or fallback` raises on any array with more than one element, so the
    # empty case is handled on the list before it becomes an array.
    sizes_a = [c.size for c in cx_a.by_rank(2)]
    sizes_b = [c.size for c in cx_b.by_rank(2)]
    if len(sizes_a) < 2 or len(sizes_b) < 2:
        return float("nan")
    return float(ks_2samp(np.array(sizes_a), np.array(sizes_b)).statistic)


def singleton_fraction(cx: Complex) -> float:
    sizes = [c.size for c in cx.by_rank(2)]
    return float(np.mean([s == 1 for s in sizes])) if sizes else 0.0


def estimate_error_rates(cx: Complex, seed: int = 0) -> dict:
    """E3.5 stand-in calibrated on the archive's own labelling.

    A false merge is a pair of distinct archive terms whose narrator sets are so similar
    that a clustering step could plausibly unify them; a false split is one archive term
    whose narrator set is separable into groups sharing no other term. Both are measured
    against the archive's curation rather than assumed.
    """
    r2 = sorted(cx.by_rank(2), key=lambda c: c.cid)
    if len(r2) < 2:
        return {"rho_merge": 0.0, "rho_split": 0.0, "n_cells": len(r2)}
    sets = [c.members for c in r2]
    n = len(sets)
    merges, pairs = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            if not a or not b:
                continue
            pairs += 1
            jac = len(a & b) / len(a | b)
            if jac >= 0.8:  # near-identical attestation profiles
                merges += 1
    splits = sum(1 for c in r2 if c.size >= 4 and len(c.segments) >= 2
                 and len(c.segments) > 2 * c.size)
    return {
        "rho_merge": float(merges / pairs) if pairs else 0.0,
        "rho_split": float(splits / n) if n else 0.0,
        "n_cells": n,
        "n_candidate_pairs": pairs,
        "definition": "rho_merge = fraction of distinct archive terms with Jaccard>=0.8 "
                      "narrator overlap; rho_split = fraction of terms whose segment "
                      "support is more than twice their narrator count",
    }
