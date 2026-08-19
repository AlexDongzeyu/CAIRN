"""Shared measurement utilities: RBO, clustered bootstrap, design effect.

Kept separate from any experiment so the same estimator is used everywhere
(the protocol's uncertainty claims depend on that).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def rbo(list1: Sequence[Any], list2: Sequence[Any], p: float = 0.9, k: int | None = None) -> float:
    """Rank-biased overlap (Webber et al. 2010), extrapolated variant.

    Correct for non-conjoint, top-weighted, ragged lists — which is what triage
    lists across granularities are. p=0.9 puts ~86% of weight in the top 10.
    """
    l1, l2 = list(list1), list(list2)
    if not l1 and not l2:
        return 1.0
    if not l1 or not l2:
        return 0.0
    k = k or max(len(l1), len(l2))
    s, ll = (l1, l2) if len(l1) <= len(l2) else (l2, l1)
    ls, ll_len = len(s), len(ll)

    seen_s: set = set()
    seen_l: set = set()
    overlap = np.zeros(k + 1)
    agreement = np.zeros(k + 1)
    for d in range(1, k + 1):
        if d <= ls:
            seen_s.add(s[d - 1])
        if d <= ll_len:
            seen_l.add(ll[d - 1])
        overlap[d] = len(seen_s & seen_l)
        agreement[d] = overlap[d] / min(d, max(ls, 1)) if d <= ls else overlap[d] / d

    x_k = overlap[k]
    summ = sum((1 - p) * (p ** (d - 1)) * (overlap[d] / d) for d in range(1, k + 1))
    return float(summ + (p**k) * (x_k / k))


def jaccard_at(list1: Sequence[Any], list2: Sequence[Any], k: int) -> float:
    a, b = set(list1[:k]), set(list2[:k])
    return len(a & b) / len(a | b) if (a | b) else 1.0


def cluster_bootstrap(
    stat_fn: Callable[[list[Any]], float],
    units: Sequence[Any],
    B: int = 10000,
    seed: int = 0,
    ci: tuple[float, float] = (2.5, 97.5),
    bca: bool = True,
) -> dict[str, float]:
    """Resample independent UNITS (narrators), not incidences.

    Treating correlated incidences as independent understates uncertainty; this is
    the estimator the protocol requires for every archive-level statistic.
    """
    rng = np.random.default_rng(seed)
    units = list(units)
    n = len(units)
    point = float(stat_fn(units))
    draws = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn([units[i] for i in idx])

    lo_p, hi_p = ci
    if not bca:
        return {"point": point, "lo": float(np.percentile(draws, lo_p)), "hi": float(np.percentile(draws, hi_p))}

    # BCa: bias correction + acceleration from jackknife over units.
    from scipy.stats import norm

    prop = float((draws < point).mean())
    prop = min(max(prop, 1.0 / (2 * B)), 1 - 1.0 / (2 * B))
    z0 = norm.ppf(prop)
    jack = np.array([stat_fn([u for j, u in enumerate(units) if j != i]) for i in range(n)]) if n <= 2000 else None
    if jack is None:
        a = 0.0
    else:
        jbar = jack.mean()
        num = ((jbar - jack) ** 3).sum()
        den = 6.0 * (((jbar - jack) ** 2).sum() ** 1.5)
        a = float(num / den) if den > 0 else 0.0

    def adj(pct: float) -> float:
        z = norm.ppf(pct / 100.0)
        zz = z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z)))
        return float(np.clip(norm.cdf(zz) * 100.0, 0.0, 100.0))

    return {
        "point": point,
        "lo": float(np.percentile(draws, adj(lo_p))),
        "hi": float(np.percentile(draws, adj(hi_p))),
        "z0": float(z0),
        "a": float(a),
    }


def cluster_subsample_ci(
    stat_fn: Callable[[list[Any]], float],
    units: Sequence[Any],
    B: int = 2000,
    seed: int = 0,
    m_frac: float = 0.632,
    z: float = 1.96,
) -> dict[str, float]:
    """Subsampling CI for statistics that are not smooth under duplicated units.

    The singleton fraction is the motivating case. Resampling narrators *with*
    replacement puts two identical copies of a narrator into the sample, and a cell
    attested only by that narrator then has attestation 2 and stops being a singleton.
    The bootstrap distribution is therefore biased downward and its interval can exclude
    the point estimate entirely.

    Drawing m < n units *without* replacement keeps every unit distinct. Under the usual
    sqrt-n scaling, sd(theta_n) is approximated by sd(theta_m) * sqrt(m/n), so the
    interval is centred on the observed statistic and widened accordingly.
    """
    rng = np.random.default_rng(seed)
    units = list(units)
    n = len(units)
    point = float(stat_fn(units))
    m = max(2, int(round(m_frac * n)))
    if n < 4:
        return {"point": point, "lo": point, "hi": point, "m": m, "n": n}

    draws = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=m, replace=False)
        draws[b] = stat_fn([units[i] for i in idx])
    sd_n = float(draws.std(ddof=1) * np.sqrt(m / n))
    return {
        "point": point,
        "lo": point - z * sd_n,
        "hi": point + z * sd_n,
        "se": sd_n,
        "m": m,
        "n": n,
        "method": "cluster subsampling (m-out-of-n without replacement, sqrt(m/n) scaled)",
    }


def design_effect(cluster_sizes: Sequence[int], values_by_cluster: Sequence[Sequence[float]]) -> dict[str, float]:
    """DEFF = 1 + (mbar - 1) * ICC, with ICC from a one-way random-effects ANOVA.

    Tells a reviewer how badly a naive edge-level CI would understate uncertainty.
    """
    groups = [np.asarray(v, dtype=float) for v in values_by_cluster if len(v) > 0]
    k = len(groups)
    if k < 2:
        return {"DEFF": 1.0, "ICC": 0.0, "mbar": float(np.mean(cluster_sizes) if len(cluster_sizes) else 0)}
    n_i = np.array([len(g) for g in groups], dtype=float)
    N = n_i.sum()
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw_den = N - k
    msw = (sum(((g - g.mean()) ** 2).sum() for g in groups) / msw_den) if msw_den > 0 else 0.0
    n0 = (N - (n_i**2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw) if (msb + (n0 - 1) * msw) > 0 else 0.0
    icc = float(np.clip(icc, 0.0, 1.0))
    mbar = float(n_i.mean())
    return {"DEFF": float(1 + (mbar - 1) * icc), "ICC": icc, "mbar": mbar, "n_clusters": k, "N": float(N)}
