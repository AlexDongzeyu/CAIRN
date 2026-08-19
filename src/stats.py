"""E7.3 / E7.5 — inference.

Almost Stochastic Order rather than t-tests, Bonferroni across model pairs, and an
explicit interaction model. The paper's central claim is a coefficient, not a win.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def aso_matrix(scores: dict[str, list[float]], confidence: float = 0.95,
               bonferroni: bool = True) -> dict:
    """Pairwise Almost Stochastic Order.

    eps_min == 0   -> A stochastically dominates B
    eps_min <  0.5 -> almost stochastic dominance, report as A >= B
    eps_min == 0.5 -> no order
    eps_min == 1   -> B dominates A
    """
    names = sorted(scores)
    n_pairs = len(names) * (len(names) - 1)
    alpha = (1 - confidence)
    adj = alpha / max(1, n_pairs) if bonferroni else alpha
    conf_adj = 1 - adj

    try:
        from deepsig import aso as _aso
        backend = "deepsig"
    except Exception:  # noqa: BLE001
        _aso, backend = None, "fallback"

    out: dict[str, float] = {}
    for a, b in itertools.permutations(names, 2):
        xa, xb = np.asarray(scores[a], float), np.asarray(scores[b], float)
        xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
        if len(xa) < 2 or len(xb) < 2:
            out[f"{a}>{b}"] = float("nan")
            continue
        if _aso is not None:
            try:
                out[f"{a}>{b}"] = float(_aso(xa, xb, confidence_level=conf_adj, show_progress_bar=False))
                continue
            except Exception:  # noqa: BLE001
                pass
        out[f"{a}>{b}"] = float(_violation_ratio(xa, xb))
    return {"eps_min": out, "backend": backend, "bonferroni_factor": n_pairs,
            "confidence_used": conf_adj}


def _violation_ratio(a: np.ndarray, b: np.ndarray, n_q: int = 1000) -> float:
    """Deterministic violation ratio of stochastic order, used if deepsig is unavailable."""
    qs = np.linspace(0.0, 1.0, n_q)
    fa, fb = np.quantile(a, qs), np.quantile(b, qs)
    diff = fb - fa
    viol = np.trapezoid(np.clip(diff, 0, None) ** 2, qs)
    total = np.trapezoid(diff**2, qs)
    return float(viol / total) if total > 0 else 0.5


def interaction_model(rows: list[dict], model_a: str, model_b: str) -> dict:
    """E7.5 — the coefficient the paper actually claims.

    The hypothesis is not "CCNN > typed star" but "the CCNN advantage grows with event
    size", which only an interaction term can express. Narrator and event are crossed
    repeated-measures sources of dependence, so both enter as random effects; seeds are
    averaged within before fitting so 10 seeds x N items are not treated as 10N
    independent observations.
    """
    import statsmodels.formula.api as smf

    df = pd.DataFrame(rows)
    df = df[df["model"].isin([model_a, model_b])].copy()
    if df.empty:
        return {"error": "no rows"}
    df = (df.groupby(["model", "narrator_id", "event_id", "event_size", "rank"], as_index=False)
            .agg(correct=("correct", "mean")))
    df["log_event_size"] = np.log1p(df["event_size"].astype(float))
    df["model_bin"] = (df["model"] == model_a).astype(float)
    df["rank"] = df["rank"].astype("category")

    res: dict = {"model_a": model_a, "model_b": model_b, "n_obs": int(len(df)),
                 "n_narrators": int(df["narrator_id"].nunique()),
                 "n_events": int(df["event_id"].nunique())}
    try:
        m = smf.mixedlm(
            "correct ~ model_bin * log_event_size",
            data=df, groups=df["narrator_id"],
            vc_formula={"event": "0 + C(event_id)"}, re_formula="~1",
        ).fit(reml=False, method="lbfgs")
        key = "model_bin:log_event_size"
        res.update({
            "converged": bool(m.converged),
            "interaction_coef": float(m.params.get(key, np.nan)),
            "interaction_se": float(m.bse.get(key, np.nan)),
            "interaction_p": float(m.pvalues.get(key, np.nan)),
            "interaction_ci": [float(m.conf_int().loc[key, 0]), float(m.conf_int().loc[key, 1])]
            if key in m.params.index else [np.nan, np.nan],
            "main_effect_model": float(m.params.get("model_bin", np.nan)),
        })
    except Exception as e:  # noqa: BLE001 - fall back to OLS with clustered SEs
        try:
            m = smf.ols("correct ~ model_bin * log_event_size", data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["narrator_id"]})
            key = "model_bin:log_event_size"
            res.update({
                "converged": False, "fallback": f"OLS+clustered SE ({type(e).__name__})",
                "interaction_coef": float(m.params.get(key, np.nan)),
                "interaction_se": float(m.bse.get(key, np.nan)),
                "interaction_p": float(m.pvalues.get(key, np.nan)),
                "interaction_ci": [float(m.conf_int().loc[key, 0]), float(m.conf_int().loc[key, 1])],
                "main_effect_model": float(m.params.get("model_bin", np.nan)),
            })
        except Exception as e2:  # noqa: BLE001
            res["error"] = f"{type(e).__name__}/{type(e2).__name__}"
    if "interaction_ci" in res and all(np.isfinite(res["interaction_ci"])):
        lo, hi = res["interaction_ci"]
        res["ci_crosses_zero"] = bool(lo <= 0 <= hi)
        res["direction"] = "positive" if lo > 0 else "negative" if hi < 0 else "indeterminate"
    return res


def stratify(rows: list[dict], by: str, bins=None) -> dict[str, list[dict]]:
    """E7.2 — every result is broken down; aggregates alone cannot support the claim."""
    out: dict[str, list[dict]] = {}
    if by == "event_size":
        bins = bins or [(2, 3), (4, 10), (11, 50), (51, 10**9)]
        for r in rows:
            v = r.get("event_size", 0)
            for lo, hi in bins:
                if lo <= v <= hi:
                    out.setdefault(f"{lo}-{hi if hi < 10**9 else '+'}", []).append(r)
                    break
    else:
        for r in rows:
            out.setdefault(str(r.get(by)), []).append(r)
    return out
