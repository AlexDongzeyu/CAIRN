"""E3 — can a size-only predictor recover rank?

P1 was registered as a Spearman threshold, rho(rank, size) > 0.9 meaning the ladder is
cardinality in disguise. The threshold is a proxy for the claim rather than the claim: a
monotone correlation of 0.471 leaves open whether a nonlinear function of size recovers rank
anyway, and a reviewer is entitled to ask. This adds the direct test and keeps the
registered criterion reported as registered.

Predictions are cross-validated, because a one-dimensional predictor fitted and scored on
the same cells would answer a different question. Balanced accuracy is the headline: the
rank distribution is skewed, so plain accuracy rewards a constant prediction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
GRANULARITIES = ["coarse", "mid", "fine"]
RANKS = (1, 2, 3)
log = make_logger("size_only")


def _fit_eval(sizes: np.ndarray, ranks: np.ndarray, seed: int = 0) -> dict:
    """Cross-validated rank-from-size recovery, against a prior-only baseline."""
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    x = np.log1p(sizes).reshape(-1, 1)
    y = ranks.astype(int)
    classes = np.unique(y)
    if len(classes) < 2 or min(np.bincount(y)[classes]) < 5:
        return {"n": int(len(y)), "skipped": "too few cells in a class"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out: dict = {"n": int(len(y)), "class_counts": {int(c): int((y == c).sum()) for c in classes}}
    for name, est in (("logistic", LogisticRegression(max_iter=2000)),
                      ("tree_depth3", DecisionTreeClassifier(max_depth=3, random_state=seed)),
                      ("prior_only", DummyClassifier(strategy="prior"))):
        pred = cross_val_predict(est, x, y, cv=cv)
        proba = cross_val_predict(est, x, y, cv=cv, method="predict_proba")
        out[name] = {
            "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
            "cross_entropy": round(float(log_loss(y, proba, labels=list(classes))), 4),
        }
    base = out["prior_only"]
    for name in ("logistic", "tree_depth3"):
        out[name]["balanced_accuracy_gain"] = round(
            out[name]["balanced_accuracy"] - base["balanced_accuracy"], 4)
        out[name]["cross_entropy_reduction"] = round(
            base["cross_entropy"] - out[name]["cross_entropy"], 4)
    return out


def _mutual_information(sizes: np.ndarray, ranks: np.ndarray, seed: int = 0) -> float:
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(np.log1p(sizes).reshape(-1, 1), ranks.astype(int),
                             discrete_features=False, random_state=seed)
    return round(float(mi[0]), 4)


def main() -> None:
    _, segments = load_corpus()
    rm_all = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    rank_maps = {"R-A": rm_all["R-A_consensus"], "R-B": rm_all["R-B_archive_native"],
                 "R-C": rm_all["R-C_adversarial"]}

    out: dict = {}
    for rm_name, rm in rank_maps.items():
        for g in GRANULARITIES:
            key = f"{rm_name}|{g}"
            cx = build_complex(segments, rm, granularity=g, rank_map_name=rm_name)
            cells = [c for c in cx.cells.values() if c.rank in RANKS]
            sizes = np.array([c.size for c in cells], dtype=float)
            ranks = np.array([c.rank for c in cells], dtype=float)

            m23 = np.isin(ranks, (2, 3))
            out[key] = {
                "all_ranks": _fit_eval(sizes, ranks),
                "rank2_vs_rank3": _fit_eval(sizes[m23], ranks[m23]),
                "mutual_information_nats": _mutual_information(sizes, ranks),
                "mutual_information_nats_rank2_vs_rank3": _mutual_information(
                    sizes[m23], ranks[m23]) if m23.sum() else float("nan"),
            }
            b = out[key]["rank2_vs_rank3"]
            if "logistic" in b:
                log(f"{key}: 2v3 balanced acc logistic={b['logistic']['balanced_accuracy']:.3f} "
                    f"tree={b['tree_depth3']['balanced_accuracy']:.3f} "
                    f"prior={b['prior_only']['balanced_accuracy']:.3f}")

    primary = out["R-A|mid"]
    out["primary"] = primary
    worst = max((v["rank2_vs_rank3"].get("tree_depth3", {}).get("balanced_accuracy", 0),
                 k) for k, v in out.items() if k != "primary" and "rank2_vs_rank3" in v)
    out["max_balanced_accuracy_rank2_vs_rank3"] = {"value": worst[0], "cell": worst[1]}
    (RES / "e_size_only.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote e_size_only.json  (max 2v3 balanced accuracy {worst[0]:.3f} at {worst[1]})")


if __name__ == "__main__":
    main()
