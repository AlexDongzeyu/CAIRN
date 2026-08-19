"""Recompute E7.3 (ASO) from the recorded runs with the gold-subset rows excluded.

The primary cell is identified by four keys, and the E8.3 gold-subset runs share all four
while being a different corpus. Pooling them gave four models n=20 drawn from two
populations and every other model n=10, so the dominance test was comparing a bimodal
sample against a unimodal one. This recomputes the test on the primary corpus alone; the
grid itself does not need re-running because every score is already recorded.

M1_dense is deterministic, so its "distribution" is one value repeated. Stochastic dominance
between a point mass and a spread is not meaningful, and it is excluded from the matrix and
reported descriptively instead.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402
from src.stats import aso_matrix  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("aso_recompute")

PRIMARY = {"granularity": "mid", "split": "narrator-disjoint",
           "neg_regime": "MNS", "rank_map": "R-A"}
DETERMINISTIC = {"M1_dense"}


def main() -> None:
    runs = json.loads((RES / "e7_1_runs.json").read_text(encoding="utf-8"))
    prim = [r for r in runs
            if all(r.get(k) == v for k, v in PRIMARY.items())
            and r.get("cell") != "E8.3_gold_subset"]
    log(f"primary-cell rows after excluding the gold subset: {len(prim)}")

    for metric in ("T1_map", "T2_auc"):
        scores: dict[str, list[float]] = {}
        for r in prim:
            if r["model"] in DETERMINISTIC:
                continue
            scores.setdefault(r["model"], []).append(r[metric])
        sizes = {k: len(v) for k, v in scores.items()}
        log(f"{metric}: sample sizes {sorted(set(sizes.values()))} across {len(scores)} models")
        if len(set(sizes.values())) > 1:
            log(f"  WARNING unequal sample sizes remain: {sizes}")

        aso = aso_matrix(scores)
        out = {"scores_summary": {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)),
                                      "n": len(v), "per_seed": v} for k, v in scores.items()},
               "excluded_models": sorted(DETERMINISTIC),
               "exclusion_reason": ("deterministic: a point mass has no stochastic order "
                                    "relation worth testing"),
               "gold_subset_excluded": True,
               **aso}
        (RES / f"e7_3_aso_{metric}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        e = aso["eps_min"]
        log(f"  {metric}: eps_min(M3>M5)={e.get('M3_typed_star>M5_ccnn')} "
            f"eps_min(M5>M3)={e.get('M5_ccnn>M3_typed_star')} "
            f"bonferroni={aso.get('bonferroni_factor')}")


if __name__ == "__main__":
    main()
