"""Recompute E11 only, testing the Betti summary that actually varies.

The original E11 tested beta_0. On this complex beta_0 is constant at 1 across every
filtration level, so the permutation null has zero variance and p = 1.0 is arithmetic
rather than evidence; the size regression likewise reproduces a constant trivially and
returns R^2 = 1.0. Both numbers were quoted in the paper as though they were findings.

beta_1 does vary, so this rebuilds the same complex under the primary cell and reruns the
permutation test and the simpler-explanation check for both summaries, recording which one
the deletion verdict actually rests on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.complexes import build_complex  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.topology import (  # noqa: E402
    permutation_test, persistence_diagram, simpler_explanation_check,
)

RES = ROOT / "data" / "results"
log = make_logger("e11_recompute")


def main() -> None:
    segments = load_corpus()[1]
    rank_maps = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    cx = build_complex(segments, rank_maps["R-A_consensus"], granularity="mid",
                       rank_map_name="R-A")
    log(f"complex rebuilt: rank-2 cells={len(cx.by_rank(2))} narrators={len(cx.narrators)}")

    diag = persistence_diagram(cx)
    b0 = diag["filtration"]["betti0"]
    b1 = diag["filtration"]["betti1"]
    log(f"beta_0 distinct values={sorted(set(b0))}  beta_1 range=({min(b1)}, {max(b1)})")

    perm = permutation_test(cx, n_perm=200, seed=0)
    log(f"perm beta_0: p={perm['betti0']['p_value']:.3f} null_std={perm['betti0']['null_std']:.4f} "
        f"degenerate={perm['betti0']['degenerate']}")
    log(f"perm beta_1: p={perm['betti1']['p_value']:.3f} null_std={perm['betti1']['null_std']:.4f} "
        f"degenerate={perm['betti1']['degenerate']}")

    simpler = simpler_explanation_check(cx, n_boot=60, seed=0)
    log(f"R2 beta_0={simpler['r2_size_and_connectivity_only']} "
        f"(constant={simpler['betti0_is_constant']})")
    log(f"R2 beta_1={simpler['r2_betti1_size_and_connectivity_only']} "
        f"(constant={simpler['betti1_is_constant']})")

    if perm.get("null_is_degenerate_by_construction"):
        decisive_p, decisive_stat = float("nan"), "none - permutation null is degenerate"
    elif perm["betti0_is_degenerate"] and not perm["betti1"]["degenerate"]:
        decisive_p, decisive_stat = perm["betti1"]["p_value"], "betti1_auc"
    else:
        decisive_p, decisive_stat = perm["p_value"], "betti0_auc"

    prev = json.loads((RES / "e11_topology.json").read_text(encoding="utf-8"))
    stability = prev.get("perturbation_stability", [])
    # With a degenerate permutation null the only informative test left is the registered
    # "simpler explanation" check, which the registration itself calls the actual kill test.
    keep = not simpler.get("topology_is_redundant", True)

    (RES / "e11_topology.json").write_text(json.dumps({
        "persistence": diag,
        "permutation_test": perm,
        "simpler_explanation_check": simpler,
        "perturbation_stability": stability,
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
        "recompute_note": ("beta_0 is constant across the filtration, and the E11.3 shuffle "
                           "reassigns member sets across cell IDs, which is an isomorphism -- "
                           "so BOTH permutation nulls have zero variance and p = 1.0 is forced "
                           "by the construction rather than measured. The verdict rests on the "
                           "registered simpler-explanation check, computed on beta_1, which "
                           "does vary and is cross-validated."),
    }, indent=2), encoding="utf-8")
    log(f"E11 recomputed: decisive={decisive_stat} p={decisive_p} keep={keep}")


if __name__ == "__main__":
    main()
