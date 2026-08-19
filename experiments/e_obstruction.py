"""Ownership obstruction: is the cell collection a set family over the narrator ground set?

A combinatorial complex is a triple (S, X, rk) with X a set of *subsets* of S, so two cells
with the same support are the same cell. This script tests that condition directly on the
built object by counting how many rank-k cells share a support with another rank-k cell.

Where the support map is not injective, no combinatorial complex over S carries the cells,
and any permutation-invariant lifting from rank 0 gives the colliding cells identical input.
The collapse factor reported here is the ratio of cells to distinct supports.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_phase5_7 import load_rank_maps  # noqa: E402
from src.complexes import GRANULARITIES, build_complex, support_injectivity  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("obstruction")


def audit(cx) -> dict:
    """Per-rank support-injectivity audit of one complex."""
    out = {"ground_set_size": len(cx.narrators)}
    for k in (1, 2, 3):
        stats = support_injectivity(cx, k)
        if stats["n_cells"]:
            out[f"rank{k}"] = stats
    return out


def main() -> None:
    segments = load_corpus()[1]
    rank_maps = load_rank_maps()

    out = {}
    for rm in sorted(rank_maps):
        for g in GRANULARITIES:
            cx = build_complex(segments, rank_maps[rm], granularity=g, rank_map_name=rm)
            out[f"{rm}|{g}"] = audit(cx)

    primary = out["R-A|mid"]
    r1 = primary["rank1"]
    # rank-1 cells are the archive's segments, so the obstruction should not move with design choices
    r1_invariant = len({(v["rank1"]["n_cells"], v["rank1"]["n_distinct_supports"])
                        for v in out.values()}) == 1
    n_design_cells = len(out)
    out["primary"] = {
        "cell": "R-A|mid",
        "ground_set_size": primary["ground_set_size"],
        "rank1_cells": r1["n_cells"],
        "rank1_distinct_supports": r1["n_distinct_supports"],
        "rank1_collapse_factor": r1["collapse_factor"],
        "rank1_supp_injective": r1["supp_injective"],
        "rank1_max_multiplicity": r1["max_multiplicity"],
        "rank1_singleton_support_cells": r1["n_singleton_support_cells"],
        "rank1_distinct_singleton_supports": r1["n_distinct_singleton_supports"],
        "rank1_multi_narrator_supports": (r1["n_distinct_supports"]
                                          - r1["n_distinct_singleton_supports"]),
        "rank2_supp_injective": primary["rank2"]["supp_injective"],
        "rank3_supp_injective": primary["rank3"]["supp_injective"],
        "rank1_invariant_across_all_cells": r1_invariant,
        "n_design_cells_checked": n_design_cells,
    }

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e_obstruction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    p = out["primary"]
    log(f"ground set |S| = {p['ground_set_size']}")
    log(f"rank-1 cells  = {p['rank1_cells']}")
    log(f"distinct supports = {p['rank1_distinct_supports']}  "
        f"(injective: {p['rank1_supp_injective']})")
    log(f"collapse factor = {p['rank1_collapse_factor']:.1f}x, "
        f"max multiplicity = {p['rank1_max_multiplicity']}")
    log(f"singleton-support cells = {p['rank1_singleton_support_cells']} over "
        f"{p['rank1_distinct_singleton_supports']} distinct singleton supports")
    log(f"invariant across all {p['n_design_cells_checked']} design cells: "
        f"{p['rank1_invariant_across_all_cells']}")


if __name__ == "__main__":
    main()
