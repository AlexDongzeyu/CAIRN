"""M10: is the exposure condition a property of the encoding, not of this archive?

The two diagnostics the paper advertises as needing no trained model are computed here over
every lifting the registered grid admits: three granularities crossed with three rank maps.
If the singleton fraction and the predicted straddle rate move with the lifting rather than
with the data, then choosing a default lifting chooses an exposure level, which is the
claim the paper wants to make about pipelines it has not run.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.perturb import singleton_fraction  # noqa: E402

from experiments.run_phase5_7 import load_rank_maps

RES = ROOT / "data" / "results"
GRANS = ("fine", "mid", "coarse")
FOLDS = (0.70, 0.15, 0.15)

log = make_logger("e_lifting")


def main() -> None:
    _, segments = load_corpus()
    rank_maps = load_rank_maps()

    rows = []
    for rm_name, rm in rank_maps.items():
        for g in GRANS:
            cx = build_complex(segments, rm, granularity=g, rank_map_name=rm_name)
            # Cells per narrator drive the exposure formula; k = 1 cannot straddle.
            k = Counter()
            for c in cx.by_rank(2):
                for n in c.members:
                    k[n] += 1
            if not k:
                continue
            exposure = sum(1.0 - sum(p ** ke for p in FOLDS) for ke in k.values()) / len(k)
            rows.append({
                "rank_map": rm_name, "granularity": g,
                "n_r2_cells": len(cx.by_rank(2)), "n_narrators": len(k),
                "singleton_fraction": float(singleton_fraction(cx)),
                "mean_k": float(sum(k.values()) / len(k)),
                "frac_k_eq_1": float(sum(1 for v in k.values() if v == 1) / len(k)),
                "predicted_exposure": float(exposure),
            })
            log(f"  {rm_name:4s} {g:7s} cells={rows[-1]['n_r2_cells']:5d} "
                f"singleton={rows[-1]['singleton_fraction']:.3f} "
                f"k=1 frac={rows[-1]['frac_k_eq_1']:.3f} "
                f"exposure={exposure:.3f}")

    es = [r["predicted_exposure"] for r in rows]
    sf = [r["singleton_fraction"] for r in rows]
    out = {"folds": list(FOLDS), "rows": rows,
           "exposure_min": min(es), "exposure_max": max(es),
           "singleton_min": min(sf), "singleton_max": max(sf),
           "n_configs": len(rows)}
    log(f"  exposure ranges {min(es):.3f}-{max(es):.3f} over {len(rows)} liftings; "
        f"singleton fraction {min(sf):.3f}-{max(sf):.3f}")
    (RES / "e_lifting.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("wrote e_lifting.json")


if __name__ == "__main__":
    main()
