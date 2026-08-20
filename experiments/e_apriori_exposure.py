"""A priori exposure: predict the straddle rate from the incidence structure alone.

Under a uniform split that assigns each of an entity's k cells independently to fold f with
probability p_f, the entity avoids straddling only by landing wholly in one fold, so

    Pr[e straddles] = 1 - sum_f p_f^{k_e},

which reduces to the familiar 1 - p^k - (1-p)^k for two folds and is exactly zero at k = 1.
Averaging over entities gives a quantity computable before any model exists.

The random split here has three folds, so the two-fold form is also reported to show how far
it misses.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

SPLITS = ROOT / "release" / "splits"
RES = ROOT / "data" / "results"
log = make_logger("exposure")


def main() -> None:
    rand = {k: v for k, v in json.loads((SPLITS / "random.json").read_text(encoding="utf-8")).items()
            if isinstance(v, list)}
    pairs = [tuple(x.split("||", 1)) for v in rand.values() for x in v if "||" in x]
    if len(pairs) != sum(len(v) for v in rand.values()):
        raise SystemExit("some incidence entries did not parse as event||narrator")

    k = Counter(n for _, n in pairs)                      # eligible cells per narrator
    total = sum(len(v) for v in rand.values())
    props = sorted((len(v) / total for v in rand.values()), reverse=True)

    def expected(ps: list[float]) -> float:
        return sum(1.0 - sum(p ** ke for p in ps) for ke in k.values()) / len(k)

    three = expected(props)
    two = expected([0.7, 0.3])

    # The splitter permutes and slices, so fold sizes are exact and the draw is without
    # replacement; this is the finite-sample law the independent form above approximates.
    sizes = [len(v) for v in rand.values()]
    hyper = sum(1.0 - sum(comb(s, ke) / comb(total, ke) for s in sizes if s >= ke)
                for ke in k.values()) / len(k)

    seen: dict[str, set[str]] = {}
    fold = {f"{e}||{n}": f for f, v in rand.items() for x in v for e, n in [x.split("||", 1)]}
    for e, n in pairs:
        seen.setdefault(n, set()).add(fold[f"{e}||{n}"])
    observed = sum(1 for v in seen.values() if len(v) > 1) / len(seen)

    singletons = sum(1 for v in k.values() if v == 1)
    out = {
        "n_narrators": len(k),
        "fold_proportions": props,
        "median_k": sorted(k.values())[len(k) // 2],
        "mean_k": sum(k.values()) / len(k),
        "n_k_equals_one": singletons,
        "frac_k_equals_one": singletons / len(k),
        "expected_three_fold": three,
        "expected_two_fold_p07": two,
        "expected_hypergeometric": hyper,
        "hypergeometric_gap_pp": abs(three - hyper) * 100.0,
        "observed": observed,
        "abs_error_three_fold": abs(three - observed),
        "abs_error_two_fold": abs(two - observed),
    }

    log(f"  narrators {len(k)}, fold proportions "
        f"{', '.join(f'{p:.3f}' for p in props)}")
    log(f"  cells per narrator: median {out['median_k']}, mean {out['mean_k']:.2f}")
    log(f"  narrators with k=1: {singletons} ({singletons / len(k):.1%}) - these cannot straddle")
    log(f"  predicted (three folds, observed proportions) {three:.4f}")
    log(f"  predicted (exact fold sizes, hypergeometric)  {hyper:.4f}"
        f"   gap {abs(three - hyper) * 100:.3f} pp")
    log(f"  predicted (two folds, p=0.7)                  {two:.4f}")
    log(f"  observed                                      {observed:.4f}")
    log(f"  |error| three-fold {abs(three - observed):.4f}   two-fold {abs(two - observed):.4f}")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e_apriori_exposure.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {RES / 'e_apriori_exposure.json'}")


if __name__ == "__main__":
    main()
