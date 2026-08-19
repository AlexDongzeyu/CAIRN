"""Narrator straddling under each split.

The three split files use different units: random.json lists event||narrator incidences,
narrator-disjoint.json lists narrators, event-disjoint.json lists events. Narrator straddling
is only comparable across them if each rule is applied to the same underlying incidence set,
so the full set is read from random.json and re-partitioned under each rule.

An earlier version extracted the owner with a regex that fell back to the whole entry when it
did not match. That silently counted unparsed entries as distinct owners and made the two
disjoint splits read 0% by construction.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

SPLITS = ROOT / "release" / "splits"
RES = ROOT / "data" / "results"
log = make_logger("straddle")


def folds(name: str) -> dict[str, list[str]]:
    d = json.loads((SPLITS / name).read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if isinstance(v, list)}


def main() -> None:
    rand, narr, even = folds("random.json"), folds("narrator-disjoint.json"), folds("event-disjoint.json")

    pairs = [tuple(x.split("||", 1)) for v in rand.values() for x in v if "||" in x]
    n_raw = sum(len(v) for v in rand.values())
    if len(pairs) != n_raw:
        raise SystemExit(f"{n_raw - len(pairs)} incidence entries did not parse as event||narrator")

    by_inc = {f"{e}||{n}": k for k, v in rand.items() for x in v for e, n in [x.split("||", 1)]}
    by_narr = {n: k for k, v in narr.items() for n in v}
    by_even = {e: k for k, v in even.items() for e in v}

    def measure(assign) -> tuple[int, int]:
        seen: dict[str, set[str]] = defaultdict(set)
        for e, n in pairs:
            f = assign(e, n)
            if f is None:
                raise SystemExit(f"no fold for incidence {e}||{n}")
            seen[n].add(f)
        return len(seen), sum(1 for v in seen.values() if len(v) > 1)

    out = {}
    for name, assign in (("random", lambda e, n: by_inc.get(f"{e}||{n}")),
                         ("narrator_disjoint", lambda e, n: by_narr.get(n)),
                         ("event_disjoint", lambda e, n: by_even.get(e))):
        total, strd = measure(assign)
        out[name] = {"n_narrators": total, "n_straddling": strd,
                     "frac_straddling": strd / total}
        log(f"  {name}: {strd}/{total} narrators straddle ({strd / total:.1%})")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "e_split_straddle.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"wrote {RES / 'e_split_straddle.json'}")


if __name__ == "__main__":
    main()
