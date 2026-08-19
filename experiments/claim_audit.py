"""Trace every number in the narrative documents back to a results file.

`findings.md` and `research-log.md` are written by hand, so their numbers can drift from
the runs -- a value copied during an earlier pass and never updated reads exactly like a
current one. This walks every numeric literal in those documents and looks for it among the
values actually stored in data/results/*.json.

A number is matched if it equals a stored value at the precision it is quoted to, which
means 0.139 matches a stored 0.13873. Unmatched numbers are reported for inspection rather
than treated as proof of error: years, counts of sections, thresholds quoted from the
protocol and round numbers legitimately appear in prose without being results. The point is
that every unmatched number gets looked at.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
DOCS = ["findings.md", "research-log.md"]
log = make_logger("claim_audit")

# Numbers that are part of the protocol or of ordinary prose, not measurements.
KNOWN_LITERALS = {
    0.67,   # pre-registered alpha floor
    0.60,   # pre-registered RBO floor / E3.3 selected threshold
    0.05,   # conventional alpha
    10.0,   # seeds, negative ratio
    16.0,   # laplacian eigenvectors
}


def flatten(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten(v, out)


def load_all_values() -> list[float]:
    vals: list[float] = []
    for p in sorted(RES.glob("*.json")):
        try:
            flatten(json.loads(p.read_text(encoding="utf-8")), vals)
        except json.JSONDecodeError:
            log(f"  skipping unparseable {p.name}")
    return vals


def quoted_numbers(text: str) -> list[tuple[str, float, int]]:
    """Return (literal, value, decimal places) for numbers worth checking."""
    out = []
    # Strip markdown links and code spans, which carry file names and ids.
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    # ISO dates would otherwise contribute a 2026 for every log entry.
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    # Sizes quoted in prose describe artefacts on disk, not measurements.
    text = re.sub(r"\d+(?:\.\d+)?\s*(?:MB|KB|GB|s\b|seconds|minutes)", " ", text, flags=re.I)
    # "171,313" is one number; splitting on the separator invented two.
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", text):
        lit = m.group(1)
        val = float(lit)
        dp = len(lit.split(".")[1]) if "." in lit else 0
        # Integers below 100 are almost always prose (phase numbers, counts of sections).
        if dp == 0 and val < 100:
            continue
        if val in KNOWN_LITERALS:
            continue
        out.append((lit, val, dp))
    return out


def matches(val: float, dp: int, pool: list[float]) -> bool:
    """Match at the precision quoted, in either percent or fraction units.

    Prose says "77.7% coverage" where the results file stores 0.777, so a value is accepted
    if either the number or the number divided by 100 is present.
    """
    for candidate, places in ((val, dp), (val / 100.0, dp + 2)):
        tol = 0.5 if places == 0 else 0.5 * (10 ** -places)
        if any(abs(v - candidate) < tol for v in pool):
            return True
    return False


def main() -> None:
    pool = load_all_values()
    log(f"claim audit against {len(pool):,} numeric values in {len(list(RES.glob('*.json')))} "
        f"results files")
    report = {}
    total = unmatched_total = 0
    for doc in DOCS:
        p = ROOT / doc
        if not p.exists():
            log(f"  {doc} absent")
            continue
        nums = quoted_numbers(p.read_text(encoding="utf-8"))
        unmatched = [(lit, dp) for lit, val, dp in nums if not matches(val, dp, pool)]
        total += len(nums)
        unmatched_total += len(unmatched)
        report[doc] = {"checked": len(nums),
                       "unmatched": [lit for lit, _ in unmatched]}
        log(f"  {doc}: {len(nums) - len(unmatched)}/{len(nums)} numbers trace to a results file")
        for lit, _ in unmatched:
            log(f"    unmatched: {lit}")

    (RES / "claim_audit.json").write_text(json.dumps(
        {"n_pool_values": len(pool), "total_checked": total,
         "total_unmatched": unmatched_total, "per_document": report}, indent=2),
        encoding="utf-8")
    log(f"{total - unmatched_total}/{total} traced; {unmatched_total} need a human look")


if __name__ == "__main__":
    main()
