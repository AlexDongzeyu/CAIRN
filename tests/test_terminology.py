"""E9.1 — terminology enforcement, as a CI check rather than as advice.

The measured quantity is *archive-conditioned attestation multiplicity*. It is a
property of what this collection happens to record, not of history. Calling it
corroboration, veracity, silence or truth would claim something the data cannot support,
so those words are banned mechanically from every paper-facing artifact.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BANNED = [
    "historical silence", "silence detection", "veracity", "contradiction score",
    "uncorroborated", "fact-check", "fact check", "reliability score",
    "who to interview", "before they die", "verify testimony", "verified testimony",
    "ground truth of history", "what really happened",
]
# "corroborat*" is allowed only inside the task name, which is a retrieval task and is
# defined in the paper as such.
CORROBORATION_OK = re.compile(r"corroborat\w*\s+retrieval|T1\s*\(?corroborat", re.I)
CORROBORATION_ANY = re.compile(r"corroborat\w*", re.I)
REQUIRED_QUALIFIER = "archive-conditioned"

PAPER_GLOBS = ["paper/**/*.tex", "paper/**/*.md", "findings.md", "to_human/**/*.md"]


def paper_files() -> list[Path]:
    out: list[Path] = []
    for g in PAPER_GLOBS:
        out += [p for p in ROOT.glob(g) if p.is_file()]
    return out


def _scan(text: str) -> dict:
    low = text.lower()
    hits = [b for b in BANNED if b in low]
    bad_corr = []
    for m in CORROBORATION_ANY.finditer(text):
        window = text[max(0, m.start() - 60): m.end() + 60]
        if not CORROBORATION_OK.search(window):
            bad_corr.append(window.strip().replace("\n", " ")[:120])
    return {"banned": hits, "unqualified_corroboration": bad_corr}


def test_no_banned_terminology_in_paper_artifacts():
    problems = {}
    for p in paper_files():
        r = _scan(p.read_text(encoding="utf-8", errors="replace"))
        if r["banned"] or r["unqualified_corroboration"]:
            problems[str(p.relative_to(ROOT))] = r
    assert not problems, f"banned terminology found: {problems}"


def test_attestation_claims_carry_the_qualifier():
    """Any file that discusses attestation multiplicity must scope it to the archive."""
    offenders = []
    for p in paper_files():
        t = p.read_text(encoding="utf-8", errors="replace")
        if "attestation multiplicity" in t.lower() and REQUIRED_QUALIFIER not in t.lower():
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"missing '{REQUIRED_QUALIFIER}' qualifier in: {offenders}"


def test_scanner_actually_catches_a_violation():
    """A check that matches nothing passes trivially; prove the scanner can fail."""
    bad = "We detect historical silence and score the veracity of each testimony."
    r = _scan(bad)
    assert "historical silence" in r["banned"] and "veracity" in r["banned"]

    unqualified = "The model improves corroboration of witness accounts."
    assert _scan(unqualified)["unqualified_corroboration"], "scanner missed a bare 'corroboration'"

    allowed = "We evaluate on T1 (corroboration retrieval), a passage-ranking task."
    assert not _scan(allowed)["unqualified_corroboration"], "scanner false-positived on the task name"


def test_scanner_sees_at_least_one_file():
    """Guards against the failure mode where the glob matches nothing and CI is green."""
    assert paper_files(), "terminology CI matched no files - the check would pass vacuously"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
