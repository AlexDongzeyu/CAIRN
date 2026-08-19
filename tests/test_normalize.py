"""Instrument checks for the transcript normalizer (E1.3).

The speaker-label coverage check exists because the first implementation parsed the
header against un-stripped text, matched nothing, and reported 0.5% coverage as though it
were a property of the archive rather than a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.normalize import (  # noqa: E402
    classify_turn_by_rules, clean_text, evaluation, initials_of, normalize_transcript,
    parse_header,
)

SAMPLE = """<html><body>
<p>Densho Digital Archive</p>
<p>Title: Jane Doe Interview</p>
<p>Narrator: Jane Doe</p>
<p>Interviewers: Alan Brown (primary), Cara Diaz (secondary)</p>
<p>Date: July 25, 1997</p>
<p>&lt;Begin Segment 1&gt;</p>
<p>AB: Could you tell us where you were born?</p>
<p>JD: I was born in Seattle in 1925, uh, and my father ran a grocery store.</p>
<p>&lt;Begin Segment 2&gt;</p>
<p>CD: And what did you do after the war?</p>
<p>JD: We went back home. It was [inaudible] hard for my mother.</p>
</body></html>"""


def test_initials():
    assert initials_of("Gene Akutsu") == "GA"
    assert initials_of("Larry Hashima") == "LH"
    assert initials_of("") == ""


def test_header_parsing_finds_narrator_and_interviewers():
    from src.normalize import canonical_text

    narr, intv = parse_header(canonical_text(SAMPLE))
    assert narr == ["Jane Doe"]
    assert [i.strip() for i in intv] == ["Alan Brown", "Cara Diaz"]


def test_turns_get_archive_roles_and_segments():
    iv = normalize_transcript("test-1", SAMPLE)
    roles = [(t.speaker_tag, t.role_gold, t.segment_no) for t in iv.turns]
    assert ("AB", "INTERVIEWER", 1) in roles
    assert ("JD", "NARRATOR", 1) in roles
    assert ("CD", "INTERVIEWER", 2) in roles
    assert all(t.role_gold != "UNKNOWN" for t in iv.turns)


def test_speaker_label_coverage_is_high():
    """Coverage near zero means the header parse broke, not that the archive is unlabelled."""
    iv = normalize_transcript("test-1", SAMPLE)
    ev = evaluation([iv])
    assert ev["label_coverage"] > 0.9, ev


def test_offsets_index_back_into_the_canonical_text():
    from src.normalize import canonical_text

    text = canonical_text(SAMPLE)
    iv = normalize_transcript("test-1", SAMPLE)
    for t in iv.turns:
        assert text[t.char_start:t.char_end] == t.raw


def test_cleaning_strips_disfluency_and_keeps_inaudible_sentinel():
    out = clean_text("I was born in Seattle, uh, and it was [inaudible] hard")
    assert "uh" not in out.split()
    assert "<INAUD>" in out
    assert "Seattle" in out


def test_cleaning_collapses_false_starts():
    assert clean_text("we we we went home") == "we went home"


def test_rules_classifier_is_blind_to_initials_but_still_informative():
    q = "And what did you do after the war? Where did you go?"
    a = ("I was born in Seattle and my father ran a grocery store for many years before "
         "we were sent away, and I remember my mother crying about it.")
    assert classify_turn_by_rules(q) == "INTERVIEWER"
    assert classify_turn_by_rules(a) == "NARRATOR"


def test_evaluation_reports_a_confusion_matrix():
    iv = normalize_transcript("test-1", SAMPLE)
    ev = evaluation([iv])
    c = ev["confusion"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == ev["n_turns_with_archive_label"]
    assert 0.0 <= ev["turn_classification_accuracy"] <= 1.0
