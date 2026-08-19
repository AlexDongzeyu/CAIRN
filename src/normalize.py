"""E1.3 — transcript normalization.

Densho transcripts carry a header naming the narrator and interviewers, `<Begin Segment N>`
markers, and turn-initial speaker initials (`LH:`, `GA:`). The initials give a ground-truth
speaker role, so the rules-based classifier the protocol asks for can be scored against the
archive's own labelling on every turn rather than on 200 hand-labelled ones.

Why this matters beyond bookkeeping: interviewer turns contain event mentions that are
*prompts*, not attestations. Counting them inflates attestation multiplicity, which is the
project's headline quantity. E8.4 ablates the exclusion; this file establishes that the
exclusion can be made reliably in the first place.

Every record keeps `raw` alongside `clean` and stores character offsets against `raw`, which
is what makes the stand-off release in E13 possible.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

SEGMENT_RE = re.compile(r"<Begin Segment (\d+)>", re.I)
TURN_RE = re.compile(r"^\s*([A-Z][A-Za-z]?[A-Z]?)\s*:\s", re.M)
FILLED_PAUSE_RE = re.compile(r"\b(?:uh+|um+|er+|ah+|mm+|hmm+)\b[,]?\s*", re.I)
BRACKET_RE = re.compile(r"\[([^\]]*)\]")
FALSE_START_RE = re.compile(r"\b(\w+)(\s+\1\b)+", re.I)
WS_RE = re.compile(r"\s+")

SECOND_PERSON = re.compile(r"\b(you|your|yours|you're|you've|did you|were you|do you)\b", re.I)
FIRST_PERSON = re.compile(r"\b(i|me|my|mine|we|us|our|i'm|i've|i'd)\b", re.I)


def initials_of(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name or ""))
    parts = [p for p in re.split(r"[\s.]+", name) if p and p[0].isalpha()]
    return "".join(p[0].upper() for p in parts[:3])


@dataclass
class Turn:
    speaker_tag: str
    role_gold: str            # NARRATOR | INTERVIEWER | UNKNOWN, from the archive's labels
    role_pred: str            # NARRATOR | INTERVIEWER, from rules only
    segment_no: int | None
    char_start: int           # offset into the raw transcript text
    char_end: int
    raw: str
    clean: str = ""
    n_sentences: int = 0


@dataclass
class NormalizedInterview:
    interview_id: str
    narrator_names: list[str]
    interviewer_names: list[str]
    turns: list[Turn] = field(default_factory=list)
    raw_chars: int = 0


def parse_header(text: str) -> tuple[list[str], list[str]]:
    narr, intv = [], []
    m = re.search(r"^Narrators?:\s*(.+)$", text, re.M)
    if m:
        narr = [x.strip() for x in re.split(r",| and ", m.group(1)) if x.strip()]
    m = re.search(r"^Interviewers?:\s*(.+)$", text, re.M)
    if m:
        raw = re.sub(r"\((?:primary|secondary|tertiary)\)", "", m.group(1))
        intv = [x.strip() for x in re.split(r",| and ", raw) if x.strip()]
    return narr, intv


def clean_text(s: str) -> str:
    """Strip disfluencies and transcriber conventions.

    `[inaudible]` becomes a sentinel rather than vanishing, so downstream mention
    detection can be told to distrust the span instead of silently trusting a gap.
    """
    def _bracket(m: re.Match) -> str:
        inner = m.group(1).lower()
        if "inaudible" in inner or "unintelligible" in inner:
            return " <INAUD> "
        return " "

    s = BRACKET_RE.sub(_bracket, s)
    s = FILLED_PAUSE_RE.sub("", s)
    s = FALSE_START_RE.sub(r"\1", s)
    return WS_RE.sub(" ", s).strip()


def classify_turn_by_rules(raw: str) -> str:
    """Rules only - deliberately blind to the speaker initials it is scored against."""
    t = raw.strip()
    words = t.split()
    n = max(1, len(words))
    q = t.count("?")
    second = len(SECOND_PERSON.findall(t)) / n
    first = len(FIRST_PERSON.findall(t)) / n

    score = 0.0
    score += 2.0 * (q > 0)                       # questions are overwhelmingly interviewer turns
    score += 3.0 * second
    score -= 3.0 * first
    score += 1.0 * (len(words) < 25)             # prompts are short; testimony runs long
    score -= 1.0 * (len(words) > 80)
    return "INTERVIEWER" if score > 1.0 else "NARRATOR"


def canonical_text(html: str) -> str:
    """Deterministic plain-text rendering that offsets are measured against.

    `strip=True` matters: without it the header's `Narrator:` label and the name land on
    separate lines and the header regexes silently match nothing, which shows up as a
    speaker-label coverage near zero rather than as an error.
    """
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True).replace("\r\n", "\n")


def normalize_transcript(interview_id: str, html: str) -> NormalizedInterview:
    text = canonical_text(html)
    narr_names, intv_names = parse_header(text)
    narr_i = {initials_of(n) for n in narr_names} - {""}
    intv_i = {initials_of(n) for n in intv_names} - {""}

    seg_marks = [(m.start(), int(m.group(1))) for m in SEGMENT_RE.finditer(text)]

    def segment_at(pos: int) -> int | None:
        cur = None
        for start, no in seg_marks:
            if start <= pos:
                cur = no
            else:
                break
        return cur

    out = NormalizedInterview(interview_id, narr_names, intv_names, raw_chars=len(text))
    matches = list(TURN_RE.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        tag = m.group(1).upper()
        gold = ("NARRATOR" if tag in narr_i else
                "INTERVIEWER" if tag in intv_i else "UNKNOWN")
        cleaned = clean_text(body)
        out.turns.append(Turn(
            speaker_tag=tag, role_gold=gold, role_pred=classify_turn_by_rules(body),
            segment_no=segment_at(m.start()), char_start=start, char_end=end,
            raw=body, clean=cleaned,
            n_sentences=len([x for x in re.split(r"(?<=[.!?])\s+", cleaned) if x.strip()]),
        ))
    return out


def evaluation(interviews: list[NormalizedInterview]) -> dict:
    """Score the rules against the archive's own speaker labels."""
    tp = fp = fn = tn = 0
    n_known = n_total = 0
    for iv in interviews:
        for t in iv.turns:
            n_total += 1
            if t.role_gold == "UNKNOWN":
                continue
            n_known += 1
            gold_i = t.role_gold == "INTERVIEWER"
            pred_i = t.role_pred == "INTERVIEWER"
            if gold_i and pred_i:
                tp += 1
            elif gold_i and not pred_i:
                fn += 1
            elif not gold_i and pred_i:
                fp += 1
            else:
                tn += 1
    acc = (tp + tn) / n_known if n_known else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    raw_chars = sum(len(t.raw) for iv in interviews for t in iv.turns)
    clean_chars = sum(len(t.clean) for iv in interviews for t in iv.turns)
    return {
        "n_interviews": len(interviews),
        "n_turns_total": n_total,
        "n_turns_with_archive_label": n_known,
        "label_coverage": n_known / n_total if n_total else 0.0,
        "turn_classification_accuracy": acc,
        "interviewer_precision": prec,
        "interviewer_recall": rec,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "pct_chars_removed_by_cleaning": (1 - clean_chars / raw_chars) if raw_chars else 0.0,
        "mean_sentences_per_turn": (
            sum(t.n_sentences for iv in interviews for t in iv.turns) / n_total
            if n_total else 0.0),
        "note": ("accuracy is measured against the archive's own speaker initials on every "
                 "labelled turn, not on a 200-turn hand-labelled sample; the rules classifier "
                 "never sees the initials"),
    }
