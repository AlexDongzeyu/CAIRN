"""Corpus loading and narrator resolution.

The rank-0 ground set is *narrators*, not interviews: a narrator may give several
interviews, and an interview may have several narrators. Densho gives a stable
`oh_id` for most narrators; we fall back to a normalized name key.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "densho"


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_markup(s: str) -> str:
    import html

    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(s or "")))).strip()


def find_boilerplate(texts: list[str], min_share: float = 0.005, min_words: int = 6) -> list[str]:
    """Sentences repeated across a large share of records are archive furniture, not content.

    Densho appends a funding acknowledgement to many segment descriptions. Left in, it
    dominates 5-gram shingles (so near-duplicate detection fires on almost every pair) and
    it dominates sentence embeddings (so every model sees the same boilerplate vector).
    Detecting it by frequency rather than by hard-coding the string keeps this honest if
    the archive changes its wording.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for t in texts:
        for sent in re.split(r"(?<=[.!?])\s+", strip_markup(t)):
            sent = sent.strip()
            if len(sent.split()) >= min_words:
                counts[sent] += 1
    n = max(1, len(texts))
    return [s for s, c in counts.items() if c / n >= min_share]


def clean_description(raw: str, boilerplate: tuple[str, ...]) -> str:
    t = strip_markup(raw)
    for b in boilerplate:
        t = t.replace(b, " ")
    return _WS_RE.sub(" ", t).strip()


def narrator_key(n: dict) -> str | None:
    """Stable narrator identifier. Prefers Densho's oral-history id."""
    oh = n.get("oh_id")
    if oh not in (None, "", 0):
        return f"oh:{oh}"
    nm = _norm_name(n.get("name") or n.get("namepart") or "")
    return f"nm:{nm}" if nm else None


@dataclass
class Segment:
    segment_id: str
    interview_id: str
    title: str
    description: str
    topics: list[dict]
    geography: list[dict]
    location: str
    extent: str
    narrators: list[str]          # resolved narrator keys
    interviewers: list[str]       # resolved interviewer keys (E8.4 ablation only)
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def text(self) -> str:
        """Archive-written segment summary; the unit of textual evidence we always have."""
        return f"{self.title}. {self.description}".strip()


@dataclass
class Interview:
    interview_id: str
    title: str
    description: str
    creation: str
    location: str
    narrators: list[str]
    topics: list[dict]
    facility: list[dict]
    rights: str
    n_segments: int
    transcript_chars: int
    raw: dict = field(repr=False, default_factory=dict)


def _jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn final line while the crawl is live
    return rows


def _roles(creators: list[dict] | None, role: str) -> list[str]:
    out = []
    for c in creators or []:
        if str(c.get("role", "")).lower() == role:
            k = narrator_key({"oh_id": c.get("oh_id"), "name": c.get("namepart")})
            if k:
                out.append(k)
    return sorted(set(out))


@lru_cache(maxsize=1)
def load_corpus(corpus_dir: str = str(CORPUS)) -> tuple[list[Interview], list[Segment]]:
    d = Path(corpus_dir)
    raw_interviews = _jsonl(d / "interviews.jsonl")
    raw_segments = _jsonl(d / "segments.jsonl")

    boiler = tuple(find_boilerplate([r.get("description") or "" for r in raw_segments]))

    interviews = [
        Interview(
            interview_id=r["interview_id"],
            title=strip_markup(r.get("title") or ""),
            description=clean_description(r.get("description") or "", boiler),
            creation=r.get("creation") or "",
            location=r.get("location") or "",
            narrators=_roles(r.get("creators"), "narrator"),
            topics=r.get("topics") or [],
            facility=r.get("facility") or [],
            rights=r.get("rights") or "",
            n_segments=int(r.get("n_segments") or 0),
            transcript_chars=int(r.get("transcript_chars") or 0),
            raw=r,
        )
        for r in raw_interviews
    ]
    by_iid = {i.interview_id: i for i in interviews}

    segments = []
    for r in raw_segments:
        narr = _roles(r.get("creators"), "narrator")
        if not narr:  # fall back to the parent interview's narrators
            narr = by_iid[r["interview_id"]].narrators if r.get("interview_id") in by_iid else []
        segments.append(
            Segment(
                segment_id=r["segment_id"],
                interview_id=r["interview_id"],
                title=strip_markup(r.get("title") or ""),
                description=clean_description(r.get("description") or "", boiler),
                topics=r.get("topics") or [],
                geography=r.get("geography") or [],
                location=r.get("location") or "",
                extent=r.get("extent") or "",
                narrators=narr,
                interviewers=_roles(r.get("creators"), "interviewer"),
                raw=r,
            )
        )
    return interviews, segments


def corpus_stats(interviews: list[Interview], segments: list[Segment]) -> dict:
    """Table 1 material."""
    import numpy as np

    narrators = sorted({n for s in segments for n in s.narrators})
    segs_per_iv = np.array([i.n_segments for i in interviews]) if interviews else np.array([0])
    tr = np.array([i.transcript_chars for i in interviews]) if interviews else np.array([0])
    return {
        "n_interviews": len(interviews),
        "n_segments": len(segments),
        "n_narrators": len(narrators),
        "n_interviews_with_transcript": int((tr > 0).sum()),
        "segments_per_interview": {
            "median": float(np.median(segs_per_iv)),
            "iqr": [float(np.percentile(segs_per_iv, 25)), float(np.percentile(segs_per_iv, 75))],
            "min": int(segs_per_iv.min()),
            "max": int(segs_per_iv.max()),
        },
        "transcript_chars": {
            "median": float(np.median(tr[tr > 0])) if (tr > 0).any() else 0.0,
            "iqr": [float(np.percentile(tr[tr > 0], 25)), float(np.percentile(tr[tr > 0], 75))]
            if (tr > 0).any() else [0.0, 0.0],
        },
        "rights": dict(__import__("collections").Counter(i.rights for i in interviews)),
        "segments_with_topics": sum(1 for s in segments if s.topics),
        "segments_with_geography": sum(1 for s in segments if s.geography),
    }
