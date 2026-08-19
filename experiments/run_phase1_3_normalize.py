"""Run PHASE 1.3 (transcript normalization) and the E3.6 ambiguity flagging it feeds.

Emits corpus/densho/normalized.jsonl as stand-off annotation: character offsets into the
archived transcript plus the derived quantities E3.6 consumes, with no copy of the text
itself. Applying clean_text() to raw[char_start:char_end] reconstructs a turn, which is
what makes the E13 release reconstructable without redistributing the testimony.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.normalize import evaluation, normalize_transcript  # noqa: E402

RES = ROOT / "data" / "results"
TRANS = ROOT / "corpus" / "densho" / "transcripts"
OUT = ROOT / "corpus" / "densho" / "normalized.jsonl"
log = make_logger("phase1_3")


def run_normalization(limit: int | None = None) -> dict:
    files = sorted(TRANS.glob("*.htm"))
    if limit:
        files = files[:limit]
    log(f"E1.3 - normalizing {len(files)} interview transcripts")

    normalized = []
    with OUT.open("w", encoding="utf-8") as f:
        for i, p in enumerate(files, 1):
            iv = normalize_transcript(p.stem, p.read_text(encoding="utf-8", errors="replace"))
            normalized.append(iv)
            for t in iv.turns:
                # Stand-off: offsets + labels only. The cleaned text is reconstructable by
                # applying clean_text() to raw[char_start:char_end], so storing it here would
                # redistribute the testimony a second time (E13.1) and triple the file size.
                f.write(json.dumps({
                    "interview_id": iv.interview_id,
                    "segment_no": t.segment_no,
                    "speaker_tag": t.speaker_tag,
                    "role_archive": t.role_gold,
                    "role_rules": t.role_pred,
                    "char_start": t.char_start,
                    "char_end": t.char_end,
                    "n_clean_chars": len(t.clean),
                    "n_sentences": t.n_sentences,
                    "has_inaud": "<INAUD>" in t.clean,
                }, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                log(f"  {i}/{len(files)}")

    ev = evaluation(normalized)
    log(f"E1.3 - turn-classification accuracy={ev['turn_classification_accuracy']:.3f} "
        f"on {ev['n_turns_with_archive_label']} labelled turns "
        f"({ev['label_coverage']:.1%} coverage); "
        f"{ev['pct_chars_removed_by_cleaning']:.1%} chars removed by cleaning")
    (RES / "e1_3_normalization.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    return ev


def run_ambiguity_flagging() -> dict:
    """E3.6 — per-cell ambiguity score; the top decile is the 'ambiguous' stratum.

    Combines the signals the protocol names, using what the archive actually exposes:
    (a) how confidently the term is linked (its corpus frequency), (b) how close the
    term sits to a merge decision (how many sibling terms share its parent path),
    (c) how many distinct surface forms its segments use, (d) whether its supporting
    text carries an <INAUD> sentinel.
    """
    _, segments = load_corpus()
    inaud_by_iv: dict[str, int] = defaultdict(int)
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("has_inaud"):
                    inaud_by_iv[r["interview_id"]] += 1

    freq: dict[str, int] = defaultdict(int)
    forms: dict[str, set] = defaultdict(set)
    inaud: dict[str, int] = defaultdict(int)
    siblings: dict[str, set] = defaultdict(set)
    for s in segments:
        for t in s.topics:
            term = t.get("term")
            if not term:
                continue
            freq[term] += 1
            forms[term].add((s.title or "").strip().lower()[:60])
            inaud[term] += inaud_by_iv.get(s.interview_id, 0)
            parent = " -- ".join([p.strip() for p in term.split("--")][:-1])
            if parent:
                siblings[parent].add(term)

    terms = sorted(freq)
    if not terms:
        return {}

    def z(v):
        v = np.asarray(v, dtype=float)
        sd = v.std()
        return (v - v.mean()) / (sd if sd > 0 else 1.0)

    link_conf = z([np.log1p(freq[t]) for t in terms])           # rarer -> less confident
    n_sib = z([len(siblings.get(" -- ".join([p.strip() for p in t.split("--")][:-1]), ())) 
               for t in terms])                                  # more siblings -> closer to a merge
    n_forms = z([len(forms[t]) / max(1, freq[t]) for t in terms])
    n_inaud = z([inaud[t] / max(1, freq[t]) for t in terms])

    score = (-link_conf) + n_sib + n_forms + n_inaud
    order = np.argsort(-score)
    cut = max(1, len(terms) // 10)
    flagged = [terms[i] for i in order[:cut]]

    out = {
        "n_terms": len(terms),
        "top_decile_cut": cut,
        "ambiguous_terms": flagged,
        "score_components": ["-log_frequency", "n_sibling_terms", "surface_form_diversity",
                             "INAUD_density"],
        "per_term_score": {terms[i]: float(score[i]) for i in order[:50]},
        "note": ("this is the E2.2 'ambiguous' stratum; agreement on it is reported "
                 "separately because it is where the construct is least stable"),
    }
    (RES / "e3_6_ambiguity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"E3.6 - flagged {cut}/{len(terms)} terms as ambiguous (top decile)")
    return out


def main() -> None:
    ev = run_normalization()
    amb = run_ambiguity_flagging()

    # fold E3.6 into the phase-3 record so the audit finds it where it expects
    p = RES / "e3_extraction.json"
    if p.exists() and amb:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["E3_6_ambiguity"] = {k: v for k, v in amb.items() if k != "per_term_score"}
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        log("E3.6 - merged into e3_extraction.json")
    log("PHASE 1.3 + 3.6 complete")
    return ev


if __name__ == "__main__":
    main()
