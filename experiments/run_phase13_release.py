"""PHASE 13 — stand-off artifact release.

Annotations are released as identifiers and offsets against the source archive, plus a
fetch script, rather than as redistributed transcript text. Densho's CC BY-NC-SA grant
would permit more, but stand-off release is the convention for testimony corpora and it
keeps the release correct even where a per-item licence is narrower.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import GRANULARITIES, build_complex  # noqa: E402
from src.corpus import corpus_stats, load_corpus  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.tasks import make_split  # noqa: E402

RES = ROOT / "data" / "results"
REL = ROOT / "release"
log = make_logger("release")


def reproduction_check() -> dict:
    """E13.2 - can the released artifacts alone reproduce Table 1?

    A separate interpreter is launched with the project source *not* importable, so the
    check can only read what a downloader would actually receive. It recomputes the corpus
    statistics from `segments_standoff.csv` + `cells.jsonl` and compares them to the
    published `e1_1_corpus_stats.json`. This is not an independent human reproduction, and
    is labelled as a self-sufficiency check rather than one.
    """
    import subprocess

    script = r'''
import csv, json, sys
from pathlib import Path
rel = Path(sys.argv[1]); out = {}
narr, ivs, segs = set(), set(), set()
with (rel / "segments_standoff.csv").open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        segs.add(row["segment_id"]); ivs.add(row["interview_id"])
        narr.update(n for n in row["narrator_ids"].split("|") if n)
out["n_segments"] = len(segs); out["n_interviews"] = len(ivs); out["n_narrators"] = len(narr)
ranks = {}
with (rel / "cells.jsonl").open(encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        if c["granularity"] == "mid":
            ranks[c["rank"]] = ranks.get(c["rank"], 0) + 1
out["cells_per_rank_mid"] = ranks
print(json.dumps(out))
'''
    tmp = REL / "_repro_check.py"
    tmp.write_text(script, encoding="utf-8")
    try:
        p = subprocess.run([sys.executable, "-I", str(tmp), str(REL)],
                           capture_output=True, text=True, timeout=600, cwd=str(REL))
        recomputed = json.loads(p.stdout.strip()) if p.returncode == 0 else {}
        err = p.stderr.strip()[-400:] if p.returncode != 0 else ""
    finally:
        tmp.unlink(missing_ok=True)

    published = json.loads((RES / "e1_1_corpus_stats.json").read_text(encoding="utf-8"))
    fields = ["n_interviews", "n_segments", "n_narrators"]
    comparison = {
        k: {"published": published.get(k), "from_release": recomputed.get(k),
            "match": published.get(k) == recomputed.get(k)}
        for k in fields
    }
    out = {
        "method": ("fresh isolated interpreter (-I), cwd inside release/, project source not "
                   "importable; reads only the released files"),
        "comparison": comparison,
        "all_match": all(v["match"] for v in comparison.values()),
        "cells_per_rank_mid": recomputed.get("cells_per_rank_mid"),
        "error": err,
        "limitation": ("this verifies the release is self-sufficient, NOT that an independent "
                       "person reproduced it; the protocol's E13.2 asks for the latter"),
    }
    (RES / "e13_2_reproduction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"E13.2 - release self-sufficiency: all_match={out['all_match']} {comparison}")
    return out


def main() -> None:
    REL.mkdir(parents=True, exist_ok=True)
    interviews, segments = load_corpus()
    maps = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    rank_maps = {"R-A": maps["R-A_consensus"], "R-B": maps["R-B_archive_native"],
                 "R-C": maps["R-C_adversarial"]}

    # cells.jsonl — one row per cell, per granularity, with provenance
    with (REL / "cells.jsonl").open("w", encoding="utf-8") as f:
        for g in GRANULARITIES:
            cx = build_complex(segments, rank_maps["R-A"], granularity=g, rank_map_name="R-A")
            for c in cx.cells.values():
                f.write(json.dumps({
                    "cell_id": c.cid, "rank": c.rank, "granularity": g, "rank_map": "R-A",
                    "label": c.label, "base_term": c.base_term,
                    "member_narrator_ids": sorted(c.members),
                    "supporting_segment_ids": sorted(c.segments),
                    "archive_conditioned_attestation_multiplicity": c.size,
                }, ensure_ascii=False) + "\n")
    log("wrote cells.jsonl")

    # incidence matrices per granularity
    for g in GRANULARITIES:
        cx = build_complex(segments, rank_maps["R-A"], granularity=g, rank_map_name="R-A")
        mats = {f"B{k}{j}": cx.incidence_matrix(k, j) for k, j in [(0, 1), (1, 2), (2, 3)]}
        np.savez_compressed(
            REL / f"incidence_{g}.npz",
            **{k: sp.csr_matrix(v).toarray().astype(np.uint8) if v.shape[0] * v.shape[1] < 5e7
               else np.array([]) for k, v in mats.items()},
            **{f"{k}_indices": v.indices for k, v in mats.items()},
            **{f"{k}_indptr": v.indptr for k, v in mats.items()},
            **{f"{k}_shape": np.array(v.shape) for k, v in mats.items()},
        )
    log("wrote incidence_*.npz")

    # splits by id
    (REL / "splits").mkdir(exist_ok=True)
    cx = build_complex(segments, rank_maps["R-A"], granularity="mid", rank_map_name="R-A")
    for kind in ("narrator-disjoint", "event-disjoint", "random"):
        s = make_split(cx, kind, seed=0)
        (REL / "splits" / f"{kind}.json").write_text(json.dumps(
            {"unit": s.unit, "train": sorted(s.train), "val": sorted(s.val), "test": sorted(s.test)},
            indent=1), encoding="utf-8")
    log("wrote splits/")

    # stand-off segment index: ids only, no redistributed text
    with (REL / "segments_standoff.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "interview_id", "narrator_ids", "topic_terms",
                    "geography_terms", "extent", "api_url"])
        for s in segments:
            w.writerow([
                s.segment_id, s.interview_id, "|".join(s.narrators),
                "|".join(t.get("term", "") for t in s.topics),
                "|".join(g.get("term", "") for g in s.geography),
                s.extent, f"https://ddr.densho.org/api/0.2/{s.segment_id}/",
            ])
    log("wrote segments_standoff.csv")

    # annotation bundle
    (REL / "annotation").mkdir(exist_ok=True)
    for name in ("e2_2_agreement.json", "e2_3_rank_maps.json"):
        (REL / "annotation" / name).write_text((RES / name).read_text(encoding="utf-8"),
                                               encoding="utf-8")
    (REL / "annotation" / "RANK_MANUAL.md").write_text(
        (ROOT / "annotation" / "RANK_MANUAL.md").read_text(encoding="utf-8"), encoding="utf-8")

    (REL / "fetch_text.py").write_text('''"""Reconstruct segment text from Densho given the released identifiers.

Usage:  python fetch_text.py segments_standoff.csv out.jsonl
Crawls politely at 1 request/second with an identifying User-Agent.
"""
import csv, json, sys, time, urllib.request

UA = "CAIRN-replication/0.1 (contact: research@example.org)"

def fetch(seg_id):
    req = urllib.request.Request(f"https://ddr.densho.org/api/0.2/{seg_id}/",
                                 headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.load(r)

def main(index_csv, out_path):
    with open(index_csv, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as o:
        for row in csv.DictReader(f):
            time.sleep(1.0)
            try:
                d = fetch(row["segment_id"])
            except Exception as e:
                print("skip", row["segment_id"], e); continue
            o.write(json.dumps({"segment_id": row["segment_id"],
                                "title": d.get("title"),
                                "description": d.get("description")}) + "\\n")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
''', encoding="utf-8")

    stats = corpus_stats(interviews, segments)
    (REL / "DATASET_CARD.md").write_text(f"""# Dataset card — CAIRN ranked complex over Densho oral history

## What this is
A ranked combinatorial complex derived from the Densho Visual History Collection
(`ddr-densho-1000`). Rank-0 cells are narrators, rank-1 cells are the archive's own
interview segments, and rank-2/rank-3 cells are descriptive terms from Densho's controlled
vocabulary, ranked by the procedure in `annotation/RANK_MANUAL.md`.

## Provenance and licensing
* Source: Densho Digital Repository, https://ddr.densho.org
* Collection: `{stats['n_interviews']}` interviews, `{stats['n_segments']}` segments,
  `{stats['n_narrators']}` distinct narrators.
* Per-item rights recorded in `LICENSE_AUDIT.csv`: {json.dumps(stats['rights'])}.
* Densho content is offered under CC BY-NC-SA 4.0 by default. This release is **stand-off**:
  identifiers, offsets and derived structure only. `fetch_text.py` reconstructs text from the
  archive. Please cite Densho, not this repository, for the testimony itself.

## Consent limitation
Interviewees consented to archival deposit and public access. They did not consent to
computational analysis, and nothing here should be read as their endorsement of it. Densho
maintains a community-consent posture over this material; researchers reusing it should
contact the archive rather than treating the CC licence as the whole of the obligation.

## What the numbers mean, and what they do not
The quantity computed here is **archive-conditioned attestation multiplicity**: the number of
distinct narrators in this collection whose segments the archive filed under a given
descriptive term. It is a property of what this archive recorded and how it described it.

**It is not a measure of historical reality.** A low value means this collection holds few
narrators described under that term. It does not mean an event was rare, unimportant,
suppressed, or untrue. Interviewer prompts are excluded from the count because they are
archival artifacts rather than witness attestations; the effect of including them is reported
as an ablation.

## Known limitations
* Rank assignment was **not** reproducible across independent operationalizations on this
  corpus (see `annotation/e2_2_agreement.json`). Any downstream claim that depends on rank
  semantics must be read under all three released rank maps (R-A, R-B, R-C).
* Triage lists are **granularity-dependent**; there is no resolution-invariant ranking.
* No expert panel validated the triage output. The application claim is "computationally
  stable", not "expert-validated".

## Files
| File | Contents |
|---|---|
| `cells.jsonl` | every cell: id, rank, granularity, members, supporting segments |
| `incidence_{{g}}.npz` | B01, B12, B23 per granularity (CSR arrays) |
| `splits/` | exact narrator-disjoint, event-disjoint and random splits by id |
| `segments_standoff.csv` | segment index with archive API URLs, no redistributed text |
| `annotation/` | frozen rank manual, agreement study, all three rank maps |
| `fetch_text.py` | reconstructs text from the archive |
""", encoding="utf-8")
    log("wrote DATASET_CARD.md and fetch_text.py")

    if (ROOT / "corpus" / "densho" / "LICENSE_AUDIT.csv").exists():
        (REL / "LICENSE_AUDIT.csv").write_text(
            (ROOT / "corpus" / "densho" / "LICENSE_AUDIT.csv").read_text(encoding="utf-8"),
            encoding="utf-8")

    reproduction_check()
    log("PHASE 13 complete")


if __name__ == "__main__":
    main()
