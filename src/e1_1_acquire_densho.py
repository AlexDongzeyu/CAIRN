"""E1.1 — Densho corpus acquisition.

Emits:
  corpus/densho/interviews.jsonl   one record per interview (entity)
  corpus/densho/segments.jsonl     one record per segment (rank-1 moment candidate)
  corpus/densho/LICENSE_AUDIT.csv  one row per interview with its rights statement
  corpus/densho/transcripts/*.htm  interview-level full transcripts

Resumable: every HTTP response is disk-cached, so re-running costs no requests.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.densho_api import API_ROOT, UA, get  # noqa: E402

OUT = ROOT / "corpus" / "densho"
TRANS = OUT / "transcripts"
COLLECTION = "ddr-densho-1000"  # Densho Visual History Collection, rights=cc


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def page_all(endpoint: str) -> list[dict]:
    """Walk a DDR paginated listing (page_size is server-fixed at 25)."""
    objs: list[dict] = []
    offset = 0
    while True:
        d = get(endpoint, {"limit": 25, "offset": offset})
        batch = d.get("objects", [])
        if not batch:
            break
        objs.extend(batch)
        offset += len(batch)
        if offset >= d.get("total", 0):
            break
    return objs


def narrators_of(creators: list[dict] | None) -> list[dict]:
    out = []
    for c in creators or []:
        if str(c.get("role", "")).lower() == "narrator":
            out.append({"name": c.get("namepart"), "oh_id": c.get("oh_id")})
    return out


def _fetch_text(url: str, tries: int = 5) -> str | None:
    """Plain HTML GET with backoff; returns None only after exhausting retries."""
    for attempt in range(tries):
        time.sleep(1.0)
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            r.raise_for_status()
            return r.text
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else None
            if sc is not None and 400 <= sc < 500 and sc != 429:
                return None
        except Exception:  # noqa: BLE001 - transient network, retry
            pass
        time.sleep(min(60.0, 2.0 * (2**attempt)))
    return None


def fetch_transcript(interview_id: str) -> tuple[str | None, str | None]:
    """Scrape the interview page for its full-transcript URL, then download it."""
    dest = TRANS / f"{interview_id}.htm"
    url_cache = TRANS / f"{interview_id}.url"
    miss = TRANS / f"{interview_id}.none"
    if dest.exists() and url_cache.exists():
        return url_cache.read_text(encoding="utf-8").strip(), dest.read_text(encoding="utf-8", errors="replace")
    if miss.exists():
        return None, None

    page = _fetch_text(f"https://ddr.densho.org/{interview_id}/")
    if page is None:
        log(f"  ! page fail {interview_id}")
        return None, None

    soup = BeautifulSoup(page, "html.parser")
    href = None
    for a in soup.find_all("a", href=True):
        if "full transcript" in a.get_text(strip=True).lower():
            href = a["href"]
            break
    if not href:
        for a in soup.find_all("a", href=True):
            if re.search(rf"{re.escape(interview_id)}-transcript-[0-9a-f]+\.htm", a["href"]):
                href = a["href"]
                break
    if not href:
        miss.parent.mkdir(parents=True, exist_ok=True)
        miss.write_text("no transcript link", encoding="utf-8")
        return None, None

    text = _fetch_text(href)
    if text is None:
        log(f"  ! transcript fail {interview_id}")
        return href, None

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    url_cache.write_text(href, encoding="utf-8")
    return href, text


def main(limit_interviews: int | None = None, fetch_transcripts: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRANS.mkdir(parents=True, exist_ok=True)

    log(f"Stage A: enumerating interviews in {COLLECTION}")
    interviews = page_all(f"{API_ROOT}/{COLLECTION}/children/")
    interviews = [i for i in interviews if i.get("model") == "entity"]
    if limit_interviews:
        interviews = interviews[:limit_interviews]
    log(f"  {len(interviews)} interview entities")

    seg_f = (OUT / "segments.jsonl").open("w", encoding="utf-8")
    int_f = (OUT / "interviews.jsonl").open("w", encoding="utf-8")
    lic_f = (OUT / "LICENSE_AUDIT.csv").open("w", encoding="utf-8", newline="")
    lic = csv.writer(lic_f)
    lic.writerow(["interview_id", "title", "rights", "contributor", "credit", "n_segments", "has_transcript"])

    n_seg_total = 0
    for k, iv in enumerate(interviews, 1):
        iid = iv["id"]
        segs = [s for s in page_all(f"{API_ROOT}/{iid}/children/") if s.get("model") == "segment"]

        transcript_url, transcript_html = (None, None)
        if fetch_transcripts:
            transcript_url, transcript_html = fetch_transcript(iid)

        rec = {
            "interview_id": iid,
            "collection_id": iv.get("collection_id") or COLLECTION,
            "title": iv.get("title"),
            "description": iv.get("description"),
            "creation": iv.get("creation"),
            "location": iv.get("location"),
            "creators": iv.get("creators"),
            "narrators": narrators_of(iv.get("creators")),
            "topics": iv.get("topics"),
            "facility": iv.get("facility"),
            "geography": iv.get("geography"),
            "rights": iv.get("rights"),
            "contributor": iv.get("contributor"),
            "credit": iv.get("credit"),
            "format": iv.get("format"),
            "genre": iv.get("genre"),
            "n_segments": len(segs),
            "transcript_url": transcript_url,
            "transcript_chars": len(transcript_html or ""),
        }
        int_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        lic.writerow([iid, iv.get("title"), iv.get("rights"), iv.get("contributor"),
                      iv.get("credit"), len(segs), bool(transcript_html)])

        for s in segs:
            seg_f.write(json.dumps({
                "segment_id": s["id"],
                "interview_id": iid,
                "collection_id": s.get("collection_id") or COLLECTION,
                "sort": s.get("index"),
                "title": s.get("title"),
                "description": s.get("description"),
                "extent": s.get("extent"),
                "creators": s.get("creators"),
                "narrators": narrators_of(s.get("creators")),
                "topics": s.get("topics"),
                "geography": s.get("geography"),
                "location": s.get("location"),
                "creation": s.get("creation"),
                "rights": s.get("rights"),
                "format": s.get("format"),
                "genre": s.get("genre"),
                "language": s.get("language"),
            }, ensure_ascii=False) + "\n")
        n_seg_total += len(segs)

        if k % 10 == 0 or k == len(interviews):
            log(f"  [{k}/{len(interviews)}] {iid} segs={len(segs)} cum_segs={n_seg_total}")
        seg_f.flush(); int_f.flush(); lic_f.flush()

    seg_f.close(); int_f.close(); lic_f.close()
    log(f"DONE: {len(interviews)} interviews, {n_seg_total} segments -> {OUT}")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    main(limit_interviews=lim, fetch_transcripts="--no-transcripts" not in sys.argv)
