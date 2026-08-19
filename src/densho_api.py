"""Densho Digital Repository (DDR) REST client.

E1.1 — primary corpus acquisition. Polite crawl: 1 req/sec, identifying UA.
All responses are cached to disk so a re-run costs no requests.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import requests

API_ROOT = "https://ddr.densho.org/api/0.2"
UA = (
    "CAIRN-research/0.1 (oral-history structure study; "
    "contact: research@example.org) python-requests"
)
RATE_LIMIT_S = 1.0
CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "densho_cache"

_last_call = [0.0]


def _throttle() -> None:
    dt = time.time() - _last_call[0]
    if dt < RATE_LIMIT_S:
        time.sleep(RATE_LIMIT_S - dt)
    _last_call[0] = time.time()


MAX_RETRIES = 6


def get(url: str, params: dict[str, Any] | None = None, *, use_cache: bool = True) -> dict:
    """GET with on-disk cache, polite throttling, and backoff on transient failures.

    4xx other than 429 are permanent and raise immediately; everything else is retried.
    """
    full = url + ("?" + urlencode(sorted((params or {}).items())) if params else "")
    key = hashlib.sha1(full.encode()).hexdigest()[:20]
    cpath = CACHE / f"{key}.json"
    if use_cache and cpath.exists():
        return json.loads(cpath.read_text(encoding="utf-8"))

    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=45)
            if 400 <= r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
            r.raise_for_status()
            data = r.json()
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(data), encoding="utf-8")
            return data
        except requests.HTTPError as e:
            sc = e.response.status_code if e.response is not None else None
            if sc is not None and 400 <= sc < 500 and sc != 429:
                raise
            last = e
        except Exception as e:  # noqa: BLE001 - network layer, retry everything transient
            last = e
        time.sleep(min(60.0, 2.0 * (2**attempt)))
    raise RuntimeError(f"GET failed after {MAX_RETRIES} attempts: {full}") from last


def search(
    fulltext: str = "",
    *,
    models: str | None = None,
    limit: int = 100,
    offset: int = 0,
    **filters: Any,
) -> dict:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if fulltext:
        params["fulltext"] = fulltext
    if models:
        params["models"] = models
    params.update(filters)
    return get(f"{API_ROOT}/search/", params)


def paginate(endpoint: str, *, limit: int = 100, cap: int | None = None, **params: Any) -> Iterator[dict]:
    """Yield every object from a paginated DDR list endpoint."""
    offset, seen = 0, 0
    while True:
        page = get(endpoint, {**params, "limit": limit, "offset": offset})
        objs = page.get("objects", [])
        if not objs:
            return
        for o in objs:
            yield o
            seen += 1
            if cap and seen >= cap:
                return
        offset += len(objs)
        if offset >= page.get("total", 0):
            return


def probe() -> dict[str, Any]:
    """Cheap connectivity + shape check. Returns a report dict."""
    report: dict[str, Any] = {}
    try:
        root = get(f"{API_ROOT}/")
        report["api_root"] = {"ok": True, "keys": sorted(root)[:20]}
    except Exception as e:  # noqa: BLE001 - probe reports any failure verbatim
        report["api_root"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    try:
        s = search(models="segment", limit=1)
        report["segments"] = {"ok": True, "total": s.get("total"), "sample": (s.get("objects") or [None])[0]}
    except Exception as e:  # noqa: BLE001
        report["segments"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    try:
        s = search(models="entity", limit=1, format="vh")
        report["entities_vh"] = {"ok": True, "total": s.get("total")}
    except Exception as e:  # noqa: BLE001
        report["entities_vh"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return report


if __name__ == "__main__":
    print(json.dumps(probe(), indent=2, default=str)[:4000])
