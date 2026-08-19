"""E6.0 — the controls that make the model comparison interpretable.

ONE frozen sentence encoder for every model. Embeddings are computed once, cached to
disk, and loaded identically everywhere. Rank-k features are built by the SAME function
for the combinatorial complex and for the star expansion, so any performance difference
is attributable to the operator rather than to the features.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "embeddings"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_encoder(name: str = ENCODER_NAME):
    from sentence_transformers import SentenceTransformer
    import torch

    return SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")


def embed_texts(texts: list[str], encoder=None, name: str = ENCODER_NAME) -> np.ndarray:
    """Cached, deterministic text embedding. The cache key covers the model and the text."""
    key = hashlib.sha1(("\u241f".join(texts) + "|" + name).encode("utf-8")).hexdigest()[:24]
    path = CACHE / f"{key}.npy"
    if path.exists():
        return np.load(path)
    enc = encoder or get_encoder(name)
    emb = np.asarray(
        enc.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=False),
        dtype=np.float32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def segment_embeddings(segments, encoder=None) -> tuple[np.ndarray, dict[str, int]]:
    texts = [s.text for s in segments]
    emb = embed_texts(texts, encoder)
    return emb, {s.segment_id: i for i, s in enumerate(segments)}


def build_features(cx, segments, encoder=None,
                   allowed_by_narrator: dict[str, set[str]] | None = None
                   ) -> dict[int, np.ndarray]:
    """Feature construction, applied identically to the CC and to G*.

    rank-0 : mean of that narrator's segment embeddings
    rank-k : mean of constituent rank-0 features, concatenated with a rank one-hot

    `allowed_by_narrator` restricts each narrator's mean to that narrator's training-side
    segments, which is what separates repeated narrator identity from cross-partition text
    aggregation. Membership is a property of the (narrator, segment) pair rather than of the
    segment: one segment can be training-side for a narrator whose incidence through it is
    in the training partition and held out for a co-narrator whose incidence is not.

    A narrator with no training-side segment falls back to their own segments. That is not a
    leak: at deployment a new interview arrives with its transcript, so a test narrator's own
    text is genuinely available, whereas a training narrator's test segments never were.
    """
    emb, sid2i = segment_embeddings(segments, encoder)
    d = emb.shape[1]

    narr_vecs: dict[str, list[int]] = {}
    narr_all: dict[str, list[int]] = {}
    for s in segments:
        for n in s.narrators:
            narr_all.setdefault(n, []).append(sid2i[s.segment_id])
            if allowed_by_narrator is None or s.segment_id in allowed_by_narrator.get(n, ()):
                narr_vecs.setdefault(n, []).append(sid2i[s.segment_id])

    n_index = {n: i for i, n in enumerate(cx.narrators)}
    X0 = np.zeros((len(cx.narrators), d), dtype=np.float32)
    n_fallback = 0
    for n in narr_all:
        if n not in n_index:
            continue
        idxs = narr_vecs.get(n)
        if not idxs:
            idxs = narr_all[n]
            n_fallback += 1
        X0[n_index[n]] = emb[idxs].mean(axis=0)
    build_features.last_fallback = (n_fallback, len(n_index))

    feats: dict[int, np.ndarray] = {}
    for k in range(4):
        cells = sorted(cx.by_rank(k), key=lambda c: c.cid)
        M = np.zeros((len(cells), d + 4), dtype=np.float32)
        for i, c in enumerate(cells):
            rows = [n_index[n] for n in c.members if n in n_index]
            if rows:
                M[i, :d] = X0[rows].mean(axis=0)
            M[i, d + k] = 1.0  # rank indicator, available to every model
        feats[k] = M
    return feats


def narrator_text_embeddings(cx, segments, encoder=None) -> np.ndarray:
    """Row-normalized narrator vectors: mean of that narrator's frozen segment embeddings."""
    emb, sid2i = segment_embeddings(segments, encoder)
    idx = {n: i for i, n in enumerate(cx.narrators)}
    M = np.zeros((len(cx.narrators), emb.shape[1]), dtype=np.float32)
    acc: dict[str, list[int]] = {}
    for s in segments:
        for n in s.narrators:
            if n in idx:
                acc.setdefault(n, []).append(sid2i[s.segment_id])
    for n, rows in acc.items():
        v = emb[rows].mean(axis=0)
        nrm = np.linalg.norm(v)
        M[idx[n]] = v / nrm if nrm > 0 else v
    return M


def cell_text_embeddings(cx, segments, k: int, encoder=None) -> np.ndarray:
    """Textual embedding of a rank-k cell: mean over the archive summaries supporting it."""
    emb, sid2i = segment_embeddings(segments, encoder)
    cells = sorted(cx.by_rank(k), key=lambda c: c.cid)
    M = np.zeros((len(cells), emb.shape[1]), dtype=np.float32)
    for i, c in enumerate(cells):
        rows = [sid2i[s] for s in c.segments if s in sid2i]
        if rows:
            v = emb[rows].mean(axis=0)
            nrm = np.linalg.norm(v)
            M[i] = v / nrm if nrm > 0 else v
    return M
