"""E5 — tasks, leakage-resistant splits, and negative sampling.

T1 corroboration retrieval (primary), T2 incidence prediction, T3 attestation regression.

Two hazards the protocol singles out are handled here explicitly:
  * circularity  - T1 ground truth comes from the same archive labelling the model sees,
                   so the query cell's membership is masked at inference and the
                   archive-curated evaluation is reported separately from any automatic one.
  * leakage      - narrator-disjoint is the primary regime; near-duplicate passages are
                   detected and logged (never silently dropped) before splitting.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

SPLITS = ("random", "narrator-disjoint", "event-disjoint")
NEG_REGIMES = ("UNS", "SNS", "MNS", "CNS", "hard")


# ------------------------------------------------------------------ near duplicates
def find_near_duplicates(segments, threshold: float = 0.8, num_perm: int = 128) -> list[tuple[str, str]]:
    """MinHash/LSH over 5-gram shingles. Oral history has genuine cross-narrator
    near-duplicates (relatives recounting one story); they are reported, not deleted.

    LSH returns *candidates*, not matches. Every candidate is re-checked against the
    actual MinHash Jaccard estimate, otherwise the count is dominated by LSH recall
    slack and vastly overstates duplication. Only the archive's content summary is
    compared - segment titles are boilerplate ("<Narrator> Segment 7") and would both
    inflate and distort the similarity.
    """
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    mh: dict[str, object] = {}
    for s in segments:
        content = (getattr(s, "description", "") or "").strip() or s.text
        toks = content.lower().split()
        if len(toks) < 8:  # too short for a 5-gram signature to mean anything
            continue
        m = MinHash(num_perm=num_perm)
        for i in range(len(toks) - 4):
            m.update(" ".join(toks[i:i + 5]).encode("utf-8"))
        mh[s.segment_id] = m
        lsh.insert(s.segment_id, m)

    owner = {s.segment_id: set(s.narrators) for s in segments}
    pairs: set[tuple[str, str]] = set()
    for sid, m in mh.items():
        for other in lsh.query(m):
            if other == sid:
                continue
            if owner.get(sid, set()) & owner.get(other, set()):
                continue  # a narrator repeating themselves is not a leak
            if m.jaccard(mh[other]) < threshold:
                continue  # LSH candidate that does not actually clear the threshold
            pairs.add(tuple(sorted((sid, other))))  # type: ignore[arg-type]
    return sorted(pairs)


# ------------------------------------------------------------------ splits
@dataclass
class Split:
    name: str
    train: set[str]
    val: set[str]
    test: set[str]
    unit: str                       # 'narrator' | 'event' | 'incidence'
    crossing_duplicates: int = 0
    notes: list[str] = field(default_factory=list)


def make_split(cx, kind: str, seed: int = 0, fracs=(0.70, 0.15, 0.15)) -> Split:
    rng = np.random.default_rng(seed)
    if kind == "narrator-disjoint":
        units = list(cx.narrators)
        unit = "narrator"
    elif kind == "event-disjoint":
        units = [c.cid for c in cx.by_rank(2)]
        unit = "event"
    elif kind == "random":
        units = [f"{c.cid}||{n}" for c in cx.by_rank(2) for n in sorted(c.members)]
        unit = "incidence"
    else:
        raise ValueError(kind)

    idx = rng.permutation(len(units))
    n_tr = int(fracs[0] * len(units))
    n_va = int(fracs[1] * len(units))
    tr = {units[i] for i in idx[:n_tr]}
    va = {units[i] for i in idx[n_tr:n_tr + n_va]}
    te = {units[i] for i in idx[n_tr + n_va:]}
    return Split(kind, tr, va, te, unit)


def incidences(cx) -> list[tuple[str, str]]:
    return [(n, c.cid) for c in cx.by_rank(2) for n in sorted(c.members)]


def partition_incidences(cx, split: Split) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {"train": [], "val": [], "test": []}
    for n, cid in incidences(cx):
        if split.unit == "narrator":
            key = n
        elif split.unit == "event":
            key = cid
        else:
            key = f"{cid}||{n}"
        bucket = "train" if key in split.train else "val" if key in split.val else "test"
        out[bucket].append((n, cid))
    return out


# ------------------------------------------------------------------ negative sampling
class NegativeSampler:
    """The four regimes of Patil, Sharma & Murty (PAKDD 2020) plus semantic hard negatives.

    They are ordered by how much realistic structure the negative retains:
    UNS/SNS ignore local structure and are known to inflate scores; MNS and CNS preserve
    local density and are the regimes on which a predictor has to actually generalize.
    """

    def __init__(self, cx, cell_emb: np.ndarray | None = None, seed: int = 0):
        self.cx = cx
        self.rng = np.random.default_rng(seed)
        self.narrators = list(cx.narrators)
        self._narr_arr = np.array(self.narrators, dtype=object)
        self.cells = sorted(cx.by_rank(2), key=lambda c: c.cid)
        self.cid2i = {c.cid: i for i, c in enumerate(self.cells)}
        self.cell_emb = cell_emb

        # narrator co-witness graph = clique expansion of the rank-2 cells
        self.co: dict[str, set[str]] = defaultdict(set)
        for c in self.cells:
            ms = list(c.members)
            for a in ms:
                self.co[a] |= set(ms)
        for a in self.co:
            self.co[a].discard(a)
        self.narr_emb: dict[str, np.ndarray] = {}

    def _uniform_outside(self, members: set[str]) -> str | None:
        """Rejection sampling: an O(N) filtered list per draw dominates runtime once the
        ground set passes a few hundred narrators."""
        if len(members) >= len(self._narr_arr):
            return None
        for _ in range(32):
            cand = str(self._narr_arr[self.rng.integers(len(self._narr_arr))])
            if cand not in members:
                return cand
        pool = [n for n in self.narrators if n not in members]
        return str(self.rng.choice(pool)) if pool else None

    def uns(self, cid: str) -> str | None:
        """Uniform NS: a narrator drawn uniformly from the ground set."""
        return self._uniform_outside(set(self.cx.cells[cid].members))

    def sns(self, cid: str) -> str | None:
        """Sized NS: uniform draw within a size-matched fake edge. Controls cardinality
        only, and is known to inflate scores; reported as the easy condition."""
        members = set(self.cx.cells[cid].members)
        n_pool = len(self._narr_arr) - len(members)
        if n_pool <= 0:
            return None
        take = min(len(members), n_pool)
        fake: set[str] = set()
        guard = 0
        while len(fake) < take and guard < 20 * take + 50:
            guard += 1
            cand = str(self._narr_arr[self.rng.integers(len(self._narr_arr))])
            if cand not in members:
                fake.add(cand)
        return str(self.rng.choice(sorted(fake))) if fake else None

    def mns(self, cid: str, walk: int = 3) -> str | None:
        """Motif NS: a short random walk on the clique expansion, so the negative sits in
        a neighbourhood with realistic local density."""
        members = set(self.cx.cells[cid].members)
        if not members:
            return None
        cur = str(self.rng.choice(sorted(members)))
        for _ in range(walk):
            nbrs = sorted(self.co.get(cur, set()))
            if not nbrs:
                break
            cur = str(self.rng.choice(nbrs))
        return cur if cur not in members else self._uniform_outside(members)

    def cns(self, cid: str) -> str | None:
        """Clique NS: remove one member and replace it with a narrator co-witnessing ALL
        the others. Hardest of the four - the negative is indistinguishable by local
        structure alone."""
        members = set(self.cx.cells[cid].members)
        if len(members) < 2:
            return self.mns(cid)
        ms = sorted(members)
        drop = str(self.rng.choice(ms))
        keep = [m for m in ms if m != drop]
        cands = set.intersection(*[self.co.get(m, set()) for m in keep]) if keep else set()
        cands -= members
        if not cands:
            return self.mns(cid)
        return str(self.rng.choice(sorted(cands)))

    def hard(self, cid: str, narr_emb: dict[str, np.ndarray] | None = None) -> str | None:
        """Semantic hard negative: the most similar narrator that is NOT incident.
        These are near-misses that lexical matching cannot separate."""
        emb = narr_emb or self.narr_emb
        members = set(self.cx.cells[cid].members)
        if not emb:
            return self.cns(cid)
        i = self.cid2i.get(cid)
        if self.cell_emb is None or i is None:
            return self.cns(cid)
        target = self.cell_emb[i]
        best, best_s = None, -np.inf
        for n, v in emb.items():
            if n in members:
                continue
            s = float(v @ target)
            if s > best_s:
                best, best_s = n, s
        return best or self._uniform_outside(members)

    def sample(self, cid: str, regime: str) -> str | None:
        return {"UNS": self.uns, "SNS": self.sns, "MNS": self.mns,
                "CNS": self.cns, "hard": self.hard}[regime](cid)

    def build(self, positives: list[tuple[str, str]], regime: str, ratio: int = 10
              ) -> tuple[list[tuple[str, str]], list[int]]:
        """Protocol E5.1: positive:negative = 1:10 for T2."""
        pairs, labels = [], []
        for n, cid in positives:
            pairs.append((n, cid)); labels.append(1)
            for _ in range(ratio):
                neg = self.sample(cid, regime)
                if neg is None:
                    continue
                pairs.append((neg, cid)); labels.append(0)
        return pairs, labels


# ------------------------------------------------------------------ T1
def build_t1_queries(cx, segments, split: Split, bucket: str = "test",
                     limit: int | None = None, seed: int = 0) -> list[dict]:
    """T1: given a held-out passage from narrator n, retrieve passages by OTHER narrators
    incident to the same rank-2 cell.

    The query passage's own cell membership is masked, so a model cannot read the answer
    off the structure it is given.

    Queries are generated in a fixed shuffled order and generation stops once `limit` is
    reached. Building every query and slicing afterwards is quadratic in cell size, which
    dominates runtime at coarse granularity where single cells span thousands of segments.
    """
    seg_by_id = {s.segment_id: s for s in segments}
    cell_of_seg: dict[str, list[str]] = defaultdict(list)
    for c in cx.by_rank(2):
        for sid in c.segments:
            cell_of_seg[sid].append(c.cid)

    target = (split.train if bucket == "train"
              else split.val if bucket == "val" else split.test)

    def in_bucket(narr: str, cid: str) -> bool:
        key = narr if split.unit == "narrator" else cid if split.unit == "event" else f"{cid}||{narr}"
        return key in target

    # narrators of each segment, precomputed once
    narr_of = {sid: set(s.narrators) for sid, s in seg_by_id.items()}

    order = sorted(cell_of_seg)
    np.random.default_rng(seed).shuffle(order)

    queries = []
    for sid in order:
        if limit is not None and len(queries) >= limit:
            break
        cids = cell_of_seg[sid]
        s = seg_by_id.get(sid)
        if not s or not s.narrators:
            continue
        qn = s.narrators[0]
        if not any(in_bucket(qn, cid) for cid in cids):
            continue
        qnarr = narr_of.get(sid, set())
        positives = set()
        for cid in cids:
            for other_sid in cx.cells[cid].segments:
                if other_sid == sid:
                    continue
                if not (narr_of.get(other_sid, set()) & qnarr):
                    positives.add(other_sid)
        if positives:
            queries.append({"qid": sid, "narrator": qn, "cells": cids,
                            "positives": sorted(positives), "masked_cells": cids})
    return queries


def t3_targets(cx) -> tuple[list[str], np.ndarray]:
    cells = sorted(cx.by_rank(2), key=lambda c: c.cid)
    return [c.cid for c in cells], np.array([c.size for c in cells], dtype=np.float32)
