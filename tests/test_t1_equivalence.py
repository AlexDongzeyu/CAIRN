"""Equivalence check: the fast T1 evaluator must agree with the reference metrics.

The vectorised path exists purely for speed. If it drifts from the straightforward
implementation the headline retrieval numbers are wrong, so the two are compared on
randomised cases rather than assumed equal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex  # noqa: E402
from src.train import (  # noqa: E402
    average_precision, evaluate_t1, ndcg_at, recall_at, reciprocal_rank,
)


class _Seg:
    def __init__(self, sid, iid, narrs, topics):
        self.segment_id, self.interview_id = sid, iid
        self.narrators, self.interviewers = narrs, []
        self.topics = [{"term": t, "id": "0"} for t in topics]
        self.geography, self.location, self.extent = [], "", ""
        self.title, self.description = sid, "text"

    @property
    def text(self):
        return self.title


def _reference(H1, r1_index, all_sids, queries):
    ap, nd, rc, mrr = [], [], [], []
    pool_idx = np.array([r1_index[f"r1:{s}"] for s in all_sids])
    for q in queries:
        qi = r1_index[f"r1:{q['qid']}"]
        sims = H1[pool_idx] @ H1[qi]
        sims = sims.copy()
        sims[all_sids.index(q["qid"])] = -np.inf
        order = np.argsort(-sims, kind="stable")
        ranked = [all_sids[i] for i in order if all_sids[i] != q["qid"]]
        pos = set(q["positives"]) & set(all_sids)
        if not pos:
            continue
        ap.append(average_precision(ranked, pos))
        nd.append(ndcg_at(ranked, pos, 10))
        rc.append(recall_at(ranked, pos, 50))
        mrr.append(reciprocal_rank(ranked, pos))
    return (float(np.mean(ap)), float(np.mean(nd)), float(np.mean(rc)), float(np.mean(mrr)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fast_t1_matches_reference_metrics(seed):
    rng = np.random.default_rng(seed)
    n_seg = 120
    segs = [_Seg(f"s{i}", f"i{i}", [f"n{i % 30}"], ["W -- A -- x"]) for i in range(n_seg)]
    cx = build_complex(segs, {"W -- A -- x": 2}, granularity="mid")
    r1_index = cx.index(1)
    all_sids = [s.segment_id for s in segs if f"r1:{s.segment_id}" in r1_index]

    d = 16
    H1 = rng.normal(size=(len(cx.cells), d)).astype(np.float32)
    H1 /= np.linalg.norm(H1, axis=1, keepdims=True)

    queries = []
    for i in range(20):
        qid = all_sids[int(rng.integers(len(all_sids)))]
        pos = list(rng.choice([s for s in all_sids if s != qid],
                              size=int(rng.integers(1, 8)), replace=False))
        queries.append({"qid": qid, "positives": sorted(pos)})

    class _Model:
        pass

    h = {1: torch.tensor(H1)}
    fast = evaluate_t1(_Model(), h, cx, queries, {s: i for i, s in enumerate(all_sids)},
                       torch.device("cpu"))
    ref_map, ref_ndcg, ref_rec, ref_mrr = _reference(
        torch.nn.functional.normalize(torch.tensor(H1), dim=-1).numpy(), r1_index, all_sids, queries)

    assert fast["T1_map"] == pytest.approx(ref_map, abs=1e-6)
    assert fast["T1_ndcg@10"] == pytest.approx(ref_ndcg, abs=1e-6)
    assert fast["T1_recall@50"] == pytest.approx(ref_rec, abs=1e-6)
    assert fast["T1_mrr"] == pytest.approx(ref_mrr, abs=1e-6)


def test_fast_t1_never_retrieves_the_query_itself():
    segs = [_Seg(f"s{i}", f"i{i}", [f"n{i}"], ["W -- A -- x"]) for i in range(20)]
    cx = build_complex(segs, {"W -- A -- x": 2}, granularity="mid")
    r1_index = cx.index(1)
    all_sids = [s.segment_id for s in segs]
    H1 = np.zeros((len(cx.cells), 8), dtype=np.float32)
    # make the query vector uniquely self-similar so a bug would rank it first
    for s in all_sids:
        H1[r1_index[f"r1:{s}"]] = np.random.default_rng(abs(hash(s)) % 2**31).normal(size=8)
    H1 /= np.linalg.norm(H1, axis=1, keepdims=True) + 1e-9
    q = {"qid": "s0", "positives": ["s5"]}
    out = evaluate_t1(object(), {1: torch.tensor(H1)}, cx, [q],
                      {s: i for i, s in enumerate(all_sids)}, torch.device("cpu"))
    # if the query were retrieved at rank 1, MRR would be capped below 1 for the true positive
    assert 0.0 < out["T1_mrr"] <= 1.0
    assert out["T1_n_queries"] == 1
