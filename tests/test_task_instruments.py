"""Instrument verification for retrieval metrics, ASO, and the negative samplers.

Same rule as tests/test_instruments.py: nothing is quoted until it has been checked
against a case whose answer is known independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences  # noqa: E402
from src.stats import _violation_ratio, aso_matrix, stratify  # noqa: E402
from src.tasks import NegativeSampler, find_near_duplicates, make_split, partition_incidences  # noqa: E402
from src.train import (  # noqa: E402
    auc_pr, auc_roc, average_precision, ndcg_at, recall_at, reciprocal_rank,
)


# ------------------------------------------------------------------ IR metrics
def test_average_precision_known_value():
    # positives at ranks 1 and 3 -> (1/1 + 2/3)/2
    assert average_precision(["a", "b", "c", "d"], {"a", "c"}) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_perfect_and_empty():
    assert average_precision(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
    assert average_precision(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_ndcg_known_value():
    # single positive at rank 2 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert ndcg_at(["a", "b", "c"], {"b"}, k=3) == pytest.approx(1 / np.log2(3))
    assert ndcg_at(["b", "a", "c"], {"b"}, k=3) == pytest.approx(1.0)


def test_ndcg_is_rank_sensitive():
    assert ndcg_at(list("abcde"), {"a"}, 5) > ndcg_at(list("abcde"), {"e"}, 5)


def test_recall_and_rr():
    assert recall_at(list("abcde"), {"a", "z"}, k=3) == pytest.approx(0.5)
    assert reciprocal_rank(list("abc"), {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(list("abc"), {"z"}) == pytest.approx(0.0)


def test_auc_of_perfect_and_random_separation():
    y = np.array([0, 0, 1, 1])
    assert auc_roc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert auc_roc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert auc_pr(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


# ------------------------------------------------------------------ ASO
def test_aso_detects_clear_dominance():
    rng = np.random.default_rng(0)
    strong = list(rng.normal(0.9, 0.01, 10))
    weak = list(rng.normal(0.5, 0.01, 10))
    out = aso_matrix({"strong": strong, "weak": weak})
    assert out["eps_min"]["strong>weak"] < 0.5
    assert out["eps_min"]["weak>strong"] > 0.5


def test_aso_finds_no_order_for_identical_distributions():
    rng = np.random.default_rng(1)
    a = list(rng.normal(0.7, 0.05, 30))
    b = list(rng.normal(0.7, 0.05, 30))
    out = aso_matrix({"a": a, "b": b})
    assert 0.15 < out["eps_min"]["a>b"] < 0.85


def test_aso_applies_bonferroni():
    out = aso_matrix({"a": [1.0, 2, 3], "b": [1.0, 2, 3], "c": [1.0, 2, 3]})
    assert out["bonferroni_factor"] == 6
    assert out["confidence_used"] > 0.95


def test_violation_ratio_fallback_is_ordered():
    rng = np.random.default_rng(0)
    hi, lo = rng.normal(1.0, 0.01, 50), rng.normal(0.0, 0.01, 50)
    assert _violation_ratio(hi, lo) < 0.5 < _violation_ratio(lo, hi)


def test_stratify_bins_event_size():
    rows = [{"event_size": s} for s in (2, 3, 7, 30, 900)]
    out = stratify(rows, "event_size")
    assert len(out["2-3"]) == 2 and len(out["4-10"]) == 1
    assert len(out["11-50"]) == 1 and len(out["51-+"]) == 1


# ------------------------------------------------------------------ splits + negatives
class _Seg:
    def __init__(self, sid, iid, narrs, topics, text="some archive summary text here"):
        self.segment_id, self.interview_id = sid, iid
        self.narrators, self.interviewers = narrs, []
        self.topics = [{"term": t, "id": "0"} for t in topics]
        self.geography, self.location, self.extent = [], "", ""
        self.title, self.description = sid, text

    @property
    def text(self):
        return f"{self.title}. {self.description}"


def _cx(n_narr=12):
    segs = []
    for i in range(n_narr):
        segs.append(_Seg(f"s{i}", f"i{i}", [f"n{i}"], ["W -- Camps -- Food"]))
    for i in range(n_narr, n_narr + 6):
        segs.append(_Seg(f"s{i}", f"i{i}", [f"n{i}"], ["W -- Camps -- School"]))
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2}
    return build_complex(segs, rank_of, granularity="mid"), segs


def test_narrator_disjoint_split_has_no_narrator_overlap():
    cx, _ = _cx(30)
    sp = make_split(cx, "narrator-disjoint", seed=0)
    assert not (sp.train & sp.test) and not (sp.val & sp.test)
    parts = partition_incidences(cx, sp)
    train_narr = {n for n, _ in parts["train"]}
    test_narr = {n for n, _ in parts["test"]}
    assert not (train_narr & test_narr), "narrator leaked across the split boundary"


def test_event_disjoint_split_has_no_event_overlap():
    cx, _ = _cx(30)
    sp = make_split(cx, "event-disjoint", seed=0)
    parts = partition_incidences(cx, sp)
    assert not ({c for _, c in parts["train"]} & {c for _, c in parts["test"]})


def test_negative_samplers_never_return_a_true_member():
    cx, _ = _cx(20)
    ns = NegativeSampler(cx, seed=0)
    cid = sorted(cx.by_rank(2), key=lambda c: -c.size)[0].cid
    members = set(cx.cells[cid].members)
    for regime in ("UNS", "SNS", "MNS", "CNS"):
        for _ in range(25):
            neg = ns.sample(cid, regime)
            assert neg is None or neg not in members, f"{regime} produced a true member"


def test_negative_ratio_is_ten_to_one():
    cx, _ = _cx(20)
    ns = NegativeSampler(cx, seed=0)
    pos = [(n, c.cid) for c in cx.by_rank(2) for n in sorted(c.members)][:5]
    pairs, labels = ns.build(pos, "UNS", ratio=10)
    assert sum(labels) == 5
    assert len(labels) - sum(labels) == 50


def test_cns_negatives_are_structurally_closer_than_uniform():
    """CNS must be harder than UNS, or the regime is not doing its job.

    The fixture must contain narrators who co-witness the target cell's members through
    a *different* cell; otherwise no clique-negative exists and both samplers degenerate
    to the uniform fallback.
    """
    segs = []
    core = [f"n{i}" for i in range(8)]
    for i, n in enumerate(core):                       # cell X = n0..n7
        segs.append(_Seg(f"sx{i}", f"ix{i}", [n], ["W -- A -- x"]))
    for i, n in enumerate(core[:7] + ["n8", "n9", "n10"]):   # cell Y shares n0..n6
        segs.append(_Seg(f"sy{i}", f"iy{i}", [n], ["W -- A -- y"]))
    for i in range(11, 40):                            # unrelated narrators
        segs.append(_Seg(f"sf{i}", f"if{i}", [f"n{i}"], ["Q -- B -- w"]))
    rank_of = {"W -- A -- x": 2, "W -- A -- y": 2, "Q -- B -- w": 2}
    cx = build_complex(segs, rank_of, granularity="mid")
    ns = NegativeSampler(cx, seed=0)
    cid = [c.cid for c in cx.by_rank(2) if c.label.endswith("-- x")][0]
    members = set(cx.cells[cid].members)

    def co_overlap(neg):
        return len(ns.co.get(neg, set()) & members) if neg else 0

    cns = [co_overlap(ns.cns(cid)) for _ in range(40)]
    uns = [co_overlap(ns.uns(cid)) for _ in range(40)]
    assert np.mean(cns) > np.mean(uns), (np.mean(cns), np.mean(uns))


def test_mns_is_between_uns_and_cns_in_hardness():
    segs = []
    core = [f"n{i}" for i in range(8)]
    for i, n in enumerate(core):
        segs.append(_Seg(f"sx{i}", f"ix{i}", [n], ["W -- A -- x"]))
    for i, n in enumerate(core[:7] + ["n8", "n9", "n10"]):
        segs.append(_Seg(f"sy{i}", f"iy{i}", [n], ["W -- A -- y"]))
    for i in range(11, 40):
        segs.append(_Seg(f"sf{i}", f"if{i}", [f"n{i}"], ["Q -- B -- w"]))
    cx = build_complex(segs, {"W -- A -- x": 2, "W -- A -- y": 2, "Q -- B -- w": 2},
                       granularity="mid")
    ns = NegativeSampler(cx, seed=1)
    cid = [c.cid for c in cx.by_rank(2) if c.label.endswith("-- x")][0]
    members = set(cx.cells[cid].members)

    def ov(neg):
        return len(ns.co.get(neg, set()) & members) if neg else 0

    assert np.mean([ov(ns.mns(cid)) for _ in range(40)]) >= np.mean([ov(ns.uns(cid)) for _ in range(40)])


def test_near_duplicate_detection_finds_a_planted_copy():
    text = "we were sent to the assembly center and slept in a horse stall for weeks on end"
    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- A -- x"], text),
        _Seg("s2", "i2", ["n2"], ["W -- A -- x"], text),          # cross-narrator copy
        _Seg("s3", "i3", ["n3"], ["W -- A -- x"], "entirely different content about fishing boats in cold water"),
    ]
    dups = find_near_duplicates(segs, threshold=0.8)
    assert ("s1", "s2") in dups
    assert not any("s3" in p for p in dups)


def test_near_duplicate_rejects_lsh_candidates_below_threshold():
    """LSH recall slack must not be counted as duplication."""
    a = "we were sent to the assembly center and slept in a horse stall for many weeks"
    b = "we were sent to the assembly center but my father stayed behind in the city alone somewhere else entirely"
    segs = [_Seg("s1", "i1", ["n1"], ["W -- A -- x"], a),
            _Seg("s2", "i2", ["n2"], ["W -- A -- x"], b)]
    assert find_near_duplicates(segs, threshold=0.9) == []


def test_near_duplicate_rate_is_sane_on_distinct_texts():
    """A detector that flags most of the corpus is broken, not informative."""
    segs = [
        _Seg(f"s{i}", f"i{i}", [f"n{i}"], ["W -- A -- x"],
             f"narrator number {i} describes working on the family farm during season {i} of the war years")
        for i in range(60)
    ]
    dups = find_near_duplicates(segs, threshold=0.8)
    assert len(dups) < 0.1 * len(segs), f"implausible duplicate rate: {len(dups)} for {len(segs)} segments"


# ------------------------------------------------------------------ leakage control
def test_masking_removes_exactly_the_held_out_incidences():
    cx, _ = _cx(24)
    sp = make_split(cx, "narrator-disjoint", seed=0)
    parts = partition_incidences(cx, sp)
    train_cx = mask_incidences(cx, set(parts["train"]))

    train_pairs = set(parts["train"])
    for c in train_cx.by_rank(2):
        for n in c.members:
            assert (n, c.cid) in train_pairs, "a held-out incidence survived masking"
    test_narr = {n for n, _ in parts["test"]}
    surviving = {n for c in train_cx.by_rank(2) for n in c.members}
    assert not (surviving & test_narr), "test narrator still visible to message passing"


def test_masking_leaves_rank1_untouched():
    """Which narrator gave a segment is never in question; only incidence is predicted."""
    cx, _ = _cx(24)
    sp = make_split(cx, "narrator-disjoint", seed=0)
    train_cx = mask_incidences(cx, set(partition_incidences(cx, sp)["train"]))
    before = {c.cid: c.members for c in cx.by_rank(1)}
    after = {c.cid: c.members for c in train_cx.by_rank(1)}
    assert before == after


def test_masking_actually_removes_something():
    """Guards against a mask that silently keeps everything."""
    cx, _ = _cx(24)
    sp = make_split(cx, "narrator-disjoint", seed=0)
    train_cx = mask_incidences(cx, set(partition_incidences(cx, sp)["train"]))
    before = sum(c.size for c in cx.by_rank(2))
    after = sum(c.size for c in train_cx.by_rank(2))
    assert after < before, (before, after)


def test_near_duplicate_ignores_same_narrator_repeats():
    text = "we were sent to the assembly center and slept in a horse stall for weeks on end"
    segs = [_Seg("s1", "i1", ["n1"], ["W -- A -- x"], text),
            _Seg("s2", "i1", ["n1"], ["W -- A -- x"], text)]
    assert find_near_duplicates(segs, threshold=0.8) == []
