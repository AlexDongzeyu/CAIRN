"""Instrument verification.

A measurement whose instrument has not been checked is not a measurement. Every
estimator whose reading we intend to quote is asserted against a case with a known
answer BEFORE it is pointed at the corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import (  # noqa: E402
    Cell, Complex, build_complex, rank_cardinality_decoupling, referent,
    support_injectivity, verify_star_roundtrip,
)
from src.corpus import clean_description, find_boilerplate, strip_markup  # noqa: E402
from src.metrics import (  # noqa: E402
    cluster_bootstrap, cluster_subsample_ci, design_effect, jaccard_at, rbo,
)
from src.ontology import (  # noqa: E402
    agreement_diagnostics, build_topic_tree, krippendorff_ordinal, q4_archive_practice, resolve,
)


# ---------------------------------------------------------------- RBO
def test_rbo_identical_lists_is_one():
    L = list("abcdefghij")
    for p in (0.8, 0.9, 0.95):
        assert rbo(L, L, p=p) == pytest.approx(1.0, abs=1e-9)


def test_rbo_disjoint_lists_is_zero():
    assert rbo(list("abcde"), list("fghij"), p=0.9) == pytest.approx(0.0, abs=1e-12)


def test_rbo_is_symmetric_and_bounded():
    a, b = list("abcdef"), list("cbadfe")
    assert rbo(a, b, p=0.9) == pytest.approx(rbo(b, a, p=0.9), abs=1e-9)
    assert 0.0 <= rbo(a, b, p=0.9) <= 1.0


def test_rbo_is_top_weighted():
    """Swapping the top two must cost more than swapping ranks 9 and 10."""
    base = list("abcdefghij")
    top_swap = ["b", "a"] + base[2:]
    deep_swap = base[:8] + ["j", "i"]
    assert rbo(base, top_swap, p=0.9) < rbo(base, deep_swap, p=0.9)


def test_rbo_prefix_agreement_beats_suffix_agreement():
    base = list("abcdefghij")
    shared_head = list("abcde") + list("zyxwv")
    shared_tail = list("zyxwv") + list("fghij")
    assert rbo(base, shared_head, p=0.9) > rbo(base, shared_tail, p=0.9)


def test_jaccard_at_k():
    assert jaccard_at(list("abcd"), list("abef"), 2) == pytest.approx(1.0)
    assert jaccard_at(list("abcd"), list("efgh"), 4) == pytest.approx(0.0)


# ------------------------------------------------------- Krippendorff alpha
def test_alpha_perfect_agreement_is_one():
    m = np.array([[2, 3, 2, 3, 2], [2, 3, 2, 3, 2], [2, 3, 2, 3, 2]], dtype=float)
    assert krippendorff_ordinal(m) == pytest.approx(1.0, abs=1e-9)


def test_alpha_systematic_disagreement_is_low():
    m = np.array([[2] * 6, [3] * 6, [2, 3, 2, 3, 2, 3]], dtype=float)
    assert krippendorff_ordinal(m) < 0.3


def test_alpha_is_not_accidentally_high_on_noise():
    rng = np.random.default_rng(0)
    m = rng.choice([2.0, 3.0], size=(3, 400))
    assert abs(krippendorff_ordinal(m)) < 0.15


def test_diagnostics_flag_the_kappa_paradox():
    """Alpha near zero with ~90% raw agreement must be diagnosable, not quoted alone."""
    terms = [f"t{i}" for i in range(100)]
    # Both annotators say "2" almost always and agree on 90 of 100 items, but the
    # marginal is so lopsided that chance-corrected agreement collapses.
    a = {t: 2 for t in terms}
    b = {t: 2 for t in terms}
    for t in terms[:5]:
        a[t] = 3
    for t in terms[5:10]:
        b[t] = 3
    labels = {"A": a, "B": b}
    d = agreement_diagnostics(labels, terms)
    assert d["mean_pairwise_raw_agreement"] == pytest.approx(0.90)
    assert d["max_marginal_skew"] > 0.6
    assert d["paradox_suspected"] is True
    assert krippendorff_ordinal(np.array([[a[t] for t in terms], [b[t] for t in terms]], float)) < 0.2


def test_diagnostics_do_not_cry_paradox_on_balanced_disagreement():
    terms = [f"t{i}" for i in range(100)]
    a = {t: (2 if i % 2 == 0 else 3) for i, t in enumerate(terms)}
    b = {t: (3 if i % 2 == 0 else 2) for i, t in enumerate(terms)}
    d = agreement_diagnostics({"A": a, "B": b}, terms)
    assert d["mean_pairwise_raw_agreement"] == pytest.approx(0.0)
    assert d["paradox_suspected"] is False


def test_diagnostics_perfect_agreement():
    terms = [f"t{i}" for i in range(50)]
    lab = {t: (2 if i % 2 else 3) for i, t in enumerate(terms)}
    d = agreement_diagnostics({"A": dict(lab), "B": dict(lab), "C": dict(lab)}, terms)
    assert d["unanimous_fraction"] == pytest.approx(1.0)
    assert d["gwet_ac1"] == pytest.approx(1.0)
    assert d["fleiss_kappa"] == pytest.approx(1.0)


# ------------------------------------------------------- clustered bootstrap
def test_cluster_bootstrap_covers_a_known_mean():
    rng = np.random.default_rng(0)
    units = list(rng.normal(5.0, 1.0, 400))
    out = cluster_bootstrap(lambda u: float(np.mean(u)), units, B=800, seed=1, bca=False)
    assert out["lo"] < 5.0 < out["hi"]
    assert out["lo"] < out["point"] < out["hi"]


def test_subsample_ci_contains_its_own_point_estimate():
    """A CI that excludes the statistic it describes is reporting a different statistic.

    The singleton fraction is exactly such a case under with-replacement resampling:
    duplicated narrators inflate cell attestation and destroy singletons.
    """
    rng = np.random.default_rng(0)
    narrators = [f"n{i}" for i in range(120)]
    cells = [frozenset(rng.choice(narrators, size=int(rng.integers(1, 9)), replace=False))
             for _ in range(200)]

    def singleton_fraction(units):
        keep = set(units)
        vals = [len(c & keep) for c in cells]
        vals = [v for v in vals if v > 0]
        return float(np.mean([v == 1 for v in vals])) if vals else 0.0

    sub = cluster_subsample_ci(singleton_fraction, narrators, B=400, seed=0)
    assert sub["lo"] <= sub["point"] <= sub["hi"], sub
    assert sub["hi"] > sub["lo"], "degenerate interval"


def test_with_replacement_bootstrap_would_have_been_biased_here():
    """Documents the failure the subsampling estimator exists to avoid.

    Fixture mirrors the observed corpus: a minority of singleton cells among many large
    ones. Under with-replacement resampling a true singleton {n} receives multiplicity
    ~Poisson(1), so it survives as a singleton only ~37% of the time while large cells
    essentially never become singletons. The interval then sits below the statistic it
    claims to describe.
    """
    rng = np.random.default_rng(1)
    narrators = [f"n{i}" for i in range(470)]
    cells = [frozenset([narrators[i]]) for i in range(29)]                       # singletons
    cells += [frozenset(rng.choice(narrators, size=int(rng.integers(4, 40)), replace=False))
              for _ in range(112)]                                               # large cells

    def sf_multiplicity(units):
        mult: dict[str, int] = {}
        for u in units:
            mult[u] = mult.get(u, 0) + 1
        vals = [sum(mult.get(n, 0) for n in c) for c in cells]
        vals = [v for v in vals if v > 0]
        return float(np.mean([v == 1 for v in vals])) if vals else 0.0

    def sf_distinct(units):
        keep = set(units)
        vals = [len(c & keep) for c in cells]
        vals = [v for v in vals if v > 0]
        return float(np.mean([v == 1 for v in vals])) if vals else 0.0

    point = sf_distinct(narrators)
    boot = cluster_bootstrap(sf_multiplicity, narrators, B=300, seed=0, bca=False)
    sub = cluster_subsample_ci(sf_distinct, narrators, B=300, seed=0)

    assert not (boot["lo"] <= point <= boot["hi"]), (
        "expected the with-replacement interval to exclude the point estimate", boot, point)
    assert sub["lo"] <= point <= sub["hi"], ("subsampling interval must contain it", sub, point)


def test_cluster_bootstrap_is_wider_than_naive_when_clustered():
    """The whole point of the estimator: correlated units must widen the interval."""
    rng = np.random.default_rng(0)
    clusters = [list(rng.normal(loc, 0.05, 20)) for loc in rng.normal(5.0, 1.0, 60)]
    flat = [x for c in clusters for x in c]
    clustered = cluster_bootstrap(lambda u: float(np.mean([x for c in u for x in c])),
                                  clusters, B=600, seed=0, bca=False)
    naive = cluster_bootstrap(lambda u: float(np.mean(u)), flat, B=600, seed=0, bca=False)
    assert (clustered["hi"] - clustered["lo"]) > (naive["hi"] - naive["lo"])


def test_design_effect_detects_clustering():
    rng = np.random.default_rng(0)
    clustered = [list(rng.normal(loc, 0.05, 20)) for loc in rng.normal(0, 1, 40)]
    independent = [list(rng.normal(0, 1, 20)) for _ in range(40)]
    assert design_effect([20] * 40, clustered)["DEFF"] > 5.0
    assert design_effect([20] * 40, independent)["DEFF"] < 2.0


# ------------------------------------------------------- decision procedure
def test_resolve_follows_the_manual():
    assert resolve(True, False, "A -- B -- C") == 2
    assert resolve(False, True, "A -- B -- C") == 3
    # both/neither -> Q4 tie-break by archive vocabulary depth
    assert resolve(True, True, "A") == 3
    assert resolve(False, False, "A -- B -- C") == 2
    assert q4_archive_practice("World War II") == 3
    assert q4_archive_practice("A -- B -- C -- D") == 2


def test_topic_tree_records_containment():
    tree = build_topic_tree(["A -- B -- C", "A -- D"])
    assert "A -- B" in tree["A"] and "A -- D" in tree["A"]
    assert "A -- B -- C" in tree["A -- B"]
    assert "A -- B -- C" not in tree  # a leaf has no children


# ------------------------------------------------------- complex machinery
class _Seg:
    def __init__(self, sid, iid, narrs, topics, geo=None):
        self.segment_id, self.interview_id = sid, iid
        self.narrators, self.interviewers = narrs, []
        self.topics = [{"term": t, "id": str(i)} for i, t in enumerate(topics)]
        self.geography = [{"term": g} for g in (geo or [])]
        self.title, self.description, self.location, self.extent = sid, "", "", ""

    @property
    def text(self):
        return self.title


def _toy():
    return [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"], ["Seattle"]),
        _Seg("s2", "i2", ["n2"], ["W -- Camps -- Food"], ["Portland"]),
        _Seg("s3", "i3", ["n3"], ["W -- Camps -- School"], ["Seattle"]),
        _Seg("s4", "i4", ["n4"], ["W -- Propaganda"], ["Seattle"]),
    ]


def test_star_expansion_is_lossless():
    segs = _toy()
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2, "W": 3}
    cx = build_complex(segs, rank_of, granularity="mid")
    out = verify_star_roundtrip(cx)
    assert out["lossless"], out


_RANKS = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2, "W": 3}


def test_support_injectivity_holds_when_every_moment_has_its_own_owner():
    cx = build_complex(_toy(), _RANKS, granularity="mid")
    out = support_injectivity(cx, 1)
    assert out["supp_injective"] is True
    assert out["n_distinct_supports"] == out["n_cells"]
    assert out["collapse_factor"] == pytest.approx(1.0)


def test_support_injectivity_flags_two_moments_sharing_one_owner():
    """Two segments by one narrator are distinct cells carrying the same support."""
    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"], ["Seattle"]),
        _Seg("s2", "i1", ["n1"], ["W -- Camps -- School"], ["Seattle"]),
        _Seg("s3", "i2", ["n2"], ["W -- Propaganda"], ["Seattle"]),
    ]
    out = support_injectivity(build_complex(segs, _RANKS, granularity="mid"), 1)
    assert out["n_cells"] == 3
    assert out["n_distinct_supports"] == 2
    assert out["max_multiplicity"] == 2
    assert out["n_cells_beyond_distinct"] == 1
    assert out["supp_injective"] is False
    assert out["collapse_factor"] == pytest.approx(1.5)


def test_training_side_membership_is_per_narrator_not_per_segment():
    """A shared segment is training-side for one speaker and held out for the other.

    Making this a property of the segment instead drops a narrator's own evidence because of
    a co-speaker's split assignment. That bug put 3 of 470 narrators on the fallback path
    under a narrator-disjoint split, where roughly a third belong there.
    """
    from src.tasks import Split

    from experiments.e_split_first import training_side_by_narrator

    segs = [
        _Seg("s1", "i1", ["n1", "n2"], ["W -- Camps -- Food"], ["Seattle"]),
        _Seg("s2", "i2", ["n1"], ["W -- Camps -- School"], ["Seattle"]),
    ]
    cx = build_complex(segs, _RANKS, granularity="mid")
    sp = Split("narrator-disjoint", train={"n1"}, val=set(), test={"n2"}, unit="narrator")
    allowed = training_side_by_narrator(cx, segs, sp)

    assert "s1" in allowed["n1"], "n1 is in train, so the shared segment is theirs to use"
    assert "s1" not in allowed.get("n2", set()), "n2 is held out, so the same segment is not"
    assert "s2" in allowed["n1"]


def test_build_features_restricts_to_allowed_segments_and_falls_back(monkeypatch):
    """The rank-0 mean uses only permitted segments, and an empty permission set falls back."""
    import src.features as feat_mod

    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"], ["Seattle"]),
        _Seg("s2", "i2", ["n1"], ["W -- Camps -- School"], ["Seattle"]),
        _Seg("s3", "i3", ["n2"], ["W -- Propaganda"], ["Seattle"]),
    ]
    emb = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    monkeypatch.setattr(feat_mod, "segment_embeddings",
                        lambda s, encoder=None: (emb, {"s1": 0, "s2": 1, "s3": 2}))

    cx = build_complex(segs, _RANKS, granularity="mid")
    idx = {n: i for i, n in enumerate(cx.narrators)}

    full = feat_mod.build_features(cx, segs)
    assert full[0][idx["n1"], 0] == pytest.approx(2.0), "unrestricted mean of 1.0 and 3.0"

    # n1 may use only s1; n2 is given no permitted segment at all and must fall back.
    restricted = feat_mod.build_features(cx, segs, allowed_by_narrator={"n1": {"s1"}})
    assert restricted[0][idx["n1"], 0] == pytest.approx(1.0)
    assert restricted[0][idx["n2"], 1] == pytest.approx(5.0), "fallback to the narrator's own"
    assert feat_mod.build_features.last_fallback == (1, 2)


def test_rank_is_not_cardinality():
    """A rank-2 cell must be able to dwarf a rank-3 cell, or the premise is false."""
    segs = _toy() + [_Seg(f"s{i}", f"i{i}", [f"n{i}"], ["W -- Camps -- Food"]) for i in range(10, 40)]
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2,
               "Tiny -- Thing": 3}
    segs.append(_Seg("sz", "iz", ["nz"], ["Tiny -- Thing"]))
    cx = build_complex(segs, rank_of, granularity="mid")
    out = rank_cardinality_decoupling(cx, ranks=(2, 3))
    assert out["extremes"]["rank2"]["largest"]["size"] > out["extremes"]["rank3"]["smallest"]["size"]
    assert out["spearman_rho"] < 0.9


def test_granularity_merges_and_splits_monotonically():
    segs = _toy()
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2}
    n = {g: len(build_complex(segs, rank_of, granularity=g).by_rank(2))
         for g in ("coarse", "mid", "fine")}
    assert n["coarse"] <= n["mid"] <= n["fine"], n


def test_referent_projection_makes_lists_conjoint():
    """Raw labels are disjoint across granularities; referents must not be."""
    segs = _toy()
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2}
    cxs = {g: build_complex(segs, rank_of, granularity=g) for g in ("coarse", "mid", "fine")}
    raw = {g: {c.label for c in cxs[g].by_rank(2)} for g in cxs}
    ref = {g: {referent(c) for c in cxs[g].by_rank(2)} for g in cxs}
    assert not (raw["coarse"] & raw["fine"]), "raw labels unexpectedly overlap"
    assert ref["coarse"] & ref["fine"], "referent projection failed to make lists conjoint"


def test_attestation_counts_distinct_narrators_only():
    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"]),
        _Seg("s2", "i1", ["n1"], ["W -- Camps -- Food"]),  # same narrator, second segment
        _Seg("s3", "i2", ["n2"], ["W -- Camps -- Food"]),
    ]
    cx = build_complex(segs, {"W -- Camps -- Food": 2}, granularity="mid")
    cell = cx.by_rank(2)[0]
    assert cell.size == 2, "repeat mentions by one narrator must not inflate attestation"


def test_interviewers_excluded_from_attestation_by_default():
    s = _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"])
    s.interviewers = ["iv1", "iv2"]
    base = build_complex([s], {"W -- Camps -- Food": 2}, granularity="mid")
    incl = build_complex([s], {"W -- Camps -- Food": 2}, granularity="mid", include_interviewers=True)
    assert base.by_rank(2)[0].size == 1
    assert incl.by_rank(2)[0].size == 3


def test_incidence_shapes_and_content():
    segs = _toy()
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2, "W": 3}
    cx = build_complex(segs, rank_of, granularity="mid")
    B01 = cx.incidence_matrix(0, 1)
    assert B01.shape == (len(cx.narrators), len(cx.by_rank(1)))
    assert B01.sum() == sum(c.size for c in cx.by_rank(1))
    B12 = cx.incidence_matrix(1, 2)
    assert B12.shape == (len(cx.by_rank(1)), len(cx.by_rank(2)))
    assert B12.sum() > 0


# ---------------------------------------------------------- corpus text hygiene
def test_markup_is_stripped():
    assert strip_markup("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_markup("a &amp; b") == "a & b"


def test_boilerplate_is_detected_by_frequency_not_hardcoded():
    """Archive furniture repeated across records must be found without naming it."""
    notice = ("This material is based upon work assisted by a grant from the "
              "Department of the Interior National Park Service.")
    texts = [f"Segment about topic {i}. {notice}" for i in range(200)]
    texts += [f"A unique description number {i} with plenty of distinct words in it." for i in range(50)]
    found = find_boilerplate(texts, min_share=0.005)
    assert any("Department of the Interior" in f for f in found)
    assert not any("unique description" in f for f in found)


def test_cleaning_removes_boilerplate_but_keeps_content():
    notice = "This material is based upon work assisted by a grant from the Department."
    raw = f"<p>Father's family background{notice}</p>"
    cleaned = clean_description(raw, (notice,))
    assert "Father's family background" in cleaned
    assert "grant" not in cleaned


def test_boilerplate_removal_collapses_the_duplicate_rate():
    """The contamination this guards against inflated duplicates by ~40x."""
    from src.tasks import find_near_duplicates

    notice = ("This material is based upon work assisted by a grant from the Department of "
              "the Interior National Park Service and administered by the office of grants.")

    class S:
        def __init__(self, sid, desc):
            self.segment_id, self.interview_id = sid, sid
            self.narrators, self.interviewers = [f"n{sid}"], []
            self.topics, self.geography = [], []
            self.location = self.extent = ""
            self.title, self.description = sid, desc

        @property
        def text(self):
            return f"{self.title}. {self.description}"

    dirty = [S(f"s{i}", f"Short topic {i}. {notice}") for i in range(40)]
    clean = [S(f"s{i}", clean_description(f"Short topic {i}. {notice}", (notice,)))
             for i in range(40)]
    assert len(find_near_duplicates(dirty, 0.8)) > 10 * max(1, len(find_near_duplicates(clean, 0.8)))


# --- the log is evidence; it must survive what the console cannot render -------------

def test_inversion_and_concordance_have_the_right_polarity():
    """A rate can be exactly right and still be reported as its own opposite.

    Every rank-3 cell here is SMALLER than every rank-2 cell, so by the stated definition
    ("the higher-ranked cell is the smaller one") every cross-rank pair is an inversion and
    none is concordant. The original implementation returned 0.0 here while the field was
    named `rank_inversion_rate`, and the paper read that number aloud as inversions.
    """
    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"]),
        _Seg("s2", "i2", ["n2"], ["W -- Camps -- Food"]),
        _Seg("s3", "i3", ["n3"], ["W -- Camps -- Food"]),
        _Seg("s4", "i4", ["n4"], ["W -- Propaganda"]),
    ]
    # rank2 "Camps -- Food" holds 3 narrators; rank3 "Propaganda" holds 1.
    rank_of = {"W -- Camps -- Food": 2, "W -- Propaganda": 3}
    cx = build_complex(segs, rank_of, granularity="mid")
    out = rank_cardinality_decoupling(cx, ranks=(2, 3))

    assert out["cross_rank_pairs"] == 1
    assert out["rank_inversion_rate"] == 1.0, "higher rank is smaller: that is an inversion"
    assert out["size_concordance_rate"] == 0.0
    assert out["rank_inversion_rate"] + out["size_concordance_rate"] + out["tie_rate"] == 1.0


def test_concordance_polarity_on_a_properly_nested_ladder():
    """The mirror image: every higher-ranked cell is larger, so nothing inverts."""
    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"]),
        _Seg("s2", "i2", ["n2"], ["W -- Propaganda"]),
        _Seg("s3", "i3", ["n3"], ["W -- Propaganda"]),
        _Seg("s4", "i4", ["n4"], ["W -- Propaganda"]),
    ]
    rank_of = {"W -- Camps -- Food": 2, "W -- Propaganda": 3}
    cx = build_complex(segs, rank_of, granularity="mid")
    out = rank_cardinality_decoupling(cx, ranks=(2, 3))
    assert out["size_concordance_rate"] == 1.0
    assert out["rank_inversion_rate"] == 0.0


def test_logging_survives_characters_the_console_cannot_encode(tmp_path, monkeypatch):
    """A completed run must not be destroyed by its own progress echo.

    Crossref returns titles containing U+2010; printing one to this machine's cp1252
    console raised UnicodeEncodeError and killed a citation run that had already
    verified eleven references. The file is UTF-8 and must record the text exactly;
    only the terminal echo is allowed to degrade.
    """
    import io

    from src import logutil

    monkeypatch.setattr(logutil, "LOGDIR", tmp_path)
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    log = logutil.make_logger("unicode_probe")
    exotic = "inter\u2010rater reliability \u2014 \u03b1 = 0.039"
    log(exotic)  # must not raise

    written = (tmp_path / "unicode_probe.log").read_text(encoding="utf-8")
    assert exotic in written


def test_degree_ks_runs_on_a_populated_complex():
    """This crashed the whole of phases 8-11 after they had already started.

    The guard was written `np.array(sizes) or np.array([0])`, which raises on any array
    with more than one element -- so it worked on the toy fixtures and failed on the
    archive. The empty case must be decided before the list becomes an array.
    """
    from src.perturb import degree_ks

    segs = [
        _Seg("s1", "i1", ["n1"], ["W -- Camps -- Food"]),
        _Seg("s2", "i2", ["n2"], ["W -- Camps -- Food"]),
        _Seg("s3", "i3", ["n3"], ["W -- Camps -- School"]),
        _Seg("s4", "i4", ["n1"], ["W -- Camps -- School"]),
        _Seg("s5", "i5", ["n4"], ["W -- Propaganda"]),
        _Seg("s6", "i6", ["n5"], ["W -- Propaganda"]),
    ]
    rank_of = {"W -- Camps -- Food": 2, "W -- Camps -- School": 2, "W -- Propaganda": 2, "W": 3}
    cx = build_complex(segs, rank_of, granularity="mid")
    assert len(cx.by_rank(2)) > 1, "fixture must exercise the multi-element path"
    assert degree_ks(cx, cx) == 0.0


def _probe():
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_mechanism import narrator_probe

    return narrator_probe


def _identity_fixture(scale: float = 1.0, informative: bool = True):
    """Segments drawn around a per-narrator centroid -- the geometry a representation has."""
    rng = np.random.default_rng(0)
    k, per, dim = 40, 20, 16
    labels = [f"n{i}" for i in range(k) for _ in range(per)]
    if not informative:
        return rng.normal(0, 1, (k * per, dim)) * scale, labels
    centroids = rng.normal(0, 3, (k, dim))
    x = np.repeat(centroids, per, axis=0) + rng.normal(0, 1, (k * per, dim))
    return x * scale, labels


def test_narrator_probe_recovers_identity_that_is_present():
    x, labels = _identity_fixture()
    r = _probe()(x, labels, np.random.default_rng(0))
    assert r["probe_accuracy"] > 0.8, r
    assert r["chance"] < 0.05


def test_narrator_probe_sits_at_chance_when_identity_is_absent():
    """Without this the probe could report a number that only tracks representation norm."""
    x, labels = _identity_fixture(informative=False)
    r = _probe()(x, labels, np.random.default_rng(0))
    assert r["probe_accuracy"] < 0.10, r


def test_narrator_probe_is_invariant_to_representation_scale():
    """Two architectures emit hidden vectors on different scales.

    An unscaled linear probe rewards the larger-magnitude one for reasons that have
    nothing to do with narrator identity -- which is exactly the comparison this probe
    is used to make, so the confound would land directly in the claim.
    """
    p = _probe()
    a = p(*_identity_fixture(scale=1.0), rng=np.random.default_rng(0))
    b = p(*_identity_fixture(scale=100.0), rng=np.random.default_rng(0))
    assert abs(a["probe_accuracy"] - b["probe_accuracy"]) < 0.02, (a, b)


# --- the citation verifier, which writes a submission artefact -----------------------

def _citer():
    import importlib

    return importlib.import_module("experiments.verify_citations")


def test_unverified_entry_keeps_the_key_it_was_cited_under():
    """An unverifiable reference must fail loudly, not silently.

    Emitting a different bib key leaves the \\citep undefined, which renders as '??'
    in the PDF -- the least visible failure available. Keeping the key puts the
    warning in the printed bibliography where an author cannot miss it.
    """
    m = _citer()
    rec = {"key": "smith2020thing", "wanted_title": "A Thing", "wanted_first_author": "Smith",
           "type": "article", "verified": False, "title_similarity": 0.4,
           "first_author_confirmed": False, "provenance": "none"}
    out = m.to_bibtex(rec)
    assert "{smith2020thing," in out, out
    assert "PLACEHOLDER_" not in out, out
    assert "UNVERIFIED" in out, out


def test_hand_checked_entry_is_labelled_as_such():
    m = _citer()
    rec = {"key": "k2004", "wanted_title": "T", "wanted_first_author": "K", "type": "book",
           "verified": False, "title_similarity": 0.4, "first_author_confirmed": False,
           "provenance": "manual",
           "manual": {"kind": "book", "checked": "checked by hand",
                      "fields": {"title": "T", "author": "K", "year": "2004"}}}
    out = m.to_bibtex(rec)
    assert out.startswith("% hand-checked"), out
    assert "@book{k2004," in out, out
    assert "UNVERIFIED" not in out, out


def test_diacritic_surname_matches_itself():
    """A non-ASCII surname must not fail its own author check.

    norm() folded the retrieved author to ASCII but the wanted surname was only
    lowercased, so Sogaard was compared against a mangled form of itself and every
    diacritic name would have been reported as a wrong author at similarity 1.00.
    """
    m = _citer()
    for name in ["S\u00f8gaard", "M\u00fcller", "Sch\u00f6lkopf", "Wei\u00df", "\u0141ukasz"]:
        assert m.norm(name) in m.norm(f"Anders {name}"), name
    # Folding must not collapse distinct names into each other.
    assert m.norm("S\u00f8gaard") == m.norm("Sogaard")
    assert m.norm("S\u00f8gaard") != m.norm("Gaard")


def test_transport_failure_is_not_recorded_as_a_bad_reference():
    """A network outage is not evidence that a citation is wrong.

    Conflating the two let a flaky lookup rewrite a correct entry as a placeholder,
    which broke the build. Provenance must distinguish 'could not check' from 'checked
    and failed', because only the latter licenses changing the bibliography.
    """
    m = _citer()
    m.TRANSPORT_ERRORS.clear()
    monkey = {"search_crossref": lambda t: [], "search_openalex": lambda t: [],
              "search_arxiv": lambda t: []}
    originals = {k: getattr(m, k) for k in monkey}
    sleep, m.time.sleep = m.time.sleep, lambda *_: None
    try:
        for k, fn in monkey.items():
            setattr(m, k, fn)
        clean = m.verify("k", "T", "S", "article", "p")
        m.TRANSPORT_ERRORS.append("openalex:HTTPError")

        def boom(_t):
            m.TRANSPORT_ERRORS.append("arxiv:TimeoutError")
            return []

        m.search_arxiv = boom
        errored = m.verify("k", "T", "S", "article", "p")
    finally:
        for k, fn in originals.items():
            setattr(m, k, fn)
        m.time.sleep = sleep
        m.TRANSPORT_ERRORS.clear()

    assert clean["provenance"] == "none", clean
    assert errored["provenance"] == "inconclusive", errored



