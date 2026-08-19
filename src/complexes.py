"""E4 — complex construction, star expansion, granularity sweep, rank/cardinality decoupling.

A cell carries a support: a subset of the rank-0 ground set (narrators). Cells are keyed by
identifier rather than by support, because distinct segments can share one narrator, so the
collection is not a set family over that ground set — `support_injectivity` measures it.
Rank is assigned from the
archive's descriptive taxonomy via E2, NEVER from |cell| — that is what separates a
combinatorial complex from a hypergraph, and E4.4 verifies it mechanically.

Rank ladder on Densho:
  0  narrator                       (stable Densho oh_id)
  1  moment   = archive segment     (the archive's own sub-interview curation)
  2  event/site = topic term / facility judged rank-2 by the rank map
  3  episode  = topic term judged rank-3 by the rank map (contains rank-2 cells)

Granularity varies ONLY the rank-2 equivalence relation, as three deterministic
functions of the same mention set:
  coarse : parent path of the rank-2 term      (merge one level up the archive tree)
  mid    : the rank-2 term itself              (PRE-DECLARED PRIMARY)
  fine   : rank-2 term x geography place       (split by archive-recorded place)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.ontology import topic_path

GRANULARITIES = ("coarse", "mid", "fine")


@dataclass(frozen=True)
class Cell:
    cid: str
    rank: int
    members: frozenset[str]          # rank-0 narrator keys
    label: str
    source: str                      # 'narrator' | 'segment' | 'topic' | 'facility'
    segments: frozenset[str] = field(default_factory=frozenset)
    base_term: str = ""              # the archive term this cell derives from

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class Complex:
    granularity: str
    rank_map: str
    narrators: list[str]
    cells: dict[str, Cell]
    include_interviewers: bool = False

    def by_rank(self, k: int) -> list[Cell]:
        return [c for c in self.cells.values() if c.rank == k]

    def attestation(self, cid: str) -> int:
        return len(self.cells[cid].members)

    def index(self, k: int) -> dict[str, int]:
        return {c.cid: i for i, c in enumerate(sorted(self.by_rank(k), key=lambda c: c.cid))}

    def incidence_matrix(self, k: int, j: int) -> sp.csr_matrix:
        """B_{k,j}: rows = rank-k cells, cols = rank-j cells; 1 iff k-cell is contained in j-cell."""
        rows_idx, cols_idx = self.index(k), self.index(j)
        rows, cols = [], []
        lower = sorted(self.by_rank(k), key=lambda c: c.cid)
        upper = sorted(self.by_rank(j), key=lambda c: c.cid)

        if k == 0:
            for c in upper:
                for n in c.members:
                    r = rows_idx.get(f"r0:{n}")
                    if r is not None:
                        rows.append(r); cols.append(cols_idx[c.cid])
        elif k == 1 and j in (2, 3):
            seg2up: dict[str, list[str]] = defaultdict(list)
            for c in upper:
                for s in c.segments:
                    seg2up[s].append(c.cid)
            for c in lower:
                for s in c.segments:  # a rank-1 cell carries exactly its own segment
                    for up in seg2up.get(s, ()):
                        rows.append(rows_idx[c.cid]); cols.append(cols_idx[up])
        elif k == 2 and j == 3:
            for c in upper:
                for lo in lower:
                    if lo.base_term and c.base_term and _is_descendant(lo.base_term, c.base_term):
                        rows.append(rows_idx[lo.cid]); cols.append(cols_idx[c.cid])
        else:
            raise ValueError(f"unsupported incidence ({k},{j})")

        m = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(rows_idx), len(cols_idx)), dtype=np.float32
        )
        return m.tocsr()

    def adjacency_matrix(self, k: int, via: int) -> sp.csr_matrix:
        """Rank-k cells adjacent through a shared rank-`via` coface (via > k)."""
        B = self.incidence_matrix(k, via)
        A = (B @ B.T).tolil()
        A.setdiag(0)
        return A.tocsr()

    def coadjacency_matrix(self, k: int, via: int) -> sp.csr_matrix:
        """Rank-k cells coadjacent through a shared rank-`via` face (via < k)."""
        B = self.incidence_matrix(via, k)
        A = (B.T @ B).tolil()
        A.setdiag(0)
        return A.tocsr()

    def summary(self) -> dict:
        out: dict = {"granularity": self.granularity, "rank_map": self.rank_map,
                     "n_narrators": len(self.narrators)}
        for k in range(4):
            cs = self.by_rank(k)
            sizes = np.array([c.size for c in cs]) if cs else np.array([0])
            out[f"rank{k}"] = {
                "n_cells": len(cs),
                "size_mean": float(sizes.mean()),
                "size_median": float(np.median(sizes)),
                "size_max": int(sizes.max()),
                "size_min": int(sizes.min()),
                "singleton_fraction": float((sizes == 1).mean()) if len(cs) else 0.0,
            }
        return out


def _is_descendant(child_term: str, parent_term: str) -> bool:
    cp, pp = topic_path(child_term), topic_path(parent_term)
    return len(cp) > len(pp) and cp[: len(pp)] == pp


def _geo_place(seg) -> str:
    g = seg.geography or []
    return str(g[0].get("term")) if g and g[0].get("term") else "UNSPECIFIED"


def _rank2_key(term: str, seg, granularity: str) -> tuple[str, str] | None:
    """The rank-2 equivalence relation. Only this varies across the sweep.

    Returns (cell_key, semantic_path). coarse merges one level up the archive tree,
    mid is the archive leaf itself, fine splits the leaf by archive-recorded place.
    """
    path = topic_path(term)
    if not path:
        return None
    full = " -- ".join(path)
    if granularity == "coarse":
        parent = " -- ".join(path[:-1]) if len(path) > 1 else path[0]
        return parent, parent
    if granularity == "mid":
        return full, full
    if granularity == "fine":
        return f"{full} @ {_geo_place(seg)}", full
    raise ValueError(granularity)


def referent(cell: Cell, depth: int = 2) -> str:
    """Project a rank-2 cell onto the coarsest level shared by every granularity.

    Raw cell labels are disjoint across granularities *by construction* (coarse keys are
    parent paths, fine keys carry a place suffix), so an overlap measure on raw labels is
    identically zero and carries no information. Comparing triage lists across resolutions
    therefore requires projecting onto a common archival referent first.
    """
    return " -- ".join(topic_path(cell.base_term or cell.label)[:depth])


def mask_incidences(cx: Complex, keep: set[tuple[str, str]]) -> Complex:
    """Rebuild the complex so message passing only sees the *training* incidences.

    Without this the model is handed the very (narrator, event) edges it is asked to
    predict: representations on both sides already encode the answer, the objective is
    trivially satisfiable, and test AUC collapses toward chance while looking like an
    architecture problem. Rank-1 cells are untouched - which narrator gave a segment is
    never in question; only the segment/narrator-to-event incidence is predicted.
    """
    import copy as _copy
    from dataclasses import replace as _replace

    cells: dict[str, Cell] = {}
    for cid, c in cx.cells.items():
        if c.rank in (2, 3):
            kept = frozenset(n for n in c.members if (n, cid) in keep)
            cells[cid] = _replace(c, members=kept)
        else:
            cells[cid] = c
    out = _copy.copy(cx)
    out.cells = cells
    return out


def build_complex(
    segments,
    rank_of_term: dict[str, int],
    granularity: str = "mid",
    rank_map_name: str = "R-A",
    include_interviewers: bool = False,
    min_cell_size: int = 1,
) -> Complex:
    """E4.1 — build the ranked combinatorial complex from archive metadata."""
    cells: dict[str, Cell] = {}
    narrators = sorted({n for s in segments for n in s.narrators})

    def speakers(s) -> list[str]:
        return sorted(set(s.narrators) | (set(s.interviewers) if include_interviewers else set()))

    for n in narrators:
        cells[f"r0:{n}"] = Cell(f"r0:{n}", 0, frozenset({n}), n, "narrator")

    # rank 1 — the archive's own segments
    for s in segments:
        mem = frozenset(speakers(s))
        if not mem:
            continue
        cells[f"r1:{s.segment_id}"] = Cell(
            f"r1:{s.segment_id}", 1, mem, s.title or s.segment_id, "segment",
            segments=frozenset({s.segment_id}),
        )

    # ranks 2 and 3 — archive topic terms, ranked by the E2 rank map
    acc: dict[tuple[int, str], dict] = {}
    for s in segments:
        mem = set(speakers(s))
        if not mem:
            continue
        for t in s.topics:
            term = t.get("term")
            if not term:
                continue
            rk = rank_of_term.get(term)
            if rk not in (2, 3):
                continue
            if rk == 2:
                keyed = _rank2_key(term, s, granularity)
                if keyed is None:
                    continue
                key, base = keyed
            else:
                key = " -- ".join(topic_path(term))
                base = key
            e = acc.setdefault((rk, key), {"members": set(), "segments": set(), "base": base})
            e["members"] |= mem
            e["segments"].add(s.segment_id)

    # rank-3 cells also absorb the narrators of their rank-2 descendants (transitive
    # attestation through the rank ladder, protocol 0.1 definition of a(x))
    r3_keys = [k for (rk, k) in acc if rk == 3]
    for (rk, key), e in list(acc.items()):
        if rk != 2:
            continue
        for r3 in r3_keys:
            if _is_descendant(e["base"], r3):
                acc[(3, r3)]["members"] |= e["members"]
                acc[(3, r3)]["segments"] |= e["segments"]

    for (rk, key), e in acc.items():
        if len(e["members"]) < min_cell_size:
            continue
        cid = f"r{rk}:{key}"
        cells[cid] = Cell(cid, rk, frozenset(e["members"]), key, "topic",
                          segments=frozenset(e["segments"]), base_term=e["base"])

    return Complex(granularity, rank_map_name, narrators, cells, include_interviewers)


# --- E4.2 star expansion -----------------------------------------------------------
def to_star(cx: Complex) -> dict:
    """Lossless bipartite/star expansion: one node per cell, typed by rank, one edge
    per containment. Node types AND rank are carried as features."""
    nodes = [
        {"cid": c.cid, "rank": c.rank, "type": ["narrator", "moment", "event", "episode"][c.rank],
         "label": c.label, "base_term": c.base_term}
        for c in sorted(cx.cells.values(), key=lambda c: (c.rank, c.cid))
    ]
    edges = []
    for c in cx.cells.values():
        if c.rank == 0:
            continue
        for n in c.members:  # membership edges preserve the exact member set
            edges.append({"src": f"r0:{n}", "dst": c.cid, "etype": f"in_r{c.rank}"})
    for c in cx.cells.values():  # provenance edges preserve segment support
        if c.rank in (2, 3):
            for s in c.segments:
                sid = f"r1:{s}"
                if sid in cx.cells:
                    edges.append({"src": sid, "dst": c.cid, "etype": f"supports_r{c.rank}"})
    return {"nodes": nodes, "edges": edges, "granularity": cx.granularity, "rank_map": cx.rank_map}


def from_star(star: dict) -> dict[str, tuple[int, frozenset[str]]]:
    """Reconstruct (rank, member set) for every cell from the star graph alone."""
    members: dict[str, set[str]] = defaultdict(set)
    ranks = {n["cid"]: n["rank"] for n in star["nodes"]}
    for n in star["nodes"]:
        if n["rank"] == 0:
            members[n["cid"]].add(n["cid"].split(":", 1)[1])
    for e in star["edges"]:
        if e["etype"].startswith("in_r"):
            members[e["dst"]].add(e["src"].split(":", 1)[1])
    return {cid: (ranks[cid], frozenset(members[cid])) for cid in ranks}


def verify_star_roundtrip(cx: Complex) -> dict:
    """E4.2 — the honest premise of the whole paper, verified rather than asserted."""
    star = to_star(cx)
    rebuilt = from_star(star)
    original = {c.cid: (c.rank, c.members) for c in cx.cells.values()}
    missing = sorted(set(original) - set(rebuilt))
    extra = sorted(set(rebuilt) - set(original))
    mismatched = sorted(cid for cid in set(original) & set(rebuilt) if original[cid] != rebuilt[cid])
    return {
        "lossless": not (missing or extra or mismatched),
        "n_cells": len(original),
        "n_star_nodes": len(star["nodes"]),
        "n_star_edges": len(star["edges"]),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_mismatched": len(mismatched),
        "examples_mismatched": mismatched[:5],
    }


def support_injectivity(cx: Complex, rank: int) -> dict:
    """Is the rank-k cell collection a set family over the ground set?

    A combinatorial complex identifies a cell with its support, so two cells sharing a
    support are one cell. Where that fails, no such complex carries the cells apart, and any
    permutation-invariant lifting from rank 0 hands the colliding cells identical input.
    """
    cells = cx.by_rank(rank)
    if not cells:
        return {"n_cells": 0}
    mult = Counter(c.members for c in cells)
    distinct = len(mult)
    singles = [c for c in cells if len(c.members) == 1]
    return {
        "n_cells": len(cells),
        "n_distinct_supports": distinct,
        "n_cells_sharing_a_support": sum(v for v in mult.values() if v > 1),
        "n_cells_beyond_distinct": len(cells) - distinct,
        "max_multiplicity": max(mult.values()),
        "supp_injective": distinct == len(cells),
        "collapse_factor": len(cells) / distinct,
        "n_singleton_support_cells": len(singles),
        "n_distinct_singleton_supports": len({c.members for c in singles}),
    }


# --- E4.4 rank/cardinality decoupling ----------------------------------------------
def rank_cardinality_decoupling(cx: Complex, ranks: tuple[int, ...] = (1, 2, 3)) -> dict:
    """E4.4 — the cheapest, most convincing experiment in the protocol.

    If rho(rank, |cell|) > 0.9 the ranks are cardinality in disguise and the premise fails.

    Reports concordance and inversion separately. An earlier version returned the
    concordance count under the name `rank_inversion_rate`, which reads as its own opposite:
    0.81 of cross-rank pairs are ordered by size *correctly*, and only 0.01 invert.
    """
    from scipy.stats import spearmanr

    cs = [c for c in cx.cells.values() if c.rank in ranks]
    rk = np.array([c.rank for c in cs], dtype=float)
    sz = np.array([c.size for c in cs], dtype=float)
    rho, p = spearmanr(rk, sz)

    inversions = 0
    total = 0
    by_rank = {k: np.array([c.size for c in cs if c.rank == k]) for k in ranks}
    for a in ranks:
        for b in ranks:
            if a >= b or not len(by_rank[a]) or not len(by_rank[b]):
                continue
    concordant = inverted = tied = 0
    total = 0
    by_rank = {k: np.array([c.size for c in cs if c.rank == k]) for k in ranks}
    for a in ranks:
        for b in ranks:
            if a >= b or not len(by_rank[a]) or not len(by_rank[b]):
                continue
            lo = np.sort(by_rank[a])          # sizes at the LOWER rank
            for h in by_rank[b]:              # h is a HIGHER-ranked cell's size
                total += len(lo)
                # searchsorted(left) counts lower-rank cells strictly smaller than h,
                # i.e. pairs where the higher-ranked cell is LARGER. That is concordance,
                # not inversion; the two were previously conflated under one name.
                concordant += int(np.searchsorted(lo, h, side="left"))
                inverted += int(len(lo) - np.searchsorted(lo, h, side="right"))
    tied = total - concordant - inverted

    # Ranks 0 and 1 are size-constrained by construction (a segment has one speaker), so a
    # global rate over all cross-rank pairs largely measures that definition. The rank-2 vs
    # rank-3 boundary is the one annotators actually dispute and the only informative test.
    pair23 = {}
    if len(by_rank.get(2, [])) and len(by_rank.get(3, [])):
        lo23 = np.sort(by_rank[2])
        c23 = i23 = t23 = 0
        for h in by_rank[3]:
            t23 += len(lo23)
            c23 += int(np.searchsorted(lo23, h, side="left"))
            i23 += int(len(lo23) - np.searchsorted(lo23, h, side="right"))
        r23 = spearmanr([2] * len(by_rank[2]) + [3] * len(by_rank[3]),
                        list(by_rank[2]) + list(by_rank[3]))
        pair23 = {
            "n_pairs": t23,
            "inversion_rate": i23 / t23 if t23 else float("nan"),
            "concordance_rate": c23 / t23 if t23 else float("nan"),
            "tie_rate": (t23 - c23 - i23) / t23 if t23 else float("nan"),
            "spearman_rho": float(r23.statistic),
        }

    extremes = {}
    for k in ranks:
        ck = [c for c in cs if c.rank == k]
        if ck:
            mx = max(ck, key=lambda c: c.size)
            mn = min(ck, key=lambda c: c.size)
            extremes[f"rank{k}"] = {
                "largest": {"label": mx.label, "size": mx.size},
                "smallest": {"label": mn.label, "size": mn.size},
            }

    ratio = None
    if "rank2" in extremes and "rank3" in extremes:
        big2 = extremes["rank2"]["largest"]["size"]
        small3 = max(1, extremes["rank3"]["smallest"]["size"])
        ratio = big2 / small3

    same_rank_ratio = {}
    for k in ranks:
        ck = sorted([c.size for c in cs if c.rank == k])
        if len(ck) >= 2 and ck[0] > 0:
            same_rank_ratio[f"rank{k}"] = ck[-1] / ck[0]

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "premise_holds": bool(rho <= 0.9),
        "cross_rank_pairs": total,
        "size_concordance_count": concordant,
        "size_concordance_rate": float(concordant / total) if total else 0.0,
        "rank_inversion_count": inverted,
        "rank_inversion_rate": float(inverted / total) if total else 0.0,
        "tie_rate": float(tied / total) if total else 0.0,
        "rank2_vs_rank3": pair23,
        "extremes": extremes,
        "largest_r2_over_smallest_r3": ratio,
        "max_size_ratio_within_rank": same_rank_ratio,
        "n_cells": len(cs),
    }


def sanity_assertions(cx: Complex) -> dict:
    """Cheap invariants that catch real construction bugs (protocol E4.1)."""
    out = {}
    B01 = cx.incidence_matrix(0, 1)
    out["B01_rows_eq_narrators"] = bool(B01.shape[0] == len(cx.narrators))
    r1 = cx.by_rank(1)
    out["no_empty_rank1"] = all(c.size > 0 for c in r1)
    out["ranks_in_range"] = set(c.rank for c in cx.cells.values()) <= {0, 1, 2, 3}
    out["members_subset_of_ground"] = all(
        set(c.members) <= set(cx.narrators) for c in cx.cells.values()
    )
    B12 = cx.incidence_matrix(1, 2)
    out["orphan_moment_fraction"] = float((np.asarray(B12.sum(axis=1)).ravel() == 0).mean()) if B12.shape[1] else 1.0
    out["all_pass"] = bool(
        out["B01_rows_eq_narrators"] and out["no_empty_rank1"]
        and out["ranks_in_range"] and out["members_subset_of_ground"]
    )
    return out


def save_complex(cx: Complex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "granularity": cx.granularity, "rank_map": cx.rank_map,
        "include_interviewers": cx.include_interviewers, "narrators": cx.narrators,
        "cells": [
            {"cid": c.cid, "rank": c.rank, "members": sorted(c.members), "label": c.label,
             "source": c.source, "segments": sorted(c.segments), "base_term": c.base_term}
            for c in cx.cells.values()
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
