"""Where does the rank disagreement actually live?

All three annotation protocols share one decision procedure, `resolve(q2, q3, term)`, and
differ only in the evidence they use to answer Q2 and Q3. When that evidence is
inconclusive -- Q2 and Q3 agree, or neither fires -- every protocol falls through to the
same deterministic tie-break, `q4_archive_practice`, which reads the archive's own facet
depth.

So the ladder should be perfectly reproducible exactly where the archive settles it, and
irreproducible exactly where the protocols have to infer. This measures that split rather
than asserting it, and reports agreement separately on each side.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.ontology import (  # noqa: E402
    a1_structural, a2_lexical, a3_distributional, build_topic_tree, krippendorff_ordinal,
    q4_archive_practice,
)

RES = ROOT / "data" / "results"
log = make_logger("p2_locus")


def _q_answers(term: str, tree, usage, encoder, terms):
    """Re-derive each protocol's Q2/Q3 verdict so we can see when the tie-break fired."""
    import re

    import numpy as np

    from src.ontology import (
        PROGRAM_MARKERS_V2, PROPER_RE, PROTO_R2_V2, PROTO_R3_V2, SITE_MARKERS_V2, topic_path,
    )

    path = topic_path(term)
    has_children = bool(tree.get(" -- ".join(path)))
    if len(path) <= 1:
        a1 = (False, True)                       # round-2 adjudication
    else:
        a1 = (not has_children, has_children)

    leaf = path[-1] if path else ""
    words = set(re.findall(r"[a-z]+", leaf.lower()))
    n_site = len(words & SITE_MARKERS_V2) + (1 if PROPER_RE.search(leaf) else 0)
    n_prog = len(words & PROGRAM_MARKERS_V2)
    a2 = (n_site > n_prog, n_prog > n_site)

    e2 = np.asarray(encoder.encode(PROTO_R2_V2, normalize_embeddings=True))
    e3 = np.asarray(encoder.encode(PROTO_R3_V2, normalize_embeddings=True))
    txt = f"{leaf}. {' '.join(usage.get(term, [])[:12])}"[:1000]
    emb = np.asarray(encoder.encode([txt], normalize_embeddings=True))
    s2, s3 = float((emb @ e2.T).max()), float((emb @ e3.T).max())
    a3 = (s2 > s3, s3 > s2)
    return {"A1_structural": a1, "A2_lexical": a2, "A3_distributional": a3}


def main() -> None:
    segments = load_corpus()[1]
    counts: dict[str, int] = defaultdict(int)
    usage: dict[str, list[str]] = defaultdict(list)
    for s in segments:
        for t in s.topics:
            term = t.get("term")
            if not term:
                continue
            counts[term] += 1
            if len(usage[term]) < 24:
                usage[term].append(s.text)

    terms = sorted(counts)
    tree = build_topic_tree(terms)
    encoder = get_encoder()
    log(f"{len(terms)} terms")

    labels = {
        "A1_structural": {t: a1_structural(t, tree, True) for t in terms},
        "A2_lexical": {t: a2_lexical(t, True) for t in terms},
        "A3_distributional": a3_distributional(terms, usage, encoder, True),
    }

    # A1's rule is complete: it sets (q2, q3) = (not has_children, has_children), which are
    # complementary, so it NEVER defers to the tie-break. Asking "do all three defer" is
    # therefore unsatisfiable by construction and would report n=0 as though it were a
    # finding. The answerable question is about the two protocols that can defer.
    defer_a2, defer_a3, defer_both = [], [], []
    for t in terms:
        qa = _q_answers(t, tree, usage, encoder, terms)
        d2 = qa["A2_lexical"][0] == qa["A2_lexical"][1]
        d3 = qa["A3_distributional"][0] == qa["A3_distributional"][1]
        if d2:
            defer_a2.append(t)
        if d3:
            defer_a3.append(t)
        if d2 and d3:
            defer_both.append(t)
    inferred_both = [t for t in terms if t not in set(defer_both)]

    def pair_agree(subset: list[str], x: str, y: str) -> float:
        if not subset:
            return float("nan")
        return sum(labels[x][t] == labels[y][t] for t in subset) / len(subset)

    def agreement(subset: list[str]) -> dict:
        import numpy as np

        if not subset:
            return {"n": 0}
        mat = np.array([[labels[a][t] for t in subset] for a in
                        ("A1_structural", "A2_lexical", "A3_distributional")], dtype=float)
        unanimous = int(sum(1 for i in range(len(subset))
                            if len(set(mat[:, i].tolist())) == 1))
        try:
            alpha = float(krippendorff_ordinal(mat))
        except Exception:  # noqa: BLE001 - alpha is undefined when every value is identical
            alpha = float("nan")
        return {"n": len(subset), "unanimous": unanimous,
                "unanimous_rate": unanimous / len(subset), "alpha": alpha,
                "A2_vs_A3": pair_agree(subset, "A2_lexical", "A3_distributional"),
                "A1_vs_A2": pair_agree(subset, "A1_structural", "A2_lexical"),
                "A1_vs_A3": pair_agree(subset, "A1_structural", "A3_distributional")}

    out = {
        "question": ("is the rank ladder reproducible where the archive's own practice "
                     "settles it, and irreproducible where a protocol must infer?"),
        "a1_never_defers": ("a1_structural sets (q2, q3) = (not has_children, has_children), "
                            "which are complementary, so it never reaches the tie-break; a "
                            "condition requiring all three to defer is unsatisfiable"),
        "both_deferring_protocols": agreement(defer_both),
        "at_least_one_inferring": agreement(inferred_both),
        "n_terms": len(terms),
        "defer_rate_A2_lexical": len(defer_a2) / len(terms),
        "defer_rate_A3_distributional": len(defer_a3) / len(terms),
        "defer_rate_both": len(defer_both) / len(terms),
        "tie_break": ("q4_archive_practice: facet path depth <= 2 -> rank 3, else rank 2; "
                      "deterministic, so any two protocols reaching it agree by construction"),
    }
    (RES / "e2_2_locus.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    b, i = out["both_deferring_protocols"], out["at_least_one_inferring"]
    log(f"defer rates: A2={out['defer_rate_A2_lexical']:.3f} "
        f"A3={out['defer_rate_A3_distributional']:.3f} both={out['defer_rate_both']:.3f}")
    log(f"both defer      n={b['n']:4d} A2vsA3={b.get('A2_vs_A3', float('nan')):.3f} "
        f"unanimous={b.get('unanimous_rate', float('nan')):.3f} "
        f"alpha={b.get('alpha', float('nan')):.3f}")
    log(f">=1 infers      n={i['n']:4d} A2vsA3={i.get('A2_vs_A3', float('nan')):.3f} "
        f"unanimous={i.get('unanimous_rate', float('nan')):.3f} "
        f"alpha={i.get('alpha', float('nan')):.3f}")


if __name__ == "__main__":
    main()
