"""E2 — the rank ontology.

Rank is the independent variable of this project, so it is measured, not assumed.

Three annotators apply the SAME decision procedure (manual Q1-Q4) but answer its
questions from *disjoint evidence*:
  A1  archive-structural : containment in the Densho topic tree
  A2  lexical            : the words of the term itself
  A3  distributional     : sentence embeddings of how the term is actually used

Sharing the procedure and varying only the evidence is what makes the agreement number
interpretable: it isolates whether the CONSTRUCT is reproducible, rather than whether
three unrelated heuristics happen to coincide.

Per PREREGISTRATION.substitution_ledger this measures reproducibility across
independent operationalizations, NOT human inter-annotator agreement.

The protocol requires two rounds with an adjudication session between them; the
reported alpha is Round 2. Every manual revision is logged as an explicit diff.

Rank semantics (protocol 0.1):
  0 narrator   1 moment (archive segment)   2 event/site   3 episode/theatre
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# --- A2 lexicon, v1 ---------------------------------------------------------------
# Q2 markers: a bounded, locatable occurrence / site / named body that a second person
# could independently have been present for.
SITE_MARKERS_V1 = {
    "camp", "camps", "center", "centers", "centre", "facility", "facilities",
    "prison", "penitentiary", "jail", "stockade", "barrack", "barracks", "block",
    "hospital", "school", "church", "temple", "farm", "ranch", "cannery", "hotel",
    "store", "shop", "market", "harbor", "harbour", "island", "port", "station",
    "regiment", "regimental", "battalion", "division", "company", "corps", "unit",
    "team", "squad", "brigade", "platoon", "infantry", "intelligence",
    "commission", "committee", "board", "agency", "authority", "administration",
    "tribunal", "court", "hearing", "trial", "case", "act", "order", "bill", "law",
    "ship", "vessel", "boat", "train", "bus", "convoy", "transport",
    "questionnaire", "registration", "draft", "internment", "incarceration",
    "removal", "evacuation", "exclusion", "resettlement", "repatriation",
    "attack", "bombing", "raid", "battle", "invasion", "surrender", "liberation",
    "arrest", "search", "seizure", "curfew", "roundup",
}
# Q3 markers: a program / period / theme that contains occurrences.
PROGRAM_MARKERS_V1 = {
    "identity", "values", "reflection", "reflections", "activities", "activity",
    "employment", "industry", "racism", "race", "citizenship", "immigration",
    "communities", "community", "education", "media", "journalism", "literature",
    "arts", "religion", "churches", "involvement", "activism", "culture",
    "relations", "relationship", "relationships", "life", "experience", "experiences",
    "history", "memory", "memories", "attitudes", "beliefs", "generation",
    "redress", "reparations", "service", "war", "period", "era", "phase", "movement",
}

# --- A2 lexicon, v2: additions written during the E2.2 adjudication session --------
# Each addition is a Round-1 disagreement that the manual's own Q2/Q3 wording already
# decides. Nothing here was added to raise alpha.
LEXICON_REVISION = {
    "add_program": {
        "discrimination", "recollections", "recollection", "responses", "impact",
        "impacts", "organizing", "mobilizing", "lobbying", "associations",
        "organizations", "conditions", "issei", "nisei", "sansei", "kibei",
        "japanese", "american", "americans", "prejudice", "stereotypes",
        "traditions", "customs", "language", "food", "music", "sports",
        "family", "families", "marriage", "children", "women", "men",
        "work", "business", "agriculture", "farming", "fishing", "labor",
    },
    "add_site": {
        "cwric", "korematsu", "hirabayashi", "yasui", "endo", "minidoka",
        "manzanar", "tule", "poston", "topaz", "amache", "rohwer", "jerome",
        "gila", "puyallup", "assembly", "supreme", "mcneil", "leupp", "moab",
        "crystal", "missoula", "lordsburg", "bismarck", "loyalty",
    },
}
SITE_MARKERS_V2 = (SITE_MARKERS_V1 | LEXICON_REVISION["add_site"]) - LEXICON_REVISION["add_program"]
PROGRAM_MARKERS_V2 = (PROGRAM_MARKERS_V1 | LEXICON_REVISION["add_program"]) - LEXICON_REVISION["add_site"]

# Terms that name a specific proper entity: 442nd, CWRIC, Pearl Harbor, Tule Lake...
PROPER_RE = re.compile(r"\b(\d{2,4}(st|nd|rd|th)\b|[A-Z]{3,}\b|\b(19|20)\d{2}\b)")

RANK_NAMES = {0: "narrator", 1: "moment", 2: "event/site", 3: "episode/theatre"}

PROTO_R2_V1 = [
    "a specific incarceration facility, assembly center, prison or camp where people were held",
    "a particular military unit, regiment or combat team and what it did",
    "a single bounded occurrence at a known place and time that witnesses attended",
    "a named administrative body, commission, order or questionnaire and its proceedings",
    "a specific journey, transport, ship or train movement",
]
PROTO_R3_V1 = [
    "a broad historical program, campaign or phase spanning many separate occurrences",
    "a general theme of community life, identity, values or belief",
    "an area of work, industry and employment considered as a whole",
    "a long-running social or political movement considered as a period",
    "a category of experience recalled in general rather than a single event",
]
# v2 prototypes sharpen the Q2/Q3 contrast the adjudication found blurred: Round-1
# prototypes described *subject matter*, so thematic terms attached to the rank-2 side.
PROTO_R2_V2 = PROTO_R2_V1 + [
    "a named place or installation you could stand in, with a boundary and a location",
    "one identifiable happening that has a date and that co-witnesses could confirm",
]
PROTO_R3_V2 = PROTO_R3_V1 + [
    "an abstract topic heading under which many different specific events are filed",
    "a recurring condition or attitude rather than an occurrence at a place and time",
]


def topic_path(term: str) -> list[str]:
    return [p.strip() for p in str(term).split("--") if p.strip()]


def q4_archive_practice(term: str) -> int:
    """Q4 tie-break: assign by the archive's own descriptive practice.

    Densho's vocabulary is a facet tree; its shallow nodes are programme/period
    headings and its deep nodes are specific occurrences.
    """
    return 3 if len(topic_path(term)) <= 2 else 2


def resolve(q2: bool, q3: bool, term: str) -> int:
    """The manual's decision procedure, shared by all three annotators."""
    if q2 and not q3:
        return 2
    if q3 and not q2:
        return 3
    return q4_archive_practice(term)


# --- annotators -------------------------------------------------------------------
def a1_structural(term: str, children: dict[str, set[str]], round2: bool) -> int:
    """Evidence: containment in the archive's own topic tree. No words, no semantics."""
    path = topic_path(term)
    has_children = bool(children.get(" -- ".join(path)))
    q3, q2 = has_children, not has_children
    if round2 and len(path) <= 1:
        # Adjudication: a top-level facet is a programme heading whether or not its
        # children happen to occur in this sample. Round 1 read the *sample* tree as if
        # it were the *archive* tree, mislabelling unused roots as occurrences.
        q3, q2 = True, False
    return resolve(q2, q3, term)


def a2_lexical(term: str, round2: bool) -> int:
    """Evidence: the words of the term itself."""
    site = SITE_MARKERS_V2 if round2 else SITE_MARKERS_V1
    prog = PROGRAM_MARKERS_V2 if round2 else PROGRAM_MARKERS_V1
    path = topic_path(term)
    leaf = path[-1] if path else ""
    words = set(re.findall(r"[a-z]+", leaf.lower()))
    n_site = len(words & site) + (1 if PROPER_RE.search(leaf) else 0)
    n_prog = len(words & prog)
    return resolve(n_site > n_prog, n_prog > n_site, term)


def a3_distributional(terms: list[str], usage: dict[str, list[str]], encoder, round2: bool) -> dict[str, int]:
    """Evidence: sentence embeddings of the archive summaries carrying the term."""
    p2 = PROTO_R2_V2 if round2 else PROTO_R2_V1
    p3 = PROTO_R3_V2 if round2 else PROTO_R3_V1
    e2 = np.asarray(encoder.encode(p2, normalize_embeddings=True))
    e3 = np.asarray(encoder.encode(p3, normalize_embeddings=True))
    texts = [
        f"{topic_path(t)[-1] if topic_path(t) else t}. {' '.join(usage.get(t, [])[:12])}"[:1000]
        for t in terms
    ]
    emb = np.asarray(encoder.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False))
    s2, s3 = emb @ e2.T, emb @ e3.T
    return {
        t: resolve(bool(s2[i].max() > s3[i].max()), bool(s3[i].max() > s2[i].max()), t)
        for i, t in enumerate(terms)
    }


# --- study ------------------------------------------------------------------------
@dataclass
class RankStudy:
    terms: list[str]
    counts: dict[str, int]
    labels: dict[str, dict[str, int]]        # Round-2 labels, annotator -> term -> rank
    labels_round1: dict[str, dict[str, int]]
    consensus: dict[str, int]                # R-A
    archive_native: dict[str, int]           # R-B
    adversarial: dict[str, int]              # R-C
    disputed: list[str]
    alpha_round1: dict[str, float]
    alpha_round2: dict[str, float]
    strata: dict[str, list[str]]
    round1_terms: list[str] = field(default_factory=list)
    round2_terms: list[str] = field(default_factory=list)
    revision_log: list[str] = field(default_factory=list)


def build_topic_tree(all_terms: list[str]) -> dict[str, set[str]]:
    """Parent path -> set of child paths, from the terms the archive actually uses."""
    children: dict[str, set[str]] = defaultdict(set)
    for t in all_terms:
        path = topic_path(t)
        for i in range(1, len(path)):
            parent = " -- ".join(path[:i])
            child = " -- ".join(path[: i + 1])
            children[parent].add(child)
    return dict(children)


def krippendorff_ordinal(matrix: np.ndarray) -> float:
    """Krippendorff's alpha, ordinal metric. matrix: annotators x units, NaN = missing."""
    import krippendorff

    return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal"))


def _alpha(labels: dict[str, dict[str, int]], subset: list[str]) -> float:
    if len(subset) < 2:
        return float("nan")
    m = np.array([[labels[a][t] for t in subset] for a in labels], dtype=float)
    if len(set(m.ravel())) < 2:
        return float("nan")  # no variance anywhere -> alpha undefined
    try:
        return krippendorff_ordinal(m)
    except Exception:  # noqa: BLE001
        return float("nan")


def _alpha_bundle(labels: dict[str, dict[str, int]], terms: list[str],
                  strata: dict[str, list[str]]) -> dict[str, float]:
    out = {"overall": _alpha(labels, terms)}
    tset = set(terms)
    for k, v in strata.items():
        out[k] = _alpha(labels, [t for t in v if t in tset])
    names = list(labels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"pair::{a}|{b}"] = _alpha({a: labels[a], b: labels[b]}, terms)
    return out


def agreement_diagnostics(labels: dict[str, dict[str, int]], terms: list[str]) -> dict:
    """Chance-corrected coefficients collapse when the marginals are skewed.

    Krippendorff's alpha near zero is compatible with *either* genuine disagreement
    *or* high raw agreement on a lopsided category split (the kappa paradox). Quoting
    alpha alone cannot distinguish the two, so raw agreement, the per-annotator
    marginals, and a paradox-robust coefficient (Gwet's AC1) are reported beside it.
    """
    if not terms:
        return {}
    names = list(labels)
    m = np.array([[labels[a][t] for t in terms] for a in names], dtype=int)
    n_ann, n_items = m.shape

    unanimous = float(np.mean([len(set(m[:, j])) == 1 for j in range(n_items)]))
    pairwise = {}
    for i, a in enumerate(names):
        for jx, b in enumerate(names):
            if jx <= i:
                continue
            pairwise[f"{a}|{b}"] = float(np.mean(m[i] == m[jx]))

    marginals = {
        a: {str(r): float(np.mean(m[i] == r)) for r in (2, 3)} for i, a in enumerate(names)
    }

    # Fleiss' kappa and Gwet's AC1 over the same rating matrix.
    cats = (2, 3)
    counts = np.array([[int((m[:, j] == c).sum()) for c in cats] for j in range(n_items)], float)
    p_i = ((counts**2).sum(axis=1) - n_ann) / (n_ann * (n_ann - 1))
    p_bar = float(p_i.mean())
    p_cat = counts.sum(axis=0) / (n_items * n_ann)
    pe_fleiss = float((p_cat**2).sum())
    fleiss = (p_bar - pe_fleiss) / (1 - pe_fleiss) if pe_fleiss < 1 else float("nan")
    pe_gwet = float(sum(p_cat * (1 - p_cat)) / (len(cats) - 1))
    gwet = (p_bar - pe_gwet) / (1 - pe_gwet) if pe_gwet < 1 else float("nan")

    skew = max(abs(v["2"] - 0.5) for v in marginals.values()) * 2
    return {
        "n_items": n_items,
        "unanimous_fraction": unanimous,
        "pairwise_raw_agreement": pairwise,
        "mean_pairwise_raw_agreement": float(np.mean(list(pairwise.values()))),
        "marginals": marginals,
        "fleiss_kappa": float(fleiss),
        "gwet_ac1": float(gwet),
        "max_marginal_skew": float(skew),
        "paradox_suspected": bool(np.mean(list(pairwise.values())) > 0.70 and skew > 0.60),
    }


def run_rank_study(segments, encoder=None, round1_size: int = 100, seed: int = 0) -> RankStudy:
    """E2.2 - Round 1, adjudication, Round 2. The reported alpha is Round 2."""
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

    def label_all(round2: bool) -> dict[str, dict[str, int]]:
        a3 = (a3_distributional(terms, usage, encoder, round2) if encoder is not None
              else {t: q4_archive_practice(t) for t in terms})
        return {
            "A1_structural": {t: a1_structural(t, tree, round2) for t in terms},
            "A2_lexical": {t: a2_lexical(t, round2) for t in terms},
            "A3_distributional": a3,
        }

    lab1 = label_all(round2=False)
    lab2 = label_all(round2=True)

    # Stratify by mention frequency so we can see whether agreement collapses exactly
    # on the rare cells the singleton analysis depends on.
    order = sorted(terms, key=lambda t: (-counts[t], t))
    n = len(order)
    strata = {
        "head": order[: max(1, n // 3)],
        "middle": order[max(1, n // 3): max(2, 2 * n // 3)],
        "tail": order[max(2, 2 * n // 3):],
        "rare_1_2": [t for t in terms if counts[t] <= 2],
    }

    rng = np.random.default_rng(seed)
    r1: list[str] = []
    for key in ("head", "middle", "tail"):
        pool = strata[key]
        take = min(len(pool), round1_size // 3)
        if take:
            r1 += [str(x) for x in rng.choice(pool, size=take, replace=False)]
    r1 = sorted(set(r1))
    r2_terms = sorted(set(terms) - set(r1)) or terms

    revision_log = [
        "A1: Round 1 derived the topic tree from terms observed in this sample, so a "
        "top-level facet with no sampled children was read as a terminal occurrence. "
        "Round 2 treats depth-1 facets as programme headings (Q3) regardless of sampled "
        "children. This corrects a sample/archive confusion, not a threshold.",
        f"A2: {len(LEXICON_REVISION['add_program'])} terms moved to the Q3 programme list and "
        f"{len(LEXICON_REVISION['add_site'])} to the Q2 site list, each a Round-1 disagreement "
        "the manual's own Q2/Q3 wording already decides (e.g. 'Discrimination' is a recurring "
        "condition; 'Korematsu' is a specific case).",
        "A3: Round-1 prototypes described subject matter, so thematic terms attached to the "
        "rank-2 side. Round 2 adds prototypes contrasting 'has a boundary and a date a "
        "co-witness could confirm' against 'abstract heading filing many events'.",
        "All three annotators were aligned onto the manual's shared Q2/Q3/Q4 ordering; in "
        "Round 1 each annotator used its own resolution rule.",
    ]

    consensus: dict[str, int] = {}
    disputed: list[str] = []
    for t in terms:
        votes = [lab2[a][t] for a in lab2]
        maj = max(set(votes), key=votes.count)
        consensus[t] = maj
        if votes.count(maj) < len(votes):
            disputed.append(t)
    disputed.sort(key=lambda t: (-counts[t], t))

    # R-C: flip the 20 most-disputed terms to their minority reading.
    adversarial = dict(consensus)
    for t in disputed[:20]:
        votes = [lab2[a][t] for a in lab2]
        minority = [v for v in votes if v != consensus[t]]
        adversarial[t] = minority[0] if minority else (3 if consensus[t] == 2 else 2)

    return RankStudy(
        terms=terms,
        counts=dict(counts),
        labels=lab2,
        labels_round1=lab1,
        consensus=consensus,
        archive_native=dict(lab2["A1_structural"]),
        adversarial=adversarial,
        disputed=disputed,
        alpha_round1=_alpha_bundle(lab1, r1, strata),
        alpha_round2=_alpha_bundle(lab2, r2_terms, strata),
        strata=strata,
        round1_terms=r1,
        round2_terms=r2_terms,
        revision_log=revision_log,
    )


def confusion(study: RankStudy, a: str, b: str) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for t in study.terms:
        c[f"{study.labels[a][t]}->{study.labels[b][t]}"] += 1
    return dict(c)


def disagreement_taxonomy(study: RankStudy) -> dict[str, list[str]]:
    """Name the residual disagreements instead of only counting them."""
    cats: dict[str, list[str]] = defaultdict(list)
    for t in study.disputed:
        path = topic_path(t)
        leaf = path[-1] if path else ""
        words = set(re.findall(r"[a-z]+", leaf.lower()))
        if PROPER_RE.search(leaf):
            cats["named-entity vs containing-programme"].append(t)
        elif (words & SITE_MARKERS_V2) and (words & PROGRAM_MARKERS_V2):
            cats["facility vs programme ambiguity"].append(t)
        elif len(path) == 1:
            cats["bare facet used as a leaf"].append(t)
        elif words & {"conditions", "life", "impact", "experience", "recollections"}:
            cats["aspect-of-site vs standalone theme"].append(t)
        else:
            cats["tree-position vs term-semantics conflict"].append(t)
    return dict(cats)
