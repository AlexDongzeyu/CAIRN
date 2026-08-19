"""Verify every reference the paper will cite, programmatically.

The paper-writing skill's hard rule: never write BibTeX from memory. Each entry is searched
against Crossref, then arXiv, and is accepted only when the retrieved title matches closely
AND the first-author surname is confirmed. Anything failing both checks is emitted as an
explicit PLACEHOLDER for human verification rather than guessed.

Two gates, learned the hard way:
  * title similarity >= 0.92 -- at 0.80 a survey happily "verifies" against a different
    survey on the same topic. Crossref returned "Deep learning for graph structured data"
    for the Hajij query and "Further results in multiset processing with neural networks"
    for the AllSet query; both would have passed a looser gate.
  * first-author surname must appear in the retrieved author list -- fabricated author
    lists are the most common defect in machine-assembled bibliographies.

Source order is Crossref -> OpenAlex -> arXiv. Crossref is authoritative for anything with
a DOI but has thin coverage of arXiv/OpenReview preprints, which is most of the topological
deep learning literature. arXiv's own API proved unreachable from this network (timeouts),
so it is kept only as a best-effort last resort and its exact-phrase form is avoided:
ti:"..." fails outright on titles containing ':' or '?'.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
OUT = ROOT / "paper"
log = make_logger("citations")

SIM_GATE = 0.92
MAILTO = "research@example.org"  # Crossref polite pool
UA = f"CAIRN-citation-audit/1.0 (mailto:{MAILTO})"

# (key, expected title, expected first-author surname, entry type, what the paper uses it for)
WANTED = [
    ("hajij2023tdl", "Topological Deep Learning: Going Beyond Graph Data",
     "Hajij", "misc", "combinatorial complexes; the rank-aware message passing we instantiate"),
    ("papillon2023architectures",
     "Architectures of Topological Deep Learning: A Survey of Message-Passing Topological Neural Networks",
     "Papillon", "misc", "taxonomy of topological message passing"),
    ("chien2022allset", "You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks",
     "Chien", "inproceedings", "M4 AllSetTransformer baseline"),
    ("wang2023edhnn", "Equivariant Hypergraph Diffusion Neural Operators",
     "Wang", "inproceedings", "M4 ED-HNN baseline"),
    ("moosavi2016lea",
     "Which Coreference Evaluation Metric Do You Trust? A Proposal for a Link-based Entity Aware Metric",
     "Moosavi", "inproceedings", "LEA, the primary coreference metric"),
    ("cybulska2014ecbplus",
     "Using a sledgehammer to crack a nut? Lexical diversity and event coreference resolution",
     "Cybulska", "inproceedings", "ECB+, the corpus named in the substitution ledger"),
    ("bugert2021generalizing",
     "Generalizing Cross-Document Event Coreference Resolution Across Multiple Corpora",
     "Bugert", "article", "cross-corpus CDEC instability motivating the E3.2 ledger entry"),
    ("kummerfeld2013error", "Error-Driven Analysis of Challenges in Coreference Resolution",
     "Kummerfeld", "inproceedings", "coreference error taxonomy behind E3.5"),
    ("webber2010rbo", "A similarity measure for indefinite rankings",
     "Webber", "article", "rank-biased overlap, used for triage-list stability"),
    ("dror2019deepdominance", "Deep Dominance - How to Properly Compare Deep Neural Models",
     "Dror", "inproceedings", "almost stochastic order testing"),
    ("ulmer2022deepsig",
     "deep-significance: Easy and Meaningful Statistical Significance Testing in the Age of Neural Networks",
     "Ulmer", "misc", "the ASO implementation used"),
    ("gwet2008ac1",
     "Computing inter-rater reliability and its variance in the presence of high agreement",
     "Gwet", "article", "AC1, reported beside Krippendorff alpha"),
    ("patil2020negative", "Negative Sampling for Hyperlink Prediction in Networks",
     "Patil", "inproceedings", "the four negative-sampling regimes"),
    ("reimers2019sbert", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
     "Reimers", "inproceedings", "the frozen sentence encoder held constant across models"),
    ("zhou2006hypergraph", "Learning with Hypergraphs: Clustering, Classification, and Embedding",
     "Zhou", "inproceedings", "normalized hypergraph Laplacian"),
    ("forman2003ricci",
     "Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature",
     "Forman", "article", "discrete curvature encoding"),
    ("krippendorff2004content", "Content Analysis: An Introduction to Its Methodology",
     "Krippendorff", "book", "Krippendorff alpha, the pre-registered agreement statistic"),
    # Related work: the existing skeptical evidence on higher-order architectures. A negative
    # result is credible as part of a literature and anomalous on its own.
    ("tang2024hypergraphmlp", "Hypergraph-MLP: Learning on Hypergraphs without Message Passing",
     "Tang", "inproceedings", "message-passing-free hypergraph learning matches HNNs"),
    ("papillon2024topotune",
     "TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks",
     "Papillon", "misc", "generalized CCNNs do not uniformly beat graph baselines"),
    ("dacrema2019worrying",
     "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches",
     "Ferrari Dacrema", "inproceedings",
     "weak baselines and evaluation design manufacture apparent progress"),
    ("zadeh1997granulation",
     "Toward a theory of fuzzy information granulation and its centrality in human reasoning "
     "and fuzzy logic",
     "Zadeh", "article", "foundational statement of information granulation"),
    ("yao2005perspectives",
     "Perspectives of granular computing",
     "Yao", "inproceedings", "granular computing as description at multiple resolutions"),
    ("pawlak1982rough", "Rough sets",
     "Pawlak", "article",
     "granulation by indiscernibility; the rank-2 equivalence relation is one"),
    ("kapoor2023leakage",
     "Leakage and the reproducibility crisis in machine-learning-based science",
     "Kapoor", "article", "leakage as a cross-field reproducibility failure"),
    # The closest ancestor of this paper's headline claim, and in a different field:
    # random splits are optimistic, not neutral. Cited so a reviewer who knows the NLP
    # evaluation literature sees we are extending it rather than rediscovering it.
    ("sogaard2021splits", "We Need To Talk About Random Splits",
     "S\u00f8gaard", "inproceedings",
     "random splits give optimistic estimates; the general form of our split result",
     "10.18653/v1/2021.eacl-main.156"),
    ("roberts2017cv",
     "Cross-validation strategies for data with temporal, spatial, hierarchical, or "
     "phylogenetic structure",
     "Roberts", "article", "group-structured splitting as established methodology"),
    ("carlsson2009topology", "Topology and data",
     "Carlsson", "article", "topological data analysis, the frame for the filtration section",
     "10.1090/s0273-0979-09-01249-x"),
    ("feng2019hgnn", "Hypergraph Neural Networks",
     "Feng", "inproceedings", "HGNN, the baseline lineage the M4 family descends from",
     "10.1609/aaai.v33i01.33013558"),
]

# Crossref, OpenAlex and arXiv index journal articles and preprints well and monographs
# badly. A canonical methods book therefore flaps between VERIFIED and UNVERIFIED across
# runs depending on what Crossref happens to return, which is a property of the instrument
# rather than of the reference. Entries below were checked by hand and are recorded with
# provenance "manual" so the automatic count is never inflated by them.
MANUAL: dict[str, dict] = {
    "krippendorff2004content": {
        "kind": "book",
        "checked": "hand-checked; Sage monograph not reliably indexed by Crossref",
        "fields": {
            "title": "Content Analysis: An Introduction to Its Methodology",
            "author": "Klaus Krippendorff",
            "year": "2004",
            "edition": "2nd",
            "publisher": "Sage Publications",
            "address": "Thousand Oaks, CA",
        },
    },
}


# NFKD decomposes accented letters into a base plus a combining mark, but a handful of
# Latin letters have no decomposition at all. Without this table the ASCII filter below
# deletes them outright, turning Søgaard into "s gaard" and Weiß into "wei".
TRANSLITERATE = str.maketrans({
    "\u00f8": "o", "\u0142": "l", "\u0111": "d", "\u00f0": "d", "\u00fe": "th",
    "\u0131": "i", "\u00e6": "ae", "\u0153": "oe", "\u00df": "ss",
})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.translate(TRANSLITERATE)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


# A transport failure is not evidence that a reference is bad. Sources that error out are
# recorded here so an unverified entry caused by a network outage is never confused with one
# caused by a genuine mismatch, and never silently overwrites a good bibliography.
TRANSPORT_ERRORS: list[str] = []


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def _get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed hosts
        return r.read()


def search_crossref(title: str) -> list[dict]:
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 5, "mailto": MAILTO})
    try:
        data = json.loads(_get(f"https://api.crossref.org/works?{q}"))
    except Exception as e:  # noqa: BLE001
        log(f"    crossref error: {type(e).__name__}")
        TRANSPORT_ERRORS.append(f"crossref:{type(e).__name__}")
        return []
    out = []
    for it in data.get("message", {}).get("items", []):
        auths = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
                 for a in it.get("author", []) or []]
        yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        # cheap retraction/correction screen -- no other audit layer covers this
        rel = " ".join((it.get("relation") or {}).keys()).lower()
        flagged = (it.get("type") == "retraction" or "update-to" in it
                   or any(w in rel for w in ("retract", "correct", "concern")))
        out.append({"title": (it.get("title") or [""])[0], "year": yr, "authors": auths,
                    "venue": (it.get("container-title") or [None])[0], "doi": it.get("DOI"),
                    "arxiv": None, "type": it.get("type"),
                    "citations": it.get("is-referenced-by-count"),
                    "retraction_flag": flagged, "source": "crossref"})
    return out


def search_openalex(title: str) -> list[dict]:
    q = urllib.parse.urlencode({"search": title, "per-page": 5, "mailto": MAILTO})
    try:
        data = json.loads(_get(f"https://api.openalex.org/works?{q}"))
    except Exception as e:  # noqa: BLE001
        log(f"    openalex error: {type(e).__name__}")
        TRANSPORT_ERRORS.append(f"openalex:{type(e).__name__}")
        return []
    out = []
    for w in data.get("results", []):
        loc = (w.get("primary_location") or {}).get("source") or {}
        ids = w.get("ids") or {}
        arx = None
        for k in ("arxiv", "mag"):
            v = str(ids.get(k) or "")
            if "arxiv" in v:
                arx = v.rsplit("/", 1)[-1]
        out.append({"title": w.get("display_name") or "",
                    "year": w.get("publication_year"),
                    "authors": [(a.get("author") or {}).get("display_name", "")
                                for a in w.get("authorships", [])],
                    "venue": loc.get("display_name"),
                    "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
                    "arxiv": arx, "type": w.get("type"),
                    "citations": w.get("cited_by_count"),
                    "retraction_flag": bool(w.get("is_retracted")),
                    "source": "openalex"})
    return out


def search_arxiv(title: str) -> list[dict]:
    clean = re.sub(r"[^A-Za-z0-9 ]", " ", title)
    q = urllib.parse.urlencode({"search_query": f'all:"{clean[:150]}"', "max_results": 5})
    try:
        xml = _get(f"https://export.arxiv.org/api/query?{q}", timeout=12)
    except Exception as e:  # noqa: BLE001
        log(f"    arxiv error: {type(e).__name__}")
        TRANSPORT_ERRORS.append(f"arxiv:{type(e).__name__}")
        return []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        eid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        pub = e.findtext("a:published", "", ns) or ""
        out.append({"title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "year": int(pub[:4]) if pub[:4].isdigit() else None,
                    "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
                    "venue": "arXiv preprint", "doi": None, "arxiv": eid, "type": "preprint",
                    "citations": None, "retraction_flag": False, "source": "arxiv"})
    return out


def fetch_doi(doi: str) -> list[dict]:
    """Resolve one DOI. A short generic title such as "Topology and data" cannot be found by
    bibliographic search, so the DOI is supplied and the title/author gate still has to pass."""
    try:
        it = json.loads(_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))["message"]
    except Exception as e:  # noqa: BLE001
        log(f"    crossref doi error: {type(e).__name__}")
        TRANSPORT_ERRORS.append(f"crossref-doi:{type(e).__name__}")
        return []
    auths = [" ".join(x for x in (a.get("given"), a.get("family")) if x)
             for a in it.get("author", []) or []]
    yr = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return [{"title": (it.get("title") or [""])[0], "year": yr, "authors": auths,
             "venue": (it.get("container-title") or [None])[0], "doi": it.get("DOI"),
             "arxiv": None, "type": it.get("type"),
             "citations": it.get("is-referenced-by-count"),
             "retraction_flag": False, "source": "crossref-doi"}]


def verify(key, title, surname, kind, purpose, doi=None) -> dict:
    log(f"  {key}")
    before = len(TRANSPORT_ERRORS)
    if doi:
        cands = fetch_doi(doi)
        tried = ["crossref-doi"]
    else:
        cands = search_crossref(title)
        tried = ["crossref"]
    time.sleep(0.6)

    def best_of(cs):
        return max(cs, key=lambda c: sim(title, c["title"]), default=None)

    best = best_of(cands)
    if best is None or sim(title, best["title"]) < SIM_GATE:
        cands += search_openalex(title)
        tried.append("openalex")
        time.sleep(0.6)
        best = best_of(cands)
    if best is None or sim(title, best["title"]) < SIM_GATE:
        cands += search_arxiv(title)
        tried.append("arxiv")
        time.sleep(3.0)  # arXiv asks for >=3s between calls
        best = best_of(cands)

    best_s = sim(title, best["title"]) if best else 0.0
    # Both sides go through norm(). Lowercasing only the wanted surname made every
    # non-ASCII name fail the check while the retrieved name was being folded to ASCII.
    author_ok = bool(best) and norm(surname) in " ".join(norm(a) for a in (best["authors"] or []))
    verified = bool(best) and best_s >= SIM_GATE and author_ok
    errored = TRANSPORT_ERRORS[before:]

    if verified:
        provenance = "lookup"
    elif key in MANUAL:
        provenance = "manual"
    elif errored:
        provenance = "inconclusive"
    else:
        provenance = "none"

    rec = {"key": key, "wanted_title": title, "wanted_first_author": surname, "type": kind,
           "purpose": purpose, "verified": verified, "title_similarity": round(best_s, 3),
           "first_author_confirmed": author_ok, "sources_tried": tried, "match": best,
           "provenance": provenance, "transport_errors": errored}
    log(f"    {'VERIFIED  ' if verified else 'UNVERIFIED'} sim={best_s:.2f} author={author_ok}"
        + (f" [{best['source']}] {best['title'][:64]}" if best else " (no candidates)"))
    if provenance == "manual":
        rec["manual"] = MANUAL[key]
        log(f"    MANUAL     lookup failed; using hand-checked record "
            f"({MANUAL[key]['checked']})")
    elif provenance == "inconclusive":
        log(f"    INCONCLUSIVE lookup could not run: {', '.join(errored)}")
    if best and best.get("retraction_flag"):
        log("    !! Crossref flags a retraction/correction relation - inspect before citing")
    return rec


def _tidy_author(name: str) -> str:
    """OpenAlex occasionally returns one author of a list as 'Last, First'."""
    if name.count(",") == 1:
        last, first = (p.strip() for p in name.split(","))
        if last and first:
            return f"{first} {last}"
    return name.strip()


def to_bibtex(rec: dict) -> str:
    if man := rec.get("manual"):
        body = ",\n".join(f"  {k:9s} = {{{v}}}" for k, v in man["fields"].items())
        return (f"% hand-checked, not machine-verifiable: {man['checked']}\n"
                f"@{man['kind']}{{{rec['key']},\n{body}\n}}\n")
    if not rec["verified"]:
        # The placeholder keeps the cited key. Emitting a different key leaves the \citep
        # undefined, which renders as a silent '??' -- the least visible possible failure.
        # Keeping the key puts the warning in the printed bibliography instead.
        return ("% EXPLICIT PLACEHOLDER - could not verify programmatically. Requires human check.\n"
                f"% author : {rec['wanted_first_author']} et al.\n"
                f"% best title similarity {rec['title_similarity']}, "
                f"first-author confirmed: {rec['first_author_confirmed']}\n"
                f"@misc{{{rec['key']},\n"
                f"  title = {{[UNVERIFIED CITATION - VERIFY BEFORE SUBMISSION] "
                f"{rec['wanted_title']}}},\n"
                f"  note  = {{Programmatic verification failed; confirm by hand.}}\n}}\n")
    m = rec["match"]
    doi = m.get("doi") or ""
    arxiv_id = m.get("arxiv") or ""
    if not arxiv_id and doi.lower().startswith("10.48550/arxiv."):
        arxiv_id = doi.split("arxiv.", 1)[1]
    venue = m.get("venue") or ""
    # OpenAlex reports preprints with venue "arXiv (Cornell University)". Recording that as
    # a booktitle would assert a proceedings that does not exist. Several of these appeared
    # at conferences, but the conference record is not what was verified here, so the entry
    # states the preprint that was.
    is_preprint = "arxiv" in venue.lower() or m.get("type") == "preprint"
    kind = "misc" if is_preprint else rec["type"]

    authors = " and ".join(_tidy_author(a) for a in m["authors"]) if m["authors"] else "Unknown"
    f = [f"  title   = {{{m['title']}}}", f"  author  = {{{authors}}}",
         f"  year    = {{{m['year']}}}"]
    if is_preprint:
        f.append(f"  howpublished = {{arXiv:{arxiv_id}}}" if arxiv_id
                 else "  howpublished = {preprint}")
    elif venue:
        f.append(f"  {'journal' if rec['type'] == 'article' else 'booktitle'} = {{{venue}}}")
    if doi:
        f.append(f"  doi     = {{{doi}}}")
    if arxiv_id:
        f.append(f"  eprint  = {{{arxiv_id}}},\n  archivePrefix = {{arXiv}}")
    return f"@{kind}{{{rec['key']},\n" + ",\n".join(f) + "\n}\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    log(f"verifying {len(WANTED)} references (title gate {SIM_GATE}, first author must match)")
    recs = [verify(*w) for w in WANTED]

    n_ok = sum(r["provenance"] == "lookup" for r in recs)
    n_manual = sum(r["provenance"] == "manual" for r in recs)
    n_incon = sum(r["provenance"] == "inconclusive" for r in recs)
    n_bad = sum(r["provenance"] == "none" for r in recs)

    # Refuse to rewrite the bibliography from a run whose lookups could not complete.
    # Overwriting here would turn a transient outage into placeholder entries in a file
    # that was previously correct, which is a silent downgrade of a submission artefact.
    if n_incon and "--force" not in sys.argv:
        log(f"ABORT: {n_incon} entr(ies) inconclusive because a source could not be reached "
            f"({len(TRANSPORT_ERRORS)} transport error(s)).")
        log("       references.bib left untouched. Re-run when the network is available, "
            "or pass --force to overwrite anyway.")
        raise SystemExit(2)

    header = ("% Bibliography for the CAIRN study.\n"
              "% Machine-verified entries were retrieved from Crossref, OpenAlex or arXiv\n"
              f"% and passed a title-similarity gate of {SIM_GATE} AND first-author\n"
              "% confirmation. Monographs the APIs do not index carry a hand-checked record,\n"
              "% labelled as such. Anything neither machine-verified nor hand-checked is left\n"
              "% as an explicit UNVERIFIED entry - never as a guess.\n\n")
    (OUT / "references.bib").write_text(header + "\n".join(to_bibtex(r) for r in recs),
                                        encoding="utf-8")
    (RES / "citation_verification.json").write_text(
        json.dumps({"gate": SIM_GATE, "n_verified": n_ok, "n_manual": n_manual,
                    "n_inconclusive": n_incon, "n_unverified": n_bad, "n_total": len(recs),
                    "transport_errors": TRANSPORT_ERRORS,
                    "records": recs}, indent=2), encoding="utf-8")

    log(f"VERIFIED by lookup {n_ok}/{len(recs)}; hand-checked {n_manual}; unverified {n_bad}")
    for r in recs:
        if r["provenance"] == "manual":
            log(f"  HAND-CHECKED {r['key']}: {r['manual']['checked']}")
        elif r["provenance"] == "none":
            log(f"  UNVERIFIED {r['key']}: needed for {r['purpose']}")


if __name__ == "__main__":
    main()
