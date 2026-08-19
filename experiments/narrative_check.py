"""Narrative diagnostics for the manuscript.

Implements the measurable checks from the conference-narrative pass: negation rate,
abstract length and number count, incantation counts, caption takeaways, hedge budget,
revoke cadence, and number density in prose. Reported before and after revision so the
effect of an edit is visible rather than asserted.

LaTeX is stripped before measuring. Tables, math and the bibliography are excluded, since
a results table legitimately contains many numerals and would swamp the prose density.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

PAPER = ROOT / "paper" / "main.tex"
OUT = ROOT / "data" / "results" / "narrative_diagnostics.json"
log = make_logger("narrative")

INCANTATIONS = ["post-hoc", "pre-registered", "preregistered", "registered", "robust",
                "novel", "importantly", "note that", "significant"]

# A scope limiter withdraws reach from a claim. Bare negation does not qualify: this paper's
# findings ARE negative ("rank is not cardinality", "the complex does not beat the star"),
# and counting those as hedges flags the contribution as timidity. The pattern therefore
# matches constructions that retract or fence, not ones that assert something negative.
HEDGE = re.compile(
    r"\b(?:however|although|though|caveat|that said|we do not claim|we make no claim|"
    r"should not be (?:read|taken|interpreted)|does not (?:mean|imply|establish|generalise|"
    r"generalize)|cannot be (?:read|taken|concluded|ruled out)|is limited to|only a|"
    r"merely|no stronger than|does not follow|beyond the scope|we cannot)\b", re.I)


def strip_latex(tex: str) -> str:
    tex = re.sub(r"%.*", "", tex)
    tex = re.sub(r"\\begin\{(tabular|table|figure|thebibliography)\}.*?\\end\{\1\}", " ",
                 tex, flags=re.S)
    tex = re.sub(r"\$[^$]*\$", " NUM ", tex)
    tex = re.sub(r"\\(?:cite\w*|ref|label|input|includegraphics|documentclass|usepackage|"
                 r"bibliographystyle|bibliography)\s*(?:\[[^\]]*\])?\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\(?:begin|end)\{[^}]*\}", " ", tex)
    tex = re.sub(r"\\[A-Za-z]+\{\}", " NUM ", tex)          # generated number macros
    tex = re.sub(r"\\[A-Za-z]+", " ", tex)
    return re.sub(r"[{}]", " ", tex)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def section_map(tex: str) -> dict[str, str]:
    parts = re.split(r"\\section\{([^}]*)\}", tex)
    out = {}
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            out[parts[i]] = parts[i + 1]
    return out


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]


def count_incantations(body: str) -> dict[str, int]:
    """Count each precision word once.

    Naive counting double-charges nested forms: every 'pre-registered' also matches
    'registered', which reported nine occurrences of a word used three times and would have
    sent an edit after phrasing that was already within budget. Longer forms are consumed
    first and their spans masked out.
    """
    text = body
    counts: dict[str, int] = {}
    for word in sorted(INCANTATIONS, key=len, reverse=True):
        pat = re.compile(r"(?<![A-Za-z-])" + re.escape(word) + r"(?![A-Za-z])", re.I)
        found = pat.findall(text)
        counts[word] = len(found)
        text = pat.sub(lambda m: " " * len(m.group(0)), text)
    return counts


def diagnose(tex: str) -> dict:
    abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    abstract_raw = abstract_m.group(1) if abstract_m else ""
    abstract = strip_latex(abstract_raw)
    abs_words = len(abstract.split())
    # A generated macro renders as one number; count macro invocations plus literal numerals.
    abs_numbers = len(re.findall(r"\\[A-Za-z]+\{\}", abstract_raw)) + \
        len(re.findall(r"(?<![\\A-Za-z])\d[\d.,]*", strip_latex(
            re.sub(r"\\[A-Za-z]+\{\}", " ", abstract_raw))))

    body = strip_latex(tex)
    sents = sentences(body)
    neg = [s for s in sents
           if re.search(r"\b(not|no|neither|nor|cannot|never)\b", s, re.I)]

    caps = re.findall(r"\\caption\{(.*?)\}\s*\n?\s*\\label", tex, re.S)
    caps += re.findall(r"\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tex)
    cap_violations = []
    for c in {re.sub(r"\s+", " ", strip_latex(c)).strip() for c in caps}:
        first = re.split(r"[.;]", c)[0]
        # A takeaway makes an assertion; a label just says what is plotted.
        if not re.search(r"\b(is|are|does|do|inverts?|holds?|fails?|beats?|leads?|shows?|"
                         r"rises?|falls?|collapses?|improves?|decides?|erases?|reproduces?|"
                         r"survives?|carries|carry|buys?|changes?|disappears?|removes?|"
                         r"eliminates?|explains?|predicts?)\b",
                         first, re.I):
            cap_violations.append(first[:90])

    inc = count_incantations(body)
    inc = {k: v for k, v in inc.items() if v}

    secs = section_map(tex)
    hedge_violations, revokes = [], []
    for name, txt in secs.items():
        clean = strip_latex(txt)
        n_hedge = sum(bool(HEDGE.search(s)) for s in sentences(clean))
        if n_hedge > 1:
            hedge_violations.append({"section": name, "scope_sentences": n_hedge})
        for p in paragraphs(clean):
            ss = sentences(p)
            if len(ss) >= 2 and HEDGE.search(ss[-1]):
                revokes.append({"section": name, "ends_with": ss[-1][:110]})

    results = " ".join(txt for name, txt in secs.items()
                       if re.search(r"P1|P2|P3|split|event layer", name, re.I))
    rclean = strip_latex(results)
    rwords = max(1, len(rclean.split()))
    density = 100 * len(re.findall(r"\bNUM\b", rclean)) / rwords

    return {
        "n_sentences": len(sents),
        "negation_rate": round(len(neg) / max(1, len(sents)), 3),
        "abstract_words": abs_words,
        "abstract_numbers": abs_numbers,
        "caption_violations": cap_violations,
        "incantations": inc,
        "hedge_budget_violations": hedge_violations,
        "revoke_cadence": revokes,
        "number_density_results": round(density, 2),
        "sections": list(secs),
    }


TARGETS = {
    "negation_rate": (0.08, "<="),
    "abstract_words": (180, "<="),
    "abstract_numbers": (5, "<="),
    "number_density_results": (6.0, "<="),
}


def report(d: dict) -> list[str]:
    lines = []
    for k, (target, op) in TARGETS.items():
        v = d[k]
        ok = v <= target if op == "<=" else v >= target
        lines.append(f"  {'PASS' if ok else 'MISS'} {k:26s} {v:<8} target {op} {target}")
    lines.append(f"  {'PASS' if not d['caption_violations'] else 'MISS'} "
                 f"{'caption_takeaways':26s} {len(d['caption_violations'])} violations")
    lines.append(f"  {'PASS' if not d['hedge_budget_violations'] else 'MISS'} "
                 f"{'hedge_budget':26s} {len(d['hedge_budget_violations'])} sections over budget")
    lines.append(f"  {'PASS' if not d['revoke_cadence'] else 'MISS'} "
                 f"{'revoke_cadence':26s} {len(d['revoke_cadence'])} paragraphs")
    over = {k: v for k, v in d["incantations"].items() if v > 3}
    lines.append(f"  {'PASS' if not over else 'MISS'} {'incantation_cap':26s} "
                 f"{over if over else 'all <= 3'}")
    return lines


def main() -> None:
    d = diagnose(PAPER.read_text(encoding="utf-8"))
    # The negation target assumes a paper whose findings are positive and whose negations are
    # therefore defensive padding. Here the negations ARE the findings -- "rank is not
    # cardinality", "the complex does not beat the star", "the split, not the architecture,
    # decides the answer". Cutting them to hit 8% would cut the contribution, so the miss is
    # recorded and explained rather than optimised away.
    d["deliberate_misses"] = [{
        "metric": "negation_rate",
        "observed": d["negation_rate"],
        "target": TARGETS["negation_rate"][0],
        "justification": ("the paper's central claims are negative findings; the flagged "
                          "sentences state results rather than hedge them"),
    }] if d["negation_rate"] > TARGETS["negation_rate"][0] else []

    log(f"narrative diagnostics over {d['n_sentences']} sentences, "
        f"{len(d['sections'])} sections")
    for line in report(d):
        log(line)
    for m in d["deliberate_misses"]:
        log(f"    deliberate miss on {m['metric']}: {m['justification']}")
    for c in d["caption_violations"]:
        log(f"    caption without a takeaway: {c}")
    for h in d["hedge_budget_violations"]:
        log(f"    hedge budget: {h['section']} has {h['scope_sentences']} scope sentences")
    for r in d["revoke_cadence"]:
        log(f"    revoke cadence in {r['section']}: ...{r['ends_with']}")
    OUT.write_text(json.dumps(d, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
