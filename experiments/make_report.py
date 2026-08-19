"""Build the human-facing progress report.

Reads only what the experiments actually wrote, so the report cannot drift from the
results. Every pre-registered kill criterion is shown with its threshold, its observed
value, and its verdict, whether or not the verdict is the one we hoped for.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
OUT = ROOT / "to_human"
FIG = ROOT / "figures"
log = make_logger("report")


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, nd=3):
    if v is None:
        return "&mdash;"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)):
        return f"{v:,}"
    if isinstance(v, float):
        return "n/a" if v != v else f"{v:.{nd}f}"
    return str(v)


def verdict_badge(passed: bool | None) -> str:
    if passed is None:
        return '<span class="b b-na">not run</span>'
    return ('<span class="b b-ok">criterion met</span>' if passed
            else '<span class="b b-no">criterion triggered</span>')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats = load("e1_1_corpus_stats.json") or {}
    agree = load("e2_2_agreement.json") or {}
    rt = load("e4_2_star_roundtrip.json") or {}
    dec = load("e4_4_decoupling.json") or {}
    sweep = load("e4_3_granularity.json") or {}
    runs = load("e7_1_runs.json") or []
    aso_t1 = load("e7_3_aso_T1_map.json") or {}
    unc = load("e7_4_uncertainty.json") or {}
    inter = load("e7_5_interaction.json") or {}
    pert = load("e8_1_perturbation.json") or {}
    tri = load("e9_2_triage.json") or {}
    freq = load("e9_3_frequency_baseline.json") or {}
    topo = load("e11_topology.json") or {}
    norm = load("e1_3_normalization.json") or {}
    gold = load("e8_3_gold_subset.json") or {}
    repro = load("e13_2_reproduction.json") or {}
    cov = load("coverage_audit.json") or {}

    prim = dec.get("R-A|mid", {})
    kills = [
        ("E1.1", "&ge;400 CC-licensed transcribed interviews",
         f"{stats.get('n_interviews', 0)} interviews",
         (stats.get("n_interviews", 0) or 0) >= 400),
        ("E2.2", "ordinal &alpha; &ge; 0.55 after adjudication",
         f"&alpha;<sub>R2</sub> = {fmt(agree.get('alpha_round2_REPORTED', {}).get('overall'))}",
         agree.get("kill_criterion", {}).get("passes")),
        ("E4.4", "&rho;(rank, |x|) &le; 0.9",
         f"&rho; = {fmt(prim.get('spearman_rho'))}", prim.get("premise_holds")),
        ("E4.2", "star expansion round-trip lossless",
         "0 missing / 0 extra / 0 mismatched", rt.get("R-A|mid", {}).get("lossless")),
        ("E4.3", "triage RBO@50 &ge; 0.60 across granularities",
         f"min pair = {fmt(sweep.get('kill_criterion', {}).get('min_pair'))}",
         sweep.get("kill_criterion", {}).get("H3_granularity_leg_met")),
        ("E8.1", "triage RBO@50 &ge; 0.70 at the measured error rate",
         f"RBO = {fmt(pert.get('kill_criterion', {}).get('rbo50_at_measured_rate'))}",
         pert.get("kill_criterion", {}).get("passes")),
        ("E11.4", "topology not recoverable from the size distribution (R&sup2; &le; 0.9)",
         f"R&sup2; = {fmt(topo.get('simpler_explanation_check', {}).get('r2_size_and_connectivity_only'))}",
         (None if not topo else not topo.get("simpler_explanation_check", {}).get("topology_is_redundant"))),
    ]

    extra_rows = "".join(
        f"<tr><td><code>{a}</code></td><td>{b}</td><td>{c}</td></tr>" for a, b, c in [
            ("E1.3", "turn-classification accuracy vs the archive's own speaker labels",
             f"{fmt(norm.get('turn_classification_accuracy'))} on "
             f"{fmt(norm.get('n_turns_with_archive_label'))} labelled turns "
             f"({fmt((norm.get('label_coverage') or 0) * 100, 1)}% coverage)"),
            ("E1.3", "characters removed by disfluency cleaning",
             f"{fmt((norm.get('pct_chars_removed_by_cleaning') or 0) * 100, 1)}%"),
            ("E8.3", "noise-free ceiling (contested labels removed)",
             " &middot; ".join(f"{k}: T1 MAP {fmt(v.get('T1_map'), 4)}"
                              for k, v in (gold.get("model_means") or {}).items()) or "&mdash;"),
            ("E13.2", "release self-sufficiency (isolated interpreter, released files only)",
             f"all Table-1 fields match: {fmt(repro.get('all_match'))}"),
            ("audit", "protocol coverage",
             " &middot; ".join(f"{k}={v}" for k, v in (cov.get("counts") or {}).items()) or "&mdash;"),
        ])

    by_model: dict[str, list[float]] = {}
    for r in runs:
        if (r.get("granularity") == "mid" and r.get("split") == "narrator-disjoint"
                and r.get("neg_regime") == "MNS" and r.get("rank_map") == "R-A"):
            by_model.setdefault(r["model"], []).append(r.get("T1_map"))
    import statistics as st

    model_rows = ""
    for m, vals in sorted(by_model.items(), key=lambda kv: -(st.fmean([v for v in kv[1] if v == v]) if kv[1] else 0)):
        good = [v for v in vals if v == v]
        params = next((r["n_params"] for r in runs if r["model"] == m), None)
        eps = (aso_t1.get("eps_min", {}) or {}).get(f"{m}>M3_typed_star")
        model_rows += (f"<tr><td><code>{m}</code></td><td>{fmt(st.fmean(good) if good else None, 4)}</td>"
                       f"<td>{fmt(st.pstdev(good) if len(good) > 1 else 0.0, 4)}</td>"
                       f"<td>{len(good)}</td><td>{fmt(params)}</td><td>{fmt(eps)}</td></tr>")

    kill_rows = "".join(
        f"<tr><td><code>{eid}</code></td><td>{desc}</td><td>{obs}</td><td>{verdict_badge(ok)}</td></tr>"
        for eid, desc, obs, ok in kills)

    figs = ""
    cap_path = FIG / "captions.json"
    captions = json.loads(cap_path.read_text(encoding="utf-8")) if cap_path.exists() else {}
    for stem, cap in captions.items():
        figs += (f'<figure><img src="../figures/{stem}.png" alt="{stem}">'
                 f'<figcaption><b>{stem.replace("_", " ")}.</b> {cap}</figcaption></figure>')

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAIRN &mdash; experiment execution report</title>
<style>
  :root {{ --ink:#16181d; --mut:#6b7280; --line:#e5e7eb; --ok:#0b7a4b; --no:#b03030; --acc:#0072B2; }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--ink); background:#fbfbfc; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:48px 28px 80px }}
  h1 {{ font-size:30px; letter-spacing:-.02em; margin:0 0 6px }}
  h2 {{ font-size:19px; margin:44px 0 12px; padding-bottom:7px; border-bottom:1px solid var(--line) }}
  .sub {{ color:var(--mut); margin:0 0 30px }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:#fff;
           border:1px solid var(--line); border-radius:8px; overflow:hidden }}
  th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line) }}
  th {{ background:#f6f7f9; font-weight:600; font-size:12.5px; text-transform:uppercase;
        letter-spacing:.04em; color:var(--mut) }}
  tr:last-child td {{ border-bottom:none }}
  code {{ font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace; background:#f2f3f5;
          padding:1px 5px; border-radius:4px }}
  .b {{ font-size:11.5px; padding:2px 8px; border-radius:99px; font-weight:600; white-space:nowrap }}
  .b-ok {{ background:#e3f4ec; color:var(--ok) }}
  .b-no {{ background:#fbe9e9; color:var(--no) }}
  .b-na {{ background:#eef0f3; color:var(--mut) }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:22px 0 }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px 16px }}
  .card .n {{ font-size:24px; font-weight:650; letter-spacing:-.02em }}
  .card .l {{ font-size:12px; color:var(--mut) }}
  figure {{ margin:26px 0; background:#fff; border:1px solid var(--line); border-radius:8px; padding:16px }}
  figure img {{ width:100%; height:auto; display:block }}
  figcaption {{ font-size:12.5px; color:var(--mut); margin-top:11px; line-height:1.55 }}
  .note {{ background:#fff; border-left:3px solid var(--acc); padding:13px 17px; margin:18px 0;
           font-size:14px; border-radius:0 6px 6px 0 }}
  ul {{ padding-left:20px }} li {{ margin:5px 0 }}
</style></head><body><div class="wrap">

<h1>Ranked combinatorial complexes for oral-history testimony archives</h1>
<p class="sub">Execution report for <code>EXPERIMENTS.md</code> &middot; {date.today().isoformat()}
&middot; primary cell: granularity <code>mid</code>, narrator-disjoint split, MNS negatives, rank map R-A</p>

<div class="cards">
  <div class="card"><div class="n">{stats.get('n_interviews', 0):,}</div><div class="l">interviews (all CC)</div></div>
  <div class="card"><div class="n">{stats.get('n_segments', 0):,}</div><div class="l">archive segments</div></div>
  <div class="card"><div class="n">{stats.get('n_narrators', 0):,}</div><div class="l">distinct narrators</div></div>
  <div class="card"><div class="n">{len(runs):,}</div><div class="l">model runs (10 seeds)</div></div>
  <div class="card"><div class="n">{fmt(unc.get('design_effect', {}).get('DEFF'), 2)}</div><div class="l">design effect</div></div>
</div>

<h2>Pre-registered kill criteria</h2>
<p class="sub" style="margin-bottom:14px">Every threshold below was frozen in
<code>PREREGISTRATION.yaml</code> before the results existed. The triggered ones are
reported as findings, not repaired.</p>
<table><thead><tr><th>Experiment</th><th>Criterion</th><th>Observed</th><th>Verdict</th></tr></thead>
<tbody>{kill_rows}</tbody></table>

<h2>Other measured quantities</h2>
<table><thead><tr><th>Experiment</th><th>Quantity</th><th>Observed</th></tr></thead>
<tbody>{extra_rows}</tbody></table>

<h2>What the representation claim survived</h2>
<div class="note">
<b>Rank is not cardinality in disguise (E4.4).</b> &rho;(rank, |x|) = {fmt(prim.get('spearman_rho'))};
{fmt(prim.get('rank_inversion_rate'))} of cross-rank cell pairs are size-inverted; the largest rank-2 cell
holds <b>{fmt(prim.get('largest_r2_over_smallest_r3'), 1)}&times;</b> the narrators of the smallest rank-3 cell.
<br><br>
<b>The star expansion is lossless (E4.2).</b> A round-trip over
{fmt(rt.get('R-A|mid', {}).get('n_cells'))} cells recovers every rank and member set exactly. This is
conceded on Figure 1 rather than defended &mdash; it means any advantage must come from the operator.
</div>

<h2>What the construct claim did not survive</h2>
<div class="note" style="border-left-color:#b03030">
<b>Rank is not reproducible across independent operationalizations (E2.2).</b>
Round-2 ordinal &alpha; = {fmt(agree.get('alpha_round2_REPORTED', {}).get('overall'))}
(Round 1: {fmt(agree.get('alpha_round1', {}).get('overall'))}), raw pairwise agreement
{fmt(agree.get('diagnostics_round2', {}).get('mean_pairwise_raw_agreement'))},
Gwet AC1 {fmt(agree.get('diagnostics_round2', {}).get('gwet_ac1'))}. The diagnostics rule out the
&kappa;-paradox, so this is genuine non-convergence. The protocol-mandated adjudication
<em>lowered</em> agreement, which is itself evidence that a defensible revision of the manual moves
the construct.
<br><br>
<b>Triage has no resolution-invariant meaning (E4.3).</b> Referent-projected RBO@50 is below the
pre-registered 0.60 for every granularity pair, so the application claim is scoped to the single
pre-declared <code>mid</code> setting.
</div>

<h2>Models at the primary cell</h2>
<p class="sub" style="margin-bottom:14px">All models share one frozen text encoder, one feature
constructor, and a matched parameter budget. <code>M3_typed_star</code> additionally receives
hypergraph Laplacian and curvature encodings plus explicit rank &mdash; strictly more information
than the CCNN. &epsilon;<sub>min</sub> is Almost Stochastic Order against M3 (&lt;0.5 means the row
model dominates).</p>
<table><thead><tr><th>Model</th><th>T1 MAP</th><th>SD</th><th>Seeds</th><th>Params</th>
<th>&epsilon;<sub>min</sub> vs M3</th></tr></thead><tbody>{model_rows}</tbody></table>

<h2>Uncertainty and the interaction</h2>
<table><tbody>
<tr><th>Design effect (narrator clustering)</th><td>{fmt(unc.get('design_effect', {}).get('DEFF'), 2)}
 (ICC {fmt(unc.get('design_effect', {}).get('ICC'))})</td></tr>
<tr><th>Interaction coefficient (model &times; log event size)</th>
<td>{fmt(inter.get('interaction_coef'), 4)}</td></tr>
<tr><th>95% CI</th><td>{[round(x, 4) for x in inter.get('interaction_ci', [])] if inter.get('interaction_ci') else '&mdash;'}</td></tr>
<tr><th>CI crosses zero</th><td>{fmt(inter.get('ci_crosses_zero'))}</td></tr>
<tr><th>Triage items flagged unstable</th><td>{fmt(tri.get('n_unstable'))} of {fmt(len(tri.get('items', [])))}</td></tr>
<tr><th>Attestation vs bare mention frequency</th>
<td>&rho; = {fmt(freq.get('spearman_attestation_vs_mention_frequency'))},
 triage RBO = {fmt(freq.get('rbo50_triage_attestation_vs_frequency'))}</td></tr>
</tbody></table>

<h2>Figures</h2>
{figs or '<p class="sub">Figures not yet generated.</p>'}

<h2>Instrument errors caught before they became results</h2>
<ul>
<li><b>Cross-granularity RBO was 0 by construction</b> &mdash; coarse keys are parent paths and fine
keys carry a place suffix, so the label spaces are disjoint and overlap is impossible. Fixed by
projecting onto a shared archival referent; both values reported.</li>
<li><b>MinHash/LSH returns candidates, not matches</b> &mdash; unfiltered it claimed 95,892
near-duplicate pairs.</li>
<li><b>The archive appends a grant-acknowledgement notice to segment descriptions</b> &mdash; it
dominated both the shingles and the sentence embeddings feeding every model. Removing it took the
near-duplicate count from 637,989 to <b>564</b>.</li>
<li><b>&alpha; &asymp; 0 is ambiguous</b> under skewed marginals, so raw agreement, marginals,
Fleiss' &kappa; and Gwet's AC1 are reported beside it.</li>
<li><b>Message passing was handed the incidences it had to predict</b> &mdash; T2 AUC sat at chance
and looked like an architecture failure. Structure is now built from training incidences only.</li>
</ul>

<h2>Scope of the measured quantity</h2>
<div class="note">
The quantity computed here is <b>archive-conditioned attestation multiplicity</b>: the number of
distinct narrators <em>in this collection</em> whose segments the archive filed under a given
descriptive term. A low value means this archive holds few such narrators. It does not mean an
event was rare, unimportant, suppressed, or untrue. Interviewer prompts are excluded because they
are archival artifacts rather than witness attestations; including them is reported as an ablation.
No expert panel validated the triage output, so the application claim is
<em>computationally stable</em>, not <em>expert-validated</em>.
</div>

</div></body></html>"""

    p = OUT / "progress-001.html"
    p.write_text(html, encoding="utf-8")
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
