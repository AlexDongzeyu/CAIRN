# Research Findings

## Research Question

Does rank-aware higher-order message passing over an archive-derived combinatorial complex
buy anything over an information-equivalent typed star graph — and is the resulting
archive-conditioned attestation-multiplicity triage list stable enough to act on?

## Current Understanding

*(updated after each outer-loop cycle)*

The Densho Visual History Collection turns out to supply the rank ladder directly, which
removes the paper's most obvious attack surface. Rank does not have to be invented: the
archive's own descriptive vocabulary is a hierarchical facet tree (`A -- B -- C`, depth 1–4),
segments are the archive's own sub-interview curation, and narrators carry stable identifiers.
So rank-1 *moments* are archive-given rather than author-imposed, and rank-2/rank-3 assignment
is a question about the archive's descriptive practice rather than about our taste.

Two structural claims survive contact with the data; one construct claim does not.

## Key Results

### C1a — rank is not cardinality in disguise (E4.4) — **PASSES**

Spearman ρ(rank, |cell|) = **0.35–0.46** across granularities, far below the pre-registered
0.9 failure threshold. The demonstration is stronger than the correlation: the largest rank-2
cell holds **79×** the narrators of the smallest rank-3 cell, and 95% of cross-rank cell pairs
are size-inverted. A rank-3 cell can be tiny (`Japanese Latin Americans`, 1 narrator) while a
rank-2 cell is large (`World War II -- Pearl Harbor and aftermath`, 79 narrators). This is the
empirical content of the representational claim and it costs twenty lines of code.

### C1b — the star expansion is lossless (E4.2) — **VERIFIED, NOT ASSUMED**

Round-trip test over 13,996 cells: every cell's rank and exact member set is recoverable from
the star graph alone. 0 missing, 0 extra, 0 mismatched, under all three rank maps. This is
conceded on the figure and in the text rather than defended — the star expansion carries the
same information, so any CCNN advantage must come from the operator, not the representation.

### C1d — the interviewer/narrator distinction can be made reliably (E1.3)

This matters because interviewer turns contain event mentions that are *prompts*, not
attestations; counting them inflates the project's headline quantity. Densho transcripts
supply ground truth — `<Begin Segment N>` markers plus turn-initial speaker initials matched
against the header's `Narrator:` / `Interviewers:` lines — so a rules-only classifier that
never sees those initials can be scored on **171,313 archive-labelled turns** (77.7% of all
turns) rather than on the 200 hand-labelled ones the protocol settles for.

Turn-classification accuracy is **0.806**; disfluency cleaning removes 13.2% of characters,
with `[inaudible]` preserved as an `<INAUD>` sentinel rather than silently dropped. Character
offsets against a canonical rendering are stored for every turn, which is what makes the
stand-off release reconstructable.

### C1c — rank is **not** reproducible across independent operationalizations (E2.2) — **KILL CRITERION TRIGGERED**

Three annotators applying the same decision procedure to disjoint evidence (archive tree
position / term lexis / distributional usage) do not converge. Round-1 ordinal α = 0.073;
after a documented adjudication and manual revision, Round-2 α = **0.039** — the revision made
agreement *worse*. Raw pairwise agreement is 0.530 and Gwet's AC1 is 0.082, so this is genuine
non-convergence rather than the κ-paradox: on a binary decision, 53% agreement is chance.
146 of 211 terms are disputed.

Agreement is worst exactly where it matters most — α is negative on the rare stratum, which is
the stratum the singleton analysis rests on.

The dominant disagreement category is named: *tree-position vs term-semantics conflict*. Whether
`World War II -- Concentration camps -- Living conditions` is a bounded site or a recurring
condition is a genuine archival question, not an annotation slip.

This is pre-registered negative result #4. It means H2's rank attribution must be reported under
all three rank maps, and any claimed rank benefit has to survive R-C.

### C2 — the CCNN does **not** beat the information-equivalent typed star GNN (E6/E7) — **negative result #1**

Primary cell (granularity `mid`, narrator-disjoint split, MNS negatives, R-A), 10 seeds, all
models sharing one frozen encoder, one feature constructor, and a parameter budget matched to
within 1.4%:

| Model | T1 MAP | T2 AUC | params |
|---|---|---|---|
| M0 feature-only MLP | 0.010 | 0.552 | 130,401 |
| M2 untyped star GNN | 0.067 | 0.785 | 130,909 |
| **M3 typed star + hypergraph encodings** | **0.137** | 0.816 | 132,219 |
| M4 AllSet | 0.013 | 0.838 | 131,041 |
| M4 ED-HNN | 0.013 | **0.853** | 131,041 |
| M4 Hypergraph-MLP | 0.010 | 0.549 | 131,041 |
| **M5 CCNN (proposed)** | 0.100 | 0.546 | 128,521 |

M3 — which receives *strictly more* information than the CCNN (hypergraph Laplacian spectra,
Forman-Ricci curvature, and explicit rank as a feature) — leads on retrieval by 37% relative,
and the rank-agnostic hypergraph models lead on incidence prediction. The pre-registered
response applies: report it, do not weaken the baseline.

Ablations of M5 point the same way. Sharing weights across ranks (A1) costs almost nothing
(0.100 → 0.094), so the *rank-specific parameterization* is not carrying the model. Removing
down-messages (A4) costs a great deal (0.100 → 0.055), so what M5 gains is ordinary
bidirectional message passing rather than anything about hierarchy.

### C3 — extraction error bounds the application (E3.5) — **KILL CRITERION TRIGGERED**

Treating the archive's professionally-curated topic assignments as gold and an unsupervised
text-similarity clusterer as the automatic system: LEA F1 peaks at 0.196 (threshold swept until
the optimum was bracketed, not to a boundary), ρ_merge = 0.841, ρ_split = 0.810,
**false-singleton rate = 0.367 > 0.30**.

The archive's topical grouping is not recoverable from segment-summary text similarity. This is
a strong argument *for* the protocol's expensive in-domain annotation step, and it means the
automated triage is a screening tool with a stated precision, not an item-level output.

Note the distinction the numbers force: the complex reported here is built directly from archive
curation, so it carries no extraction error by construction. The E3.5 rates describe what anyone
applying this method to an *unstructured* archive would face, which is why E8.1 is calibrated at
both the archive-internal ambiguity rate and the measured automatic-extractor rate.

### C4 — the split effect has a demonstrated cause: a moment IS its narrator (E-NEW-1/3) — **exploratory**

Run after the split result, so labelled exploratory rather than pre-registered.

`build_features` constructs a rank-*k* cell as the mean of its constituent rank-0 narrator
features. A rank-1 moment has exactly one narrator, so **a moment's input vector is its
narrator's vector** — verified directly: the maximum elementwise spread between two moments by
the same narrator is `0.000e+00`. Only message passing over shared events can separate them.

Two consequences, both measured:

1. *The identity probe needs its control to be readable.* A linear probe recovers narrator
   identity from the **input** features at accuracy **1.000** (majority-class floor 0.011 over
   440 narrators, n = 12,846 — the probe keeps narrators with at least four moments). That is a
   tautology: the classes are identical vectors. After message passing the models still sit near
   that ceiling: 0.861 ± 0.037 (typed star) and 0.902 ± 0.064 (complex) narrator-disjoint;
   0.932 ± 0.026 and 0.943 ± 0.007 random. The complex is higher in both, but seed spreads
   overlap, so the reading is *neither architecture discards narrator identity* — not a ranking
   between them.
2. *Destroying identity destroys the advantage.* Shuffling narrator labels (segments-per-narrator
   preserved) and re-running both splits: the complex's gain from a random split falls from
   **+0.290** to **−0.006** MAP. The typed star, which never had a gain, does not lose one
   (−0.033 → −0.155). The comparison is made on the split *gap*, because anonymisation changes
   which passages count as positives but transforms both splits identically.

So the random split does not merely correlate with the complex's advantage — removing what it
leaks removes the advantage. The leak is mechanical: a random split places *the same input
vector* on both sides of the partition.

**Two confirmatory tests that failed.** First, we predicted the leak would scale with how much
of a narrator sits in training: positive slope of per-narrator advantage on log segment count
under the random split, flat under narrator-disjoint. Neither half held. Random slope +0.0155, CI
[−0.0095, 0.0404], p = 0.225 — crosses zero. The narrator-disjoint control is not flat but
strongly negative, −0.169, CI [−0.241, −0.097]: the complex falls further behind on prolific
narrators exactly when it cannot have seen them. The two slopes differ by +0.185, which would
suggest the random split removes a size penalty rather than adding an advantage — but that
comparison was not registered, has no interval, and needs a slope-by-split interaction we did
not fit. It is an open observation, not evidence.

Second, and more consequentially, we intervened on the hypothesised cause. If the leak were
carried by the moment vector being identical to its narrator's, replacing it should close the
gap. We gave each rank-1 cell its own passage embedding (manipulation check: 13315/13315 rows
replaced, within-narrator spread 0.000e+00 → 2.793e-01). **The gap did not close**: for the
complex it went from +0.3147 to +0.3852, and the typed star was unchanged (−0.0242 → −0.0184).

That refutation is a correction, not a footnote. Only rank-1 features changed; the rank-0
narrator layer and every message path through it remained, and that is what the random split
feeds on. Anonymisation destroys the narrator layer and *does* close the gap. So the supported
claim is narrower than first written: the leak lives in the narrator layer the ladder is built
on, not in the moment vector. The paper's introduction, conclusion, mechanism section and two
figure captions had already asserted the stronger chain and were corrected.

**Also disclosed.** The dense baseline is not information-matched with the trained models: it is
scored on each passage's own frozen embedding, while every trained model sees only its narrator's
mean. It carries strictly more textual information and no structure. The typed-star-vs-complex
comparison — the paper's actual claim — is unaffected, since both sit on identical inputs.

## Patterns and Insights

- **The archive's curation is the asset.** Everything reproducible here (segments, narrator ids,
  facet tree, geography) came from Densho's own descriptive work. Everything fragile (which facet
  is a "site" vs a "programme", and whether topical grouping can be recovered from text) is where
  we had to impose a reading.
- **Cardinality decoupling and rank reproducibility are independent properties.** The complex is
  demonstrably not a hypergraph-with-labels (E4.4 passes decisively) *and* the labels are not
  reproducible (E2.2 fails). Both can be true, and the protocol was right to test them separately.
- **Structure helps; rank-aware structure does not.** Every structural model beats the
  structure-free MLP on incidence prediction by a wide margin (0.55 → 0.79–0.85). But the ordering
  among structural models does not favour the rank ladder, and M5's own ablations attribute its
  performance to message-passing directionality rather than to hierarchy.
- **Verify the instrument, then verify the verification.** Roughly a dozen measurement bugs were
  caught before they became published claims, and several were caught only by testing the checker
  rather than trusting it: an inversion rate that measured its own opposite, a CI that excluded
  its point estimate, a `??` detector firing inside compressed PDF streams, a permutation null
  that could not move because the shuffle was an isomorphism, and a probe whose "finding" was
  1.000 by construction. Twice the *test* was wrong rather than the instrument, so checking the
  fixture before changing the code saved a correct implementation from being "repaired".
- **A null result and a degenerate test look identical from the outside.** E11 reported p = 1.000
  and R² = 1.0 and read as a clean deletion. Both numbers were forced by construction. The
  verdict survived re-testing on the statistic that varies, but the route to it was not evidence.
- **Pooled populations are this project's most persistent defect class — three instances.** The
  ASO test pooled the gold subset with the primary corpus; the gold-subset table withheld the
  model that won on it; and the E7.5 interaction model filtered on *model* rather than on
  *condition*, fitting over every granularity, split, negative regime and rank map at once. The
  third is the worst, because the pool contained the random split — the one condition where the
  ordering that coefficient describes is reversed. Refitting on the primary cell alone made the
  effect four times larger (−0.085 vs −0.020). Every one of the three was found by asking "what
  population is this number actually computed over?", which is now a standing check.


## Lessons and Constraints

Instrument errors caught before they became "findings" — each of these would have produced a
confident, wrong claim:

- **Cross-granularity RBO is identically 0 by construction.** Coarse keys are parent paths and
  fine keys carry a place suffix, so the label spaces are disjoint and overlap is impossible.
  The comparison only means something after projecting both lists onto a shared archival
  referent. Reported both ways, with the reason.
- **MinHash/LSH returns *candidates*, not matches.** Unfiltered, it reported 95,892
  cross-narrator near-duplicate pairs among 11,696 segments (~8 per segment). Re-checking each
  candidate against the actual Jaccard estimate is mandatory; so is comparing content summaries
  rather than boilerplate titles (`<Narrator> Segment 7`).
- **α near zero does not imply disagreement.** With lopsided marginals a chance-corrected
  coefficient collapses even at high raw agreement, so α is never quoted without raw agreement,
  per-annotator marginals, and Gwet's AC1 beside it.
- **A degenerate fixture can hide a broken sampler.** The first CNS test passed no narrator who
  co-witnessed the target cell through a *different* cell, so CNS and UNS both fell back to
  uniform and looked identical. The sampler also had to drop a *random* member, not always the
  last one.
- **Vectorising the retrieval metrics changed the runtime from ~8 min to seconds**, so the fast
  path is checked against the loop implementation on randomised cases before it is trusted.
- Environment: pip silently downgraded numpy to 1.26.4 on Python 3.14, which breaks the sklearn
  ABI without an obvious error message.

## Open Questions

- Does the CCNN advantage (if any) survive R-C, the adversarial rank map? Pre-registered: if
  the sign flips, the effect is annotation-driven and no rank benefit may be claimed.
- Does ablation A2 (shuffled rank labels, distribution preserved) tie true ranks? If so the
  model exploits a partition, not a hierarchy.
- Is the archive-conditioned attestation triage list stable across granularity once projected
  onto a shared referent, and under perturbation calibrated to the measured error rates?

## Optimization Trajectory

*(populated from data/results/e7_1_runs.json)*
