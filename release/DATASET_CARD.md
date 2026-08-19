# Dataset card — CAIRN ranked complex over Densho oral history

## What this is
A ranked combinatorial complex derived from the Densho Visual History Collection
(`ddr-densho-1000`). Rank-0 cells are narrators, rank-1 cells are the archive's own
interview segments, and rank-2/rank-3 cells are descriptive terms from Densho's controlled
vocabulary, ranked by the procedure in `annotation/RANK_MANUAL.md`.

## Provenance and licensing
* Source: Densho Digital Repository, https://ddr.densho.org
* Collection: `526` interviews, `13315` segments,
  `470` distinct narrators.
* Per-item rights recorded in `LICENSE_AUDIT.csv`: {"cc": 526}.
* Densho content is offered under CC BY-NC-SA 4.0 by default. This release is **stand-off**:
  identifiers, offsets and derived structure only. `fetch_text.py` reconstructs text from the
  archive. Please cite Densho, not this repository, for the testimony itself.

## Consent limitation
Interviewees consented to archival deposit and public access. They did not consent to
computational analysis, and nothing here should be read as their endorsement of it. Densho
maintains a community-consent posture over this material; researchers reusing it should
contact the archive rather than treating the CC licence as the whole of the obligation.

## What the numbers mean, and what they do not
The quantity computed here is **archive-conditioned attestation multiplicity**: the number of
distinct narrators in this collection whose segments the archive filed under a given
descriptive term. It is a property of what this archive recorded and how it described it.

**It is not a measure of historical reality.** A low value means this collection holds few
narrators described under that term. It does not mean an event was rare, unimportant,
suppressed, or untrue. Interviewer prompts are excluded from the count because they are
archival artifacts rather than witness attestations; the effect of including them is reported
as an ablation.

## Known limitations
* Rank assignment was **not** reproducible across independent operationalizations on this
  corpus (see `annotation/e2_2_agreement.json`). Any downstream claim that depends on rank
  semantics must be read under all three released rank maps (R-A, R-B, R-C).
* Triage lists are **granularity-dependent**; there is no resolution-invariant ranking.
* No expert panel validated the triage output. The application claim is "computationally
  stable", not "expert-validated".

## Files
| File | Contents |
|---|---|
| `cells.jsonl` | every cell: id, rank, granularity, members, supporting segments |
| `incidence_{g}.npz` | B01, B12, B23 per granularity (CSR arrays) |
| `splits/` | exact narrator-disjoint, event-disjoint and random splits by id |
| `segments_standoff.csv` | segment index with archive API URLs, no redistributed text |
| `annotation/` | frozen rank manual, agreement study, all three rank maps |
| `fetch_text.py` | reconstructs text from the archive |
