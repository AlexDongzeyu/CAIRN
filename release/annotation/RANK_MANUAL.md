# RANK_MANUAL — assigning rank to archive descriptive terms

**Corpus.** Densho Digital Repository, collection `ddr-densho-1000` (Densho Visual History
Collection), 526 interviews, 13,315 archive-curated segments, `rights: cc`.

**Scope.** This manual assigns a rank to each *descriptive term* in the archive's controlled
vocabulary. Rank then propagates to the cells derived from that term. It is deliberately a
judgement about the archive's descriptive practice, not about history.

---

## 0. The one rule that must never be broken

> **Do not use the number of narrators who mention `x` at any point in this procedure.**

A camp mentioned by 300 narrators and a specific transport mentioned by 3 are **both rank 2**.
A theme mentioned by 2 narrators and a war phase mentioned by 400 are **both rank 3**.

Rank is assigned from the archive's descriptive taxonomy, never from `|x|`. E4.4 verifies this
mechanically: if Spearman ρ(rank, |cell|) exceeds 0.9 the whole premise has failed.

---

## 1. Rank definitions

| Rank | Name | One-sentence definition |
|---|---|---|
| 0 | **narrator** | One interviewee, identified by a stable Densho oral-history id. |
| 1 | **moment** | One narrated episode localized in one testimony — operationally, one archive segment. |
| 2 | **event / site** | A specific battle, camp, transport, ship, unit, body, or incarceration facility with a bounded spatiotemporal extent that a second person could independently have been present for. |
| 3 | **episode / theatre** | An administrative, military, or historiographic programme, campaign, or phase that *contains* rank-2 cells and is not itself a single locatable occurrence. |

Rank 0 and rank 1 are given by the archive and require no judgement: rank-0 cells are narrators,
rank-1 cells are Densho's own sub-interview segments. **The archive did the segmentation, not us** —
this is the reason Densho is the primary corpus. Only ranks 2 and 3 are annotated.

---

## 2. Decision procedure

Applied in order. Each annotator answers the same questions from different evidence.

```
Given a candidate descriptive term x:

  Q1. Is x localized to a single narrator's account with no plausible co-witness
      by construction (a private thought, a family conversation, a solitary act)?
                                                              -> rank 1 (moment)

  Q2. Does x denote a named or definitely-describable place, facility, vessel,
      unit action, body, or transport with a bounded spatiotemporal extent that a
      second person could independently have been present for?
                                                              -> rank 2 (event/site)

  Q3. Does x denote an administrative, military, or historiographic programme /
      campaign / phase that CONTAINS rank-2 cells and is not itself a single
      locatable occurrence?                                   -> rank 3 (episode)

  Q4. If x satisfies both Q2 and Q3, or neither, assign by the ARCHIVE's own
      descriptive practice: Densho's vocabulary is a facet tree whose shallow
      nodes are programme/period headings and whose deep nodes are specific
      occurrences. Depth <= 2 -> rank 3; depth >= 3 -> rank 2.
```

`Q4` is a tie-break, not a shortcut. Reaching for it before answering Q2 and Q3 collapses the
manual into "count the dashes".

---

## 3. Positive examples (rank 2)

| Term | Why |
|---|---|
| `World War II -- Military service -- 442nd Regimental Combat Team` | A specific named unit; co-witnesses served in it. |
| `World War II -- Resistance and dissidence -- Supreme Court cases -- Fred Korematsu` | A specific case with a bounded record. |
| `World War II -- Administration -- Registration and the "loyalty questionnaire"` | A specific administrative instrument administered at known places and times. |
| `World War II -- Temporary Assembly Centers -- Living conditions` | Bound to a definite facility type at a definite phase; co-presence is possible. |
| `Redress and reparations -- Commission on Wartime Relocation and Internment of Civilians (CWRIC) -- Formation and work` | A named body with proceedings a second person could attend. |

## 4. Positive examples (rank 3)

| Term | Why |
|---|---|
| `World War II` | A war phase containing many rank-2 occurrences. |
| `Redress and reparations` | A long-running movement, not a locatable occurrence. |
| `Identity and values` | A theme under which many distinct events are filed. |
| `Industry and employment` | An area of work considered as a whole. |
| `Race and racism` | A recurring condition, not an occurrence at a place and time. |

## 5. Non-examples

**Not rank 2:** `Reflections on the past` (a stance, no extent) · `Immigration and citizenship`
(a policy domain) · `Community activities` (a category heading).

**Not rank 3:** `442nd Regimental Combat Team` (one unit) · `Pearl Harbor and aftermath --
Arrest, searches, and seizures` (bounded actions) · `Minidoka` (one facility).

---

## 6. Boundary cases and their adjudicated answers

| Case | Answer | Reason |
|---|---|---|
| `World War II -- Concentration camps -- Living conditions` | **contested** | Q2 reads "camps" as a definite facility type; Q3 reads "living conditions" as a recurring condition. Both defensible; this term is in the disputed set and is flipped under R-C. |
| `Race and racism -- Discrimination` | rank 3 | "Discrimination" names a recurring condition, not one occurrence. Q3 fires, Q2 does not. |
| `World War II -- Pearl Harbor and aftermath -- Personal recollections` | rank 3 | The head is an occurrence, but the leaf modifier turns it into a *mode of recall*. Q2 fails on the leaf. |
| `Identity and values -- Kibei` | rank 3 | Names a generational category of people, not a bounded happening. |
| `Community activities -- Associations and organizations` | rank 3 | A class of bodies, not one body. Contrast with CWRIC, which is one named body. |
| `Religion and churches` (used as a bare depth-1 leaf) | rank 3 | Q4 applies: a top-level facet is a programme heading even when the archive files a segment directly under it. |

---

## 7. Annotators and what counts as independence

Per `PREREGISTRATION.yaml → substitution_ledger → E2_2_annotators`, no human annotators were
available. Three **mechanical** annotators apply the procedure above from disjoint evidence:

| Annotator | Evidence it may use | Evidence it cannot see |
|---|---|---|
| **A1 structural** | containment in the archive's facet tree | the words of the term; how it is used |
| **A2 lexical** | the words of the term itself | tree position; usage |
| **A3 distributional** | sentence embeddings of the archive summaries carrying the term | tree position; the term's own lexis |

**This measures reproducibility of the rank construct across independent operationalizations.
It is not human inter-annotator agreement, and H1 is correspondingly weakened.**

---

## 8. Revision history

**v1 → v2 (adjudication session, logged in `data/results/e2_2_agreement.json`)**

1. **A1.** v1 derived the facet tree from terms observed *in this sample*, so a top-level facet
   with no sampled children was read as a terminal occurrence. v2 treats depth-1 facets as
   programme headings regardless of sampled children. This corrects a sample/archive confusion.
2. **A2.** 39 terms moved to the Q3 programme list and 25 to the Q2 site list — each one a
   Round-1 disagreement that the manual's own Q2/Q3 wording already decides.
3. **A3.** v1 prototypes described *subject matter*, so thematic terms attached to the rank-2
   side. v2 adds prototypes contrasting "has a boundary and a date a co-witness could confirm"
   against "abstract heading filing many events".
4. All three annotators were aligned onto the shared Q2/Q3/Q4 ordering; in v1 each used its own
   resolution rule.

**The revision did not raise agreement.** Round-2 α is reported as the headline regardless, and
the fact that a defensible adjudication moved agreement in the wrong direction is itself
evidence about the construct — see `findings.md`.
