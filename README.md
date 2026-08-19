# CAIRN

**Ownership Obstruction and Evaluation Design in Higher-Order Models of Oral History**

Across 526 oral-history interviews, 13,315 described moments realise only 455 distinct
supports over 470 narrators. No combinatorial complex over those narrators can hold them
apart — a collapse we call the **ownership obstruction**. This repository holds the code,
the stand-off data release and the audit record behind that result.

The paper is [`paper/main.pdf`](paper/main.pdf).

---

## Where things are

| Path | What it holds |
|---|---|
| [`release/`](release/) | **The data release.** Stand-off segments, incidence matrices, splits, annotation. Start here. |
| [`src/`](src/) | Library: complexes, encodings, models, tasks, metrics, topology. |
| [`experiments/`](experiments/) | One script per experiment, plus the paper and figure builders. |
| [`tests/`](tests/) | Instrument tests. `python -m pytest tests/ -q` |
| [`data/results/`](data/results/) | Every experiment's output as JSON. Each number in the paper is read from one of these. |
| [`paper/`](paper/) | LaTeX sources and the compiled PDF. |
| [`figures/`](figures/) | The three figures the paper includes. |
| [`PREREGISTRATION.yaml`](PREREGISTRATION.yaml) | Registered defaults and predictions, fixed before the runs, plus the substitution ledger. |
| [`research-log.md`](research-log.md) | The defect log: every instrument error caught before it became a finding, dated. |
| [`findings.md`](findings.md) | Standing conclusions, checked against `data/results/` by `experiments/claim_audit.py`. |

Section XII of the paper promises the release carries the cell definitions, incidence
matrices at all three granularities, the splits, the annotation manual, a licence audit, the
defect log, the substitution record and a fetch script. The first five and the last are in
`release/`; the defect log is `research-log.md` and the substitution record is the
`substitution_ledger` block of `PREREGISTRATION.yaml`.

## The data release

`release/` is stand-off: it carries structure and offsets, not interview text, because the
archive's material is not ours to redistribute. [`release/fetch_text.py`](release/fetch_text.py)
resolves the text from the source archive; [`release/DATASET_CARD.md`](release/DATASET_CARD.md)
documents provenance and [`release/LICENSE_AUDIT.csv`](release/LICENSE_AUDIT.csv) records
per-item terms.

```
release/
  cells.jsonl                 cells of the combinatorial complex
  incidence_{coarse,mid,fine}.npz    incidence at the three ranks
  segments_standoff.csv       segment offsets into the source transcripts
  splits/                     random, narrator-disjoint, event-disjoint
  annotation/                 rank manual, agreement, three rank maps
```

## Reproducing

```bash
python -m pytest tests/ -q          # instrument tests
python experiments/make_figures.py  # figures
python experiments/make_paper.py --compile   # numbers.tex, main.tex, main.pdf
```

Every number printed in the paper is a macro resolved from `data/results/*.json` by the
`SPEC` table in [`experiments/make_paper.py`](experiments/make_paper.py). Nothing is typed
in by hand, so a changed result changes the paper. The build fails if a macro has no source.

Two per-item result dumps (`e7_per_item_primary.json`, `e_split_first.json`, ~33 MB
together) are regenerable from `experiments/` and are not tracked.

## Headline numbers

| | |
|---|---|
| Rank-1 collapse | 455 distinct inputs, 29.3-fold |
| Narrators straddling a random split | 64.8%, predicted in closed form at 64.1% |
| T1 MAP, random split | 0.388 complex vs 0.106 star |
| T1 MAP, narrator-disjoint | 0.098 complex vs 0.139 star |
| Rank-2 assignment agreement | α = 0.039 across three operationalisations of one manual |

The magnitude of the reversal is a finding about this archive. The obstruction itself is
structural, and reaches any collection whose items are singly owned.
