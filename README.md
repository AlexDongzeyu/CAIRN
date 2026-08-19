# CAIRN: Ownership Obstruction and Evaluation Design in Higher-Order Models of Oral History

## File structure

| Path | Content |
|---|---|
| [`release/`](release/) | **The data release.** Stand-off segments, incidence matrices, splits, annotation. Start here. |
| [`src/`](src/) | Library: complexes, encodings, models, tasks, metrics, topology. |
| [`experiments/`](experiments/) | One script per experiment, plus the paper and figure builders. |
| [`tests/`](tests/) | Instrument tests. `python -m pytest tests/ -q` |
| [`data/results/`](data/results/) | Every experiment's output as JSON. Each number in the paper is read from one of these. |

## Data release

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

## Headline numbers

| | |
|---|---|
| Rank-1 collapse | 455 distinct inputs, 29.3-fold |
| Narrators straddling a random split | 64.8%, predicted in closed form at 64.1% |
| T1 MAP, random split | 0.388 complex vs 0.106 star |
| T1 MAP, narrator-disjoint | 0.098 complex vs 0.139 star |
| Rank-2 assignment agreement | α = 0.039 across three operationalisations of one manual |
