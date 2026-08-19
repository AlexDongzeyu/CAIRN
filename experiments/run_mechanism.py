"""E-NEW: does the rank ladder buy event understanding, or an identity shortcut?

The split result shows the complex gaining sharply under a random split while the typed star
does not. The natural explanation is that rank 0 (narrator) with rank 1 (that narrator's own
moments) creates a short path from a test segment back to its own narrator's training
segments, and that a rank-aware operator makes that path cheap. That explanation has been
asserted; these two experiments test it.

E-NEW-3  narrator-identity probe.
  Freeze each trained model, take its rank-1 representations for held-out segments, and fit
  a linear probe to predict which narrator produced the segment. If the complex encodes
  identity more strongly than the typed star, its representations are more decodable. The
  probe is deliberately linear: it measures what is linearly available, not what some
  classifier could extract with enough capacity.

E-NEW-1  identity-path ablation.
  Rebuild the archive with narrator identity destroyed -- every segment reassigned to a
  random narrator, preserving the number of segments per narrator -- and re-run both models
  under both splits. If the shortcut explanation is right, the complex's random-split
  advantage should collapse, because there is no longer a stable identity to recognise.

Both are confirmatory tests of a stated mechanism, run after the fact and labelled as
exploratory in the results file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import build_features, get_encoder, segment_embeddings  # noqa: E402
from src.hypergraph_encodings import hypergraph_encodings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.tasks import NegativeSampler, build_t1_queries, make_split, partition_incidences  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("mechanism")


def load_rank_maps() -> dict[str, dict[str, int]]:
    d = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    return {"R-A": d["R-A_consensus"], "R-B": d["R-B_archive_native"],
            "R-C": d["R-C_adversarial"]}


def load_baseline_split_scores() -> dict[str, dict[str, float]]:
    """Real-narrator T1 MAP per model per split, from the main grid.

    Read from the same file the paper's split table reads, so the anonymised comparison
    cannot silently be made against a differently-configured run.
    """
    return {m: {s: float(np.mean(vs)) for s, vs in d.items()}
            for m, d in _baseline_per_seed().items()}


def _baseline_per_seed() -> dict[str, dict[str, list[float]]]:
    runs = json.loads((RES / "e7_1_runs.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, list[float]]] = {}
    for r in runs:
        if (r.get("granularity") != "mid" or r.get("neg_regime") != "MNS"
                or r.get("rank_map") != "R-A" or r.get("cell") == "E8.3_gold_subset"):
            continue
        v = r.get("T1_map")
        if isinstance(v, (int, float)) and v == v:
            out.setdefault(r["model"], {}).setdefault(r["split"], []).append(float(v))
    return out


def gap_ci(per_seed: dict[str, list[float]], b: int = 10000, seed: int = 0) -> dict:
    """Interval for (random - narrator-disjoint) by resampling seeds within each split.

    Seeds are the only replication this arm has, so the interval describes optimisation
    noise in the gap and is labelled as such wherever it is reported.
    """
    a, c = per_seed.get("random", []), per_seed.get("narrator-disjoint", [])
    if not a or not c:
        return {}
    rng = np.random.default_rng(seed)
    a, c = np.asarray(a, float), np.asarray(c, float)
    draws = np.array([rng.choice(a, len(a), replace=True).mean()
                      - rng.choice(c, len(c), replace=True).mean() for _ in range(b)])
    return {"point": float(a.mean() - c.mean()),
            "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)),
            "n_seeds": int(min(len(a), len(c)))}

PARAM_BUDGET = 130_000
MODELS = ("M3_typed_star", "M5_ccnn")
SPLITS = ("narrator-disjoint", "random")
SEEDS = (0, 1, 2, 3, 4)


def anonymise(segments, rng):
    """Destroy narrator identity while preserving the segments-per-narrator distribution.

    Reassigns the narrator labels themselves rather than dropping rank 0, so the complex
    keeps exactly the same shape and the only thing that changes is whether the identity is
    stable across segments.
    """
    import copy

    segs = [copy.copy(s) for s in segments]
    all_narr = sorted({n for s in segs for n in (s.narrators or [])})
    counts = [len(s.narrators or []) for s in segs]
    pool = []
    for s in segs:
        pool.extend(s.narrators or [])
    rng.shuffle(pool)
    k = 0
    for s, c in zip(segs, counts):
        s.narrators = pool[k:k + c]
        k += c
    return segs, len(all_narr)


def prepare(segments, rank_maps, split_kind, encoder, device, seed=0):
    cx = build_complex(segments, rank_maps["R-A"], granularity="mid", rank_map_name="R-A")
    feats = build_features(cx, segments, encoder)
    sp = make_split(cx, split_kind, seed=seed)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=seed)
    data = {p: ns.build(parts[p], "MNS", ratio=10) for p in ("train", "val", "test")}
    train_cx = mask_incidences(cx, set(parts["train"]))
    H = train_cx.incidence_matrix(0, 2)
    enc0 = (hypergraph_encodings(H, k_spectral=16) if H.shape[1]
            else np.zeros((H.shape[0], 21), np.float32))
    bp = make_bundle(train_cx, feats, device)
    be = make_bundle(train_cx, feats, device, extra={0: enc0})
    _, sid2i = segment_embeddings(segments, encoder)
    data["t1_queries"] = build_t1_queries(cx, segments, sp, bucket="test", limit=400)
    data["seg_index"] = sid2i
    return cx, sp, data, bp, be, data["t1_queries"]


def narrator_probe(h1: np.ndarray, seg_narrator: list[str], rng) -> dict:
    """Linear probe: how much narrator identity is linearly readable from a segment vector?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    keep = [i for i, n in enumerate(seg_narrator) if n]
    X = h1[keep]
    y = np.array([seg_narrator[i] for i in keep])

    # A narrator needs at least two segments to appear on both sides of the probe split.
    uniq, counts = np.unique(y, return_counts=True)
    ok = set(uniq[counts >= 4])
    m = np.array([v in ok for v in y])
    X, y = X[m], y[m]
    if len(set(y)) < 2:
        return {"probe_accuracy": float("nan"), "chance": float("nan"), "n": 0}

    # The probe split is held fixed across seeds on purpose: the quantity of interest is
    # how the representation varies with the training seed, not how the probe split does.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    # Standardise inside the probe. Two architectures produce hidden vectors on different
    # scales, and an unscaled linear probe rewards the larger-magnitude one for reasons that
    # have nothing to do with narrator identity -- which is the whole comparison here.
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(sc.transform(Xtr), ytr)
    acc = float(clf.score(sc.transform(Xte), yte))
    # Chance for a majority-class guess, which is the honest floor on imbalanced labels.
    _, c = np.unique(yte, return_counts=True)
    return {"probe_accuracy": acc, "chance": float(c.max() / c.sum()),
            "n_classes": int(len(set(y))), "n": int(len(y))}


def main() -> None:
    # --post-only recomputes the parts that need no training -- the frozen-encoder control
    # probes and the split-gap summary -- and merges them into an existing result file.
    # The full run produces the same fields; this exists so that adding a control does not
    # cost twenty retrainings.
    post_only = "--post-only" in sys.argv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segments = load_corpus()[1]
    rank_maps = load_rank_maps()
    log(f"device={device} segments={len(segments)} post_only={post_only}")

    encoder = get_encoder()
    out: dict = {"labelled": "EXPLORATORY - run after the split result, to test a stated "
                            "mechanism rather than a pre-registered hypothesis",
                 "probe": [], "anonymised": []}
    if post_only:
        prev = json.loads((RES / "e_mechanism.json").read_text(encoding="utf-8"))
        out = prev
        out["probe"] = [r for r in prev.get("probe", []) if not r.get("is_control")]

    # ---- E-NEW-3: identity probe on the real archive ---------------------------------
    for split_kind in SPLITS:
        cx, sp, data, bp, be, queries = prepare(segments, rank_maps, split_kind, encoder, device)
        # Feature rows are built as sorted(cx.by_rank(k), key=cid) in features.py. The label
        # vector MUST use that same order or the probe scores a shuffled pairing and reports
        # chance as a finding. Keep only cells owned by exactly one narrator; a shared cell
        # has no single identity to decode.
        rank1 = sorted(cx.by_rank(1), key=lambda c: c.cid)
        rows = [i for i, c in enumerate(rank1) if len(c.members) == 1]
        labels = [next(iter(rank1[i].members)) for i in rows]
        log(f"  {split_kind}: rank-1 cells={len(rank1)} single-narrator={len(rows)} "
            f"distinct-narrators={len(set(labels))}")

        # Control. The frozen sentence encoder already carries narrator idiolect, so a high
        # probe score on a trained representation means nothing until we know what the raw
        # input scores. Anything at or below this line was inherited, not learned.
        for tag, bnd in (("frozen_encoder_typed", be), ("frozen_encoder_plain", bp)):
            X1 = np.asarray(bnd.X[1].detach().cpu().numpy() if hasattr(bnd.X[1], "detach")
                            else bnd.X[1])
            if X1.shape[0] != len(rank1):
                raise RuntimeError(f"{tag}: {X1.shape[0]} input rows vs {len(rank1)} cells")
            base = narrator_probe(X1[rows], labels, np.random.default_rng(0))
            rec = {"split": split_kind, "model": tag,
                   "probe_accuracy_mean": base["probe_accuracy"], "probe_accuracy_std": 0.0,
                   "chance": base["chance"], "n_classes": base["n_classes"],
                   "n": base["n"], "n_seeds": 1, "is_control": True}
            out["probe"].append(rec)
            log(f"  PROBE {split_kind:18s} {tag:22s} acc={rec['probe_accuracy_mean']:.3f} "
                f"chance={rec['chance']:.3f}  [input control]")

        if post_only:
            continue

        for model in MODELS:
            accs = []
            bundle = be if model == "M3_typed_star" else bp
            dims = {k: bundle.X[k].shape[1] for k in bundle.X}
            hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
            for seed in SEEDS:
                set_all_seeds(seed)
                cfg = RunConfig(model=model, granularity="mid", rank_map="R-A",
                                split=split_kind, neg_regime="MNS", seed=seed, hidden=hidden)
                res = train_eval(cfg, cx, bundle, data, device, eval_cx=cx, return_hidden=True)
                h = res.get("hidden", {})
                if 1 not in h:
                    log(f"  {model}: no rank-1 representation returned; skipping probe")
                    break
                H = np.asarray(h[1])
                if H.shape[0] != len(rank1):
                    raise RuntimeError(
                        f"{model}: rank-1 representation has {H.shape[0]} rows but the "
                        f"complex has {len(rank1)} rank-1 cells; the label alignment "
                        f"assumption in this script no longer holds")
                accs.append(narrator_probe(H[rows], labels, np.random.default_rng(seed)))
            if accs:
                rec = {"split": split_kind, "model": model,
                       "probe_accuracy_mean": float(np.mean([a["probe_accuracy"] for a in accs])),
                       "probe_accuracy_std": float(np.std([a["probe_accuracy"] for a in accs])),
                       "chance": float(np.mean([a["chance"] for a in accs])),
                       "n_classes": accs[0]["n_classes"], "n": accs[0]["n"],
                       "n_seeds": len(accs)}
                out["probe"].append(rec)
                log(f"  PROBE {split_kind:18s} {model:16s} "
                    f"acc={rec['probe_accuracy_mean']:.3f}+-{rec['probe_accuracy_std']:.3f} "
                    f"chance={rec['chance']:.3f}")

    (RES / "e_mechanism.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- E-NEW-1: destroy narrator identity, re-run both splits ----------------------
    if not post_only:
        anon_segments, n_narr = anonymise(segments, np.random.default_rng(0))
        log(f"E-NEW-1 - narrator identity shuffled across {n_narr} narrators")
        for split_kind in SPLITS:
            cx, sp, data, bp, be, queries = prepare(anon_segments, rank_maps, split_kind,
                                                    encoder, device)
            for model in MODELS:
                vals = []
                bundle = be if model == "M3_typed_star" else bp
                dims = {k: bundle.X[k].shape[1] for k in bundle.X}
                hidden = match_hidden_to_budget(model, dims, bundle, PARAM_BUDGET)
                for seed in SEEDS:
                    set_all_seeds(seed)
                    cfg = RunConfig(model=model, granularity="mid", rank_map="R-A",
                                    split=split_kind, neg_regime="MNS", seed=seed,
                                    hidden=hidden)
                    r = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
                    vals.append(r["T1_map"])
                rec = {"split": split_kind, "model": model, "condition": "narrator-anonymised",
                       "T1_map_mean": float(np.mean(vals)), "T1_map_std": float(np.std(vals)),
                       "T1_map_per_seed": [float(v) for v in vals],
                       "n_seeds": len(vals)}
                out["anonymised"].append(rec)
                log(f"  ANON  {split_kind:18s} {model:16s} "
                    f"T1_MAP={rec['T1_map_mean']:.4f}+-{rec['T1_map_std']:.4f}")
                (RES / "e_mechanism.json").write_text(json.dumps(out, indent=2),
                                                      encoding="utf-8")

    # ---- the quantity the claim actually rests on ------------------------------------
    # Anonymisation changes which passages count as positives, so an anonymised score is
    # not comparable to a real one directly. What IS comparable is the gap BETWEEN splits,
    # because both splits are transformed the same way. If narrator identity is what the
    # random split leaks, that gap must shrink when identity is destroyed.
    baseline = load_baseline_split_scores()
    summary = {}
    for model in MODELS:
        anon = {r["split"]: r["T1_map_mean"] for r in out["anonymised"] if r["model"] == model}
        base = baseline.get(model, {})
        if {"random", "narrator-disjoint"} <= set(anon) and \
           {"random", "narrator-disjoint"} <= set(base):
            g0 = base["random"] - base["narrator-disjoint"]
            g1 = anon["random"] - anon["narrator-disjoint"]
            summary[model] = {
                "gap_real_narrators": float(g0), "gap_anonymised": float(g1),
                "gap_reduction": float(g0 - g1),
                "gap_reduction_frac": float((g0 - g1) / g0) if g0 else float("nan"),
                "gap_real_ci": gap_ci(_baseline_per_seed().get(model, {})),
                "gap_anon_ci": gap_ci({r["split"]: r.get("T1_map_per_seed", [])
                                       for r in out["anonymised"] if r["model"] == model}),
                "baseline": base, "anonymised": anon}
            log(f"  GAP   {model:16s} real={g0:+.4f} anon={g1:+.4f} "
                f"reduction={(g0 - g1):+.4f}")
    out["split_gap"] = summary
    (RES / "e_mechanism.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    log("mechanism experiments complete")


if __name__ == "__main__":
    main()
