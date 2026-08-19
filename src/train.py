"""Training and evaluation for T1/T2/T3 under the E6.0 controls.

One objective trains every model (incidence BCE with the sampled negatives); T1 and T3
are then read off the learned representations. Using a single objective keeps the
comparison honest - no model gets a task-specific decoder the others lack.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src.models import Bundle, build_model
from src.seeds import set_all_seeds


# ------------------------------------------------------------------ IR metrics
def average_precision(ranked: list[str], positives: set[str]) -> float:
    if not positives:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in positives:
            hits += 1
            s += hits / i
    return s / len(positives)


def ndcg_at(ranked: list[str], positives: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in positives)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(positives)) + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at(ranked: list[str], positives: set[str], k: int = 50) -> float:
    return len(set(ranked[:k]) & positives) / len(positives) if positives else 0.0


def reciprocal_rank(ranked: list[str], positives: set[str]) -> float:
    for i, d in enumerate(ranked, 1):
        if d in positives:
            return 1.0 / i
    return 0.0


def auc_roc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


def auc_pr(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")


# ------------------------------------------------------------------ training
@dataclass
class RunConfig:
    model: str
    granularity: str
    rank_map: str
    split: str
    neg_regime: str
    seed: int
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    lr: float = 5e-3
    weight_decay: float = 1e-4
    epochs: int = 800
    patience: int = 120
    extra: dict = field(default_factory=dict)


def _pairs_to_idx(pairs, labels, n_index, c_index):
    a, b, y = [], [], []
    for (n, cid), lab in zip(pairs, labels):
        if n in n_index and cid in c_index:
            a.append(n_index[n]); b.append(c_index[cid]); y.append(lab)
    return np.array(a), np.array(b), np.array(y, dtype=np.float32)


def train_eval(cfg: RunConfig, cx, bundle: Bundle, data: dict, device, eval_cx=None,
               return_hidden: bool = False, collect_t1: bool = False,
               val_queries: list | None = None, map_every: int = 25) -> dict:
    """data: {'train'/'val'/'test': (pairs, labels)} plus optional T1 queries.

    `bundle` must be built from the TRAINING structure only; `eval_cx` (defaulting to cx)
    supplies cell indexing and metadata for scoring.

    Passing `val_queries` additionally tracks a checkpoint chosen by validation T1 MAP and
    reports the test metrics it would have produced. Checkpoint selection is otherwise by
    validation AUC, which is the criterion the published runs used.
    """
    set_all_seeds(cfg.seed)
    eval_cx = eval_cx or cx
    n_index = {n: i for i, n in enumerate(eval_cx.narrators)}
    c_index = eval_cx.index(2)
    dims = {k: bundle.X[k].shape[1] for k in bundle.X}

    kw = dict(cfg.extra)
    model = build_model(cfg.model, dims, bundle, cfg.hidden, cfg.layers, cfg.dropout, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    tensors = {}
    for part in ("train", "val", "test"):
        a, b, y = _pairs_to_idx(*data[part], n_index, c_index)
        tensors[part] = (
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(b, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )

    best_val, best_state, bad = -np.inf, None, 0
    best_map, best_map_state, best_map_epoch, best_epoch = -np.inf, None, -1, -1
    for ep in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        h = model(bundle)
        a, b, y = tensors["train"]
        loss = F.binary_cross_entropy_with_logits(model.score(h, a, b, 0, 2), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            h = model(bundle)
            av, bv, yv = tensors["val"]
            sv = model.score(h, av, bv, 0, 2).cpu().numpy()
            v = auc_roc(yv.cpu().numpy(), sv)
            if val_queries and ep % map_every == 0:
                m = evaluate_t1(model, h, eval_cx, val_queries, data["seg_index"],
                                device)["T1_map"]
                if np.isfinite(m) and m > best_map:
                    best_map, best_map_epoch = m, ep
                    best_map_state = {k: t.detach().clone()
                                      for k, t in model.state_dict().items()}
        if np.isnan(v):
            v = -np.inf
        if v > best_val:
            best_val, bad, best_epoch = v, 0, ep
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        h = model(bundle)
        at, bt, yt = tensors["test"]
        st = model.score(h, at, bt, 0, 2).cpu().numpy()
        yt_np = yt.cpu().numpy()

        out = {
            "model": cfg.model, "granularity": cfg.granularity, "rank_map": cfg.rank_map,
            "split": cfg.split, "neg_regime": cfg.neg_regime, "seed": cfg.seed,
            "n_params": model.n_params(), "hidden": cfg.hidden,
            "T2_auc": auc_roc(yt_np, st), "T2_aupr": auc_pr(yt_np, st),
            "val_auc": float(best_val) if np.isfinite(best_val) else float("nan"),
        }
        out.update(_t2_ranking(st, yt_np, bt.cpu().numpy()))
        if data.get("t1_queries"):
            out.update(evaluate_t1(model, h, eval_cx, data["t1_queries"], data["seg_index"],
                                   device, collect=collect_t1))
        out.update(_t3(h, eval_cx))
        out["per_item"] = _per_item(st, yt_np, at.cpu().numpy(), bt.cpu().numpy(), eval_cx)
        if return_hidden:
            out["hidden"] = {k: v.detach().cpu().numpy() for k, v in h.items()}

    if val_queries is not None:
        out["auc_selected_epoch"] = int(best_epoch)
        out["map_selected_epoch"] = int(best_map_epoch)
        out["val_map_at_selection"] = float(best_map) if np.isfinite(best_map) else float("nan")
        if best_map_state is not None:
            model.load_state_dict(best_map_state)
            model.eval()
            with torch.no_grad():
                h = model(bundle)
                at, bt, yt = tensors["test"]
                st = model.score(h, at, bt, 0, 2).cpu().numpy()
                out["T2_auc_mapsel"] = auc_roc(yt.cpu().numpy(), st)
                if data.get("t1_queries"):
                    t1 = evaluate_t1(model, h, eval_cx, data["t1_queries"],
                                     data["seg_index"], device)
                    out["T1_map_mapsel"] = t1["T1_map"]
                    out["T1_ndcg@10_mapsel"] = t1["T1_ndcg@10"]
    return out


def evaluate_m1_dense(cx, segments, data, seg_emb, cell_emb, narr_emb, cfg: RunConfig) -> dict:
    """E6.2 - the non-structural retrieval floor, scored with the same metric code.

    M1 has no parameters and no training loop, so it is evaluated directly: T1 ranks
    candidate passages by cosine over frozen segment embeddings, T2 scores an incidence
    by cosine between the narrator vector and the event's text vector.
    """
    n_index = {n: i for i, n in enumerate(cx.narrators)}
    c_index = cx.index(2)
    a, b, y = _pairs_to_idx(*data["test"], n_index, c_index)
    s = np.einsum("ij,ij->i", narr_emb[a], cell_emb[b]) if len(a) else np.array([])

    out = {
        "model": "M1_dense", "granularity": cfg.granularity, "rank_map": cfg.rank_map,
        "split": cfg.split, "neg_regime": cfg.neg_regime, "seed": cfg.seed,
        "n_params": 0, "hidden": 0,
        "T2_auc": auc_roc(y, s) if len(s) else float("nan"),
        "T2_aupr": auc_pr(y, s) if len(s) else float("nan"),
        "val_auc": float("nan"),
    }
    out.update(_t2_ranking(s, y, b))

    r1_index = cx.index(1)
    fake_h = {1: torch.tensor(seg_emb_to_rank1(cx, segments, seg_emb), dtype=torch.float32)}
    if data.get("t1_queries"):
        out.update(evaluate_t1(None, fake_h, cx, data["t1_queries"], data["seg_index"],
                               torch.device("cpu")))
    out["T3_spearman"] = float("nan")
    out["per_item"] = _per_item(s, y, a, b, cx) if len(s) else []
    return out


def seg_emb_to_rank1(cx, segments, seg_emb) -> np.ndarray:
    """Place each segment's frozen embedding at its rank-1 cell index."""
    r1_index = cx.index(1)
    sid2i = {s.segment_id: i for i, s in enumerate(segments)}
    M = np.zeros((len(r1_index), seg_emb.shape[1]), dtype=np.float32)
    for cid, row in r1_index.items():
        sid = cid.split(":", 1)[1]
        if sid in sid2i:
            M[row] = seg_emb[sid2i[sid]]
    return M


def _t2_ranking(scores, y, group) -> dict:
    """MRR and Hits@k computed within each cell's candidate list."""
    by: dict[int, list[tuple[float, int]]] = {}
    for s, lab, g in zip(scores, y, group):
        by.setdefault(int(g), []).append((float(s), int(lab)))
    rr, hits = [], {1: [], 5: [], 10: []}
    for lst in by.values():
        lst.sort(key=lambda t: -t[0])
        pos = [i for i, (_, lab) in enumerate(lst, 1) if lab == 1]
        if not pos:
            continue
        rr.append(1.0 / pos[0])
        for k in hits:
            hits[k].append(1.0 if pos[0] <= k else 0.0)
    return {"T2_mrr": float(np.mean(rr)) if rr else float("nan"),
            **{f"T2_hits@{k}": (float(np.mean(v)) if v else float("nan")) for k, v in hits.items()}}


def evaluate_t1(model, h, cx, queries, seg_index, device, collect: bool = False) -> dict:
    """T1 retrieval over learned rank-1 (moment) representations.

    The query's own rank-2 membership is never supplied to the scorer, so the model
    cannot read the answer off the structure it was given.

    Metrics are computed with vectorised rank arithmetic rather than by materialising a
    ranked id list per query; the list version is O(queries x corpus) in Python and
    dominates total runtime once the corpus passes a few thousand segments.
    """
    H1 = F.normalize(h[1], dim=-1).cpu().numpy()
    r1_index = cx.index(1)
    all_sids = [sid for sid in seg_index if f"r1:{sid}" in r1_index]
    if not all_sids or not queries:
        return {"T1_map": float("nan"), "T1_ndcg@10": float("nan"),
                "T1_recall@50": float("nan"), "T1_mrr": float("nan"), "T1_n_queries": 0}
    pool_idx = np.array([r1_index[f"r1:{s}"] for s in all_sids])
    pos_in_pool = {s: i for i, s in enumerate(all_sids)}
    P = H1[pool_idx]

    ap, nd, rc, mrr = [], [], [], []
    pq: list[dict] = []
    disc = 1.0 / np.log2(np.arange(2, 2 + len(all_sids)))
    for q in queries:
        qk = f"r1:{q['qid']}"
        if qk not in r1_index:
            continue
        rows = [pos_in_pool[s] for s in q["positives"] if s in pos_in_pool]
        if not rows:
            continue
        sims = P @ H1[r1_index[qk]]
        self_i = pos_in_pool.get(q["qid"])
        if self_i is not None:
            sims[self_i] = -np.inf          # never retrieve the query itself
        order = np.argsort(-sims, kind="stable")

        rel = np.zeros(len(all_sids), dtype=bool)
        rel[rows] = True
        rel_ranked = rel[order]
        hits = np.cumsum(rel_ranked)
        ranks = np.arange(1, len(all_sids) + 1)
        n_pos = int(rel_ranked.sum())
        ap_q = float((hits[rel_ranked] / ranks[rel_ranked]).sum() / n_pos)
        ap.append(ap_q)
        if collect:
            pq.append({"qid": q["qid"], "narrator": q["narrator"], "ap": ap_q})
        nd_dcg = float((rel_ranked[:10] * disc[:10]).sum())
        nd_idcg = float(disc[: min(10, n_pos)].sum())
        nd.append(nd_dcg / nd_idcg if nd_idcg > 0 else 0.0)
        rc.append(float(rel_ranked[:50].sum() / n_pos))
        first = int(np.argmax(rel_ranked)) + 1 if rel_ranked.any() else 0
        mrr.append(1.0 / first if first else 0.0)

    return {"T1_map": float(np.mean(ap)) if ap else float("nan"),
            "T1_ndcg@10": float(np.mean(nd)) if nd else float("nan"),
            "T1_recall@50": float(np.mean(rc)) if rc else float("nan"),
            "T1_mrr": float(np.mean(mrr)) if mrr else float("nan"),
            "T1_n_queries": len(ap),
            **({"T1_per_query": pq} if collect else {})}


def _t3(h, cx) -> dict:
    """Diagnostic: is attestation structure learnable at all?"""
    from scipy.stats import spearmanr

    cells = sorted(cx.by_rank(2), key=lambda c: c.cid)
    if len(cells) < 3:
        return {"T3_spearman": float("nan")}
    H2 = h[2].cpu().numpy()
    y = np.array([c.size for c in cells], dtype=float)
    pred = H2.mean(axis=1)
    rho, _ = spearmanr(pred, y)
    return {"T3_spearman": float(rho)}


def _per_item(scores, y, a_idx, b_idx, cx) -> list[dict]:
    """Per-incidence outcomes for the E7.5 mixed-effects interaction model."""
    cells = sorted(cx.by_rank(2), key=lambda c: c.cid)
    thr = float(np.median(scores))
    rows = []
    for s, lab, ai, bi in zip(scores, y, a_idx, b_idx):
        c = cells[int(bi)]
        rows.append({
            "correct": int((s >= thr) == (lab == 1)),
            "score": float(s),
            "label": int(lab),
            "narrator_id": cx.narrators[int(ai)],
            "event_id": c.cid,
            "event_size": int(c.size),
            "rank": int(c.rank),
        })
    return rows
