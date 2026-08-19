"""PHASE 10 — cross-archive transfer.

Per PREREGISTRATION.substitution_ledger the transfer archive is a Densho collection
disjoint from `ddr-densho-1000`, which makes this cross-*collection* rather than
cross-institution transfer. That is stated in the paper rather than glossed.

The scientific question is not "does the model transfer" but "does the CONCLUSION
transfer" — is the sign and rough magnitude of the interaction coefficient the same?

Nothing in this file may tune anything. Every threshold, rank map and hyperparameter is
taken from the primary archive as-is, and the date of first contact is logged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complexes import build_complex, mask_incidences, rank_cardinality_decoupling  # noqa: E402
from src.corpus import corpus_stats, load_corpus  # noqa: E402
from src.densho_api import API_ROOT, get  # noqa: E402
from src.e1_1_acquire_densho import narrators_of, page_all  # noqa: E402
from src.features import build_features, get_encoder, segment_embeddings  # noqa: E402
from src.hypergraph_encodings import hypergraph_encodings  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import make_bundle, match_hidden_to_budget  # noqa: E402
from src.ontology import agreement_diagnostics, run_rank_study  # noqa: E402
from src.seeds import SEEDS, set_all_seeds  # noqa: E402
from src.stats import interaction_model  # noqa: E402
from src.tasks import NegativeSampler, build_t1_queries, make_split, partition_incidences  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
OUTDIR = ROOT / "corpus" / "densho_transfer"
log = make_logger("phase10")

# Held-out collections: never touched during any design decision on ddr-densho-1000.
TRANSFER_COLLECTIONS = ["ddr-densho-1007", "ddr-densho-1010", "ddr-densho-1014", "ddr-densho-1011"]
PARAM_BUDGET = 130_000


def harvest(collections: list[str], cap_interviews: int = 200,
            budget_s: float = 900.0) -> int:
    """Fetch the held-out collections.

    Logs a heartbeat every 10 interviews and stops at a wall-clock budget. An earlier run
    of this phase sat silent for 44 minutes with no way to tell whether it was throttled,
    retrying, or wedged, because progress was only reported once per collection. A harvest
    that stops early is recoverable; one that cannot be observed is not.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_iv = n_seg = 0
    with (OUTDIR / "interviews.jsonl").open("w", encoding="utf-8") as fi, \
         (OUTDIR / "segments.jsonl").open("w", encoding="utf-8") as fs:
        for col in collections:
            if time.time() - t0 > budget_s:
                log(f"  harvest budget of {budget_s:.0f}s reached; stopping before {col}")
                break
            log(f"  listing {col} ...")
            try:
                ivs = [o for o in page_all(f"{API_ROOT}/{col}/children/") if o.get("model") == "entity"]
            except Exception as e:  # noqa: BLE001
                log(f"  collection {col} unavailable: {type(e).__name__}")
                continue
            log(f"  {col}: {len(ivs)} interviews")
            for iv in ivs:
                if n_iv >= cap_interviews or time.time() - t0 > budget_s:
                    break
                iid = iv["id"]
                try:
                    segs = [s for s in page_all(f"{API_ROOT}/{iid}/children/")
                            if s.get("model") == "segment"]
                except Exception:  # noqa: BLE001
                    continue
                if not segs:
                    continue
                n_iv += 1
                fi.write(json.dumps({
                    "interview_id": iid, "collection_id": col, "title": iv.get("title"),
                    "description": iv.get("description"), "creation": iv.get("creation"),
                    "location": iv.get("location"), "creators": iv.get("creators"),
                    "narrators": narrators_of(iv.get("creators")), "topics": iv.get("topics"),
                    "facility": iv.get("facility"), "geography": iv.get("geography"),
                    "rights": iv.get("rights"), "contributor": iv.get("contributor"),
                    "credit": iv.get("credit"), "format": iv.get("format"),
                    "genre": iv.get("genre"), "n_segments": len(segs),
                    "transcript_url": None, "transcript_chars": 0,
                }, ensure_ascii=False) + "\n")
                for s in segs:
                    n_seg += 1
                    fs.write(json.dumps({
                        "segment_id": s["id"], "interview_id": iid, "collection_id": col,
                        "sort": s.get("index"), "title": s.get("title"),
                        "description": s.get("description"), "extent": s.get("extent"),
                        "creators": s.get("creators"), "narrators": narrators_of(s.get("creators")),
                        "topics": s.get("topics"), "geography": s.get("geography"),
                        "location": s.get("location"), "creation": s.get("creation"),
                        "rights": s.get("rights"), "format": s.get("format"),
                        "genre": s.get("genre"), "language": s.get("language"),
                    }, ensure_ascii=False) + "\n")
                if n_iv % 10 == 0:
                    log(f"    {n_iv} interviews / {n_seg} segments "
                        f"({time.time() - t0:.0f}s elapsed)")
            if n_iv >= cap_interviews or time.time() - t0 > budget_s:
                break
    log(f"  harvested {n_iv} interviews / {n_seg} segments in {time.time() - t0:.0f}s")
    return n_iv


def run_grid(segments, rank_map, encoder, device, seeds, models=("M3_typed_star", "M5_ccnn")):
    cx = build_complex(segments, rank_map, granularity="mid", rank_map_name="R-A")
    if len(cx.by_rank(2)) < 5:
        return cx, [], []
    feats = build_features(cx, segments, encoder)
    sp = make_split(cx, "narrator-disjoint", seed=0)
    parts = partition_incidences(cx, sp)
    ns = NegativeSampler(cx, seed=0)
    data = {p: ns.build(parts[p], "MNS", ratio=10) for p in ("train", "val", "test")}
    train_cx = mask_incidences(cx, set(parts["train"]))
    H = train_cx.incidence_matrix(0, 2)
    enc0 = hypergraph_encodings(H, 16) if H.shape[1] else np.zeros((H.shape[0], 21), np.float32)
    bp = make_bundle(train_cx, feats, device)
    be = make_bundle(train_cx, feats, device, extra={0: enc0})
    _, sid2i = segment_embeddings(segments, encoder)
    data["t1_queries"] = build_t1_queries(cx, segments, sp, "test", limit=400)
    data["seg_index"] = sid2i

    rows, items = [], []
    for m in models:
        bundle = be if m == "M3_typed_star" else bp
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(m, dims, bundle, PARAM_BUDGET)
        for s in seeds:
            cfg = RunConfig(model=m, granularity="mid", rank_map="R-A",
                            split="narrator-disjoint", neg_regime="MNS", seed=s, hidden=hidden)
            out = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
            out["model"] = m
            pit = out.pop("per_item", [])
            for r in pit:
                r["model"] = m
                r["seed"] = s
            items.extend(pit)
            rows.append(out)
        mine = [r for r in rows if r["model"] == m]
        log(f"    {m:16s} T1_MAP={np.nanmean([r['T1_map'] for r in mine]):.4f} "
            f"T2_AUC={np.nanmean([r['T2_auc'] for r in mine]):.4f}")
    return cx, rows, items


def main() -> None:
    set_all_seeds(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_contact = time.strftime("%Y-%m-%d %H:%M:%S")
    log(f"E10 - FIRST CONTACT WITH THE TRANSFER ARCHIVE: {first_contact}")

    log("E10.1 - harvesting held-out collections")
    n = harvest(TRANSFER_COLLECTIONS)
    if n == 0:
        (RES / "e10_transfer.json").write_text(json.dumps(
            {"status": "transfer archive unavailable", "first_contact": first_contact},
            indent=2), encoding="utf-8")
        return

    interviews_t, segments_t = load_corpus.__wrapped__(str(OUTDIR))
    stats_t = corpus_stats(interviews_t, segments_t)
    log(f"  transfer: {stats_t['n_interviews']} interviews / {stats_t['n_segments']} segments / "
        f"{stats_t['n_narrators']} narrators")

    encoder = get_encoder()
    maps = json.loads((RES / "e2_3_rank_maps.json").read_text(encoding="utf-8"))
    rank_map_primary = maps["R-A_consensus"]

    # E10.1 - agreement on the transfer archive using the UNMODIFIED manual
    study_t = run_rank_study(segments_t, encoder=encoder)
    diag_t = agreement_diagnostics(study_t.labels, study_t.round2_terms)
    agree_primary = json.loads((RES / "e2_2_agreement.json").read_text(encoding="utf-8"))
    alpha_home = agree_primary["alpha_round2_REPORTED"]["overall"]
    alpha_away = study_t.alpha_round2["overall"]
    log(f"E10.1 - alpha home={alpha_home:.3f} away={alpha_away:.3f} drop={alpha_home - alpha_away:.3f}")

    # E10.2 - zero-shot structural transfer, no retuning of anything
    unseen = [t for t in study_t.terms if t not in rank_map_primary]
    merged = dict(rank_map_primary)
    for t in unseen:  # terms the primary archive never used still need a rank
        merged[t] = study_t.consensus[t]
    cx_t = build_complex(segments_t, merged, granularity="mid", rank_map_name="R-A")
    dec_t = rank_cardinality_decoupling(cx_t)
    sizes_t = np.array([c.size for c in cx_t.by_rank(2)]) if cx_t.by_rank(2) else np.array([0])

    log("E10.2 - zero-shot model transfer")
    _, rows_t, items_t = run_grid(segments_t, merged, encoder, device, SEEDS)

    # E10.3 - does the CONCLUSION transfer?
    inter_away = interaction_model(items_t, "M5_ccnn", "M3_typed_star") if items_t else {}
    inter_home = json.loads((RES / "e7_5_interaction.json").read_text(encoding="utf-8")) \
        if (RES / "e7_5_interaction.json").exists() else {}

    home_ci = inter_home.get("interaction_ci") or [float("nan")] * 2
    away_ci = inter_away.get("interaction_ci") or [float("nan")] * 2
    home_dir = inter_home.get("direction")
    away_dir = inter_away.get("direction")
    transfers = bool(home_dir and away_dir and home_dir == away_dir
                     and home_dir != "indeterminate")

    (RES / "e10_transfer.json").write_text(json.dumps({
        "first_contact_with_transfer_archive": first_contact,
        "collections": TRANSFER_COLLECTIONS,
        "substitution_note": ("PREREGISTRATION substitution_ledger E1_2: the transfer archive is a "
                              "held-out Densho collection, so this is cross-collection rather than "
                              "cross-institution transfer"),
        "transfer_corpus_stats": stats_t,
        "E10_1_meta_rank": {
            "alpha_home": alpha_home, "alpha_transfer": alpha_away,
            "alpha_drop": alpha_home - alpha_away,
            "raw_agreement_transfer": diag_t.get("mean_pairwise_raw_agreement"),
            "n_terms_transfer": len(study_t.terms),
            "n_terms_unseen_in_primary": len(unseen),
        },
        "E10_2_zero_shot": {
            "rank_cardinality": {"spearman_rho": dec_t["spearman_rho"],
                                 "premise_holds": dec_t["premise_holds"],
                                 "rank_inversion_rate": dec_t["rank_inversion_rate"]},
            "n_rank2_cells": len(cx_t.by_rank(2)),
            "singleton_fraction": float((sizes_t == 1).mean()) if len(sizes_t) else None,
            "attestation_mean": float(sizes_t.mean()) if len(sizes_t) else None,
            "attestation_max": int(sizes_t.max()) if len(sizes_t) else None,
            "model_means": {m: {
                "T1_map": float(np.nanmean([r["T1_map"] for r in rows_t if r["model"] == m])),
                "T2_auc": float(np.nanmean([r["T2_auc"] for r in rows_t if r["model"] == m])),
            } for m in {r["model"] for r in rows_t}},
        },
        "E10_3_conclusion_transfer": {
            "interaction_home": {"coef": inter_home.get("interaction_coef"), "ci": home_ci,
                                 "direction": home_dir},
            "interaction_transfer": {"coef": inter_away.get("interaction_coef"), "ci": away_ci,
                                     "direction": away_dir},
            "conclusion_transfers": transfers,
            "kill_criterion": ("if the interaction CI crosses zero on the transfer archive while "
                               "being clearly signed on the primary, the finding is archive-specific "
                               "and must be reported that way"),
            "transfer_ci_crosses_zero": inter_away.get("ci_crosses_zero"),
        },
    }, indent=2), encoding="utf-8")
    log(f"E10.3 - home dir={home_dir} transfer dir={away_dir} conclusion_transfers={transfers}")
    log("PHASE 10 complete")


if __name__ == "__main__":
    main()
