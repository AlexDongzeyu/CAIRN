"""E4 — paired bootstrap over narrators, and an interval for the difference-in-differences.

Figure 4's intervals are over ten seeds, which measure optimisation noise. The sampling unit
the claim is about is the query, and queries cluster by narrator: one prolific narrator can
contribute dozens. So the interval that belongs beside the headline is a cluster bootstrap
that resamples narrators.

The paper's central quantity has never carried an interval at all. It is not either split's
gap but the difference between them,

    (M5_random - M5_disjoint) - (M3_random - M3_disjoint),

and that is what row `did` reports. Resampling is done once per replicate over the union of
narrators, and each split then contributes whatever queries the sampled narrators own; the
two splits hold out different narrators, so a replicate is not required to populate both
equally.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logutil import make_logger  # noqa: E402

RES = ROOT / "data" / "results"
N_BOOT = 10_000
log = make_logger("paired_bootstrap")


def _ci(v: np.ndarray, alpha: float = 0.05) -> list[float]:
    v = v[np.isfinite(v)]
    if not len(v):
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(v, 100 * alpha / 2)), 4),
            round(float(np.percentile(v, 100 * (1 - alpha / 2))), 4)]


def main() -> None:
    payload = json.loads((RES / "e_split_first.json").read_text(encoding="utf-8"))
    pq = payload["per_query"]

    # Average over seeds first: a query's score is a property of the model, not of a seed.
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    narr_of: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in pq:
        key = (r["feature_mode"], r["split"], r["model"])
        acc[key][r["qid"]].append(r["ap"])
        narr_of[key][r["qid"]] = r["narrator"]

    mean_ap = {k: {q: float(np.mean(v)) for q, v in d.items()} for k, d in acc.items()}

    modes = payload["feature_modes"]
    splits = payload["splits"]
    rng = np.random.default_rng(0)
    out: dict = {"n_boot": N_BOOT, "paired": {}, "did": {}, "means": {}, "exposure": payload["exposure"]}

    # Seed-averaged T1 MAP per condition, in a shape the paper's macro paths can address.
    for (fmode, split, model), d in mean_ap.items():
        out["means"].setdefault(fmode, {}).setdefault(split, {})[model] = round(
            float(np.mean(list(d.values()))), 4)
    # The quantity E1 turns on: how much the random split lifts each model over the
    # narrator-disjoint one, under each feature construction.
    for fmode in modes:
        m = out["means"].get(fmode, {})
        if "random" in m and "narrator-disjoint" in m:
            out.setdefault("split_lift", {})[fmode] = {
                model: round(m["random"][model] - m["narrator-disjoint"][model], 4)
                for model in m["random"] if model in m["narrator-disjoint"]
            }

    for fmode in modes:
        # queries by narrator, per split, shared across models within a split
        for challenger in ("M3_strong", "M3_matched"):
            for split in splits:
                k5 = (fmode, split, "M5_ccnn")
                k3 = (fmode, split, challenger)
                if k5 not in mean_ap or k3 not in mean_ap:
                    continue
                shared = sorted(set(mean_ap[k5]) & set(mean_ap[k3]))
                by_narr: dict[str, list[str]] = defaultdict(list)
                for q in shared:
                    by_narr[narr_of[k5][q]].append(q)
                narrs = sorted(by_narr)
                a5 = np.array([mean_ap[k5][q] for q in shared])
                a3 = np.array([mean_ap[k3][q] for q in shared])
                pos = {q: i for i, q in enumerate(shared)}
                idx_of_narr = [np.array([pos[q] for q in by_narr[n]]) for n in narrs]

                boot = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    pick = rng.integers(0, len(narrs), len(narrs))
                    sel = np.concatenate([idx_of_narr[i] for i in pick])
                    boot[b] = a3[sel].mean() - a5[sel].mean()
                obs = float(a3.mean() - a5.mean())
                out["paired"][f"{fmode}|{split}|{challenger}-M5"] = {
                    "observed_difference": round(obs, 4),
                    "ci95": _ci(boot),
                    "n_queries": len(shared),
                    "n_narrators": len(narrs),
                    "M3_map": round(float(a3.mean()), 4),
                    "M5_map": round(float(a5.mean()), 4),
                }
                log(f"{fmode:11s} {split:17s} {challenger}-M5 = {obs:+.4f} "
                    f"CI{_ci(boot)}  ({len(shared)} queries, {len(narrs)} narrators)")

            # difference-in-differences: how much more the random split helps M5 than M3
            if not {"random", "narrator-disjoint"} <= set(splits):
                continue
            packs = {}
            ok = True
            for split in ("random", "narrator-disjoint"):
                k5, k3 = (fmode, split, "M5_ccnn"), (fmode, split, challenger)
                if k5 not in mean_ap or k3 not in mean_ap:
                    ok = False
                    break
                shared = sorted(set(mean_ap[k5]) & set(mean_ap[k3]))
                by_narr = defaultdict(list)
                for q in shared:
                    by_narr[narr_of[k5][q]].append(q)
                packs[split] = (
                    np.array([mean_ap[k5][q] for q in shared]),
                    np.array([mean_ap[k3][q] for q in shared]),
                    {q: i for i, q in enumerate(shared)},
                    by_narr,
                )
            if not ok:
                continue
            union = sorted(set(packs["random"][3]) | set(packs["narrator-disjoint"][3]))
            lookup = {
                split: [np.array([packs[split][2][q] for q in packs[split][3].get(n, ())],
                                 dtype=int) for n in union]
                for split in ("random", "narrator-disjoint")
            }
            boot = np.full(N_BOOT, np.nan)
            for b in range(N_BOOT):
                pick = rng.integers(0, len(union), len(union))
                gaps = {}
                for split in ("random", "narrator-disjoint"):
                    sel = np.concatenate([lookup[split][i] for i in pick]) if len(pick) else np.array([], int)
                    if not len(sel):
                        gaps = {}
                        break
                    a5, a3 = packs[split][0], packs[split][1]
                    gaps[split] = a5[sel].mean() - a3[sel].mean()
                if len(gaps) == 2:
                    boot[b] = gaps["random"] - gaps["narrator-disjoint"]
            obs_gaps = {
                s: float(packs[s][0].mean() - packs[s][1].mean())
                for s in ("random", "narrator-disjoint")
            }
            did = obs_gaps["random"] - obs_gaps["narrator-disjoint"]
            out["did"][f"{fmode}|M5-{challenger}"] = {
                "gap_random": round(obs_gaps["random"], 4),
                "gap_narrator_disjoint": round(obs_gaps["narrator-disjoint"], 4),
                "difference_in_differences": round(did, 4),
                "ci95": _ci(boot),
                "n_narrators_resampled": len(union),
                "replicates_usable": int(np.isfinite(boot).sum()),
            }
            log(f"{fmode:11s} DiD M5-{challenger} = {did:+.4f} CI{_ci(boot)}")

    (RES / "e_paired_bootstrap.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    log("wrote e_paired_bootstrap.json")


if __name__ == "__main__":
    main()
