"""E-NEW-2: does the leak scale with how much identity there is to leak?

The mechanism section shows that a rank-1 moment's input vector is its narrator's vector,
so a random split places identical vectors on both sides of the partition. If that is the
mechanism, the complex's advantage under a random split should be larger for narrators who
contribute more moments -- more of that narrator's vector sits in training.

A single number ("the gap collapses under anonymisation") is one observation. A slope makes
it a dose-response, which is much harder to explain away as an artefact of one condition.

Fits per-narrator advantage (complex minus typed star, per-incidence correctness under the
random split) against log segments-per-narrator, with narrator random effects. Reports the
same regression under the narrator-disjoint split as a control: there the narrator is unseen
in training, so there is nothing to leak and the slope should be flat.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.train import RunConfig, train_eval  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("dose_response")
MODELS = ("M5_ccnn", "M3_typed_star")
SEEDS = tuple(range(10))
SPLITS = ("random", "narrator-disjoint")


def collect(segments, rank_maps, split_kind, encoder, device) -> list[dict]:
    cx, bp, be, data, _sp, _dense = prepare_cell(
        segments, rank_maps, PRIMARY["granularity"], split_kind, PRIMARY["neg"],
        PRIMARY["rank_map"], encoder, device, {})
    items: list[dict] = []
    for m in MODELS:
        bundle = be if m == "M3_typed_star" else bp
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(m, dims, bundle, PARAM_BUDGET)
        for s in SEEDS:
            set_all_seeds(s)
            cfg = RunConfig(model=m, granularity=PRIMARY["granularity"],
                            rank_map=PRIMARY["rank_map"], split=split_kind,
                            neg_regime=PRIMARY["neg"], seed=s, hidden=hidden)
            out = train_eval(cfg, cx, bundle, data, device, eval_cx=cx)
            for r in out.pop("per_item", []):
                r["model"] = m
                r["seed"] = s
                items.append(r)
        log(f"  {split_kind:18s} {m:16s} done ({len(items)} rows so far)")
    return items


def fit(items: list[dict], seg_per_narrator: dict[str, int]) -> dict:
    """Per-narrator advantage regressed on log segments-per-narrator."""
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame(items)
    if df.empty:
        return {"error": "no rows"}
    # Average over seeds and incidences within (model, narrator) before differencing, so
    # ten seeds are not treated as ten independent narrators.
    per = (df.groupby(["model", "narrator_id"], as_index=False)
             .agg(correct=("correct", "mean"), n_items=("correct", "size")))
    wide = per.pivot(index="narrator_id", columns="model", values="correct").dropna()
    if wide.empty or not {"M5_ccnn", "M3_typed_star"} <= set(wide.columns):
        return {"error": "missing a model"}
    adv = (wide["M5_ccnn"] - wide["M3_typed_star"]).rename("advantage").reset_index()
    adv["n_segments"] = adv["narrator_id"].map(seg_per_narrator)
    adv = adv.dropna()
    adv = adv[adv["n_segments"] > 0]
    if len(adv) < 10:
        return {"error": f"only {len(adv)} narrators"}
    adv["log_segments"] = np.log(adv["n_segments"].astype(float))

    m = smf.ols("advantage ~ log_segments", data=adv).fit(cov_type="HC3")
    ci = m.conf_int().loc["log_segments"]
    return {
        "n_narrators": int(len(adv)),
        "slope": float(m.params["log_segments"]),
        "slope_se": float(m.bse["log_segments"]),
        "slope_p": float(m.pvalues["log_segments"]),
        "slope_ci": [float(ci[0]), float(ci[1])],
        "ci_crosses_zero": bool(ci[0] <= 0 <= ci[1]),
        "mean_advantage": float(adv["advantage"].mean()),
        "median_segments": float(adv["n_segments"].median()),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segments = load_corpus()[1]
    rank_maps = load_rank_maps()
    encoder = get_encoder()

    seg_per_narrator: dict[str, int] = defaultdict(int)
    for s in segments:
        for n in s.narrators:
            seg_per_narrator[n] += 1
    log(f"device={device} narrators={len(seg_per_narrator)} "
        f"median segments/narrator={np.median(list(seg_per_narrator.values())):.0f}")

    out: dict = {"labelled": "EXPLORATORY - dose-response test of the stated mechanism",
                 "prediction": ("positive slope under the random split, where a narrator's "
                                "own segments are in training; flat under narrator-disjoint, "
                                "where they are not")}
    for split_kind in SPLITS:
        items = collect(segments, rank_maps, split_kind, encoder, device)
        res = fit(items, dict(seg_per_narrator))
        out[split_kind] = res
        log(f"  SLOPE {split_kind:18s} {res.get('slope', float('nan')):+.5f} "
            f"CI={[round(x, 5) for x in res.get('slope_ci', [float('nan')] * 2)]} "
            f"p={res.get('slope_p', float('nan')):.3g} n={res.get('n_narrators')}")
        (RES / "e_dose_response.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    r, nd = out.get("random", {}), out.get("narrator-disjoint", {})
    if "slope" in r and "slope" in nd:
        out["slope_difference"] = float(r["slope"] - nd["slope"])
        out["mechanism_supported"] = bool(
            not r.get("ci_crosses_zero", True) and r["slope"] > 0
            and r["slope"] > nd["slope"])
        log(f"  slope difference (random - narrator-disjoint) = {out['slope_difference']:+.5f}")
        log(f"  mechanism_supported = {out['mechanism_supported']}")
    (RES / "e_dose_response.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("dose-response complete")


if __name__ == "__main__":
    main()
