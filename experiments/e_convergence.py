"""Does M5 train? An overfit check and a validation-AUC trace, for the primary cell.

M5's link AUC sits near chance while a parameter-free baseline reaches 0.725, and a reader is
entitled to read that as an optimisation failure rather than a finding. Two cheap measurements
separate those readings: whether the model drives training loss toward zero on a small
held-in subset with regularisation off, and whether its validation AUC is flat from the first
epoch. A model that overfits on demand is training, and a flat validation curve is then a
statement about what the architecture extracts, not about the optimiser.

M3 runs under the identical protocol as the positive control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from run_phase5_7 import PARAM_BUDGET, PRIMARY, load_rank_maps, prepare_cell  # noqa: E402
from src.corpus import load_corpus  # noqa: E402
from src.features import get_encoder  # noqa: E402
from src.logutil import make_logger  # noqa: E402
from src.models import build_model, match_hidden_to_budget  # noqa: E402
from src.seeds import set_all_seeds  # noqa: E402
from src.train import _pairs_to_idx, auc_roc  # noqa: E402

RES = ROOT / "data" / "results"
log = make_logger("convergence")
MODELS = ("M5_ccnn", "M3_typed_star")
SUBSET = 200
EPOCHS = 600


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segments = load_corpus()[1]
    cx, bp, be, data, _sp, _dense = prepare_cell(
        segments, load_rank_maps(), PRIMARY["granularity"], PRIMARY["split"], PRIMARY["neg"],
        PRIMARY["rank_map"], get_encoder(), device, {})
    n_index = {n: i for i, n in enumerate(cx.narrators)}
    c_index = cx.index(2)
    log(f"device={device}: overfit on {SUBSET} incidences, dropout 0, {EPOCHS} epochs")

    out: dict = {"subset_size": SUBSET, "epochs": EPOCHS, "device": device, "models": {}}
    for name in MODELS:
        bundle = be if name == "M3_typed_star" else bp
        dims = {k: bundle.X[k].shape[1] for k in bundle.X}
        hidden = match_hidden_to_budget(name, dims, bundle, PARAM_BUDGET)
        set_all_seeds(0)
        model = build_model(name, dims, bundle, hidden, 2, 0.0).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)

        ta, tb, ty = _pairs_to_idx(*data["train"], n_index, c_index)
        ta = torch.tensor(ta[:SUBSET], dtype=torch.long, device=device)
        tb = torch.tensor(tb[:SUBSET], dtype=torch.long, device=device)
        ty = torch.tensor(ty[:SUBSET], dtype=torch.float32, device=device)
        va, vb, vy = _pairs_to_idx(*data["val"], n_index, c_index)
        va_t = torch.tensor(va, dtype=torch.long, device=device)
        vb_t = torch.tensor(vb, dtype=torch.long, device=device)

        losses, trace = [], []
        for ep in range(EPOCHS):
            model.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model.score(model(bundle), ta, tb, 0, 2), ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))
            if ep % 25 == 0 or ep == EPOCHS - 1:
                model.eval()
                with torch.no_grad():
                    s = model.score(model(bundle), va_t, vb_t, 0, 2).cpu().numpy()
                trace.append([ep, float(auc_roc(np.asarray(vy, dtype=float), s))])

        first, last = trace[0][1], trace[-1][1]
        out["models"][name] = {
            "train_loss_start": losses[0], "train_loss_final": losses[-1],
            "train_loss_min": min(losses), "val_auc_first": first, "val_auc_final": last,
            "val_auc_max": max(v for _, v in trace), "val_auc_gain": last - first,
            "val_auc_trace": trace, "hidden": hidden}
        log(f"  {name:16s} train loss {losses[0]:.4f} -> {losses[-1]:.4f} "
            f"(min {min(losses):.4f}); val AUC {first:.3f} -> {last:.3f}")

    (RES / "e_convergence.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("convergence check complete")


if __name__ == "__main__":
    main()
