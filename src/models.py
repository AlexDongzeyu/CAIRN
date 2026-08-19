"""E6 — the model zoo.

Every model consumes the SAME frozen text features and the SAME structure; they differ
only in the operator applied to it. Parameter counts are reported and matched.

  M0  feature-only MLP                      no structure at all (the skeptical control)
  M1  dense retrieval                       cosine over frozen text, no parameters
  M2  untyped star GNN                      G* with node types erased (the weak baseline)
  M3  typed star GNN + hypergraph encodings CRITICAL BASELINE - strictly more information
  M4  AllSet / ED-HNN / Hypergraph-MLP      higher-order but rank-agnostic
  M5  CCNN                                  rank-aware up/down/within message passing

M3 receives explicit rank as a feature AND hypergraph Laplacian/curvature encodings, so
it is not handicapped relative to M5. That is deliberate: it is the comparison the claim
has to survive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ("M0_mlp", "M1_dense", "M2_untyped_star", "M3_typed_star",
               "M4_allset", "M4_edhnn", "M4_hgmlp", "M5_ccnn")
ABLATIONS = ("A1_shared_weights", "A2_shuffled_ranks", "A3_collapse_r2r3",
             "A4_no_down", "A5_no_moments")


def sp_to_torch(m: sp.spmatrix, device) -> torch.Tensor:
    m = m.tocoo()
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    val = torch.tensor(m.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, m.shape).coalesce().to(device)


def row_normalize(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m, dtype=np.float32)
    d = np.asarray(m.sum(axis=1)).ravel()
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    return sp.diags(inv) @ m


@dataclass
class Bundle:
    """Everything a model may read. Identical content for the CC and for G*."""
    X: dict[int, torch.Tensor]                 # rank -> node features
    B: dict[tuple[int, int], torch.Tensor]     # (k,j) -> row-normalized incidence
    Bt: dict[tuple[int, int], torch.Tensor]    # transposes
    ranks_present: tuple[int, ...]
    device: torch.device


def make_bundle(cx, feats: dict[int, np.ndarray], device, extra: dict[int, np.ndarray] | None = None
                ) -> Bundle:
    X, B, Bt = {}, {}, {}
    for k in range(4):
        f = feats[k]
        if extra and k in extra and extra[k].shape[0] == f.shape[0]:
            f = np.hstack([f, extra[k]])
        X[k] = torch.tensor(f, dtype=torch.float32, device=device)
    for (k, j) in [(0, 1), (1, 2), (2, 3), (1, 3), (0, 2), (0, 3)]:
        try:
            m = cx.incidence_matrix(k, j)
        except ValueError:
            continue
        if m.shape[0] == 0 or m.shape[1] == 0:
            continue
        B[(k, j)] = sp_to_torch(row_normalize(m), device)
        Bt[(k, j)] = sp_to_torch(row_normalize(m.T), device)
    return Bundle(X, B, Bt, tuple(range(4)), device)


class _Head(nn.Module):
    """Shared scoring head so no model gains from a fancier decoder."""

    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * d, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b, a * b], dim=-1)).squeeze(-1)


class BaseModel(nn.Module):
    def __init__(self, dims: dict[int, int], hidden: int):
        super().__init__()
        self.hidden = hidden
        self.inp = nn.ModuleDict({str(k): nn.Linear(d, hidden) for k, d in dims.items()})
        self.head = _Head(hidden)

    def encode_inputs(self, bundle: Bundle) -> dict[int, torch.Tensor]:
        return {k: self.inp[str(k)](bundle.X[k]) for k in bundle.X}

    def score(self, h: dict[int, torch.Tensor], ai: torch.Tensor, bi: torch.Tensor,
              ra: int, rb: int) -> torch.Tensor:
        return self.head(h[ra][ai], h[rb][bi])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class M0_MLP(BaseModel):
    """No structure. If this is competitive, structure is not doing work."""

    def __init__(self, dims, hidden=64, layers=2, dropout=0.1):
        super().__init__(dims, hidden)
        self.mlp = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, bundle: Bundle):
        h = self.encode_inputs(bundle)
        for k in h:
            for lin in self.mlp:
                h[k] = self.drop(F.relu(lin(h[k])))
        return h


class M2_UntypedStar(BaseModel):
    """G* with node types erased: one weight matrix for every relation."""

    def __init__(self, dims, hidden=64, layers=2, dropout=0.1):
        super().__init__(dims, hidden)
        self.lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.self_lin = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, bundle: Bundle):
        h = self.encode_inputs(bundle)
        for lin, slin in zip(self.lin, self.self_lin):
            new = {k: slin(v) for k, v in h.items()}
            for (k, j), Bkj in bundle.B.items():
                if k in h and j in h:
                    new[j] = new[j] + lin(torch.sparse.mm(bundle.Bt[(k, j)], h[k]))
                    new[k] = new[k] + lin(torch.sparse.mm(Bkj, h[j]))
            h = {k: self.drop(F.relu(v)) for k, v in new.items()}
        return h


class M3_TypedStar(BaseModel):
    """Typed (relation-specific) star GNN. Receives hypergraph encodings and explicit
    rank in its inputs, i.e. strictly more feature information than the CCNN."""

    def __init__(self, dims, hidden=64, layers=2, dropout=0.1, rel_keys=()):
        super().__init__(dims, hidden)
        self.rel_keys = [f"{k}_{j}" for (k, j) in rel_keys]
        self.up = nn.ModuleList([nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in self.rel_keys})
                                 for _ in range(layers)])
        self.down = nn.ModuleList([nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in self.rel_keys})
                                   for _ in range(layers)])
        self.self_lin = nn.ModuleList([nn.ModuleDict({str(k): nn.Linear(hidden, hidden) for k in dims})
                                       for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, bundle: Bundle):
        h = self.encode_inputs(bundle)
        for li in range(len(self.up)):
            new = {k: self.self_lin[li][str(k)](v) for k, v in h.items()}
            for (k, j), Bkj in bundle.B.items():
                r = f"{k}_{j}"
                if r not in self.up[li] or k not in h or j not in h:
                    continue
                new[j] = new[j] + self.up[li][r](torch.sparse.mm(bundle.Bt[(k, j)], h[k]))
                new[k] = new[k] + self.down[li][r](torch.sparse.mm(Bkj, h[j]))
            h = {k: self.drop(F.relu(v)) for k, v in new.items()}
        return h


class M4_HigherOrder(BaseModel):
    """Rank-agnostic higher-order models. `variant` selects AllSet-style attention,
    ED-HNN-style equivariant diffusion, or the message-passing-free Hypergraph-MLP."""

    def __init__(self, dims, hidden=64, layers=2, dropout=0.1, variant="allset"):
        super().__init__(dims, hidden)
        self.variant = variant
        self.v2e = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.e2v = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.att = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, bundle: Bundle):
        h = self.encode_inputs(bundle)
        if self.variant == "hgmlp":       # skeptical control: no message passing at all
            for lin in self.v2e:
                h = {k: self.drop(F.relu(lin(v))) for k, v in h.items()}
            return h
        # treat rank-2 cells as hyperedges over rank-0 narrators
        key = (0, 2) if (0, 2) in bundle.B else (0, 1)
        Bkj, Btkj = bundle.B[key], bundle.Bt[key]
        for li in range(len(self.v2e)):
            e = torch.sparse.mm(Btkj, h[key[0]])
            e = F.relu(self.v2e[li](e))
            if self.variant == "allset":
                e = e * torch.sigmoid(self.att[li](e))
            v = torch.sparse.mm(Bkj, e)
            v = F.relu(self.e2v[li](v))
            if self.variant == "edhnn":   # equivariant residual diffusion
                v = v + h[key[0]]
            h = dict(h)
            h[key[0]] = self.drop(v)
            h[key[1]] = self.drop(e)
        return h


class M5_CCNN(BaseModel):
    """Rank-aware higher-order message passing.

    Per layer: up r0->r1->r2->r3, down r3->r2->r1->r0, and within-rank exchange through
    (co)adjacency. Weight matrices are rank-specific; sharing them destroys the very
    thing the experiment is testing (ablation A1).

    Within-rank exchange is computed as B (B^T H) rather than by forming the adjacency
    B B^T, which is algebraically identical and avoids materialising a 13k x 13k matrix
    for the moment layer.
    """

    def __init__(self, dims, hidden=64, layers=2, dropout=0.1,
                 share_weights=False, use_down=True, use_within=True, skip_moments=False):
        super().__init__(dims, hidden)
        self.share, self.use_down, self.use_within = share_weights, use_down, use_within
        self.skip_moments = skip_moments
        n_rel = 1 if share_weights else 3
        self.up = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_rel)])
                                 for _ in range(layers)])
        self.dn = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_rel)])
                                 for _ in range(layers)])
        self.wn = nn.ModuleList([nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_rel)])
                                 for _ in range(layers)])
        self.wi = nn.ModuleList([nn.ModuleDict({str(k): nn.Linear(hidden, hidden) for k in dims})
                                 for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

    def _w(self, mods, i):
        return mods[0] if self.share else mods[i]

    def forward(self, bundle: Bundle):
        h = self.encode_inputs(bundle)
        # A5 removes the intermediate moment rank, so narrators attach to events directly.
        ladder = [(0, 2), (2, 3)] if self.skip_moments else [(0, 1), (1, 2), (2, 3)]
        for li in range(len(self.up)):
            new = {k: self.wi[li][str(k)](v) for k, v in h.items()}
            for i, (k, j) in enumerate(ladder):
                if (k, j) not in bundle.B or k not in h or j not in h:
                    continue
                Bt, B = bundle.Bt[(k, j)], bundle.B[(k, j)]
                new[j] = new[j] + self._w(self.up[li], i)(torch.sparse.mm(Bt, h[k]))
                if self.use_down:
                    new[k] = new[k] + self._w(self.dn[li], i)(torch.sparse.mm(B, h[j]))
                if self.use_within:
                    # rank-k cells exchange with each other through their shared rank-j cofaces
                    within = torch.sparse.mm(B, torch.sparse.mm(Bt, h[k]))
                    new[k] = new[k] + self._w(self.wn[li], i)(within)
            h = {k: self.drop(F.relu(v)) for k, v in new.items()}
        return h


class M1_Dense:
    """Parameter-free retrieval floor: cosine over the frozen text embeddings.

    Included because if a bi-encoder with no structure at all is competitive, the
    structural models are not earning their keep. It is scored with exactly the same
    metric code as every other model.
    """

    def __init__(self, seg_emb: np.ndarray, cell_emb: np.ndarray, narr_emb: np.ndarray):
        self.seg_emb = seg_emb
        self.cell_emb = cell_emb
        self.narr_emb = narr_emb

    def t2_scores(self, narr_idx: np.ndarray, cell_idx: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", self.narr_emb[narr_idx], self.cell_emb[cell_idx])

    def n_params(self) -> int:
        return 0


def build_model(name: str, dims, bundle: Bundle, hidden=64, layers=2, dropout=0.1, **kw):
    rel_keys = tuple(bundle.B.keys())
    if name == "M0_mlp":
        return M0_MLP(dims, hidden, layers, dropout)
    if name == "M2_untyped_star":
        return M2_UntypedStar(dims, hidden, layers, dropout)
    if name == "M3_typed_star":
        return M3_TypedStar(dims, hidden, layers, dropout, rel_keys=rel_keys)
    if name == "M4_allset":
        return M4_HigherOrder(dims, hidden, layers, dropout, variant="allset")
    if name == "M4_edhnn":
        return M4_HigherOrder(dims, hidden, layers, dropout, variant="edhnn")
    if name == "M4_hgmlp":
        return M4_HigherOrder(dims, hidden, layers, dropout, variant="hgmlp")
    if name == "M5_ccnn":
        return M5_CCNN(dims, hidden, layers, dropout, **kw)
    raise ValueError(name)


def match_hidden_to_budget(name: str, dims, bundle: Bundle, target_params: int,
                           layers: int = 2, dropout: float = 0.1,
                           lo: int = 8, hi: int = 256, **kw) -> int:
    """Pick the hidden width whose parameter count lands closest to a shared budget.

    E6.0 requires the main comparison to be matched within +-15%. Relation-specific
    models (M3) otherwise carry roughly twice the parameters of the shared-weight models
    at the same width, which would make any difference uninterpretable.
    """
    best, best_gap = lo, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = build_model(name, dims, bundle, mid, layers, dropout, **kw).n_params()
        gap = abs(n - target_params)
        if gap < best_gap:
            best, best_gap = mid, gap
        if n < target_params:
            lo = mid + 1
        elif n > target_params:
            hi = mid - 1
        else:
            return mid
    return best
