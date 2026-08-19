"""E6.4 — hypergraph-level structural encodings for the critical baseline.

These are what make M3 (typed star GNN) a *strictly stronger* competitor than the CCNN:
they hand a plain graph model higher-order structural information that standard
message passing cannot derive on its own. If the CCNN still wins after M3 receives
these, the win is attributable to the rank-aware operator rather than to rank
information being unavailable to the baseline.

Implemented:
  * normalized hypergraph Laplacian eigenvector encodings (spectral)
  * Forman-Ricci curvature on the clique expansion (discrete curvature)
  * incidence-degree and cell-size profiles
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def hypergraph_laplacian(H: sp.csr_matrix) -> sp.csr_matrix:
    """Zhou et al. normalized hypergraph Laplacian from an incidence matrix H (V x E)."""
    dv = np.asarray(H.sum(axis=1)).ravel()
    de = np.asarray(H.sum(axis=0)).ravel()
    dv_inv_sqrt = sp.diags(np.divide(1.0, np.sqrt(dv), out=np.zeros_like(dv), where=dv > 0))
    de_inv = sp.diags(np.divide(1.0, de, out=np.zeros_like(de), where=de > 0))
    S = dv_inv_sqrt @ H @ de_inv @ H.T @ dv_inv_sqrt
    return sp.eye(H.shape[0], format="csr") - S


def laplacian_encoding(H: sp.csr_matrix, k: int = 16, seed: int = 0) -> np.ndarray:
    """First k non-trivial eigenvectors of the hypergraph Laplacian, sign-fixed."""
    n = H.shape[0]
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)
    L = hypergraph_laplacian(H).astype(np.float64)
    kk = min(k + 1, max(1, n - 1))
    try:
        rng = np.random.default_rng(seed)
        vals, vecs = eigsh(L, k=kk, which="SM", v0=rng.normal(size=n), maxiter=5000, tol=1e-6)
    except Exception:  # noqa: BLE001 - dense fallback for tiny or ill-conditioned problems
        vals, vecs = np.linalg.eigh(L.toarray())
        vals, vecs = vals[:kk], vecs[:, :kk]
    order = np.argsort(vals)
    vecs = vecs[:, order][:, 1:]  # drop the trivial constant eigenvector
    out = np.zeros((n, k), dtype=np.float32)
    take = min(k, vecs.shape[1])
    if take > 0:
        V = vecs[:, :take]
        # sign convention: make the largest-magnitude entry positive, else the encoding
        # is only defined up to a sign and becomes a source of seed noise
        for j in range(take):
            if V[np.argmax(np.abs(V[:, j])), j] < 0:
                V[:, j] = -V[:, j]
        out[:, :take] = V.astype(np.float32)
    return out


def forman_ricci(H: sp.csr_matrix) -> np.ndarray:
    """Node-level Forman-Ricci curvature on the clique expansion of the hypergraph.

    Cheap, deterministic, and sensitive to exactly the higher-order density that
    pairwise message passing washes out.
    """
    A = (H @ H.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray((A > 0).sum(axis=1)).ravel().astype(np.float64)
    n = A.shape[0]
    curv = np.zeros(n, dtype=np.float32)
    indptr, indices = A.indptr, A.indices
    for i in range(n):
        nbrs = indices[indptr[i]:indptr[i + 1]]
        if len(nbrs) == 0:
            continue
        # Forman curvature of edge (i,j) on an unweighted graph: 4 - deg_i - deg_j
        curv[i] = float(np.mean(4.0 - deg[i] - deg[nbrs]))
    return curv


def degree_profile(H: sp.csr_matrix) -> np.ndarray:
    """Incidence degree and the size profile of the cells a node participates in."""
    dv = np.asarray(H.sum(axis=1)).ravel()
    de = np.asarray(H.sum(axis=0)).ravel()
    Hc = H.tocsr()
    feats = np.zeros((H.shape[0], 4), dtype=np.float32)
    for i in range(H.shape[0]):
        cols = Hc.indices[Hc.indptr[i]:Hc.indptr[i + 1]]
        sizes = de[cols] if len(cols) else np.array([0.0])
        feats[i] = [dv[i], sizes.mean(), sizes.max(), sizes.min()]
    return feats


def hypergraph_encodings(H: sp.csr_matrix, k_spectral: int = 16, seed: int = 0) -> np.ndarray:
    """Full encoding block: [laplacian eigenvectors | forman curvature | degree profile]."""
    if H.shape[0] == 0 or H.shape[1] == 0:
        return np.zeros((H.shape[0], k_spectral + 5), dtype=np.float32)
    lap = laplacian_encoding(H, k=k_spectral, seed=seed)
    curv = forman_ricci(H).reshape(-1, 1)
    prof = degree_profile(H)
    block = np.hstack([lap, curv, prof]).astype(np.float32)
    mu, sd = block.mean(axis=0, keepdims=True), block.std(axis=0, keepdims=True)
    return (block - mu) / np.where(sd > 0, sd, 1.0)
