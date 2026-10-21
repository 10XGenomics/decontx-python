"""Benchmark decontx on synthetic sparse data: wall time and peak RSS.

Usage: python benchmarks/bench.py [n_cells ...]
"""

import resource
import sys
import time

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from decontx import decontx


def make_adata(n_cells, n_genes=20000, n_clusters=10, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, n_clusters, size=n_cells)
    profiles = rng.gamma(0.1, 1.0, size=(n_clusters, n_genes)).astype(np.float32)
    profiles /= profiles.sum(axis=1, keepdims=True)
    indptr = [0]
    indices, data = [], []
    for k in z:
        depth = rng.integers(2000, 8000)
        g = rng.choice(n_genes, size=depth, p=profiles[k].astype(np.float64) / profiles[k].astype(np.float64).sum())
        u, c = np.unique(g, return_counts=True)
        indices.append(u)
        data.append(c)
        indptr.append(indptr[-1] + len(u))
    X = csr_matrix(
        (np.concatenate(data).astype(np.float32), np.concatenate(indices), np.array(indptr)),
        shape=(n_cells, n_genes),
    )
    adata = AnnData(X)
    adata.obs["leiden"] = z.astype(str)
    return adata


def main():
    sizes = [int(a) for a in sys.argv[1:]] or [10000]
    # warm up numba compilation
    decontx(make_adata(200, 500, 3), copy=True, verbose=False, max_iter=3)
    for n in sizes:
        adata = make_adata(n)
        t = time.perf_counter()
        decontx(adata, copy=False, verbose=False)
        dt = time.perf_counter() - t
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
        print(f"n_cells={n} nnz={adata.X.nnz} time={dt:.2f}s peak_rss={rss:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
