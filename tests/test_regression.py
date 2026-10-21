"""Fixed-seed regression test guarding numerical behavior during optimization.

Regenerate the reference with:  python tests/test_regression.py
"""

import os
import numpy as np
from scipy.sparse import csr_matrix
from anndata import AnnData
from decontx import decontx

REF = os.path.join(os.path.dirname(__file__), "regression_ref.npz")


def _make(n_cells=300, n_genes=120, n_clusters=4, n_batches=2, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, n_clusters, size=n_cells)
    profiles = rng.gamma(0.5, 1.0, size=(n_clusters, n_genes))
    profiles /= profiles.sum(axis=1, keepdims=True)
    depth = rng.integers(200, 800, size=n_cells)
    X = np.stack([rng.multinomial(d, profiles[k]) for d, k in zip(depth, z)])
    adata = AnnData(csr_matrix(X.astype(np.float32)))
    adata.obs["leiden"] = z.astype(str)
    adata.obs["batch"] = rng.integers(0, n_batches, size=n_cells).astype(str)
    return adata


def _run():
    out = {}
    for name, kw in {
        "plain": {},
        "nodelta": {"estimate_delta": False},
        "batch": {"batch_key": "batch"},
    }.items():
        res = decontx(_make(), copy=True, verbose=False, max_iter=60, convergence=0.0, **kw)
        out[name + "_contam"] = res.obs["decontX_contamination"].values.astype(np.float64)
        out[name + "_counts"] = res.layers["decontX_counts"].toarray()
    return out


def test_regression():
    ref = np.load(REF)
    got = _run()
    for k in ref.files:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-6, atol=1e-8, err_msg=k)


if __name__ == "__main__":
    np.savez_compressed(REF, **_run())
    print("wrote", REF)
