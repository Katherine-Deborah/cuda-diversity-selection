import sys; sys.path.insert(0, "src")
import numpy as np
from features import sym_features, checks

def test_swap_and_shape():
    r = np.random.default_rng(0)
    u = lambda: (lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32))(r.normal(size=(5, 8)))
    P, A, B = u(), u(), u()
    X = sym_features(P, A, B)
    assert X.shape == (5, 24) and np.array_equal(X, sym_features(P, B, A))
    assert checks(P, A, B, X)["n_diff_norm_lt_eps"] == 0
    assert checks(P, A, A, sym_features(P, A, A))["n_diff_norm_lt_eps"] == 5  # identical responses flagged
