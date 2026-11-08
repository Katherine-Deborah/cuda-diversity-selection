import sys; sys.path.insert(0, "src")
import numpy as np
from normalize import feat_final, feat_comp, unit
from kcenter import kcenter

def _u(r, n=6, h=8):
    x = r.normal(size=(n, h)); return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

def test_norm_and_swap_and_zero():
    r = np.random.default_rng(0); P, A, B = _u(r), _u(r), _u(r)
    for f in (feat_final, feat_comp):
        X = f(P, A, B)
        assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5) and np.array_equal(X, f(P, B, A))
        Z = f(P, A, A)  # identical responses -> q = 0 must stay 0, no NaN
        assert np.isfinite(Z).all() and not Z[:, 16:].any()
    assert not unit(np.zeros((1, 4), np.float32)).any()

def test_kcenter_ties_and_distinct():
    X = np.array([[0, 0], [1, 0], [1, 0], [0, 5]], np.float32)
    assert kcenter(X, 3) == [0, 3, 1]  # tie between rows 1 and 2 -> smallest index
