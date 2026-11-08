"""Plan step 3: symmetric features x = [p, (a+b)/2, |a-b|]  (unnormalized; normalization is step 4).
Run: python src/features.py  -> data/pilot_ids.json, data/pilot_emb.npz (P,A,B), data/pilot_features.npy, data/features_report.json
"""
import json, random, numpy as np

def sym_features(P, A, B):
    """P,A,B: (n,h) unit-normalized prompt / chosen / rejected embeddings -> (n,3h) float32."""
    return np.concatenate([P, (A + B) / 2, np.abs(A - B)], axis=1).astype(np.float32)

def checks(P, A, B, X, eps=1e-6):
    n, h = P.shape
    assert X.shape == (n, 3 * h) and X.dtype == np.float32
    assert np.isfinite(X).all(), "non-finite feature"
    assert np.array_equal(X, sym_features(P, B, A)), "swap changes features"  # exact: + and |.| are commutative in fp
    q = np.linalg.norm(X[:, 2 * h:], axis=1)
    return dict(n=n, dim=3 * h, swap_invariant=True, finite=True,
                diff_norm_min=float(q.min()), diff_norm_mean=float(q.mean()),
                n_diff_norm_lt_eps=int((q < eps).sum()), n_diff_norm_lt_1e2=int((q < 1e-2).sum()))

if __name__ == "__main__":
    from embed import load, encode_context, encode_response
    rs = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
    rs = random.Random(0).sample(rs, 2000)  # pilot pool: 2000 train pairs, seed 0
    json.dump([r["id"] for r in rs], open("data/pilot_ids.json", "w"))
    m = load("minilm")
    P = encode_context(m, [r["context"] for r in rs])
    A = encode_response(m, [r["chosen"] for r in rs]); B = encode_response(m, [r["rejected"] for r in rs])
    X = sym_features(P, A, B)
    rep = checks(P, A, B, X)
    np.savez("data/pilot_emb.npz", P=P, A=A, B=B); np.save("data/pilot_features.npy", X)
    rep["groups"] = len({r["group"] for r in rs}); rep["storage_mb"] = X.nbytes / 2**20
    json.dump(rep, open("data/features_report.json", "w"), indent=2); print(json.dumps(rep, indent=1))
