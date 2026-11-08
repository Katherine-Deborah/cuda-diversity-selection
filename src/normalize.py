"""Plan step 4: two normalization candidates + analysis on the pilot embeddings. -> data/normalize_report.json"""
import json, numpy as np
from features import sym_features
from kcenter import kcenter
EPS = 1e-6  # x / max(||x||, EPS): a (near-)zero vector stays ~zero instead of blowing up

def unit(X): return (X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), EPS)).astype(np.float32)

def feat_final(P, A, B):  # F: concat then unit-normalize
    return unit(sym_features(P, A, B))

def feat_comp(P, A, B):  # C: unit-normalize p, m, q separately, concat, unit-normalize (each part ends with norm 1/sqrt3)
    h = P.shape[1]; X = sym_features(P, A, B)
    return unit(np.concatenate([unit(X[:, :h]), unit(X[:, h:2 * h]), unit(X[:, 2 * h:])], 1))

if __name__ == "__main__":
    z = np.load("data/pilot_emb.npz"); P, A, B = z["P"], z["A"], z["B"]; h = P.shape[1]
    X = sym_features(P, A, B); rep = {}
    nrm = {n: np.linalg.norm(X[:, i * h:(i + 1) * h], axis=1) for i, n in enumerate("pmq")}
    rep["component_norms"] = {n: dict(min=float(v.min()), p5=float(np.percentile(v, 5)), mean=float(v.mean()),
                              p95=float(np.percentile(v, 95)), max=float(v.max())) for n, v in nrm.items()}
    rep["n_q_norm_lt_1e-6"] = int((nrm["q"] < 1e-6).sum()); rep["n_q_norm_lt_0.1"] = int((nrm["q"] < 0.1).sum())
    rep["n_m_norm_lt_0.1"] = int((nrm["m"] < 0.1).sum())
    F, C = feat_final(P, A, B), feat_comp(P, A, B)
    assert np.isfinite(F).all() and np.isfinite(C).all()
    rep["unit_norm_ok"] = {n: bool(np.allclose(np.linalg.norm(M, axis=1), 1, atol=1e-5)) for n, M in (("final", F), ("comp", C))}
    # identity: squared Euclidean == 2 - 2cos for unit vectors
    i, j = np.random.default_rng(0).integers(0, len(F), (2, 500))
    rep["sqdist_eq_2-2cos_maxerr"] = float(np.abs(((F[i] - F[j]) ** 2).sum(1) - (2 - 2 * (F[i] * F[j]).sum(1))).max())
    # how much of the pairwise squared distance does each component carry? (share of mean sq distance)
    sh = {}
    for n, M in (("final", F), ("comp", C)):
        d = ((M[i] - M[j]) ** 2); tot = d.sum(1).mean()
        sh[n] = {c: float(d[:, k * h:(k + 1) * h].sum(1).mean() / tot) for k, c in enumerate("pmq")}
    rep["sqdist_share"] = sh
    # does the choice change the selection? k-center (k=200) overlap + kNN(10) overlap
    sF, sC = set(kcenter(F, 200)), set(kcenter(C, 200)); rep["kcenter_overlap_k200"] = len(sF & sC) / 200
    G1, G2 = F @ F.T, C @ C.T; np.fill_diagonal(G1, -9); np.fill_diagonal(G2, -9)
    k1, k2 = np.argsort(-G1, 1)[:, :10], np.argsort(-G2, 1)[:, :10]
    rep["knn10_overlap"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(k1, k2)]))
    # baselines for the k-center overlap: same features, different start row; and chance level
    rep["kcenter_overlap_same_features_other_start"] = len(sF & set(kcenter(F, 200, start=1))) / 200
    rep["kcenter_overlap_chance"] = 200 / len(F)
    # the amplification risk: effect of C on the lowest-|q| rows
    low = np.argsort(nrm["q"])[:5]; rep["lowest_q_norms"] = [float(nrm["q"][t]) for t in low]
    rep["zero_q_row_after_comp_has_q_part"] = float(np.linalg.norm(C[low[0], 2 * h:]))
    json.dump(rep, open("data/normalize_report.json", "w"), indent=2); print(json.dumps(rep, indent=1))
