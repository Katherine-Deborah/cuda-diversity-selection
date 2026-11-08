"""Plan step 5: time the NumPy reference k-center; fit O(N*k*d) and extrapolate. -> data/kcenter_timing.json"""
import json, time, numpy as np
from normalize import feat_final
from kcenter import kcenter
z = np.load("data/pilot_emb.npz"); F = feat_final(z["P"], z["A"], z["B"]); d = F.shape[1]
rows = []
for N, k in [(2000, 200), (2000, 50), (4000, 400), (8000, 800)]:  # N>2000 = tiled+noised pilot rows, timing only
    X = F if N == 2000 else (np.tile(F, (N // 2000, 1)) + np.random.default_rng(0).normal(0, 1e-3, (N, d))).astype(np.float32)
    t = time.perf_counter(); kcenter(X, k); dt = time.perf_counter() - t
    rows.append(dict(N=N, k=k, d=d, sec=dt, ns_per_Nkd=dt / (N * k * d) * 1e9)); print(rows[-1])
c = float(np.median([r["ns_per_Nkd"] for r in rows])) * 1e-9
proj = {f"N={N},k={N//10}": c * N * (N // 10) * d for N in (20000, 50000, 133535)}
json.dump(dict(runs=rows, projected_sec_numpy_cpu=proj), open("data/kcenter_timing.json", "w"), indent=2); print(proj)
