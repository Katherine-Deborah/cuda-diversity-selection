"""Reference farthest-first (greedy k-center), squared Euclidean, float32. Start = row 0; ties -> smallest index."""
import numpy as np

def kcenter(X, k, start=0):
    X = np.ascontiguousarray(X, dtype=np.float32)
    sel = [start]; d = ((X - X[start]) ** 2).sum(1); d[start] = -1  # -1 masks selected rows
    for _ in range(k - 1):
        j = int(d.argmax())  # argmax returns the first max -> smallest index on ties
        sel.append(j); d = np.minimum(d, ((X - X[j]) ** 2).sum(1)); d[sel] = -1
    return sel
