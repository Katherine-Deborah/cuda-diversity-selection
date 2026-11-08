"""Plan step 1 (qualitative + sanity): nearest neighbours + context->own-response retrieval.
Writes data/nn_check.md and data/nn_check.json. Same 1000-pair sample as encoder_study.py."""
import json, random, numpy as np, torch
from sentence_transformers import SentenceTransformer
rs = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
rs = random.Random(0).sample(rs, 1000)
ctx = [r["context"] for r in rs]; ch = [r["chosen"] for r in rs]; rj = [r["rejected"] for r in rs]
last_h = [c.rsplit("\n\nHuman:", 1)[-1].strip() for c in ctx]  # final user request, for display
picks = random.Random(1).sample(range(1000), 6)
md, out = [], {}
for name, hf in [("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("bge-small", "BAAI/bge-small-en-v1.5")]:
    m = SentenceTransformer(hf, device="cuda")
    enc = lambda t: m.encode(t, batch_size=64, normalize_embeddings=True)
    res = {}
    for side in ("right",):  # default truncation (cut the end) for the retrieval metrics
        m.tokenizer.truncation_side = side
        C = enc(ctx)
        if side == "right":
            A, B = enc(ch), enc(rj)
            S = C @ C.T; np.fill_diagonal(S, -1)
            nn = S.argmax(1)
            res["ctx_nn_sim_mean"] = float(S.max(1).mean())
            md.append(f"# {name}\n")
            for i in picks:
                md.append(f"**Q**: {last_h[i][:150]}\n  - NN (sim {S[i, nn[i]]:.2f}): {last_h[nn[i]][:150]}\n")
            R = C @ np.vstack([A, B]).T  # context vs all 2000 responses
            own = np.stack([np.arange(1000), np.arange(1000) + 1000], 1)
            rank = np.array([(R[i] > R[i, own[i]].max()).sum() for i in range(1000)])  # 0 = own response is top
            res["own_resp_top1"] = float((rank == 0).mean()); res["own_resp_top10"] = float((rank < 10).mean())
            # response-response: is a pair's chosen closer to its rejected than to random responses?
            ab = (A * B).sum(1); rnd = (A * np.roll(B, 1, 0)).sum(1)
            res["chosen_vs_rejected_sim"] = float(ab.mean()); res["chosen_vs_random_sim"] = float(rnd.mean())
    # left-vs-right difference on contexts that actually exceed the limit
    lens = np.array([len(x) for x in m.tokenizer(ctx)["input_ids"]]); long = lens > m.max_seq_length
    m.tokenizer.truncation_side = "right"; Cr = enc(ctx); m.tokenizer.truncation_side = "left"; Cl = enc(ctx)
    res["n_long_ctx"] = int(long.sum())
    res["long_ctx_cos_left_vs_right"] = float((Cr[long] * Cl[long]).sum(1).mean()) if long.any() else None
    out[name] = res; print(name, json.dumps(res, indent=1)); del m; torch.cuda.empty_cache()
open("data/nn_check.md", "w", encoding="utf-8").write("\n".join(md))
json.dump(out, open("data/nn_check.json", "w"), indent=2)
