"""Re-measure retrieval + truncation effect with the final rules. -> data/remeasure.json"""
import json, random, numpy as np, torch
from embed import load, encode_context, encode_response, _enc
rs = random.Random(0).sample([json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")], 1000)
ctx = [r["context"] for r in rs]; ch = [r["chosen"] for r in rs]; rj = [r["rejected"] for r in rs]
out = {}
for name in ("minilm", "bge-small"):
    m = load(name); A, B = encode_response(m, ch), encode_response(m, rj)
    lens = np.array([len(x) for x in m.tokenizer(ctx)["input_ids"]]); long = lens > m.max_seq_length
    res = {"n_long_ctx": int(long.sum())}
    for side in ("right", "left"):
        C = _enc(m, ctx, side); R = C @ np.vstack([A, B]).T
        rank = np.array([(R[i] > max(R[i, i], R[i, i + 1000])).sum() for i in range(1000)])
        res[side] = dict(top1_all=float((rank == 0).mean()), top10_all=float((rank < 10).mean()),
                         top1_long=float((rank[long] == 0).mean()), top10_long=float((rank[long] < 10).mean()))
    out[name] = res; print(name, json.dumps(res, indent=1)); del m; torch.cuda.empty_cache()
json.dump(out, open("data/remeasure.json", "w"), indent=2)
