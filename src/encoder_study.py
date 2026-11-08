"""Plan step 1: token lengths/truncation + embed time/throughput/peak GPU mem on 1000 train pairs.
Writes data/encoder_study.json"""
import json, random, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
rs = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
rs = random.Random(0).sample(rs, 1000)
texts = {"context": [r["context"] for r in rs],
         "response": [r["chosen"] for r in rs] + [r["rejected"] for r in rs]}
out = {}
for name, hf in [("minilm", "sentence-transformers/all-MiniLM-L6-v2"), ("bge-small", "BAAI/bge-small-en-v1.5")]:
    m = SentenceTransformer(hf, device="cuda")
    m.half() if False else None  # float32 per plan
    lim = m.max_seq_length
    res = {"max_seq_length": lim, "dim": m.get_sentence_embedding_dimension()}
    for k, t in texts.items():
        lens = np.array([len(x) for x in m.tokenizer(t, add_special_tokens=True)["input_ids"]])
        res[k] = dict(tok_mean=float(lens.mean()), tok_p95=float(np.percentile(lens, 95)),
                      tok_max=int(lens.max()), trunc_rate=float((lens > lim).mean()))
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    allt = texts["context"] + texts["response"]
    e = m.encode(allt, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    torch.cuda.synchronize(); dt = time.time() - t0
    res.update(embed_s=dt, texts_per_s=len(allt) / dt, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
               finite=bool(np.isfinite(e).all()))
    out[name] = res; print(name, json.dumps(res, indent=1))
    del m; torch.cuda.empty_cache()
json.dump(out, open("data/encoder_study.json", "w"), indent=2)
