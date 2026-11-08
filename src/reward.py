"""Plan step 7 pilot: DistilBERT + scalar head, pairwise loss -softplus(-(r+ - r-)).
Usage: python src/reward.py overfit | python src/reward.py compare
Writes data/reward_pilot.json"""
import json, random, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoModel, AutoTokenizer
from normalize import feat_final
from kcenter import kcenter

NAME, REV = "distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"
L, R = 256, 128          # total tokens incl. [CLS],[SEP],[SEP]; response capped at R tokens, context gets the rest (left-truncated)
LR, EFF_BS, MICRO = 2e-5, 32, 8   # micro = pairs per forward (2*micro sequences)
tok = AutoTokenizer.from_pretrained(NAME, revision=REV); dev = "cuda"

def encode(ctx, resp):
    r = tok(resp, add_special_tokens=False)["input_ids"][:R]
    c = tok(ctx, add_special_tokens=False)["input_ids"]; c = c[max(0, len(c) - (L - 3 - len(r))):]
    return [tok.cls_token_id] + c + [tok.sep_token_id] + r + [tok.sep_token_id]

def batchify(seqs):
    n = max(map(len, seqs)); ids = torch.zeros(len(seqs), n, dtype=torch.long); m = torch.zeros_like(ids)
    for i, s in enumerate(seqs): ids[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return ids.to(dev), m.to(dev)

class RM(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = AutoModel.from_pretrained(NAME, revision=REV); s.head = nn.Linear(768, 1)
    def forward(s, ids, m): return s.head(s.enc(input_ids=ids, attention_mask=m).last_hidden_state[:, 0]).squeeze(-1)

def scores(model, pairs):  # pairs: list of (chosen_ids, rejected_ids) -> (r+, r-)
    ids, m = batchify([p[0] for p in pairs] + [p[1] for p in pairs]); r = model(ids, m); n = len(pairs)
    return r[:n], r[n:]

def prep(rs): return [(encode(r["context"], r["chosen"]), encode(r["context"], r["rejected"])) for r in rs]

@torch.no_grad()
def evaluate(model, data):
    model.eval(); rp, rm = [], []
    for i in range(0, len(data), 16):
        a, b = scores(model, data[i:i + 16]); rp.append(a.float().cpu()); rm.append(b.float().cpu())
    rp, rm = torch.cat(rp), torch.cat(rm)
    return dict(acc=float((rp > rm).float().mean()), loss=float(Fn.softplus(-(rp - rm)).mean()))

def train(train_data, val_data, epochs, seed=0, log=None):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)   # same seed => same head init for every method
    model = RM().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = random.Random(seed); hist = [dict(epoch=0, **evaluate(model, val_data))] if val_data else []
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); steps = 0
    for ep in range(1, epochs + 1):
        model.train(); idx = list(range(len(train_data))); g.shuffle(idx); losses = []
        for s in range(0, len(idx), EFF_BS):
            batch = [train_data[i] for i in idx[s:s + EFF_BS]]; opt.zero_grad()
            for j in range(0, len(batch), MICRO):
                mb = batch[j:j + MICRO]; a, b = scores(model, mb)
                loss = Fn.softplus(-(a - b)).sum() / len(batch); loss.backward(); losses.append(loss.item() * len(batch) / len(mb))
            opt.step(); steps += 1
            if steps % 10 == 0: print(f'  step {steps} loss {np.mean(losses[-4:]):.3f} t={time.time()-t0:.0f}s', flush=True)
        h = dict(epoch=ep, train_loss=float(np.mean(losses)), steps=steps, **(evaluate(model, val_data) if val_data else {}))
        hist.append(h); print(h, flush=True)
    return dict(history=hist, train_sec=time.time() - t0, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20)

if __name__ == "__main__":
    mode = sys.argv[1]; rows = {}
    for f in ("train", "val"):
        for l in open(f"data/{f}.jsonl", encoding="utf-8"): r = json.loads(l); rows[r["id"]] = r
    pool_ids = json.load(open("data/pilot_ids.json")); pool = [rows[i] for i in pool_ids]
    val = [r for r in rows.values() if r["split"] == "val"]; val = random.Random(0).sample(val, 1000)
    out = {}
    if mode == "overfit":   # does the loss go down on a tiny batch?
        res = train(prep(pool[:16]), None, epochs=1)  # warm-up for memory only
        torch.manual_seed(0); model = RM().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR); d = prep(pool[:16]); ls = []
        for _ in range(30):
            model.train(); opt.zero_grad(); a, b = scores(model, d[:8]); l = Fn.softplus(-(a - b)).mean(); l.backward(); opt.step(); ls.append(l.item())
        out["overfit_16"] = dict(first=ls[0], last=ls[-1], decreased=ls[-1] < ls[0] * 0.5); print(out)
    elif mode == "retime":  # step speed only: 12 optimizer steps on pilot pairs, GPU clock logged
        import subprocess
        clk = lambda: subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        d = prep(pool[:12 * EFF_BS]); torch.manual_seed(0); model = RM().to(dev); opt = torch.optim.AdamW(model.parameters(), lr=LR); model.train()
        torch.cuda.reset_peak_memory_stats(); ts = []
        for s_ in range(12):
            t = time.time(); opt.zero_grad(); b = d[s_ * EFF_BS:(s_ + 1) * EFF_BS]
            for j in range(0, len(b), MICRO):
                a, c = scores(model, b[j:j + MICRO]); (Fn.softplus(-(a - c)).sum() / len(b)).backward()
            opt.step(); torch.cuda.synchronize(); ts.append(time.time() - t)
            if s_ in (2, 11): print("clock/power:", clk(), flush=True)
        out = dict(step_sec=ts, median_step_sec=float(np.median(ts[2:])), peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20)
        print(out)
    elif mode == "signal":  # is there ANY learning signal with more data? all 2000 pilot pairs, 2 epochs
        out["all2000"] = train(prep(pool), prep(val), epochs=2)
    else:
        z = np.load("data/pilot_emb.npz"); F = feat_final(z["P"], z["A"], z["B"])
        sel = {"random200": random.Random(0).sample(range(2000), 200), "kcenter200": kcenter(F, 200)}
        vd = prep(val); out["val_n"] = len(vd)
        out["untrained_val"] = None
        for name, ix in sel.items():
            print("==", name); out[name] = train(prep([pool[i] for i in ix]), vd, epochs=3)
    json.dump(out, open(f"data/reward_pilot_{mode}.json", "w"), indent=2)
