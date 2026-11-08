"""Parse HH-RLHF into (context, chosen, rejected) records + prompt-group split.
Run: python src/parse.py   -> data/{train,val,test}.jsonl, data/parse_report.json
"""
import hashlib, json, re, sys
from collections import Counter
from pathlib import Path

SEP = "\n\nAssistant:"
SPLIT_SEED = 0
FRAC = (0.8, 0.1, 0.1)  # train/val/test by prompt group


def parse_pair(chosen: str, rejected: str):
    """Return (context, chosen_resp, rejected_resp) or (None, reason)."""
    ci, ri = chosen.rfind(SEP), rejected.rfind(SEP)
    if ci < 0 or ri < 0:
        return None, "no_assistant_turn"
    ctx, a = chosen[:ci], chosen[ci + len(SEP):].strip()
    ctx_r, b = rejected[:ri], rejected[ri + len(SEP):].strip()
    if ctx != ctx_r:
        return None, "context_mismatch"
    if not a or not b:
        return None, "empty_response"
    if a == b:
        return None, "identical_responses"
    if SEP in a or SEP in b or "\n\nHuman:" in a or "\n\nHuman:" in b:
        return None, "extra_turn_in_response"
    last = re.sub(r"^Human:\s*", "", ctx.rsplit("\n\nHuman:", 1)[-1].strip())
    for t in (a, b):  # degenerate: assistant just echoes the user's last message (or starts with "Human:")
        if t.startswith("Human:") or re.sub(r"^Human:\s*", "", t) == last:
            return None, "echoes_user"
    return (ctx.strip(), a, b), None


def group_id(ctx: str) -> str:
    # prompt group = first human turn (all continuations of one conversation share it)
    first = re.split(r"\n\nAssistant:", ctx.strip(), maxsplit=1)[0]
    return hashlib.sha1(first.encode()).hexdigest()[:16]


def split_of(gid: str) -> str:
    u = int(hashlib.sha1(f"{SPLIT_SEED}:{gid}".encode()).hexdigest()[:8], 16) / 16**8
    return "train" if u < FRAC[0] else "val" if u < FRAC[0] + FRAC[1] else "test"


def build():
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf")
    rows, drops = [], Counter()
    for src in ("train", "test"):
        for i, ex in enumerate(ds[src]):
            r, why = parse_pair(ex["chosen"], ex["rejected"])
            if r is None:
                drops[why] += 1
                continue
            ctx, a, b = r
            g = group_id(ctx)
            rows.append(dict(id=f"{src}-{i}", group=g, split=split_of(g), source=src,
                             context=ctx, chosen=a, rejected=b))
    return rows, drops


if __name__ == "__main__":
    rows, drops = build()
    out = Path(__file__).resolve().parent.parent / "data"
    by = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    for s, rs in by.items():
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
    groups = {s: {r["group"] for r in rs} for s, rs in by.items()}
    overlap = {f"{a}&{b}": len(groups[a] & groups[b]) for a, b in
               [("train", "val"), ("train", "test"), ("val", "test")]}
    rep = dict(counts={s: len(rs) for s, rs in by.items()},
               groups={s: len(g) for s, g in groups.items()},
               dropped=dict(drops), cross_split_group_overlap=overlap)
    (out / "parse_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    assert not any(overlap.values()), "prompt groups leak across splits"
