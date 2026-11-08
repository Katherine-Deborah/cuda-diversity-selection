"""Dump 20 parsed examples for manual review -> data/inspect_20.md"""
import json, random
rs = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rs)
with open("data/inspect_20.md", "w", encoding="utf-8") as f:
    for r in rs[:20]:
        f.write(f"## {r['id']} (group {r['group']})\n**CONTEXT**\n{r['context'][-600:]}\n\n**CHOSEN**: {r['chosen'][:300]}\n\n**REJECTED**: {r['rejected'][:300]}\n\n---\n")
# invariants over the whole train set
bad = [r["id"] for r in rs if r["chosen"] in r["context"][-len(r["chosen"]):] and len(r["chosen"]) > 20]
print("chosen text duplicated at end of context:", len(bad))
print("ctx ends with Human turn:", sum("\n\nAssistant:" not in r["context"].rsplit("\n\nHuman:", 1)[-1] for r in rs) / len(rs))
