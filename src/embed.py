"""Encoding rules (plan step 2): contexts keep the RECENT turns (left-truncate), responses keep the start."""
import numpy as np
from sentence_transformers import SentenceTransformer

ENCODERS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge-small": "BAAI/bge-small-en-v1.5"}

def load(name, device="cuda"):
    return SentenceTransformer(ENCODERS[name], device=device)

def _enc(m, texts, side, bs=64):
    m.tokenizer.truncation_side = side  # HF truncates before adding [CLS]/[SEP], so specials are kept
    return m.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)

def encode_context(m, texts): return _enc(m, texts, "left")
def encode_response(m, texts): return _enc(m, texts, "right")

if __name__ == "__main__":  # self-check: left truncation keeps the END, specials survive
    m = load("minilm")
    m.tokenizer.truncation_side = "left"
    ids = m.tokenizer("word " * 400 + "THEEND", truncation=True, max_length=m.max_seq_length)["input_ids"]
    toks = m.tokenizer.convert_ids_to_tokens(ids)
    assert len(ids) == 256 and toks[0] == "[CLS]" and toks[-1] == "[SEP]" and toks[-3:-1] == ["thee", "##nd"], toks[-4:]
    print("ok", toks[:2], toks[-3:])
