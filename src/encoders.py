"""MedCPT retrieval models (NCBI): article encoder, query encoder, cross-encoder reranker.
Trained on PubMed search logs, so they suit medical text. Same models for every LLM -> fair comparison."""
import time

import numpy as np


def _load(name, kind, device):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    cls = AutoModelForSequenceClassification if kind == "cross" else AutoModel
    model = cls.from_pretrained(name).to(device).eval()
    if device == "cuda":
        model = model.half()
    return tok, model, torch


def _batched(n_items, lengths, batch_size):
    """Longest items first, so memory problems show up in the first batch."""
    order = sorted(range(n_items), key=lambda i: -lengths[i])
    for s in range(0, n_items, batch_size):
        yield order[s:s + batch_size]


class Encoder:
    def __init__(self, name, kind, device, max_length, batch_size):
        self.tok, self.model, self.torch = _load(name, kind, device)
        self.device, self.max_length, self.batch_size, self.name = device, max_length, batch_size, name

    def _run(self, first, second, lengths, fn, label):
        n = len(first)
        result = [None] * n
        t0, done = time.time(), 0
        for idx in _batched(n, lengths, self.batch_size):
            a = [first[i] for i in idx]
            b = [second[i] for i in idx] if second is not None else None
            enc = self.tok(a, b, truncation=True, padding=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                vals = fn(enc)
            for j, i in enumerate(idx):
                result[i] = vals[j]
            done += len(idx)
            if done % (self.batch_size * 100) < self.batch_size or done == n:
                print(f"  {label}: {done}/{n} ({time.time() - t0:.0f}s)", flush=True)
        return result

    def embed(self, first, second=None, label="encode"):
        lengths = [len(x) + (len(second[i]) if second else 0) for i, x in enumerate(first)]
        fn = lambda enc: self.model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
        return np.asarray(self._run(first, second, lengths, fn, label), dtype=np.float32)

    def score(self, queries, articles, label="rerank"):
        lengths = [len(q) + len(a) for q, a in zip(queries, articles)]
        fn = lambda enc: self.model(**enc).logits[:, 0].float().cpu().numpy()
        return np.asarray(self._run(queries, articles, lengths, fn, label), dtype=np.float32)


def article_encoder(cfg, device):
    c = cfg["index"]
    return Encoder(c["article_encoder"], "bi", device, c["article_max_length"], c["batch_size"])


def query_encoder(cfg, device):
    c = cfg["index"]
    return Encoder(c["query_encoder"], "bi", device, c["query_max_length"], c["batch_size"])


def cross_encoder(cfg, device):
    c = cfg["rerank"]
    return Encoder(c["cross_encoder"], "cross", device, c["max_length"], c["batch_size"])
