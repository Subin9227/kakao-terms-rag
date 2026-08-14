"""Dense retrieval + RRF fusion over the article corpus.

Needs sentence-transformers (venv locally, preinstalled on Colab). BM25 in
retrieval.py stays stdlib-only so the lexical half never depends on this.
"""
import re

from retrieval import Retriever, load_corpus

# e5 models were trained with these prefixes; using them wrong costs real accuracy.
_E5_PREFIX = re.compile(r"(^|/)(multilingual-)?e5-", re.I)


def article_text(article):
    return "{} 제{}조({})\n{}".format(
        article["doc"], article["article_no"], article.get("title", ""), article.get("text", "")
    )


class DenseIndex:
    def __init__(self, corpus, model_name, device=None, batch_size=16):
        from sentence_transformers import SentenceTransformer

        self.corpus = corpus
        self.model_name = model_name
        self.needs_prefix = bool(_E5_PREFIX.search(model_name))
        self.model = SentenceTransformer(model_name, device=device)
        # 제출 셀의 절단 길이와 맞춘다. 여기서만 8192를 쓰면 측정치가 전이되지 않는다.
        self.model.max_seq_length = 2048
        passages = [article_text(a) for a in corpus]
        if self.needs_prefix:
            passages = ["passage: " + p for p in passages]
        self.embeddings = self.model.encode(
            passages, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    def search(self, query, top_k=4):
        if self.needs_prefix:
            query = "query: " + query
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ q
        order = scores.argsort()[::-1][:top_k]
        return [
            (self.corpus[i]["doc"], self.corpus[i]["article_no"], float(scores[i])) for i in order
        ]


def rrf(rankings, top_k=4, k=60, weights=None):
    """Reciprocal rank fusion. Rank-only, so BM25 and cosine scales never need calibrating."""
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc, article_no, _score) in enumerate(ranking, 1):
            fused[(doc, article_no)] = fused.get((doc, article_no), 0.0) + weight / (k + rank)
    ranked = sorted(fused.items(), key=lambda item: -item[1])
    return [(doc, no, score) for (doc, no), score in ranked[:top_k]]


class Reranker:
    """교차 인코더로 dense 후보를 다시 매긴다. MRR 은 1순위만 보상하므로 여기가 표적이다."""

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device=None, max_length=1024):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16).to(device or "cpu").eval()
        self.device = device or "cpu"
        self.max_length = max_length

    def rerank(self, query, corpus_by_key, candidates, top_k=4, batch_size=8):
        pairs = [(query, article_text(corpus_by_key[(d, n)])) for d, n, _ in candidates]
        scores = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            enc = self.tok([p[0] for p in chunk], [p[1] for p in chunk], padding=True,
                           truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                out = self.model(**enc).logits.view(-1).float()
            scores.extend(out.tolist())
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [(candidates[i][0], candidates[i][1], scores[i]) for i in order[:top_k]]


class HybridRetriever:
    def __init__(self, corpus=None, model_name="nlpai-lab/KURE-v1", device=None,
                 granularity="article", tokenizer="word", pool=12):
        self.corpus = corpus if corpus is not None else load_corpus()
        self.lexical = Retriever(self.corpus)
        self.dense = DenseIndex(self.corpus, model_name, device=device)
        self.granularity = granularity
        self.tokenizer = tokenizer
        self.pool = pool  # candidates per retriever before fusion

    def search(self, query, top_k=4, mode="rrf", dense_weight=1.0):
        if mode == "bm25":
            return self.lexical.search(query, top_k, self.granularity, self.tokenizer)
        if mode == "dense":
            return self.dense.search(query, top_k)
        return rrf(
            [
                self.lexical.search(query, self.pool, self.granularity, self.tokenizer),
                self.dense.search(query, self.pool),
            ],
            top_k=top_k, weights=[1.0, dense_weight],
        )
