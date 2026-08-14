import json
import math
import re
from collections import Counter


K1 = 1.2
B = 0.75
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?=[①②③④⑤⑥⑦⑧⑨⑩])|(?m:(?=^\s*\d+\s*\.\s*))"
)


def load_corpus(path="corpus.json"):
    with open(path, encoding="utf-8") as source:
        corpus = json.load(source)
    if not isinstance(corpus, list):
        raise ValueError("corpus must be a JSON list")
    return corpus


def word_tokens(text):
    return _WORD_RE.findall(text.lower())


def ngram_tokens(text):
    text = re.sub(r"\s+", "", text.lower())
    return [text[i : i + n] for n in (2, 3) for i in range(len(text) - n + 1)]


def tokenize(text, tokenizer="word"):
    if tokenizer == "word":
        return word_tokens(text)
    if tokenizer == "ngram":
        return ngram_tokens(text)
    if tokenizer == "hybrid":
        return word_tokens(text) + ngram_tokens(text)
    raise ValueError("tokenizer must be one of: word, ngram, hybrid")


def build_units(corpus, granularity="article"):
    if granularity not in ("article", "clause"):
        raise ValueError("granularity must be article or clause")

    units = []
    for article in corpus:
        parent = (article["doc"], article["article_no"])
        title = article.get("title", "")
        if granularity == "article":
            units.append({
                "doc": parent[0],
                "article_no": parent[1],
                "title": title,
                "text": title + "\n" + article.get("text", ""),
                "parent": parent,
            })
            continue

        parts = [part.strip() for part in _CLAUSE_SPLIT_RE.split(article.get("text", "")) if part.strip()]
        merged = []
        for part in parts or [""]:
            if len(part) < 40 and merged:
                merged[-1] += "\n" + part
            else:
                merged.append(part)
        for unit_no, part in enumerate(merged, 1):
            units.append({
                "doc": parent[0],
                "article_no": parent[1],
                "title": title,
                "text": title + "\n" + part,
                "parent": parent,
                "unit_no": unit_no,
            })
    return units


class BM25:
    def __init__(self, units, tokenizer, k1=K1, b=B):
        self.units = units
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents = [Counter(tokenize(unit["text"], tokenizer)) for unit in units]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0
        document_frequency = Counter()
        for document in self.documents:
            document_frequency.update(document)
        self.document_frequency = document_frequency

    def scores(self, query):
        query_terms = Counter(tokenize(query, self.tokenizer))
        if not query_terms:
            return [0.0] * len(self.documents)
        total_documents = len(self.documents)
        scores = []
        for document, length in zip(self.documents, self.lengths):
            score = 0.0
            length_ratio = length / self.average_length if self.average_length else 0
            for term, query_frequency in query_terms.items():
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += idf * (frequency * (self.k1 + 1) / denominator) * min(query_frequency, 1)
            scores.append(score)
        return scores


class Retriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self._indexes = {}
        self._article_order = {
            (article["doc"], article["article_no"]): index
            for index, article in enumerate(corpus)
        }

    def search(self, query, top_k=4, granularity="article", tokenizer="word"):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        key = (granularity, tokenizer)
        if key not in self._indexes:
            self._indexes[key] = BM25(build_units(self.corpus, granularity), tokenizer)
        index = self._indexes[key]
        article_scores = {}
        for unit, score in zip(index.units, index.scores(query)):
            parent = unit["parent"]
            article_scores[parent] = max(score, article_scores.get(parent, 0.0))
        ranked = sorted(
            article_scores.items(),
            key=lambda item: (-item[1], self._article_order[item[0]]),
        )
        return [(doc, article_no, score) for (doc, article_no), score in ranked[:top_k]]


def search(query, top_k=4, granularity="article", tokenizer="word", corpus=None):
    return Retriever(load_corpus() if corpus is None else corpus).search(
        query, top_k, granularity, tokenizer
    )
