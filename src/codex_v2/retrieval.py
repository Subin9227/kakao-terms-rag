# -*- coding: utf-8 -*-
"""조 단위 BM25 검색기. 1번 셀에 그대로 들어갈 코드이므로 외부 패키지를 쓰지 않는다.

토크나이저 구성 요소를 켜고 끌 수 있게 해 두었다. 어느 조합이 실제로 MRR 을
올리는지는 eval_retrieval.py 가 공개 10문항 + 검증 18문항으로 측정한다.
"""

import math
import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_HANGUL_RE = re.compile(r"[가-힣]+")

# 긴 조사부터 검사해야 '에게서'가 '에'로 잘못 잘리지 않는다.
_JOSA = (
    "으로써", "에게서", "이라고", "에서는", "으로는", "에서", "에게", "으로", "라고",
    "까지", "부터", "보다", "처럼", "이나", "마다", "조차", "한테", "와의", "과의",
    "의", "가", "이", "은", "는", "을", "를", "와", "과", "로", "도", "만", "에",
)


def tokenize(text, josa=True, bigram=True):
    tokens = []
    for word in _WORD_RE.findall(unicodedata.normalize("NFC", str(text))):
        tokens.append(word)
        if not _HANGUL_RE.fullmatch(word):
            continue
        if josa and len(word) > 2:
            for particle in _JOSA:
                if word.endswith(particle) and len(word) - len(particle) >= 2:
                    tokens.append(word[: -len(particle)])
                    break
        if bigram and len(word) >= 2:
            tokens.extend(word[i:i + 2] for i in range(len(word) - 1))
    return tokens


class BM25:
    def __init__(self, corpus_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(corpus_tokens)
        self.lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.lengths) / self.n_docs
        self.term_freqs = [Counter(doc) for doc in corpus_tokens]
        df = Counter()
        for doc in corpus_tokens:
            df.update(set(doc))
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }

    def scores(self, query_tokens):
        out = [0.0] * self.n_docs
        for i, tf in enumerate(self.term_freqs):
            dl, total = self.lengths[i], 0.0
            for term in query_tokens:
                freq = tf.get(term)
                if not freq:
                    continue
                total += self.idf[term] * freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out[i] = total
        return out


class Index:
    """조항 목록을 받아 BM25 로 검색한다."""

    # 기본값은 공개 10 + 검증 18문항으로 측정한 최적 조합이다. 조사 제거는
    # 검증 MRR 을 0.8380 -> 0.8333 으로 떨어뜨려 끈다.
    def __init__(self, articles, josa=False, bigram=True):
        self.articles = articles
        self.opts = {"josa": josa, "bigram": bigram}
        self.bm25 = BM25([tokenize(self._text(a), **self.opts) for a in articles])

    @staticmethod
    def _text(article):
        # 문서명과 조 제목을 본문 앞에 붙인다. '위치정보'처럼 문서명에만 있는 말이
        # 질문에 나오면 그 문서 쪽 조항이 올라온다.
        return f"{article['document']} {article['title']} {article['text']}"

    def named_document(self, question):
        """질문이 약관 이름을 직접 부른 경우 그 이름을 돌려준다.

        '카카오 통합 약관'과 '카카오 통합서비스약관'이 섞이지 않도록 이름 전체로
        맞추고, 여러 개가 걸리면 가장 긴 이름을 쓴다.
        """
        text = re.sub(r"\s+", "", unicodedata.normalize("NFC", question))
        hits = [d for d in {a["document"] for a in self.articles}
                if re.sub(r"\s+", "", d) in text]
        return max(hits, key=len) if hits else None

    def apply_doc_boost(self, question, scores, doc_boost):
        """질문이 부른 약관 쪽을 끌어올린다. 잘라내지 않고 가산만 한다 —
        같은 내용이 여러 약관에 나란히 실린 문항이 실제로 있다."""
        named = self.named_document(question) if doc_boost else None
        if not named:
            return scores
        # 점수 폭을 기준으로 가산한다. max 만 쓰면 코사인 유사도처럼 음수가 나올 수
        # 있는 점수에서 가산점이 오히려 감점이 된다.
        bonus = doc_boost * (max(scores) - min(scores))
        return [s + bonus if a["document"] == named else s
                for s, a in zip(scores, self.articles)]

    def rank(self, question, doc_boost=0.5):
        """전체 조항을 점수 순으로 세운 인덱스 목록. 동점은 인덱스 순으로 고정한다."""
        scores = self.apply_doc_boost(
            question, self.bm25.scores(tokenize(question, **self.opts)), doc_boost
        )
        return sorted(range(len(self.articles)), key=lambda i: (-scores[i], i))

    def search(self, question, top_k=4, doc_boost=0.5):
        return [self.articles[i] for i in self.rank(question, doc_boost)[:top_k]]
