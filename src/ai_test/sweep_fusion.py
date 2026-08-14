"""융합 가중치 스윕 — 공개 10문항과 합성 144문항에서 동시에 잰다.

공개 세트는 BM25가 만점이고 합성 세트는 dense가 압도한다. 한쪽만 보고 고르면
반대쪽에서 무너지므로 두 세트를 같은 표에 놓고 결정한다.

    .venv/bin/python sweep_fusion.py
"""
from dense_retrieval import HybridRetriever
from eval_all import load_questions, metrics
from retrieval import load_corpus

SETS = ["gold_questions_public10.json", "bench_questions.json"]
WEIGHTS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def main():
    corpus = load_corpus()
    retriever = HybridRetriever(corpus)
    questions = {path: load_questions(path) for path in SETS}

    def row(label, fn):
        cells = []
        for path in SETS:
            m = metrics(questions[path], fn)
            cells.append("{:.3f}/{:.3f}/{:.3f}".format(m["recall@4"], m["full@4"], m["mrr"]))
        print("{:<22} {:>24} {:>24}".format(label, *cells))

    print("{:<22} {:>24} {:>24}".format("config (recall/full/mrr)", "public10", "bench144"))
    row("bm25 only", lambda q, k: retriever.search(q, k, "bm25"))
    row("dense only", lambda q, k: retriever.search(q, k, "dense"))
    for w in WEIGHTS:
        row("rrf dense_w={}".format(w),
            lambda q, k, w=w: retriever.search(q, k, "rrf", dense_weight=w))


if __name__ == "__main__":
    main()
