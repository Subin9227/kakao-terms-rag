"""Compare BM25 / dense / RRF on any question file sharing the gold schema.

    .venv/bin/python eval_all.py gold_questions_public10.json bench_questions.json
"""
import json
import sys

from retrieval import load_corpus


def load_questions(path):
    data = json.load(open(path, encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def metrics(questions, search_fn, top_k=4):
    recall = full = mrr = hit1 = 0.0
    misses = []
    for q in questions:
        gold = {(g["doc"], g["article"]) for g in q["gold_articles"]}
        got = [(d, n) for d, n, _ in search_fn(q["question"], top_k)]
        found = gold & set(got)
        recall += len(found) / len(gold)
        full += 1.0 if found == gold else 0.0
        hit1 += 1.0 if got and got[0] in gold else 0.0
        for rank, key in enumerate(got, 1):
            if key in gold:
                mrr += 1.0 / rank
                break
        if found != gold:
            misses.append((q["id"], q.get("ptype"), sorted(gold - found), got))
    n = len(questions)
    return {"recall@4": recall / n, "full@4": full / n, "mrr": mrr / n, "hit@1": hit1 / n,
            "n": n, "misses": misses}


def by_ptype(questions, search_fn, top_k=4):
    groups = {}
    for q in questions:
        groups.setdefault(q.get("ptype", "?"), []).append(q)
    return {k: metrics(v, search_fn, top_k) for k, v in sorted(groups.items())}


def main(paths, model_name="nlpai-lab/KURE-v1"):
    corpus = load_corpus()
    from dense_retrieval import HybridRetriever

    retriever = HybridRetriever(corpus, model_name=model_name)
    for path in paths:
        questions = load_questions(path)
        print("\n=== {}  (n={})  model={}".format(path, len(questions), model_name))
        print("{:<8} {:>9} {:>8} {:>7} {:>7}".format("mode", "recall@4", "full@4", "mrr", "hit@1"))
        results = {}
        for mode in ("bm25", "dense", "rrf"):
            m = metrics(questions, lambda q, k, mode=mode: retriever.search(q, k, mode))
            results[mode] = m
            print("{:<8} {:>9.3f} {:>8.3f} {:>7.3f} {:>7.3f}".format(
                mode, m["recall@4"], m["full@4"], m["mrr"], m["hit@1"]))
        if any(q.get("ptype") for q in questions):
            print("-- per ptype (recall@4 / full@4)")
            for mode in ("bm25", "dense", "rrf"):
                row = by_ptype(questions, lambda q, k, mode=mode: retriever.search(q, k, mode))
                print("   {:<6} ".format(mode) + "  ".join(
                    "{}={:.3f}/{:.3f}(n={})".format(t, v["recall@4"], v["full@4"], v["n"])
                    for t, v in row.items()))
        worst = results["rrf"]["misses"]
        if worst:
            print("-- rrf misses ({}):".format(len(worst)))
            for qid, ptype, missing, got in worst[:15]:
                print("   {} [{}] missing={} got={}".format(qid, ptype, missing, got[:4]))


if __name__ == "__main__":
    args = sys.argv[1:] or ["gold_questions_public10.json"]
    model = "nlpai-lab/KURE-v1"
    if args[0].startswith("--model="):
        model = args.pop(0).split("=", 1)[1]
    main(args, model)
