import json

from retrieval import Retriever, load_corpus


CONFIGS = (
    ("article", "word"),
    ("article", "ngram"),
    ("article", "hybrid"),
    ("clause", "word"),
    ("clause", "ngram"),
    ("clause", "hybrid"),
)


def _gold_set(question):
    return {(item["doc"], item["article"]) for item in question["gold_articles"]}


def _evaluate_question(retriever, question, granularity, tokenizer):
    gold = _gold_set(question)
    retrieved = retriever.search(question["question"], 4, granularity, tokenizer)
    retrieved_keys = [(doc, article_no) for doc, article_no, _ in retrieved]
    hits = set(retrieved_keys) & gold
    reciprocal_rank = next(
        (1.0 / rank for rank, key in enumerate(retrieved_keys, 1) if key in gold),
        0.0,
    )
    return {
        "qid": question["id"],
        "gold": gold,
        "retrieved": retrieved,
        "recall": len(hits) / len(gold),
        "full": len(hits) == len(gold),
        "mrr": reciprocal_rank,
        "hit1": bool(retrieved_keys and retrieved_keys[0] in gold),
    }


def _metrics(breakdown):
    total = len(breakdown)
    return {
        "recall@4": sum(item["recall"] for item in breakdown) / total,
        "full@4": sum(item["full"] for item in breakdown) / total,
        "mrr": sum(item["mrr"] for item in breakdown) / total,
        "hit@1": sum(item["hit1"] for item in breakdown) / total,
    }


def _format_key(key):
    return "(%s, %s)" % key


def _print_table(rows):
    print("granularity  tokenizer   recall@4   full@4      mrr   hit@1")
    print("------------ ---------- --------- --------- ------- -------")
    for row in rows:
        metrics = row["metrics"]
        print(
            "%-12s %-10s %9.3f %9.3f %7.3f %7.3f"
            % (
                row["granularity"],
                row["tokenizer"],
                metrics["recall@4"],
                metrics["full@4"],
                metrics["mrr"],
                metrics["hit@1"],
            )
        )


def _print_breakdown(config, breakdown):
    print("\nBEST CONFIG: granularity=%s, tokenizer=%s" % config)
    for item in breakdown:
        gold = sorted(item["gold"], key=lambda key: (key[0], key[1]))
        print("%s  gold=%s" % (item["qid"], "{" + ", ".join(_format_key(key) for key in gold) + "}"))
        retrieved_keys = set()
        for rank, (doc, article_no, score) in enumerate(item["retrieved"], 1):
            key = (doc, article_no)
            retrieved_keys.add(key)
            print("  %d. %s score=%.6f%s" % (rank, _format_key(key), score, "  GOLD" if key in item["gold"] else ""))
        for key in gold:
            if key not in retrieved_keys:
                print("  MISS %s" % _format_key(key))


def main(corpus_path="corpus.json", gold_path="gold_questions_public10.json"):
    corpus = load_corpus(corpus_path)
    with open(gold_path, encoding="utf-8") as source:
        questions = json.load(source)["questions"]
    retriever = Retriever(corpus)

    rows = []
    config_results = {}
    for order, (granularity, tokenizer) in enumerate(CONFIGS):
        breakdown = [
            _evaluate_question(retriever, question, granularity, tokenizer)
            for question in questions
        ]
        metrics = _metrics(breakdown)
        row = {
            "granularity": granularity,
            "tokenizer": tokenizer,
            "metrics": metrics,
            "breakdown": breakdown,
            "order": order,
        }
        rows.append(row)
        config_results["%s/%s" % (granularity, tokenizer)] = metrics

    rows.sort(
        key=lambda row: (
            -row["metrics"]["recall@4"],
            -row["metrics"]["full@4"],
            -row["metrics"]["mrr"],
            -row["metrics"]["hit@1"],
            row["order"],
        )
    )
    _print_table(rows)
    best = rows[0]
    _print_breakdown((best["granularity"], best["tokenizer"]), best["breakdown"])
    return {
        "configs": config_results,
        "best_config": {
            "granularity": best["granularity"],
            "tokenizer": best["tokenizer"],
            **best["metrics"],
        },
        "breakdown": best["breakdown"],
    }


if __name__ == "__main__":
    main()
