"""후보 답변 파일들을 운영진 채점식(MRR 20 + 키팩트 F1 30)으로 줄세운다.

recall 만 보던 이전 지표와 결론이 갈릴 수 있어 판정은 이 표로 한다. LLM 50점은
재현할 수 없으므로 50점 만점 기준이다.

    python3 rank_official.py [gold] [answers...]
"""
import glob
import json
import sys

from official_score import MRR_AT, article_no, f1, norm


def score(path, gold):
    by_qid = {a["qid"]: a for a in json.load(open(path, encoding="utf-8"))["answers"]}
    acc = {"f1": [], "prec": [], "rec": [], "mrr": [], "len": []}
    for qid, q in gold.items():
        got = by_qid.get(qid)
        if not got:
            continue
        answer = got.get("answer") or ""
        s, p, r = f1(answer, " ".join(q["key_facts"]))
        want = {(norm(g["doc"]), g["article"]) for g in q["gold_articles"]}
        rank = next((i for i, (d, n) in enumerate(got.get("retrieved", []), 1)
                     if (norm(d), article_no(n)) in want), None)
        acc["f1"].append(s)
        acc["prec"].append(p)
        acc["rec"].append(r)
        acc["mrr"].append(MRR_AT.get(rank, 0.0))
        acc["len"].append(len(answer))
    n = len(acc["f1"]) or 1
    mean = {k: sum(v) / n for k, v in acc.items()}
    mean["score"] = mean["mrr"] * 20 + mean["f1"] * 30
    return mean


def main(gold_path, paths):
    gold = {q["id"]: q for q in json.load(open(gold_path, encoding="utf-8"))["questions"]}
    rows = sorted(((score(p, gold), p) for p in paths), key=lambda x: -x[0]["score"])
    print("{:<12} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6}".format(
        "variant", "점수/50", "F1", "prec", "rec", "MRR", "길이"))
    for m, path in rows:
        print("{:<12} {:>8.2f} {:>7.4f} {:>7.3f} {:>7.3f} {:>7.4f} {:>6.0f}".format(
            path.replace("answers_bench_", "").replace(".json", ""),
            m["score"], m["f1"], m["prec"], m["rec"], m["mrr"], m["len"]))


if __name__ == "__main__":
    args = sys.argv[1:]
    gold = args[0] if args else "bench_subset.json"
    files = args[1:] or sorted(glob.glob("answers_bench_*.json"))
    main(gold, files)
