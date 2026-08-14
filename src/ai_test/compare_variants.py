"""후보 답변 파일들을 같은 표로 놓고 비교한다.

    python3 compare_variants.py bench_subset.json answers_bench_*.json
"""
import collections
import glob
import json
import sys

from score_answers import article_no, hard_covered, norm_doc, soft_recall


def score(path, gold):
    data = json.load(open(path, encoding="utf-8"))
    by_qid = {a["qid"]: a for a in data["answers"]}
    per_type = collections.defaultdict(lambda: [0.0, 0.0, 0])
    hard = soft = ret = 0.0
    lengths = []
    for q in gold:
        got = by_qid.get(q["id"])
        if got is None:
            continue
        answer = got.get("answer") or ""
        facts = q["key_facts"]
        h = sum(hard_covered(f, answer) for f in facts) / len(facts)
        s = sum(soft_recall(f, answer) for f in facts) / len(facts)
        want = {(norm_doc(g["doc"]), g["article"]) for g in q["gold_articles"]}
        have = {(norm_doc(d), article_no(n)) for d, n in got.get("retrieved", [])}
        hard += h
        soft += s
        ret += len(want & have) / len(want)
        lengths.append(len(answer))
        row = per_type[q.get("ptype", "?")]
        row[0] += h
        row[1] += s
        row[2] += 1
    n = len(lengths) or 1
    perf = (data.get("meta") or {}).get("performance") or {}
    return {"n": len(lengths), "hard": hard / n, "soft": soft / n, "ret": ret / n,
            "chars": sum(lengths) / n, "p50": perf.get("p50_latency_s"),
            "per_type": {t: (v[0] / v[2], v[1] / v[2], v[2]) for t, v in sorted(per_type.items())}}


def main(gold_path, paths):
    gold = json.load(open(gold_path, encoding="utf-8"))["questions"]
    rows = [(p, score(p, gold)) for p in paths]
    rows.sort(key=lambda r: -r[1]["soft"])
    print("{:<28} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "file", "n", "hard", "soft", "ret", "chars", "p50_s"))
    for path, r in rows:
        print("{:<28} {:>4} {:>7.3f} {:>7.3f} {:>7.3f} {:>7.0f} {:>7}".format(
            path.replace("answers_bench_", "").replace(".json", ""),
            r["n"], r["hard"], r["soft"], r["ret"], r["chars"], r["p50"] or "-"))
    print("\nptype별 soft")
    types = sorted({t for _, r in rows for t in r["per_type"]})
    print("{:<28} ".format("file") + " ".join("{:>14}".format(t) for t in types))
    for path, r in rows:
        cells = []
        for t in types:
            v = r["per_type"].get(t)
            cells.append("{:>14}".format("{:.3f}(n={})".format(v[1], v[2]) if v else "-"))
        print("{:<28} ".format(path.replace("answers_bench_", "").replace(".json", "")) + " ".join(cells))


if __name__ == "__main__":
    args = sys.argv[1:]
    gold = args[0] if args else "bench_subset.json"
    files = args[1:] or sorted(glob.glob("answers_bench_*.json"))
    main(gold, files)
