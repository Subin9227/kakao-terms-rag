"""Score an answers file against the gold set — local proxy for the operators' LLM judge.

    python3 score_answers.py answers_public_1.json [gold_questions_public10.json]

Two answer metrics, deliberately mechanical so runs stay comparable:
  hard  — key fact counted as stated if the answer contains a >=12-char contiguous
          substring of it. Catches exact figures, periods and legal citations, which
          is what the reference answers hinge on.
  soft  — character-trigram recall of the key fact inside the answer. Credits a
          faithful paraphrase that hard misses.
Plus retrieval recall over the gold articles the response actually returned.
"""
import json
import re
import sys
import unicodedata

# 12자로 잡았더니 "6개월간 보관됩니다"처럼 어미만 다른 정답을 0점 처리했다. 8자로 낮추고,
# 판정 기준은 soft(문자 3그램 재현율)를 우선한다 — 실제 채점은 의미 일치를 보는 LLM 심사다.
MIN_SPAN = 8


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text)))


def norm_doc(text):
    return norm(text)


def trigrams(text):
    text = norm(text)
    return {text[i:i + 3] for i in range(len(text) - 2)}


def hard_covered(fact, answer):
    fact, answer = norm(fact), norm(answer)
    return any(fact[i:i + MIN_SPAN] in answer for i in range(len(fact) - MIN_SPAN + 1))


def soft_recall(fact, answer):
    fact_grams = trigrams(fact)
    if not fact_grams:
        return 0.0
    return len(fact_grams & trigrams(answer)) / len(fact_grams)


def article_no(value):
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def main(answers_path, gold_path="gold_questions_public10.json"):
    answers = json.load(open(answers_path, encoding="utf-8"))
    gold = json.load(open(gold_path, encoding="utf-8"))["questions"]
    by_qid = {a["qid"]: a for a in answers["answers"]}
    gold_by_id = {q["id"]: q for q in gold}

    missing = [q["id"] for q in gold if q["id"] not in by_qid]
    extra = [qid for qid in by_qid if qid not in gold_by_id]
    if missing or extra:
        print("[warn] qid mismatch — missing={} extra={}".format(missing, extra))

    rows, hard_all, soft_all, ret_all = [], [], [], []
    for qid, q in gold_by_id.items():
        got = by_qid.get(qid)
        if got is None:
            continue
        answer = got.get("answer") or ""
        facts = q["key_facts"]
        hard = sum(hard_covered(f, answer) for f in facts) / len(facts)
        soft = sum(soft_recall(f, answer) for f in facts) / len(facts)
        want = {(norm_doc(g["doc"]), g["article"]) for g in q["gold_articles"]}
        have = {(norm_doc(d), article_no(n)) for d, n in got.get("retrieved", [])}
        ret = len(want & have) / len(want)
        rows.append((qid, q["ptype"], hard, soft, ret, len(answer), got.get("error")))
        hard_all.append(hard)
        soft_all.append(soft)
        ret_all.append(ret)

    print("{:<5} {:<6} {:>6} {:>6} {:>6} {:>6}  {}".format(
        "qid", "ptype", "hard", "soft", "ret", "chars", "error"))
    for qid, ptype, hard, soft, ret, length, err in rows:
        print("{:<5} {:<6} {:>6.2f} {:>6.2f} {:>6.2f} {:>6}  {}".format(
            qid, ptype, hard, soft, ret, length, err or ""))
    n = len(rows) or 1
    print("\nMEAN  hard={:.3f}  soft={:.3f}  retrieval={:.3f}  (n={})".format(
        sum(hard_all) / n, sum(soft_all) / n, sum(ret_all) / n, len(rows)))

    perf = (answers.get("meta") or {}).get("performance")
    if perf:
        print("PERF  p50={}s p95={}s throughput={} req/s success={}/{}".format(
            perf.get("p50_latency_s"), perf.get("p95_latency_s"),
            perf.get("throughput_rps"), perf.get("success"),
            perf.get("requests")))
    viol = (answers.get("meta") or {}).get("doc_name_violations")
    print("DOC VIOLATIONS:", len(viol) if viol else 0)


if __name__ == "__main__":
    main(*sys.argv[1:])
