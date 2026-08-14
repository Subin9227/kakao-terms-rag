"""운영진 채점식을 로컬에서 근사한다 — MRR 20 + 키팩트 F1 30 + LLM 50.

MRR은 정의가 공개돼 있어 정확히 재현된다(1위 1.0, 2위 0.5, 3위 1/3, 4위 0.25, 밖이면 0).
키팩트 F1은 토크나이저가 공개되지 않아 근사한다. 12팀 연습 채점 결과 10문항으로 후보를
맞춰 본 결과 문자 2그램 다중집합 F1이 평균오차 0.025로 가장 가까웠다.
LLM 50점은 재현 불가라 계산에서 빼고, 나머지 50점 만점 기준으로만 보고한다.

    python3 official_score.py answers_public_9.json [gold_questions_public10.json]
"""
import json
import re
import sys
import unicodedata
from collections import Counter

MRR_AT = {1: 1.0, 2: 0.5, 3: 1.0 / 3, 4: 0.25}


def norm(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(text)))


def char2(text):
    text = norm(text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


def f1(pred, ref):
    p, r = char2(pred), char2(ref)
    if not p or not r:
        return 0.0, 0.0, 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0, 0.0, 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall), precision, recall


def article_no(value):
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def main(answers_path, gold_path="gold_questions_public10.json"):
    answers = json.load(open(answers_path, encoding="utf-8"))
    gold = json.load(open(gold_path, encoding="utf-8"))["questions"]
    by_qid = {a["qid"]: a for a in answers["answers"]}

    rows, mrr_all, f1_all = [], [], []
    for q in gold:
        got = by_qid.get(q["id"])
        if got is None:
            continue
        answer = got.get("answer") or ""
        want = {(norm(g["doc"]), g["article"]) for g in q["gold_articles"]}
        rank = None
        for i, (doc, no) in enumerate(got.get("retrieved", []), 1):
            if (norm(doc), article_no(no)) in want:
                rank = i
                break
        mrr = MRR_AT.get(rank, 0.0)
        score, precision, recall = f1(answer, " ".join(q["key_facts"]))
        rows.append((q["id"], q["ptype"], rank, mrr, score, precision, recall, len(answer)))
        mrr_all.append(mrr)
        f1_all.append(score)

    print("{:<5} {:<6} {:>4} {:>5} {:>6} {:>6} {:>6} {:>6}".format(
        "qid", "ptype", "rank", "mrr", "f1", "prec", "rec", "chars"))
    for qid, ptype, rank, mrr, score, precision, recall, length in rows:
        print("{:<5} {:<6} {:>4} {:>5.2f} {:>6.3f} {:>6.3f} {:>6.3f} {:>6}".format(
            qid, ptype, rank if rank else 0, mrr, score, precision, recall, length))

    n = len(rows) or 1
    mrr_mean, f1_mean = sum(mrr_all) / n, sum(f1_all) / n
    print("\nMRR      {:.4f} → {:.3f} / 20".format(mrr_mean, mrr_mean * 20))
    print("키팩트F1 {:.4f} → {:.3f} / 30".format(f1_mean, f1_mean * 30))
    print("소계     {:.3f} / 50  (LLM 50점 제외)".format(mrr_mean * 20 + f1_mean * 30))


if __name__ == "__main__":
    main(*sys.argv[1:])
