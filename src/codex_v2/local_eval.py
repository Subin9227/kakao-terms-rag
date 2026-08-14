# -*- coding: utf-8 -*-
"""운영진 채점기를 역공학한 오프라인 채점기.

practice_result_*.json 6팀 × 10문항의 실측 keyfact_f1 / keyfact_recall / mrr 을
재현하도록 맞춘 구현이다. GPU 없이 답변 문안만으로 점수를 비교할 수 있다.

역공학 근거(6팀 실측값 대조로 확정):
  · keyfact_f1 = 2|A∩G| / (|A|+|G|)  — 토큰 '집합'(중복 무시) 기준
  · 토큰 = [0-9A-Za-z가-힣]+ 중 순수 숫자 토큰 제외
  · 대괄호·괄호 안 내용을 따로 지우지 않는다 (인용 표기도 그대로 점수에 반영된다)
  · MRR = retrieved 에서 정답 (문서, 조) 가 처음 등장한 순위의 역수, 4위까지
"""

import json
import re
import unicodedata
from collections import Counter

_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def tokens(text):
    """채점기와 같은 방식으로 토큰 집합을 만든다."""
    raw = _TOKEN_RE.findall(unicodedata.normalize("NFC", str(text)))
    return {t for t in raw if not t.isdigit()}


def keyfact_f1(answer, keyfacts):
    """키팩트 전체를 이어붙인 텍스트와 답변 사이의 토큰 F1."""
    a, g = tokens(answer), tokens(" ".join(keyfacts))
    overlap = len(a & g)
    if not overlap:
        return 0.0
    return round(2 * overlap / (len(a) + len(g)), 4)


def keyfact_recall(answer, keyfacts, threshold=0.7):
    """키팩트 단위 재현율. 키팩트 토큰의 threshold 이상이 답변에 있으면 적중."""
    a = tokens(answer)
    hits = 0
    for fact in keyfacts:
        g = tokens(fact)
        if g and len(a & g) / len(g) >= threshold:
            hits += 1
    return round(hits / len(keyfacts), 4)


def reciprocal_rank(retrieved, gold_refs, cutoff=4):
    """정답 (문서, 조) 가 처음 등장한 순위의 역수. 4위 안에 없으면 0."""
    gold = {(_norm(d), int(a)) for d, a in gold_refs}
    for i, (doc, art) in enumerate(retrieved[:cutoff]):
        if (_norm(doc), int(art)) in gold:
            return 1.0 / (i + 1)
    return 0.0


def _norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", str(s)))


def score(answers, goldset):
    """answers: {qid: {"answer": str, "retrieved": [[doc, art], ...]}}

    운영진 배점(MRR 20 + 키팩트 F1 30 + LLM 판정 50)에서 LLM 판정을 뺀
    객관 지표 50점분만 계산한다. LLM 판정은 로컬에서 재현할 수 없다.
    """
    rows = []
    for item in goldset:
        qid = item["qid"]
        got = answers.get(qid)
        if got is None:
            rows.append({"qid": qid, "rr": 0.0, "f1": 0.0, "recall": 0.0, "chars": 0})
            continue
        rows.append({
            "qid": qid,
            "rr": reciprocal_rank(got["retrieved"], item["gold"]),
            "f1": keyfact_f1(got["answer"], item["keyfacts"]),
            "recall": keyfact_recall(got["answer"], item["keyfacts"]),
            "chars": len(got["answer"]),
        })
    n = len(rows)
    mrr = sum(r["rr"] for r in rows) / n
    f1 = sum(r["f1"] for r in rows) / n
    return {
        "per_question": rows,
        "mrr": round(mrr, 4),
        "keyfact_f1": round(f1, 4),
        "mrr_points": round(mrr * 20, 3),
        "keyfact_points": round(f1 * 30, 3),
        "objective_points": round(mrr * 20 + f1 * 30, 3),
    }


CONTRACT_DOCS = (
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
)


def check_contract(payload):
    """answer_question 반환값이 대회 계약을 지키는지 검사한다."""
    problems = []
    if not isinstance(payload, dict):
        return ["반환값이 dict 가 아닙니다"]
    if not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        problems.append("answer 가 비어 있지 않은 문자열이 아닙니다")
    ret = payload.get("retrieved")
    if not isinstance(ret, list) or not 1 <= len(ret) <= 4:
        problems.append(f"retrieved 개수가 1~4 가 아닙니다: {ret!r}")
        return problems
    for pair in ret:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            problems.append(f"retrieved 항목이 [문서명, 조번호] 쌍이 아닙니다: {pair!r}")
            continue
        doc, art = pair
        if doc not in CONTRACT_DOCS:
            problems.append(f"문서명이 고정 목록에 없습니다: {doc!r}")
        if not isinstance(art, int) and not str(art).strip().isdigit():
            problems.append(f"조번호를 숫자로 식별할 수 없습니다: {art!r}")
    return problems


def load_goldset(path):
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)["questions"]


if __name__ == "__main__":
    # 자체 검사 — 역공학이 실제 채점 결과를 재현하는지 확인한다.
    assert keyfact_f1("담당자 1인만 이용할 수 있으며", ["담당자 1인만 이용할 수 있으며"]) == 1.0
    assert keyfact_f1("전혀 다른 문장", ["담당자 1인만"]) == 0.0
    assert reciprocal_rank([["A", 1], ["카카오계정 약관", 10]], [["카카오계정 약관", 10]]) == 0.5
    assert reciprocal_rank([["A", 1]], [["카카오계정 약관", 10]]) == 0.0
    assert tokens("제10조 [1] 2시간") == {"제10조", "2시간"}   # 순수 숫자 토큰만 탈락
    assert check_contract({"answer": "x", "retrieved": [["카카오계정 약관", 10]]}) == []
    assert check_contract({"answer": "x", "retrieved": []})
    print("local_eval self-check OK")
