"""평가기 셀프체크 — API 호출 없음.

핵심 검증: 운영진이 공개 연습 채점에 쓴 12개 팀 실측값을 우리 계산이 재현하는가.
    python3 test_evaluator.py
"""
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import (WEIGHTS, judge_total, keyfact_scores, load_answers, load_goldset,
                       mrr_contrib, score_candidate, validate_output)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold_questions_public10.json")
DATA = os.path.join(HERE, "json자료")


def _teams():
    for f in sorted(glob.glob(os.path.join(DATA, "practice_result_*.json"))):
        if "(1)" in f:
            continue
        t = re.search(r"practice_result_(\d+)", f).group(1)
        ans = os.path.join(DATA, f"answers_public_{t}.json")
        if os.path.exists(ans):
            yield t, json.load(open(f, encoding="utf-8")), ans


def test_mrr_exact():
    """MRR 은 운영진 값과 전수 일치해야 한다."""
    gold = load_goldset(GOLD)
    n = 0
    for _, res, ans_path in _teams():
        _, answers = load_answers(ans_path)
        for pq in res["objective"]["per_question"]:
            got = mrr_contrib(answers[pq["qid"]]["retrieved"], gold[pq["qid"]]["gold_articles"])
            assert abs(got - pq["mrr_contrib"]) < 1e-9, f"{pq['qid']}: {got} != {pq['mrr_contrib']}"
            n += 1
    print(f"  MRR 전수 일치 {n}건")


def test_judge_axis_weights():
    """4축 → 0~100 환산이 운영진 값과 일치해야 한다."""
    worst, n = 0.0, 0
    for _, res, _ in _teams():
        for pq in res["judges"]["gemini"]["per_question"]:
            axes = {k: v["score"] for k, v in pq["axes"].items()}
            worst = max(worst, abs(judge_total(axes) - pq["total_0_100"]))
            n += 1
    assert worst < 1e-9, f"축 가중치 불일치: 최대잔차 {worst}"
    print(f"  판정 축 환산 일치 {n}건 (최대잔차 {worst:.1e})")


def test_total_reproduces_official():
    """로컬 지표 + 운영진 judge 점수로 운영진 총점을 재현한다."""
    gold = load_goldset(GOLD)
    diffs, mine, real = [], [], []
    for _, res, ans_path in _teams():
        _, answers = load_answers(ans_path)
        jt = {p["qid"]: p["total_0_100"] for p in res["judges"]["gemini"]["per_question"]}
        n = len(gold)
        mrr = sum(mrr_contrib(answers[q]["retrieved"], gold[q]["gold_articles"]) for q in gold) / n
        kf = sum(keyfact_scores(answers[q]["answer"], gold[q]["key_facts"])[1] for q in gold) / n
        jd = sum(jt[q] for q in gold) / n
        got = WEIGHTS["mrr"] * mrr * 100 + WEIGHTS["keyfact"] * kf * 100 + WEIGHTS["judge"] * jd
        diffs.append(abs(got - res["total_0_100"]))
        mine.append(got)
        real.append(res["total_0_100"])

    mae, worst = statistics.mean(diffs), max(diffs)
    rank = lambda xs: [sorted(xs, reverse=True).index(v) + 1 for v in xs]
    rm, rr = rank(mine), rank(real)
    k = len(rm)
    rho = 1 - 6 * sum((a - b) ** 2 for a, b in zip(rm, rr)) / (k * (k * k - 1))

    assert mae <= 5.0, f"MAE {mae:.3f} > 5.0"
    assert rho >= 0.9, f"순위 상관 {rho:.4f} < 0.9"
    print(f"  총점 재현 {k}팀: MAE {mae:.3f}점 · 최대 {worst:.3f}점 · 스피어만 {rho:.4f}")


def test_dynamic_shape():
    """문항 수·문항 번호를 바꿔도 코드가 따라가야 한다."""
    gold = load_goldset(GOLD)
    subset = {k: gold[k] for k in list(gold)[:3]}
    renamed = {f"X{i:03d}": dict(v, id=f"X{i:03d}") for i, v in enumerate(subset.values())}
    answers = {qid: {"answer": " ".join(q["key_facts"]), "retrieved": [[a["doc"], a["article"]]
               for a in q["gold_articles"]]} for qid, q in renamed.items()}
    r = score_candidate("BLIND99", answers, renamed, lambda p: {"accuracy": 10, "grounding": 10,
                        "completeness": 10, "clarity": 10}, cache={})
    assert r["status"] == "completed" and r["total"] > 90, r
    print(f"  임의 문항 3개·임의 id: total={r['total']:.3f} status={r['status']}")


def test_missing_retrieved_renormalizes():
    """retrieved 가 없으면 MRR 을 빼고 가중치를 재정규화한다."""
    gold = load_goldset(GOLD)
    answers = {qid: {"answer": " ".join(q["key_facts"]), "retrieved": []} for qid, q in gold.items()}
    perfect = lambda p: {"accuracy": 10, "grounding": 10, "completeness": 10, "clarity": 10}
    r = score_candidate("BLIND98", answers, gold, perfect, cache={}, log=lambda *a: None)
    assert r["status"] == "completed" and r["total"] > 70, r
    print(f"  retrieved 없음: total={r['total']:.3f} (MRR 0점으로 깎이지 않음)")


def test_uniform_weights_across_candidates():
    """한 후보만 retrieved 가 없어도 전원 같은 가중치를 써야 비교가 성립한다."""
    import tempfile

    gold = load_goldset(GOLD)
    perfect_answers = {qid: {"answer": " ".join(q["key_facts"]),
                             "retrieved": [[a["doc"], a["article"]] for a in q["gold_articles"]]}
                       for qid, q in gold.items()}
    tmp = tempfile.mkdtemp()
    paths = []
    for i, keep in enumerate([True, False], start=1):          # 1명은 retrieved 있음, 1명은 없음
        items = [{"qid": q, "answer": v["answer"],
                  "retrieved": v["retrieved"] if keep else []} for q, v in perfect_answers.items()]
        p = os.path.join(tmp, f"answers_BLIND0{i}.json")
        json.dump({"answers": items}, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        paths.append(p)

    import evaluator
    keep_cache, evaluator.CACHE_PATH = evaluator.CACHE_PATH, os.path.join(tmp, "c.json")
    keep_gap, evaluator.REQUEST_GAP_SEC = evaluator.REQUEST_GAP_SEC, 0
    try:
        out = evaluator.run(GOLD, paths, os.path.join(tmp, "o.json"),
                            lambda p: {"accuracy": 10, "grounding": 10, "completeness": 10,
                                       "clarity": 10}, log=lambda *a: None)
    finally:
        evaluator.CACHE_PATH, evaluator.REQUEST_GAP_SEC = keep_cache, keep_gap

    a, b = out["results"]
    assert abs(a["total"] - b["total"]) < 1e-9, \
        f"동일 품질인데 점수가 다름: {a['total']} vs {b['total']} (가중치가 후보별로 갈림)"
    print(f"  retrieved 유/무 혼재: 두 후보 total 동일 {a['total']:.3f}")


def test_failure_status():
    """전 문항 판정 실패면 failed + total null."""
    gold = load_goldset(GOLD)
    answers = {qid: {"answer": "답변", "retrieved": []} for qid in gold}

    def broken(prompt):
        raise RuntimeError("판정 불가")

    import evaluator
    keep, evaluator.MAX_RETRIES = evaluator.MAX_RETRIES, 1
    keep_gap, evaluator.RETRY_BASE_SEC = evaluator.RETRY_BASE_SEC, 0
    try:
        r = score_candidate("BLIND97", answers, gold, broken, cache={}, log=lambda *a: None)
    finally:
        evaluator.MAX_RETRIES, evaluator.RETRY_BASE_SEC = keep, keep_gap
    assert r["status"] == "failed" and r["total"] is None, r
    print(f"  전 문항 실패: status={r['status']} total={r['total']}")


def test_output_schema():
    """출력 형식 검증기가 위반을 실제로 잡는가."""
    ok = {"results": [{"blind_id": f"BLIND0{i}", "total": 70.0 + i, "status": "completed"}
                      for i in range(1, 6)]}
    assert validate_output(ok) == [], validate_output(ok)

    bad = {"results": [{"blind_id": "BLIND01", "total": 70.0, "status": "completed", "rank": 1}],
           "schema_version": 1}
    assert len(validate_output(bad)) >= 2
    dup = {"results": [{"blind_id": "BLIND01", "total": 70.0, "status": "completed"},
                       {"blind_id": "BLIND01", "total": 71.0, "status": "completed"}]}
    assert any("중복" in p for p in validate_output(dup))
    nullbad = {"results": [{"blind_id": "BLIND01", "total": 0.0, "status": "failed"}]}
    assert any("null" in p for p in validate_output(nullbad))
    print("  스키마 검증기: 정상 통과 / rank·schema_version·중복·failed-total 위반 검출")


def test_no_hardcoded_question_ids():
    """평가기 본체에 문항 id·문항 수가 박혀 있으면 안 된다."""
    src = open(os.path.join(HERE, "evaluator.py"), encoding="utf-8").read()
    hits = re.findall(r'["\']P\d{2}["\']', src)
    assert not hits, f"문항 id 하드코딩: {hits}"
    assert "range(10)" not in src and "range(30)" not in src
    print("  하드코딩 없음: 문항 id 0건, 문항 수 0건")


if __name__ == "__main__":
    for fn in [test_mrr_exact, test_judge_axis_weights, test_total_reproduces_official,
               test_dynamic_shape, test_missing_retrieved_renormalizes, test_uniform_weights_across_candidates, test_failure_status,
               test_output_schema, test_no_hardcoded_question_ids]:
        print(f"{fn.__name__}:")
        fn()
    print("\n전부 통과")
