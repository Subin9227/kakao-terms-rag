# -*- coding: utf-8 -*-
"""GPU 없이 돌리는 사전 점검.

생성 모델만 가짜로 바꾸고 나머지(약관 적재, 검색, 유형 분류, 후처리, 회피 안전망,
반환 계약)를 공개 10문항 + 검증 18문항 전부에 대해 실제로 돌린다.
Colab 에 올리기 전에 여기서 깨지는 것을 먼저 잡는다.
"""

import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import generation as gen                                  # noqa: E402
from local_eval import check_contract, reciprocal_rank    # noqa: E402
from retrieval import Index                               # noqa: E402

ARTICLES = json.load(open(os.path.join(HERE, "terms.json"), encoding="utf-8"))
INDEX = Index(ARTICLES, josa=False, bigram=True)

# 노트북에서 템플릿이 채우는 전역을 여기서 대신 채운다.
gen.TOP_K = 4
gen.search = lambda q, top_k=4: INDEX.search(q, top_k, doc_boost=0.5)

FAKE = {"normal": "담당자 1인만 이용할 수 있으며, 다른 사람에게 공유하는 것은 금지됩니다.",
        "refusal": "제공된 조항에서는 확인되지 않습니다.",
        "empty": "   ",
        "noisy": "답변: 위 조항에 따르면, 금지됩니다. 금지됩니다. 这是中文"}
MODE = {"value": "normal"}
gen.generate = lambda messages, max_new_tokens: FAKE[MODE["value"]]


def questions():
    with open(os.path.join(HERE, "goldset.json"), encoding="utf-8") as fp:
        for q in json.load(fp)["questions"]:
            yield q["qid"], q["question"], q["gold"]
    with open(os.path.join(HERE, "valset.json"), encoding="utf-8") as fp:
        for q in json.load(fp):
            yield q["id"], q["q"], q["gold"]


def check_postprocess():
    """후처리가 점수를 깎는 두 가지 실수를 막는지 본다."""
    out = gen.postprocess("답변: 위 조항에 따르면, 변경・제한・중지할 수 있습니다. "
                          "변경・제한・중지할 수 있습니다. 这是中文")
    assert out.startswith("변경"), out              # 머리말·군더더기 제거
    assert out.count("변경") == 1, out              # 같은 문장 반복 제거
    assert "・" in out, out                         # 가운뎃점은 남아야 세 낱말로 세어진다
    assert "这" not in out, out                     # 한자 혼입만 제거


def main():
    check_postprocess()
    items = list(questions())
    kinds, problems, rr_total = Counter(), [], 0.0
    for qid, text, gold in items:
        kinds[gen.classify(text)] += 1
        for mode in FAKE:
            MODE["value"] = mode
            out = gen.answer_question(text)
            bad = check_contract(out)
            if bad:
                problems.append((qid, mode, bad))
            if not out["answer"].strip():
                problems.append((qid, mode, ["answer 가 비었습니다"]))
            if mode in ("refusal", "empty") and "확인되지" in out["answer"]:
                problems.append((qid, mode, ["회피 안전망이 동작하지 않았습니다"]))
        MODE["value"] = "normal"
        rr_total += reciprocal_rank(gen.answer_question(text)["retrieved"], gold)

    print(f"문항 {len(items)}개 × 생성 시나리오 {len(FAKE)}가지")
    print(f"유형 분류: {dict(kinds)}")
    print(f"검색 MRR(전체 28문항) {rr_total / len(items):.4f}")
    print(f"계약 위반 {len(problems)}건")
    for qid, mode, bad in problems[:10]:
        print(f"  {qid} [{mode}] {bad}")
    # 28문항이 건드리지 않는 입력들. 계약을 깨지 않고 무언가를 돌려주기만 하면 된다.
    MODE["value"] = "normal"
    edge = ["hello, what is this?", "?????", "12345", "가", "약관",
            "카카오계정 약관과 카카오 통합 약관 중 어느 쪽이 우선하나요?",
            "회사는 " * 400 + "무엇을 하나요?"]
    for mode in ("normal", "refusal"):
        MODE["value"] = mode
        for text in edge:
            out = gen.answer_question(text)
            bad = check_contract(out)
            if bad:
                problems.append((text[:24], mode, bad))
    print(f"경계 입력 {len(edge)}가지 × 2시나리오 통과")

    MODE["value"] = "normal"
    ctx = gen.build_context(INDEX.search("서비스 중단 시 복구 지연 공지", 4, doc_boost=0.5))
    print(f"컨텍스트 길이 표본 {len(ctx):,}자")
    longest = max(len(gen.build_context(INDEX.search(t, 4, doc_boost=0.5)))
                  for _, t, _ in items)
    print(f"컨텍스트 최댓값 {longest:,}자")
    assert not problems, "계약 위반이 남아 있습니다"
    print("test_offline OK")


if __name__ == "__main__":
    main()
