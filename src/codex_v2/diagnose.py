# -*- coding: utf-8 -*-
"""답변 파일 하나를 문항별로 뜯어본다.

사용: python3 v2/diagnose.py <answers_public_*.json>

키팩트 F1 은 토큰 집합 F1 이므로 손실은 두 갈래뿐이다.
  · 빠진 낱말(분자 손실) — 근거 문장을 덜 옮겼거나 표현을 바꿔 썼다.
  · 남는 낱말(분모 손실) — 조항에 없는 말을 새로 썼거나 묻지 않은 내용을 붙였다.
어느 쪽인지 보여야 프롬프트를 어디로 고칠지 정할 수 있다.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from local_eval import load_goldset, score, tokens  # noqa: E402


def main(path):
    goldset = load_goldset(os.path.join(HERE, "goldset.json"))
    with open(path, encoding="utf-8") as fp:
        answers = {a["qid"]: a for a in json.load(fp)["answers"]}
    result = score(answers, goldset)

    print(f"MRR {result['mrr']:.4f} ({result['mrr_points']}점) · "
          f"키팩트 F1 {result['keyfact_f1']:.4f} ({result['keyfact_points']}점) · "
          f"객관 합계 {result['objective_points']}점 / 50")
    print("=" * 88)
    for item in goldset:
        qid = item["qid"]
        row = next(r for r in result["per_question"] if r["qid"] == qid)
        got = answers.get(qid)
        if got is None:
            print(f"{qid} 답변 없음")
            continue
        a, g = tokens(got["answer"]), tokens(" ".join(item["keyfacts"]))
        missing, extra = sorted(g - a), sorted(a - g)
        print(f"{qid}  F1 {row['f1']:.4f}  RR {row['rr']:.2f}  {row['chars']:>4}자  "
              f"겹침 {len(a & g)} / 답변 {len(a)} / 정답 {len(g)}")
        if missing:
            print(f"     빠진 낱말 {len(missing):>3}: {' '.join(missing[:18])}")
        if extra:
            print(f"     남는 낱말 {len(extra):>3}: {' '.join(extra[:18])}")
    print("=" * 88)
    worst = sorted(result["per_question"], key=lambda r: r["f1"])[:3]
    print("가장 낮은 3문항: " + ", ".join(f"{r['qid']} {r['f1']:.3f}" for r in worst))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
