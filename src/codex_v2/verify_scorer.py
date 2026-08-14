# -*- coding: utf-8 -*-
"""복원한 채점기가 6팀의 실제 채점 결과를 재현하는지 확인한다."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eval import load_goldset, score  # noqa: E402

DOWNLOADS = os.path.expanduser("~/Downloads")
HERE = os.path.dirname(os.path.abspath(__file__))
TEAMS = (12, 13, 14, 17, 2, 5)


def main():
    goldset = load_goldset(os.path.join(HERE, "goldset.json"))
    print(f"{'팀':>3}  {'MRR 실측/복원':>16}  {'F1 실측/복원':>16}  {'객관 50점 실측/복원':>22}")
    print("-" * 68)
    deltas = []
    for team in TEAMS:
        with open(f"{DOWNLOADS}/answers_public_{team}.json", encoding="utf-8") as fp:
            answers = {a["qid"]: a for a in json.load(fp)["answers"]}
        with open(f"{DOWNLOADS}/practice_result_{team}.json", encoding="utf-8") as fp:
            obj = json.load(fp)["objective"]
        got = score(answers, goldset)
        real_pts = obj["mrr"] * 20 + obj["keyfact_f1"] * 30
        deltas.append(abs(got["objective_points"] - real_pts))
        print(f"{team:>3}  {obj['mrr']:>7.4f}/{got['mrr']:<8.4f}  "
              f"{obj['keyfact_f1']:>7.4f}/{got['keyfact_f1']:<8.4f}  "
              f"{real_pts:>10.3f}/{got['objective_points']:<11.3f}")
    print("-" * 68)
    print(f"객관 50점 기준 평균 오차 {sum(deltas) / len(deltas):.3f}점, 최대 {max(deltas):.3f}점")


if __name__ == "__main__":
    main()
