# -*- coding: utf-8 -*-
"""검색기만 따로 측정한다. 공개 10문항 + 자체 검증 18문항.

MRR 은 20점 배점이고 상위 팀이 모두 1.0 이라 방어 지표다. 공개 10문항이 1.0 인 것은
당연하고, 판단은 검증 18문항(공개 문항이 쓰지 않은 조항) 쪽을 보고 한다.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieval import Index  # noqa: E402


def load_questions():
    with open(os.path.join(HERE, "goldset.json"), encoding="utf-8") as fp:
        public = [{"id": q["qid"], "q": q["question"], "gold": q["gold"]}
                  for q in json.load(fp)["questions"]]
    with open(os.path.join(HERE, "valset.json"), encoding="utf-8") as fp:
        val = [{"id": q["id"], "q": q["q"], "gold": q["gold"]} for q in json.load(fp)]
    return public, val


def measure(index, questions, top_k=4):
    rr_total, hit, misses = 0.0, 0, []
    for item in questions:
        gold = {(d, int(a)) for d, a in item["gold"]}
        got = [(h["document"], h["article"]) for h in index.search(item["q"], top_k)]
        rank = next((i + 1 for i, ref in enumerate(got) if ref in gold), None)
        rr_total += 1.0 / rank if rank else 0.0
        if rank:
            hit += 1
        else:
            misses.append((item["id"], got[:2]))
    n = len(questions)
    return rr_total / n, hit, n, misses


def main():
    with open(os.path.join(HERE, "terms.json"), encoding="utf-8") as fp:
        articles = json.load(fp)
    public, val = load_questions()

    print(f"조항 {len(articles)}개 | 공개 {len(public)}문항 | 검증 {len(val)}문항")
    print(f"{'토크나이저':22} {'공개 MRR':>9} {'검증 MRR':>9} {'검증 top4':>10}")
    print("-" * 56)
    best = None
    for josa in (False, True):
        for bigram in (False, True):
            index = Index(articles, josa=josa, bigram=bigram)
            pub_mrr, _, _, _ = measure(index, public)
            val_mrr, hit, n, misses = measure(index, val)
            name = "어절" + ("+조사" if josa else "") + ("+bigram" if bigram else "")
            print(f"{name:22} {pub_mrr:9.4f} {val_mrr:9.4f} {hit:>7}/{n}")
            if best is None or val_mrr > best[0]:
                best = (val_mrr, name, misses)
    print("-" * 56)
    print(f"최고 조합: {best[1]} (검증 MRR {best[0]:.4f})")
    if best[2]:
        print("검증셋 미적중:")
        for qid, got in best[2]:
            print(f"  {qid} -> {got}")


if __name__ == "__main__":
    main()
