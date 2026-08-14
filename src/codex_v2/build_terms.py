# -*- coding: utf-8 -*-
"""1번 셀에 넣을 약관 원문(4종 72조)을 만든다.

HTML 3종은 이미 깨끗한 terms_chunks.jsonl 을 그대로 쓰고,
PDF 에서 뽑은 '카카오 통합 약관'만 줄바꿈으로 끊긴 어절을 되붙인다.

되붙이는 판단은 사전으로 한다. 줄 끝 어절 + 줄 첫 어절을 공백 없이 이었을 때
나머지 약관 어디에도 없는 말이 되면 원래 띄어쓰기였던 것이고, 실제로 쓰이는
말이 되면 PDF 가 어절 중간을 자른 것이다.
  · '따' + '릅니다' -> '따릅니다' (사전에 있음)  -> 붙인다
  · '및' + '규칙'   -> '및규칙'   (사전에 없음)  -> 띄운다

이 파일은 제출본이 아니라 제출본을 만드는 도구다. 결과기 코랩은 여기서 나온
문자열을 셀 안에 그대로 담고 실행 중에 파일을 읽지 않는다.
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
PDF = os.path.join(ROOT, "kakao_integrated_terms_archive_2022-08-25.pdf")
CHUNKS = os.path.join(ROOT, "terms_chunks.jsonl")

ARCHIVE = "카카오 통합 약관"
EXPECTED = {
    "카카오계정 약관": 17,
    "카카오 통합서비스약관": 18,
    ARCHIVE: 21,
    "카카오 위치정보 이용약관": 16,
}
EFFECTIVE = {
    "카카오계정 약관": "2026-05-29",
    "카카오 통합서비스약관": "2026-05-29",
    ARCHIVE: "2022-08-25",
    "카카오 위치정보 이용약관": "2026-07-16",
}
# PDF 본문에서 조 표제는 언제나 그 줄 하나를 통째로 차지한다.
# 본문 안의 '제9조에 의한' 같은 참조는 줄 전체가 아니므로 걸리지 않는다.
HEADING_RE = re.compile(r"^제\s*(\d+)\s*조\s*(.{0,40})$")
CHAPTER_RE = re.compile(r"^제\s*\d+\s*장\b")


def load_clean_chunks():
    with open(CHUNKS, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def build_vocab(rows):
    """PDF 가 아닌 문서에서 실제로 쓰인 어절 사전."""
    vocab = set()
    for row in rows:
        if row["document"] == ARCHIVE:
            continue
        vocab.update(re.findall(r"[0-9A-Za-z가-힣]+", row["title"] + " " + row["text"]))
    return vocab


def repair_wraps(lines, vocab):
    """줄 경계에서 끊긴 어절을 사전으로 판단해 되붙인다."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        tail = re.search(r"[0-9A-Za-z가-힣]+$", out)
        head = re.match(r"[0-9A-Za-z가-힣]+", line)
        if tail and head and (tail.group() + head.group()) in vocab:
            out += line          # 어절 중간이 잘린 것 -> 공백 없이 잇는다
        else:
            out += " " + line
    return out


def split_articles(lines, vocab):
    """줄 단위로 조를 자른 뒤, 각 조의 본문만 되붙이기를 적용한다."""
    articles, current = [], None
    for line in lines:
        stripped = line.strip()
        if not stripped or CHAPTER_RE.match(stripped):
            continue
        head = HEADING_RE.match(stripped)
        if head:
            current = {"article": int(head.group(1)), "title": head.group(2).strip(), "body": []}
            articles.append(current)
        elif current is not None:
            current["body"].append(stripped)
    return [
        {"article": a["article"], "title": a["title"], "text": repair_wraps(a["body"], vocab)}
        for a in articles if a["body"]
    ]


def main():
    rows = load_clean_chunks()
    vocab = build_vocab(rows)

    raw = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", PDF, "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    # 쪽번호 줄만 걷어낸다.
    lines = [l for l in raw.splitlines() if not re.fullmatch(r"\s*\d+\s*", l)]
    fixed = split_articles(lines, vocab)

    out = []
    for row in rows:
        if row["document"] == ARCHIVE:
            continue
        out.append({k: row[k] for k in ("document", "article", "title", "text")})
    for art in sorted(fixed, key=lambda a: a["article"]):
        out.append({"document": ARCHIVE, **art})

    counts = {}
    for row in out:
        counts[row["document"]] = counts.get(row["document"], 0) + 1
    print("조항 수:", counts)
    bad = {d: (counts.get(d), n) for d, n in EXPECTED.items() if counts.get(d) != n}
    if bad:
        print("기대치와 다름:", bad, file=sys.stderr)

    for row in out:
        row["effective_date"] = EFFECTIVE[row["document"]]
    dest = os.path.join(HERE, "terms.json")
    with open(dest, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    total = sum(len(r["text"]) for r in out)
    print(f"저장: {dest}  조항 {len(out)}개  본문 {total:,}자")

    # 되붙이기가 실제로 먹었는지 눈으로 확인할 표본
    for probe in ("따릅니다", "2시간 이상", "존속하게 됩니다"):
        hit = sum(probe in r["text"] for r in out if r["document"] == ARCHIVE)
        print(f"  검사 {probe!r}: {ARCHIVE} 안에서 {hit}건")


if __name__ == "__main__":
    main()
