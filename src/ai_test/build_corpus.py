import html
import json
import re
import unicodedata
import zipfile


ARTICLE_RE = re.compile(
    r"^\s*제\s*(\d+)\s*조\s*(?:[（(]\s*([^）)]*?)\s*[）)]|(\s+.+?))\s*$"
)
TAG_RE = re.compile(r"<(/?)\s*([A-Za-z][\w:-]*)(?:\s+[^>]*)?>")
HEADING_RE = re.compile(
    r"<(?P<tag>strong|h[1-6])\b"
    r"(?=[^>]*\bclass\s*=\s*['\"][^'\"]*\btit_subject\b[^'\"]*['\"])"
    r"[^>]*>(?P<body>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
BLOCK_TAG_RE = re.compile(
    r"</?(?:br|p|li|ul|ol|div|h[1-6]|section|article|tr|td)\b[^>]*>",
    re.IGNORECASE,
)
CHROME_LINES = {
    "언어지원",
    "언어지원 메뉴",
    "메뉴 선택됨",
    "KOR",
    "ENG",
    "JPN",
    "다운로드",
    "인쇄하기",
    "공지사항",
    "권리침해신고안내",
    "개인정보 처리방침",
}


def normalize_line(value):
    value = html.unescape(unicodedata.normalize("NFC", value))
    value = value.replace("\u200b", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_heading(value):
    match = ARTICLE_RE.fullmatch(normalize_line(value))
    if not match:
        return None
    title = match.group(2) or match.group(3)
    return int(match.group(1)), normalize_line(title)


def clean_text(value):
    lines = []
    for raw_line in value.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue
        if re.fullmatch(r"(?:공고일자|시행일자)\s*:.*", line):
            break
        if line in CHROME_LINES:
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_html(value):
    value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL)
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
    value = BLOCK_TAG_RE.sub("\n", value)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value)


def terms_section(value):
    value = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", "", value, flags=re.IGNORECASE | re.DOTALL)
    opening = re.search(
        r"<div\b[^>]*class\s*=\s*['\"][^'\"]*\bwrap_terms\b[^'\"]*['\"][^>]*>",
        value,
        re.IGNORECASE,
    )
    if not opening:
        raise AssertionError("terms container not found")

    depth = 1
    for tag in TAG_RE.finditer(value, opening.end()):
        if tag.group(2).lower() != "div":
            continue
        if tag.group(1):
            depth -= 1
            if depth == 0:
                section = value[opening.end() : tag.start()]
                return re.sub(
                    r"<div\b[^>]*class\s*=\s*['\"][^'\"]*\bwrap_btn\b[^'\"]*['\"][^>]*>.*?</div\s*>",
                    "",
                    section,
                    flags=re.IGNORECASE | re.DOTALL,
                )
        else:
            depth += 1
    raise AssertionError("terms container is not closed")


def parse_html(value):
    section = terms_section(value)
    matches = []
    for match in HEADING_RE.finditer(section):
        heading = parse_heading(strip_html(match.group("body")))
        if heading:
            matches.append((match, heading))

    articles = []
    for index, (match, (article_no, title)) in enumerate(matches):
        end = matches[index + 1][0].start() if index + 1 < len(matches) else len(section)
        text = clean_text(strip_html(section[match.end() : end]))
        articles.append({"article_no": article_no, "title": title, "text": text})
    return articles


def read_docx_text(path):
    with zipfile.ZipFile(path) as archive:
        value = archive.read("word/document.xml").decode("utf-8")
    value = re.sub(r"</w:p\s*>", "\n", value)
    value = re.sub(r"<w:tab\s*/?>", " ", value)
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value)


def parse_docx(value):
    articles = []
    current = None
    body = []

    def flush():
        if current is not None:
            current["text"] = clean_text("\n".join(body))
            articles.append(current.copy())

    for raw_line in value.splitlines():
        line = normalize_line(raw_line)
        heading = parse_heading(line)
        if heading:
            flush()
            article_no, title = heading
            current = {"article_no": article_no, "title": title}
            body.clear()
        elif current is not None:
            body.append(line)
    flush()
    return articles


def fact_found(fact, text):
    fact = normalize_line(fact)
    text = normalize_line(text)
    return any(fact[index : index + 12] in text for index in range(len(fact) - 11))


def main():
    sources = [
        ("raw/account.html", "카카오계정 약관", "2026년 5월 29일", "html"),
        ("raw/location.html", "카카오 위치정보 이용약관", "2026년 7월 16일", "html"),
        ("raw/unified.html", "카카오 통합 약관", "2022년 8월 25일", "html"),
        ("raw/unified_service.docx", "카카오 통합서비스약관", "2026년 5월 29일", "docx"),
    ]

    corpus = []
    counts = {}
    for path, doc, effective_date, kind in sources:
        if kind == "html":
            with open(path, encoding="utf-8") as source_file:
                source_text = source_file.read()
            articles = parse_html(source_text)
        else:
            source_text = read_docx_text(path)
            articles = parse_docx(source_text)
        assert effective_date in source_text, f"{doc}: 시행일자 missing"
        counts[doc] = len(articles)
        corpus.extend({"doc": doc, **article} for article in articles)
        print(f"[date] {doc}: {effective_date} OK")

    expected_names = {doc for _, doc, _, _ in sources}
    assert expected_names == {entry["doc"] for entry in corpus}
    assert counts["카카오계정 약관"] == 17
    assert counts["카카오 위치정보 이용약관"] == 16
    assert counts["카카오 통합서비스약관"] == 18
    print(
        "[counts] "
        f"카카오계정 약관={counts['카카오계정 약관']}, "
        f"카카오 위치정보 이용약관={counts['카카오 위치정보 이용약관']}, "
        f"카카오 통합 약관(unified.html)={counts['카카오 통합 약관']}, "
        f"카카오 통합서비스약관={counts['카카오 통합서비스약관']}"
    )

    assert all(len(entry["text"]) > 30 for entry in corpus)
    shortest = sorted(corpus, key=lambda entry: len(entry["text"]))[:5]
    for entry in shortest:
        print(f"[shortest] {entry['doc']}, {entry['article_no']}, {len(entry['text'])}")
    print(f"[total] articles={len(corpus)}, characters={sum(len(entry['text']) for entry in corpus)}")

    with open("gold_questions_public10.json", encoding="utf-8") as gold_file:
        gold = json.load(gold_file)
    corpus_by_key = {(entry["doc"], entry["article_no"]): entry for entry in corpus}
    missing = []
    percentages = []
    for question in gold["questions"]:
        pieces = []
        gold_text = []
        for reference in question["gold_articles"]:
            key = (reference["doc"], reference["article"])
            entry = corpus_by_key.get(key)
            ok = entry is not None and bool(entry["text"].strip())
            pieces.append(f"({key[0]}, {key[1]}) {'OK' if ok else 'MISSING'}")
            if ok:
                gold_text.append(entry["text"])
            else:
                missing.append((question["id"], key))
        print(f"[gold] {question['id']}: " + "; ".join(pieces))

        joined = "\n".join(gold_text)
        found = sum(fact_found(fact, joined) for fact in question["key_facts"])
        total = len(question["key_facts"])
        fraction = found / total if total else 1.0
        percentages.append(fraction)
        print(f"[facts] {question['id']}: {found}/{total} ({fraction:.1%})")

    assert not missing, f"gold article references missing: {missing}"
    print(f"[facts] overall mean: {sum(percentages) / len(percentages):.1%}")

    with open("corpus.json", "w", encoding="utf-8") as corpus_file:
        json.dump(corpus, corpus_file, ensure_ascii=False, indent=2)
        corpus_file.write("\n")
    print("[result] corpus.json written")


if __name__ == "__main__":
    main()
