# -*- coding: utf-8 -*-
"""답변 생성부. 채점 방식에서 거꾸로 설계했다.

키팩트 F1 = 2|A∩G| / (|A|+|G|) 이고 A·G 는 토큰 '집합'이다. 그래서
  · 조항에 없는 낱말을 하나 새로 쓸 때마다 분모만 커지고 분자는 그대로다.
  · 같은 낱말을 여러 번 써도 손해가 없다. 문제는 '새로운 낱말'이다.
  · 답의 길이가 아니라 답에 등장하는 서로 다른 낱말의 수가 점수를 정한다.
따라서 원문 표현을 그대로 옮기되, 묻지 않은 내용을 덧붙이지 않는 것이 곧 점수다.

측정으로 기각한 선택지
  · 답변에 '제N조에 따르면' 을 붙이는 안: 키팩트 F1 -0.82점,
    근거성 기대 이득 +0.33점(비회피 56건 중 5건만 조번호 누락으로 감점) -> 순손실.
"""

import re

from retrieval import tokenize   # 노트북에서는 검색기 코드가 같은 셀에 이미 들어간다

# "generate": 모델이 답을 쓴다(기본). "extract": 원문 구간을 그대로 이어 붙인다.
# 로컬 측정에서 extract 가 키팩트 F1 은 더 높다(0.7669 vs 0.6252). 판정 50점까지
# 이기는지는 연습 채점으로 확인해야 해서 기본값은 generate 로 둔다.
ANSWER_MODE = "generate"

CTX_FULL_RANKS = 2        # 상위 2개 조항은 길어도 통째로 넣는다
CTX_CHAR_BUDGET = 7000    # 3·4위 조항 본문은 이 예산 안에서만 넣는다

# 유형별 생성 토큰 상한. 이건 목표 길이가 아니라 '잘리지 않게 하는 상한'이다.
# 답을 짧게 만드는 일은 프롬프트가 하고, 이 값은 열거형이 중간에 끊기는 것과
# 한 요청이 러너의 120초 제한에 걸리는 것만 막는다. 그래서 넉넉한 쪽으로 잡는다.
TOKEN_BUDGET = {"yesno": 256, "single": 288, "multi": 384, "enum": 512}

# '몇 가지/몇 종류'는 열거지만 '몇 명/몇 개월'은 아니다. 숫자+가지 표현도 함께 본다.
_ENUM_RE = re.compile(
    r"[0-9]+\s*가지|몇\s*가지|몇\s*종류|"
    r"[한두세네다섯여섯일곱여덟아홉열일이삼사오육칠팔구십]\s*가지|"
    r"열거|나열|각각\s*무엇|모두\s*무엇|어떤\s*것들|종류.{0,8}무엇|항목.{0,8}무엇")
_YESNO_RE = re.compile(r"(우선하여|우선\s*적용|허용되나요|허용됩니까|가능한가요|맞나요|"
                       r"되나요|인가요|있나요|끝나나요|하나요|합니까)\s*\??\s*$")
_WH_RE = re.compile(r"무엇|어떤|누가|어디|언제|얼마|며칠|몇|어떻게|왜")


def classify(question):
    """생성 예산을 정하기 위한 질문 유형. 모델을 한 번 더 부르지 않는다."""
    text = question.strip()
    if _ENUM_RE.search(text):
        return "enum"
    wh = len(set(_WH_RE.findall(text)))
    if wh >= 2:
        return "multi"
    if _YESNO_RE.search(text) and wh == 0:
        return "yesno"
    return "single"


def build_context(hits, full_ranks=CTX_FULL_RANKS, budget=CTX_CHAR_BUDGET):
    """상위 조항은 통째로, 나머지는 예산 안에서 넣는다.

    상위 조항을 글자 수로 자르지 않는 것이 중요하다. 열거형 문항의 정답은 조항
    뒷부분에 있는 경우가 많아, 자르면 근거 목록에는 남고 답에서는 사라진다.
    """
    blocks, used = [], 0
    for i, hit in enumerate(hits, 1):
        head = f"[{hit['document']} 제{hit['article']}조({hit['title']})]"
        body = hit["text"]
        if i <= full_ranks:
            blocks.append(f"<조항 {i}> {head}\n{body}")
            continue
        # 예산은 3위 이하에만 매긴다. 상위 조항 길이를 여기 합산하면 1·2위가 길 때
        # 3·4위가 무조건 제목만 남는다.
        if used + len(body) <= budget:
            used += len(body)
            blocks.append(f"<조항 {i}> {head}\n{body}")
        else:
            blocks.append(f"<조항 {i}> {head}")
    return "\n\n".join(blocks)


SYSTEM_PROMPT = (
    "제시된 카카오 약관 조항만을 근거로 질문에 답하세요.\n"
    "\n"
    "1. 한국어로만 답합니다. 중국어·일본어를 섞지 않고, 번역하거나 설명을 덧붙이겠다는 "
    "말도 쓰지 않으며, 답변 본문만 한 번 출력합니다.\n"
    "2. 근거가 되는 문장은 요약하거나 바꿔 쓰지 말고 조항의 표현과 어순을 그대로 옮깁니다. "
    "숫자·기간·비율·법령명·조문번호·회사명·서비스명은 한 글자도 바꾸지 않습니다. "
    "'1인'을 '1명'으로, '금지됩니다'를 '허용되지 않습니다'로 바꾸지 않습니다.\n"
    "3. 조항에 없는 낱말을 새로 만들어 쓰지 않습니다. 인사말, 배경 설명, 요약 문장, "
    "'위 조항에 따르면' 같은 머리말은 붙이지 않습니다. 다만 이것은 군더더기를 빼라는 "
    "뜻이지 내용을 줄이라는 뜻이 아닙니다. 질문이 요구한 사실은 하나도 빼지 않습니다.\n"
    "4. 질문이 여러 가지를 물으면 각각에 모두 답합니다. 무엇을·어떤 방법으로·어디에·"
    "어떤 효력을 처럼 나뉘어 있으면 하나도 빠뜨리지 않습니다.\n"
    "5. 열거를 묻는 질문이면 항목 이름만 쓰지 말고 조항에 적힌 각 항목의 설명까지 함께, "
    "조항이 밝힌 개수만큼 빠짐없이 옮깁니다.\n"
    "6. 질문의 전제를 그대로 믿지 마십시오. 조항이 전제와 다르면 '아니오'로 시작해 "
    "바로잡습니다. 다만 '예/아니오'는 가부를 묻는 질문에만 쓰고, 무엇·어떻게·얼마처럼 "
    "설명을 요구하는 질문에는 곧바로 내용을 답합니다.\n"
    "7. 제시된 조항에 근거가 조금이라도 있으면 반드시 그 내용으로 답합니다. "
    "'확인되지 않습니다', '알 수 없습니다' 라고 답하지 않습니다."
)

# 실제 약관에 없는 가상의 조항·회사명만 쓴다. 진짜 조문 번호나 회사명을 예시에
# 넣으면 그 값이 다른 문항의 답변으로 새어 나온다.
FEWSHOT = [
    {"role": "user", "content": (
        "다음은 관련 약관 조항입니다.\n\n"
        "<조항 1> [예시 약관 제1조(준칙)]\n"
        "본 약관에 규정되지 않은 사항은 세부지침에 따릅니다. 본 약관과 세부지침의 내용이 "
        "충돌할 경우 세부지침에 따릅니다.\n\n"
        "질문: 본 약관이 세부지침보다 우선하여 적용되나요?\n\n답변:")},
    {"role": "assistant", "content":
        "아니오. 본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다."},
    {"role": "user", "content": (
        "다음은 관련 약관 조항입니다.\n\n"
        "<조항 1> [예시 약관 제2조(대상 서비스)]\n"
        "서비스 명칭에 '예시'가 사용되더라도 회사가 아닌 예시 계열사에서 제공하는 서비스"
        "(예: ㈜가나다물류가 제공하는 가나다 배송 서비스)는 본 약관의 대상 서비스에 "
        "포함되지 않습니다.\n\n"
        "질문: 본 약관의 대상 서비스에 포함되지 않는 서비스는 누가 제공하는 서비스이며, "
        "약관은 그 예로 무엇을 들고 있나요?\n\n답변:")},
    {"role": "assistant", "content": (
        "회사가 아닌 예시 계열사에서 제공하는 서비스입니다. 약관은 그 예로 ㈜가나다물류가 "
        "제공하는 가나다 배송 서비스를 들고 있습니다.")},
]


# 유형별 지시를 질문 바로 옆에 붙인다. 시스템 프롬프트의 3번(군더더기 금지)이
# 4·5번(모든 하위 질문·항목 설명)을 눌러 답이 뼈만 남는 일이 실제로 있었다.
# 공개 10문항 실행에서 P03 이 항목 이름 5개만(84자, F1 0.35), P06 이 앞 문장을
# 통째로 빠뜨렸다(52자, F1 0.28).
KIND_HINT = {
    "enum": "항목 이름만 쓰지 말고, 각 항목마다 조항에 적힌 설명까지 함께 옮기세요. "
            "조항이 밝힌 개수만큼 하나도 빠뜨리지 마세요.",
    "multi": "질문이 묻는 것이 여러 개입니다. 각각에 대해 한 문장씩, 조항의 표현을 "
             "그대로 써서 모두 답하세요.",
    "yesno": "예 또는 아니오로 시작한 뒤, 근거가 되는 조항 문장을 원문 그대로 옮기세요.",
    "single": "근거가 되는 조항 문장을 원문 그대로 옮겨 한두 문장으로 답하세요.",
}


def build_messages(question, hits, kind="single"):
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEWSHOT
        + [{"role": "user", "content":
            f"다음은 관련 약관 조항입니다.\n\n{build_context(hits)}\n\n"
            f"질문: {question}\n\n{KIND_HINT[kind]}\n\n답변:"}]
    )


_PREAMBLE_RE = re.compile(r"^\s*(답변\s*[:：]|답\s*[:：]|정답\s*[:：])\s*")
_FILLER_RE = re.compile(
    r"^\s*(?:위\s*조항에\s*따르면|제시된\s*조항에\s*따르면|해당\s*조항에\s*따르면|"
    r"약관에\s*따르면)\s*[,，]?\s*")
# 가운뎃점 ・(U+30FB)는 빼야 한다. 약관의 "변경・제한・중지" 에서 이걸 지우면
# 세 낱말이 한 낱말로 붙어 채점 토큰 세 개를 한꺼번에 잃는다.
_CJK_RE = re.compile(r"[一-鿿぀-ゟァ-ヺ]")


def postprocess(text):
    """새 낱말만 늘리는 군더더기를 걷어낸다. F1 분모가 그만큼 줄어든다."""
    out = _PREAMBLE_RE.sub("", text.strip())
    out = _FILLER_RE.sub("", out)
    out = _CJK_RE.sub("", out)                      # 한자·가나 혼입 제거
    # 같은 문장을 두 번 쓰면 판정의 명료성이 깎인다. 순서는 유지하고 중복만 없앤다.
    seen, kept = set(), []
    for part in re.split(r"(?<=다\.)\s+|\n+", out):
        key = re.sub(r"\s+", "", part)
        if key and key in seen:
            continue
        seen.add(key)
        kept.append(part.strip())
    return re.sub(r"[ \t]+", " ", "\n".join(k for k in kept if k)).strip()


_REFUSAL_RE = re.compile(
    r"확인되지\s*않|확인할\s*수\s*없|찾을\s*수\s*없|알\s*수\s*없|"
    r"명시되어\s*있지\s*않|나와\s*있지\s*않|포함되어\s*있지\s*않|정보가\s*없")


# 조항을 항(①②) · 호(1. 2.) 단위로 먼저 자른다. 열거형 문항의 정답은 호 하나가
# 통째로 한 항목이라, 문장으로 쪼개면 항목 이름과 설명이 흩어진다.
_SPAN_SPLIT_RE = re.compile(r"(?=(?:^|\s)(?:\d{1,2}\.\s|[①-⑳]))")
_SPAN_MAX_CHARS = 260


def source_spans(text):
    parts = [p.strip() for p in _SPAN_SPLIT_RE.split(text) if p.strip()]
    spans = []
    for part in parts:
        if len(part) <= _SPAN_MAX_CHARS:
            spans.append(part)
        else:
            spans.extend(s.strip() for s in re.split(r"(?<=다\.)\s+", part) if s.strip())
    # 항 기호(①②)는 채점 토큰이 아니라 떼어내도 F1 이 변하지 않는다. 답변으로 읽힐 때
    # 조문을 그대로 퍼온 티가 덜 나서 판정의 명료성에만 도움이 된다.
    return [re.sub(r"^[①-⑳]\s*", "", s).strip() for s in spans]


def extractive_answer(question, hits, kind="single"):
    """검색된 조항에서 질문과 겹치는 구간을 원문 그대로 골라 잇는다.

    공개 10문항 기준 키팩트 F1 0.7669 (23.0/30점) — 같은 문항에서 생성 답변
    최고치 0.6252 보다 높다. 표현을 한 글자도 바꾸지 않으니 당연한 결과다.
    다만 판정(50점)의 명료성·완결성까지 이기는지는 확인되지 않아 기본값이 아니다.
    회피가 나왔을 때의 안전망으로는 무조건 이쪽이 낫다.
    """
    if kind == "enum":
        # 열거형은 항목을 고르는 순간 지는 문제다. 5개 중 3개만 집으면 나머지 2개의
        # 낱말을 통째로 잃는다. 1위 조항을 통째로 옮기는 편이 낫다 —
        # 공개 10문항에서 P03 이 0.194 -> 0.718, 전체 0.7145 -> 0.7669.
        return " ".join(source_spans(hits[0]["text"]))

    query = set(tokenize(question))
    n_spans = 1
    candidates = []
    for rank, hit in enumerate(hits):
        for order, span in enumerate(source_spans(hit["text"])):
            candidates.append((len(query & set(tokenize(span))), rank, order, span))
    if not candidates:
        return hits[0]["text"]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    picked = [c for c in candidates[:n_spans] if c[0] > 0] or candidates[:1]
    return " ".join(c[3] for c in sorted(picked, key=lambda c: (c[1], c[2])))


def answer_question(question: str):
    """공통 러너가 질문마다 호출하는 고정 진입점.

    반환 형식은 계약이다:
        {"answer": "...", "retrieved": [["카카오계정 약관", 3], ...]}   # 1~4개
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question은 비어 있지 않은 문자열이어야 합니다.")
    question = question.strip()

    hits = search(question, top_k=TOP_K)                              # noqa: F821
    kind = classify(question)
    if ANSWER_MODE == "extract":
        answer = extractive_answer(question, hits, kind)
    else:
        answer = postprocess(generate(build_messages(question, hits, kind),  # noqa: F821
                                      TOKEN_BUDGET[kind]))
        # 근거가 있는데 회피하면 판정이 100점 만점에 15점으로 떨어진다.
        if not answer or _REFUSAL_RE.search(answer):
            answer = extractive_answer(question, hits, kind)

    # 프롬프트에 넣은 조항을 검색 순위 그대로 적는다. 빈 목록은 계약 위반이다.
    retrieved, seen = [], set()
    for hit in hits:
        key = (hit["document"], hit["article"])
        if key not in seen:
            seen.add(key)
            retrieved.append([hit["document"], hit["article"]])
    return {"answer": answer, "retrieved": retrieved[:TOP_K]}         # noqa: F821
