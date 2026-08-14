"""evaluator.py 를 단일 소스로 삼아 제출용 Colab 노트북을 만든다.

    python3 build_notebook.py <팀번호>     ->  evaluator_<팀번호>.ipynb

로직을 노트북에 손으로 옮겨 적지 않는다. evaluator.py 만 고치면 노트북이 따라온다.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = sys.argv[1] if len(sys.argv) > 1 else "TEAM"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.rstrip().splitlines(keepends=True)}


src = open(os.path.join(HERE, "evaluator.py"), encoding="utf-8").read()
# evaluator.py 의 셀 마커로 쪼갠다. 첫 조각은 markdown 헤더라 따로 뗀다.
chunks = [c for c in re.split(r"^# %%.*$", src, flags=re.M)]
chunks = chunks[1:] if not chunks[0].strip() else chunks   # 선두 빈 조각 제거
header = "\n".join(re.sub(r"^#\s?", "", l).rstrip() for l in chunks[0].strip().splitlines())
header = re.sub(r"^# .*\n", "", header).strip()            # 제목 줄은 아래에서 따로 붙인다
config_cell = chunks[1].strip()
logic_cell = "\n".join(c.strip() for c in chunks[2:]).strip()

cells = [
    md(f"# 학생 평가기 — 교차 평가 채점기 (팀 {TEAM})\n\n"
       + header
       + "\n\n**실행 순서**: 위에서부터 셀을 차례로 실행하세요. "
         "파일은 셀에서 직접 업로드하며 구글 드라이브를 마운트하지 않습니다."),

    md("## 1. 패키지 설치"),
    code("!pip install -q google-genai"),

    md("## 2. 설정\n\n판정 모델과 가중치. 발급받은 키가 서빙하는 정확한 model id 로 맞추세요."),
    code(config_cell),

    md("## 3. 평가기 본체\n\n"
       "문항 수·문항 번호·질문 내용은 코드에 없습니다. 전부 업로드한 파일에서 읽습니다."),
    code(logic_cell),

    md("## 4. 파일 업로드\n\n"
       "골드셋 1개 · 익명 답변 파일 5개 · (선택) 약관 파일을 한 번에 선택해 올리세요.\n"
       "파일명이 아니라 파일 내용을 보고 종류를 자동으로 구분합니다."),
    code("""from google.colab import files

uploaded = files.upload()
gold_path, answer_paths, term_paths = classify_uploads(list(uploaded))

print("골드셋 :", gold_path)
print("답변   :", answer_paths)
print("약관   :", term_paths or "(없음 — 판정은 핵심내용 기준으로만 진행)")
assert gold_path, "골드셋 파일을 찾지 못했습니다"
assert answer_paths, "답변 파일을 찾지 못했습니다\""""),

    md("## 5. API 키 입력\n\n키는 노트북에 저장되지 않습니다. 실행할 때마다 입력합니다."),
    code("""import getpass

api_key = getpass.getpass("Gemini API 키: ")
judge = make_gemini_judge(api_key)

# 연결 확인 — 1회만 호출
print("연결 확인:", judge("[질문]\\n확인\\n\\n[정답 핵심내용]\\n- 확인\\n\\n[답변] <<<확인>>>"))"""),

    md("## 6. 채점 실행\n\n"
       "문항을 순차로 판정합니다. 중간에 끊겨도 `judge_cache.json` 에 남은 결과는 재사용되므로 "
       "다시 실행하면 이어서 진행합니다."),
    code(f'out = run(gold_path, answer_paths, "eval_{TEAM}.json", judge, term_paths)\n'
         f'\nfor r in out["results"]:\n'
         f'    print(r["blind_id"], r["total"], r["status"])'),

    md("## 7. 제출 전 검증 및 다운로드"),
    code(f"""problems = validate_output(out)
if problems:
    print("형식 문제 발견:")
    for p in problems:
        print(" -", p)
else:
    print("형식 검증 통과 — 제출 가능")

files.download("eval_{TEAM}.json")"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = os.path.join(HERE, f"evaluator_{TEAM}.ipynb")
json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"생성: {out_path}  (셀 {len(cells)}개)")
