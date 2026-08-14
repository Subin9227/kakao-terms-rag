# -*- coding: utf-8 -*-
"""제출용 2셀 노트북을 조립한다.

1번 셀 = cell1_template.py 에 약관 원문(terms.json), 검색기(retrieval.py),
         생성부(generation.py)를 끼워 넣은 것.
2번 셀 = 운영진 공통 러너를 기존 노트북에서 그대로 가져오고 _SP_TEAM 한 줄만 바꾼 것.

검색기와 생성부를 따로 파일로 두는 이유는 로컬에서 그대로 import 해서 측정하기
위해서다. 제출본에는 한 셀로 합쳐져 들어간다.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RUNNER_SOURCE = os.path.join(ROOT, "result_generator_qwen7b_4bit_t4.ipynb")
TEAM = "9"


def module_body(path):
    """모듈에서 셸로 옮길 본문만 뽑는다 — 맨 앞 인코딩 줄과 독스트링은 뺀다."""
    with open(path, encoding="utf-8") as fp:
        text = fp.read()
    text = re.sub(r"^#\s*-\*-.*?-\*-\s*\n", "", text)
    text = re.sub(r'^\s*"""(?:.|\n)*?"""\s*\n', "", text, count=1)
    # 로컬 측정용 import. 노트북에서는 같은 셀 안에 이미 정의돼 있다.
    text = re.sub(r"(?m)^from (retrieval|local_eval) import .*\n", "", text)
    return text.strip("\n")


def build_cell1():
    with open(os.path.join(HERE, "cell1_template.py"), encoding="utf-8") as fp:
        template = fp.read()
    with open(os.path.join(HERE, "terms.json"), encoding="utf-8") as fp:
        articles = json.load(fp)

    # r''' ... ''' 안에 넣는다. raw 라서 json 이 넣은 역슬래시 이스케이프가 그대로 산다.
    # 원문을 한 글자도 바꾸지 않는 것이 목적이다 — 채점이 표현 일치로 이루어지기 때문이다.
    payload = json.dumps(articles, ensure_ascii=False, separators=(",", ":"))
    assert "'''" not in payload and not payload.endswith("\\"), "약관 원문에 문자열 리터럴을 깨는 문자가 있음"

    retrieval = module_body(os.path.join(HERE, "retrieval.py"))
    # import 와 정규식 상수는 템플릿이 이미 갖고 있거나 여기서만 쓴다. 그대로 옮긴다.
    generation = module_body(os.path.join(HERE, "generation.py"))

    cell = template.replace("__TERMS_JSON__", payload)
    cell = cell.replace("__RETRIEVAL__", retrieval)
    cell = cell.replace("__GENERATION__", generation)
    for marker in ("__TERMS_JSON__", "__RETRIEVAL__", "__GENERATION__"):
        assert marker not in cell, f"치환되지 않은 자리표시자: {marker}"
    return cell


def build_cell2():
    with open(RUNNER_SOURCE, encoding="utf-8") as fp:
        source = "".join(json.load(fp)["cells"][1]["source"])
    line = '_SP_TEAM = ""          # 예: "1"  ← 운영진이 알려준 팀 식별자(숫자)를 그대로 적습니다'
    assert line in source, "공통 러너에서 _SP_TEAM 줄을 찾지 못했습니다"
    return source.replace(line, line.replace('_SP_TEAM = ""', f'_SP_TEAM = "{TEAM}"'), 1)


def main():
    cell1, cell2 = build_cell1(), build_cell2()
    notebook = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": cell1.splitlines(keepends=True)},
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": cell2.splitlines(keepends=True)},
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    dest = os.path.join(ROOT, f"result_generator_{TEAM}.ipynb")
    with open(dest, "w", encoding="utf-8") as fp:
        json.dump(notebook, fp, ensure_ascii=False, indent=1)

    compile(cell1, "cell1", "exec")          # 문법 오류를 여기서 잡는다
    compile(cell2, "cell2", "exec")
    print(f"저장: {dest}")
    print(f"  1번 셀 {len(cell1):,}자 / {len(cell1.splitlines())}줄")
    print(f"  2번 셀 {len(cell2):,}자 / {len(cell2.splitlines())}줄  (_SP_TEAM=\"{TEAM}\")")


if __name__ == "__main__":
    main()
