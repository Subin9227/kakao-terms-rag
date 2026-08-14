# -*- coding: utf-8 -*-
import base64
import copy
import gzip
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORPUS_PATH = ROOT / "corpus.json"
CELL1_PATH = ROOT / "cell1.py"
TEMPLATE_PATH = ROOT / "[학생용]_결과기_개발_기본_틀_ipynb의_사본.ipynb"
TEAM = "9"   # 운영진이 알려준 팀 식별자. 2번 셀에서 고쳐도 되는 유일한 줄이다.
# 제출 파일명은 운영진 규칙을 그대로 따른다.
OUTPUT_PATH = ROOT / ("result_generator_%s.ipynb" % TEAM)
BLOB_RE = re.compile(r'(?m)^CORPUS_BLOB = "[^"\n]*"$')


def _cell_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _corpus_blob(corpus):
    compact = json.dumps(corpus, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(gzip.compress(compact, mtime=0)).decode("ascii")


def main():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    blob = _corpus_blob(corpus)
    cell1_source = CELL1_PATH.read_text(encoding="utf-8")
    updated_source, replacements = BLOB_RE.subn(
        lambda _match: "CORPUS_BLOB = " + json.dumps(blob), cell1_source, count=1
    )
    assert replacements == 1, "cell1.py must contain exactly one CORPUS_BLOB assignment"
    if updated_source != cell1_source:
        CELL1_PATH.write_text(updated_source, encoding="utf-8")
    cell1_source = updated_source

    # 로컬 측정은 prompt.py 로, 제출은 cell1.py 로 돈다. 둘이 어긋나면 재 본 적 없는
    # 프롬프트를 제출하게 되므로 여기서 막는다.
    import prompt

    rules = re.search(r'ANSWER_RULES = """(.*?)"""', cell1_source, re.S)
    assert rules, "cell1.py에서 ANSWER_RULES를 찾지 못했습니다"
    matched = [name for name, text in prompt.PROMPTS.items() if text == rules.group(1)]
    assert matched, "cell1.py의 ANSWER_RULES가 prompt.py의 어떤 판본과도 일치하지 않습니다"
    print("prompt variant: {}".format(matched[0]))

    template = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    assert len(template.get("cells", [])) == 2, "template must contain exactly two cells"
    original_cell2 = template["cells"][1]
    assert original_cell2.get("cell_type") == "code"
    cell2_text = _cell_text(original_cell2)
    cell2_text, replaced = re.subn(r'^_SP_TEAM = ""', '_SP_TEAM = "%s"' % TEAM,
                                   cell2_text, count=1, flags=re.M)
    assert replaced == 1, "2번 셀에서 _SP_TEAM 한 줄을 찾지 못했습니다"
    original_cell2 = copy.deepcopy(original_cell2)
    original_cell2["source"] = [cell2_text]
    output = copy.deepcopy(template)
    output["cells"] = [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [cell1_source],
        },
        copy.deepcopy(original_cell2),
    ]
    assert len(output["cells"]) == 2
    assert all(cell["cell_type"] == "code" for cell in output["cells"])
    # 2번 셀은 _SP_TEAM 한 줄 외에는 배포본과 같아야 한다.
    assert _cell_text(output["cells"][1]) == cell2_text
    template_lines = _cell_text(template["cells"][1]).splitlines()
    output_lines = cell2_text.splitlines()
    diff = [i for i, (a, b) in enumerate(zip(template_lines, output_lines)) if a != b]
    assert len(template_lines) == len(output_lines) and len(diff) == 1 \
        and template_lines[diff[0]].startswith("_SP_TEAM ="), \
        "2번 셀이 _SP_TEAM 외의 줄에서 달라졌습니다: %s" % diff
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print("output path: {}".format(OUTPUT_PATH))
    print("cell 1 size: {} bytes".format(len(cell1_source.encode("utf-8"))))
    print("cell 2 size: {} bytes".format(len(cell2_text.encode("utf-8"))))
    print("blob size: {} bytes".format(len(blob.encode("ascii"))))


if __name__ == "__main__":
    main()
