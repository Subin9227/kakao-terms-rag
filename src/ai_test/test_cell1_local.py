"""제출용 cell1.py를 실제로 실행해 공통 러너와 끝까지 붙여 본다.

맥에는 CUDA가 없으므로 디바이스 지정 세 줄만 MPS로 바꿔 exec 한다. 바꾸는 부분을
명시적으로 세고 하나라도 어긋나면 즉시 실패시켜, "대충 돌려보고 통과"를 막는다.
검증되는 것은 계약(반환 형식·문서명·1~4개·러너 저장)이며, 속도는 T4에서 따로 잰다.

    .venv/bin/python test_cell1_local.py
"""
import json
import os
import sys
import tempfile

from test_runner_contract import runner_source

PATCHES = [
    ('DEVICE = torch.device("cuda:0")', 'DEVICE = torch.device("mps")'),
    ('    device_map={"": 0},\n', ''),
    ('torch.cuda.get_device_name(0)', '"MPS(local)"'),
]


def load_cell1():
    source = open("cell1.py", encoding="utf-8").read()
    for old, new in PATCHES:
        assert old in source, "cell1.py에서 패치 대상을 찾지 못했다: %r" % old
        source = source.replace(old, new)
    namespace = {"__name__": "cell1_local"}
    exec(compile(source, "cell1.py", "exec"), namespace)
    model = namespace["GENERATOR_MODEL"]
    model.to("mps")
    return namespace


def main():
    ns = load_cell1()
    answer_question = ns["answer_question"]
    app = ns["app"]

    out_dir = tempfile.mkdtemp(prefix="cell1-")
    runner_ns = {"__name__": "__main__", "answer_question": answer_question, "app": app}
    exec(compile(runner_source(out_dir), "<cell2>", "exec"), runner_ns)

    path = os.path.join(out_dir, "answers_public_1.json")
    data = json.load(open(path, encoding="utf-8"))
    meta = data.get("meta") or {}
    allowed = set(ns["OFFICIAL_DOCUMENT_NAMES"])
    assert len(data["answers"]) == 10
    assert not [a for a in data["answers"] if a.get("error")], "실패 문항 존재"
    assert not meta.get("doc_name_violations"), meta["doc_name_violations"]
    for a in data["answers"]:
        assert isinstance(a["answer"], str) and a["answer"].strip()
        assert 1 <= len(a["retrieved"]) <= 4
        for doc, art in a["retrieved"]:
            assert doc in allowed, doc
            assert isinstance(art, int)
    json.dump(data, open("answers_local_cell1.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[cell1 계약 검증 통과] 답변 10건 · 문서명위반 0 · 저장:", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
