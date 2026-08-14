"""공통 러너(2번 셀)를 GPU 없이 로컬에서 그대로 돌려 계약 위반을 먼저 잡는다.

러너 소스는 손대지 않는다. 맥에는 /content/ 를 만들 수 없어 출력 경로 한 줄만
임시 디렉터리로 바꾸고, 답변은 실제 검색기 + 고정 문자열 스텁으로 채운다.
LLM 없이도 retrieved 정규화·문서명 대조·성능 루프·저장 형식이 전부 검증된다.

    .venv/bin/python test_runner_contract.py
"""
import json
import os
import re
import sys
import tempfile

NB = "[학생용]_결과기_개발_기본_틀_ipynb의_사본.ipynb"


def runner_source(output_dir, team="1"):
    nb = json.load(open(NB, encoding="utf-8"))
    src = "".join(nb["cells"][1]["source"])
    src, n1 = re.subn(r'_SP_TEAM = ""', '_SP_TEAM = "%s"' % team, src, count=1)
    src, n2 = re.subn(r'_SP_OUTPUT_DIR = "/content/"',
                      '_SP_OUTPUT_DIR = %r' % output_dir, src, count=1)
    src, n3 = re.subn(r"_SP_AUTO_DOWNLOAD = True", "_SP_AUTO_DOWNLOAD = False", src, count=1)
    assert n1 == n2 == n3 == 1, "러너 소스 패치 실패 — 배포본이 바뀌었는지 확인"
    return src


def build_stub():
    """실제 BM25 검색 + 근거 본문을 그대로 잘라 붙이는 답변 스텁."""
    from retrieval import Retriever, load_corpus

    corpus = load_corpus()
    by_key = {(a["doc"], a["article_no"]): a for a in corpus}
    retriever = Retriever(corpus)

    def answer_question(question: str):
        hits = retriever.search(question, top_k=3)
        parts = [by_key[(d, n)]["text"][:200] for d, n, _ in hits]
        return {"answer": " ".join(parts), "retrieved": [[d, n] for d, n, _ in hits]}

    return answer_question


def main():
    from fastapi import FastAPI, HTTPException
    import threading

    app = FastAPI()
    lock = threading.Lock()
    answer_question = build_stub()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/answer")
    def answer_api(payload: dict):
        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            raise HTTPException(status_code=400, detail="question must be a non-empty string")
        with lock:
            return answer_question(question.strip())

    out_dir = tempfile.mkdtemp(prefix="runner-")
    namespace = {"__name__": "__main__", "answer_question": answer_question, "app": app}
    exec(compile(runner_source(out_dir), "<cell2>", "exec"), namespace)

    path = os.path.join(out_dir, "answers_public_1.json")
    data = json.load(open(path, encoding="utf-8"))
    meta = data.get("meta") or {}
    assert data["team"] == "1"
    assert len(data["answers"]) == 10, len(data["answers"])
    assert not [a for a in data["answers"] if a.get("error")], "실패 문항 존재"
    assert not meta.get("doc_name_violations"), meta.get("doc_name_violations")
    assert not meta.get("timeout_qids"), meta.get("timeout_qids")
    for a in data["answers"]:
        assert isinstance(a["answer"], str) and a["answer"]
        assert 1 <= len(a["retrieved"]) <= 4
        for doc, art in a["retrieved"]:
            assert isinstance(art, int), (doc, art)
    print("\n[계약 검증 통과] 파일:", path)
    print("  문항", len(data["answers"]), "· 문서명위반", len(meta.get("doc_name_violations") or []),
          "· 타임아웃", len(meta.get("timeout_qids") or []))
    print("  성능:", json.dumps(
        {k: v for k, v in (meta.get("performance") or {}).items()
         if k in ("success", "fail", "throughput_rps", "p50_latency_s", "p95_latency_s")},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
