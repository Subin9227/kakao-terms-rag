# -*- coding: utf-8 -*-
# =====================================================================================
#  결과기 1번 셀 — 팀 구현
# =====================================================================================
#  ┌─ 유지하는 계약 ────────────────────────────────────────────────────────────────┐
#  │ · answer_question(question: str) 함수 이름과 입력 형식                         │
#  │ · 반환값: {"answer": 문자열, "retrieved": [[문서명, 조번호], ...]} 1~4개        │
#  │ · 전역 FastAPI app, GET /health, POST /answer                                 │
#  │ · Qwen2.5-Instruct 계열 생성 모델을 Colab T4 에서 로컬 실행                    │
#  │ · 새 Colab T4 런타임에서 위에서 아래로 한 번 실행하면 끝난다                   │
#  └────────────────────────────────────────────────────────────────────────────────┘
#
#  설계
#    · 약관 4종 72개 조를 이 셀에 문자열로 담는다. 채점 당일 kakao.com 이 막히거나
#      원문이 바뀌어도 결과가 흔들리지 않는다. 드라이브·업로드·크롤링을 쓰지 않는다.
#    · 청크는 조(條) 단위다. retrieved 계약이 [문서명, 조번호] 라 조보다 잘게 쪼개면
#      근거를 되짚을 수 없고, 더 크게 묶으면 열거형 문항에서 뒷부분을 잃는다.
#    · 검색은 BM25(어절+음절 bigram) 와 임베딩을 순위 융합(RRF)한다. 질문이 약관
#      이름을 직접 부르면 그 약관 쪽에 가산점을 준다.
#    · 프롬프트에 넣는 조항과 retrieved 에 적는 조항을 같게 맞춘다.
#    · 문항 수·문항 번호를 코드에 넣지 않는다. 질문 문자열만 받아 처리한다.
# =====================================================================================


# -------------------------------------------------------------------------------------
# 0. 고정 기준 — 문서명과 생성 모델 계열
# -------------------------------------------------------------------------------------
OFFICIAL_DOCUMENT_NAMES = (
    "카카오계정 약관",
    "카카오 위치정보 이용약관",
    "카카오 통합서비스약관",
    "카카오 통합 약관",
)

REQUIRED_GENERATION_MODEL_FAMILY = "Qwen2.5-Instruct"


# =====================================================================================
# 1. 팀별 자유 구현 영역
# =====================================================================================
import gc
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter

# torch 를 import 하기 전에 잡아야 효과가 있다. 긴 프롬프트에서 생기는 메모리
# 단편화로 OOM 이 나는 것을 줄인다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# True: 7B 4bit (품질) / False: 3B fp16 (안정). T4 는 Turing 이라 bf16 을 못 쓴다.
USE_7B = True

# accelerate 는 device_map 로드에 필요하고, bitsandbytes 는 4bit 양자화에 필요하다.
# -U 를 붙이지 않으므로 이미 깔려 있으면 그대로 두고 torch 도 건드리지 않는다.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "accelerate"], check=False)
if USE_7B:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

GEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" if USE_7B else "Qwen/Qwen2.5-3B-Instruct"
EMB_MODEL_ID = "BAAI/bge-m3"          # 런타임 안에서 로컬 실행. 원격 API 가 아니다.

TOP_K = 4                 # retrieved 계약 상한. MRR 은 첫 적중 순위만 보므로 꽉 채운다.
DOC_NAME_BOOST = 0.5      # 질문이 약관 이름을 부를 때 그 약관에 주는 가산점 비율
SEED = 42                 # 컨텍스트 예산과 생성 상한은 1-4 답변 조립부에 있다

torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ----- 1-1. 약관 원문 4종 72조 -------------------------------------------------------
# 출처: 카카오계정 약관(2026-05-29), 카카오 위치정보 이용약관(2026-07-16),
#       카카오 통합서비스약관(2026-05-29), 카카오 통합 약관(2022-08-25 아카이브).
ARTICLES = json.loads(r'''__TERMS_JSON__''')

_counts = Counter(a["document"] for a in ARTICLES)
_EXPECTED = {"카카오계정 약관": 17, "카카오 통합서비스약관": 18,
             "카카오 통합 약관": 21, "카카오 위치정보 이용약관": 16}
assert set(_counts) == set(OFFICIAL_DOCUMENT_NAMES), f"문서명 불일치: {sorted(_counts)}"
for _doc, _n in _EXPECTED.items():
    assert _counts[_doc] == _n, f"{_doc}: 조 {_n}개여야 하는데 {_counts[_doc]}개"
for _doc in OFFICIAL_DOCUMENT_NAMES:
    _nums = sorted(a["article"] for a in ARTICLES if a["document"] == _doc)
    assert _nums == list(range(1, len(_nums) + 1)), f"{_doc}: 조번호가 끊김 {_nums}"
print(f"[인덱스] 조항 {len(ARTICLES)}개 — "
      + " / ".join(f"{d} {_counts[d]}" for d in OFFICIAL_DOCUMENT_NAMES))


# ----- 1-2. 검색 ---------------------------------------------------------------------
__RETRIEVAL__


BM25_INDEX = Index(ARTICLES, josa=False, bigram=True)
print("[인덱스] BM25 준비 완료")

# 임베딩은 BM25 가 놓치는 어휘 불일치 문항을 위한 보강이다. 없어도 결과기는 동작해야
# 하므로, 내려받기나 로드가 실패하면 BM25 단독으로 조용히 물러난다.
DENSE = None
try:
    _emb_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_ID)
    _emb_model = AutoModel.from_pretrained(EMB_MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()

    @torch.inference_mode()
    def _embed(texts, max_length=2048, batch_size=4):
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = _emb_tokenizer(
                texts[i:i + batch_size], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(_emb_model.device)
            hidden = _emb_model(**batch).last_hidden_state[:, 0]   # BGE 계열은 CLS 풀링
            vectors.append(torch.nn.functional.normalize(hidden, p=2, dim=-1))
        return torch.cat(vectors)

    _CHUNK_VECTORS = _embed([Index._text(a) for a in ARTICLES]).float().to("cpu")
    # 인덱싱이 끝나면 임베딩 모델을 CPU 로 내린다. 질의는 한 번에 한 문장이라
    # CPU 로도 금방 끝나고, 생성 모델에 VRAM 을 더 내줄 수 있다.
    _emb_model = _emb_model.float().to("cpu")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    DENSE = (_embed, _CHUNK_VECTORS)
    print(f"[인덱스] 임베딩 준비 완료 {tuple(_CHUNK_VECTORS.shape)} (모델은 CPU 로 이동)")
except Exception as exc:                                          # noqa: BLE001
    # 실패한 임베딩 모델이 GPU 에 남아 있으면 바로 뒤 7B 로드가 OOM 난다.
    DENSE = None
    for _name in ("_emb_model", "_CHUNK_VECTORS"):
        globals().pop(_name, None)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"[인덱스] 임베딩 사용 안 함 ({type(exc).__name__}: {exc}) — BM25 단독으로 진행")


def search(question, top_k=TOP_K, rrf_k=60):
    """BM25 와 임베딩의 '순위'를 융합한다.

    점수를 직접 더하지 않는 이유는 BM25 점수와 코사인 유사도가 스케일이 달라
    정규화 방식에 따라 결과가 요동치기 때문이다. RRF 는 그 문제를 피한다.
    """
    global DENSE
    bm25_order = BM25_INDEX.rank(question, doc_boost=DOC_NAME_BOOST)
    dense_order = None
    if DENSE is not None:
        try:
            embed, vectors = DENSE
            scores = (vectors @ embed([question], max_length=512)[0].float()).tolist()
            scores = BM25_INDEX.apply_doc_boost(question, scores, DOC_NAME_BOOST)
            dense_order = sorted(range(len(ARTICLES)), key=lambda i: (-scores[i], i))
        except Exception as exc:                                  # noqa: BLE001
            # 질의 한 번이 실패해도 답변은 나와야 한다. 이후 질문은 BM25 로만 간다.
            DENSE = None
            print(f"[검색] 임베딩 질의 실패 ({type(exc).__name__}: {exc}) — BM25 단독으로 전환")
    if dense_order is None:
        order = bm25_order[:top_k]
    else:
        fused = {}
        for ranking in (bm25_order, dense_order):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        # 동점은 인덱스 순으로 고정한다. 실행마다 결과가 달라지지 않게 한다.
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
    return [ARTICLES[i] for i in order]


# ----- 1-3. 생성 모델 ----------------------------------------------------------------
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
if USE_7B:
    # 7B fp16 은 15GB 라 T4(16GB)에 들어가지 않는다. 4bit 로 올린다.
    _load_kwargs = dict(quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,   # T4 는 Turing 이라 bf16 미지원
    ))
else:
    _load_kwargs = dict(torch_dtype=DTYPE)

def _load_generator(model_id, load_kwargs):
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,          # "auto" 를 쓰지 않는다. CPU 로 조용히 흘러가면 한 요청이 분 단위가 된다.
        attn_implementation="sdpa", # FlashAttention2 는 Ampere 이상 전용. T4 에서는 sdpa.
        **load_kwargs,
    ).eval()


try:
    gen_model = _load_generator(GEN_MODEL_ID, _load_kwargs)
except Exception as exc:                                          # noqa: BLE001
    # bitsandbytes 설치 실패나 VRAM 부족으로 4bit 로드가 깨져도 셀은 끝까지 실행돼야
    # 한다. 3B fp16 은 T4 에 확실히 들어간다.
    print(f"[모델] {GEN_MODEL_ID} 로드 실패 ({type(exc).__name__}: {exc}) — 3B fp16 으로 내려갑니다")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()   # 실패한 7B 할당을 비우지 않으면 3B 도 못 올라간다
    GEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    gen_model = _load_generator(GEN_MODEL_ID, dict(torch_dtype=DTYPE))
if gen_tokenizer.pad_token_id is None:
    gen_tokenizer.pad_token = gen_tokenizer.eos_token


@torch.inference_mode()
def generate(messages, max_new_tokens):
    text = gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gen_tokenizer(text, return_tensors="pt").to(DEVICE)
    output = gen_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,            # 그리디 — 실행마다 같은 답이 나온다
        repetition_penalty=1.05,
        pad_token_id=gen_tokenizer.pad_token_id,
    )
    out = gen_tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    del inputs, output
    if DEVICE == "cuda":
        torch.cuda.empty_cache()    # 문항 간 메모리 누적을 막는다
    return out


# ----- 1-4. 답변 조립 ----------------------------------------------------------------
__GENERATION__


# 워밍업 — CUDA 커널 초기화 비용을 첫 채점 요청이 아니라 여기서 태운다.
_warm = answer_question("이 약관의 목적은 무엇인가요?")
assert isinstance(_warm["answer"], str) and 1 <= len(_warm["retrieved"]) <= 4
print(f"[모델] {GEN_MODEL_ID} 로드 및 워밍업 완료 — 근거 {_warm['retrieved']}")


# =====================================================================================
# 2. 고정 FastAPI 연결 영역 — 삭제하거나 경로를 바꾸지 않습니다
# =====================================================================================
# 2번 공통 러너는 아래 app 을 localhost 에서 실행하고 다음 주소를 호출합니다.
#   · GET  /health : 결과기 서버 준비 여부 확인
#   · POST /answer : {"question": "..."}을 보내 answer_question() 결과 수신
#
# 동시 요청에서 하나의 GPU 생성 모델이 충돌하지 않도록 Lock 을 사용합니다.
import threading


def _install_server_packages():
    """공통 러너와 연결하는 데 필요한 가벼운 서버 패키지만 설치합니다."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn"],
        check=True,
    )


_install_server_packages()

from fastapi import FastAPI, HTTPException  # noqa: E402


app = FastAPI(title="KTB AI Performance Result Generator")
_GENERATION_LOCK = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/answer")
def answer_api(payload: dict):
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=400, detail="question must be a non-empty string")
    with _GENERATION_LOCK:
        return answer_question(question.strip())


print("[1번 셀 준비] 2번 공통 러너를 실행하세요.")
