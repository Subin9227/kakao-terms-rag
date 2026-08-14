# Claude 대화 (2/2) — codex 3라운드 토론, 노트북 조립, 실행 결과 비교

`claude_01.md` 에서 이어짐.

## 2026-08-10 (계속)

---

[Claude]

**Phase 1 — codex 토론.** `codex exec --sandbox read-only -m gpt-5.6-luna -c model_reasoning_effort=max`
로 3라운드를 돌렸습니다. 프롬프트와 응답은 `ai-test-codex/v2/debate/` 에 전부 남아 있습니다.

라운드 1 안건 5개: ① 약관 내장 vs 런타임 크롤링 ② 7B-4bit vs 3B-fp16 ③ 답변에 조항번호 인용
④ 답변 길이 정책 ⑤ retrieved 개수.

**codex 라운드 1 응답 (요지, 원문은 `debate/round1_codex.md`)**

> Recommendation: embed the pinned 72-article corpus, use josa-stripped syllable-bigram BM25,
> keep Qwen2.5-7B-Instruct 4-bit, pass all four selected articles to the model, and add only a
> minimal `제N조` citation.
>
> Preserve the raw PDF spacing in the prompt because the reconstructed keyfacts contain
> artifacts such as `2시간 이 상`, `따 릅니다`.
>
> The judge gain from grounding 7→10 is `50 × 0.25 × (3 / 10) = 3.75 points`.
> For P01, adding one non-gold token `제10조` gives `32 / 33 = 0.9697`, so the F1 loss is only
> `0.0303` for that question.
>
> The current mismatch is risky: retrieval returns four, but generation sees only `hits[:2]`.
> A correct article at rank 3 or 4 can receive MRR credit while being unavailable to the answer generator.
>
> On a true zero-hit case, do not return `retrieved: []`; the current code violates the 1–4 contract there.
>
> Biggest omitted issue: This is fundamentally a source-span selection problem, not ordinary
> free-form generation. The highest-upside experiment is a copy-first path: split retrieved
> articles into numbered clauses, have Qwen select clause IDs, and concatenate the original clauses.

---

[Claude]

codex 라운드1 수신. 핵심 지적을 검증했습니다 — 특히 "BM25만으로 H14·H15를 놓친다"는 주장.

먼저 인용 삽입의 실제 이득을 60개 관측으로 직접 측정했습니다.

```
$ python3 (grounding vs 인용 여부)
인용 있음 17건 grounding 평균 9.47
인용 없음 43건 grounding 평균 8.72

grounding<10 인 경우:
  T12 P08 grounding=7 인용=False acc=10  34자
  T13 P06 grounding=7 인용=False acc=10  81자
  T13 P08 grounding=7 인용=False acc=10  35자
  T17 P08 grounding=7 인용=True  acc=7  214자
  T17 P09 grounding=7 인용=False acc=10  84자
  T17 P10 grounding=7 인용=True  acc=7  137자
  T2  P09 grounding=7 인용=True  acc=9  245자
  T5  P05 grounding=0 인용=False acc=0   18자
  T5  P06 grounding=7 인용=False acc=10  81자
  T5  P08 grounding=0 인용=False acc=0   19자
  T5  P09 grounding=0 인용=False acc=0   18자
  T5  P10 grounding=0 인용=False acc=0   18자

grounding 분포: Counter({10: 48, 7: 8, 0: 4})
```

인용 없이도 grounding 10이 48건입니다. 회피 4건(전부 5팀)을 빼면 비회피 56건 중 조항번호
누락으로 감점된 것은 5건뿐입니다. 그리고 인용을 넣은 3건은 인용에도 불구하고 7점입니다.

F1 비용을 직접 쟀습니다.

```
$ python3 (인용 삽입 F1 비용)
T12: 원본 F1 0.6632 | 제N조 0.6360 (-0.815점) | 문서명+제N조 0.6085 (-1.640점)
T13: 원본 F1 0.6276 | 제N조 0.5997 (-0.838점) | 문서명+제N조 0.5708 (-1.705점)

grounding 기대 이득: 비회피 56건 중 5건이 조번호 누락으로 7점 -> 8.9% x 3.75점 = +0.33점
```

**측정 결론: 인용 삽입은 −0.82점(F1) vs +0.33점(grounding) = 순손실.** codex의 산술은
3.75점이 모든 문항에 적용된다고 가정했는데, 실제로는 8.9%에만 적용됩니다. 권고를 기각합니다.

---

[Claude]

라운드 2 반론을 작성해 보냈습니다. 정정 4건.

1. PDF 공백 아티팩트는 제 버그였고 이미 고쳤다. 키팩트 26/26, eric fact 54/54가 정제본에 원문 그대로 존재한다.
2. 조사 제거는 이득이 아니라 손해다(0.8380 → 0.8333, top4 18/18 → 17/18).
3. 네가 놓친 가장 싼 검색 개선이 있다 — 문서명 부스트로 검증 MRR 0.8380 → 0.9074.
4. 남은 비1위 3건이 전부 어휘 불일치형이므로, 네가 기각한 BGE-M3가 오히려 정당화된다(상한 0.0926 × 20 = 1.85점).

**codex 라운드 2 응답 (원문은 `debate/round2_codex.md`)**

> Concede all four corrections. Ship cleaned text, `josa=False + bigram=True`, `doc_boost=0.5`,
> and BGE-M3 as an optional local ranker. Round 1's raw-spacing and josa recommendations were wrong.
>
> One implementation footgun: `Index` still defaults to `josa=True` and `doc_boost=0.0`; pass both explicitly.
>
> I concede. The residual cases are exactly where dense retrieval is justified, and the observed
> ceiling is about `0.0926 × 20 = 1.852` MRR points.
>
> The single disqualification-level risk is violating the fresh-runtime boundary: using a remote
> embedding endpoint or relying on a preloaded BGE cache, Drive file, or external artifact.
>
> [copy-first 단일 패스 설계 제안]
> ```
> TYPE=DIRECT
> CLAUSES=R1C2,R1C3
> ```
> 파서가 위 형식만 받아들이고, 벗어나면 결정적 폴백으로 간다.

`Index` 기본값을 측정 최적값(josa=False, doc_boost=0.5)으로 바꿨습니다. 재측정 결과 동일합니다.

---

[Claude]

codex의 copy-first 아이디어가 옳은지 모델 없이 결정적으로 재 봤습니다. 조항을 항(①②)·호(1. 2.)
단위로 자르고, 질문과 토큰이 겹치는 구간을 골라 원문 그대로 이어붙입니다.

```
$ python3 (추출식 상한 측정)
span  1개(열거는 x3): 키팩트 F1 0.7145  (21.44점)
span  2개(열거는 x3): 키팩트 F1 0.5761  (17.28점)
span  3개(열거는 x3): 키팩트 F1 0.4866  (14.60점)
span  4개(열거는 x3): 키팩트 F1 0.4247  (12.74점)
span  6개(열거는 x3): 키팩트 F1 0.3279  (9.84점)

참고 — 우리 최고 실측 0.6252(18.76점) / 12팀 0.6723(20.17점)
```

**모델을 전혀 쓰지 않은 추출식이 우리 최고 생성 답변과 12팀을 모두 앞섭니다.**
`ANSWER_MODE = "extract"` 로 넣고, 회피가 나왔을 때의 안전망으로도 씁니다.

열거형이 약점이라 고쳤습니다.

```
$ python3 (열거 전략 비교)
현재(3구간)      전체 F1 0.7145 (21.44점)  P03 0.1942
all_top1        전체 F1 0.7669 (23.01점)  P03 0.7182
items_top1      전체 F1 0.7622 (22.87점)  P03 0.6707
```

열거형은 항목을 고르는 순간 지는 문제입니다. 5개 중 3개만 집으면 나머지 2개의 낱말을 통째로
잃습니다. 1위 조항을 통째로 옮기는 쪽으로 바꿨습니다.

```
$ python3 (출하 코드 extract 모드 채점)
extract: MRR 1.0000 · F1 0.7669 · 객관 43.008/50
  P01:0.97 P02:0.62 P03:0.72 P04:0.74 P05:0.90 P06:0.78 P07:0.51 P08:1.00 P09:0.75 P10:0.68
```

일반화도 봤습니다.

```
검증 18문항 사실 포함률(extract): 0.741
  !! H11 0/2 (0.00)   97자     ← 검색 2위
  !! H14 0/5 (0.00)  502자     ← 검색 3위
  !! H15 0/3 (0.00)  246자     ← 검색 2위
  (나머지 12문항 1.00)
```

**추출식의 약점이 드러납니다. 1위 조항이 틀리면 통째로 틀립니다.** 생성 방식은 4개 조항을
다 보므로 이 위험이 없습니다. 그래서 기본값은 `generate` 로 두었습니다.

---

[Claude]

**Phase 2 — 노트북 조립.** `v2/build_notebook.py` 가 `cell1_template.py` 에 약관(terms.json),
검색기(retrieval.py), 생성부(generation.py)를 끼워 넣어 2셀 노트북을 만듭니다. 2번 셀은
기존 노트북에서 그대로 가져오고 `_SP_TEAM` 한 줄만 바꿉니다.

```
$ python3 v2/build_notebook.py
저장: result_generator_9.ipynb
  1번 셀 77,718자 / 590줄
  2번 셀 23,375자 / 565줄  (_SP_TEAM="9")
```

```
$ python3 (계약·규정 검사)
셀 수 2 | 둘 다 code True
2번 셀 원본 대비 다른 줄: 1
  - _SP_TEAM = ""          # 예: "1"  ← ...
  + _SP_TEAM = "9"          # 예: "1"  ← ...
2번 셀 줄 수 동일: True
드라이브: 없음 / torch.compile: 없음 / input(): 없음 / 네트워크 호출: 없음 / pip -U: 없음 / 문항번호 하드코딩: 없음
Qwen2.5 계열: ['Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-3B-Instruct']
answer_question 서명: def answer_question(question: str)
```

```
$ python3 v2/test_offline.py
문항 28개 × 생성 시나리오 4가지
유형 분류: {'single': 10, 'enum': 5, 'multi': 10, 'yesno': 3}
검색 MRR(전체 28문항) 0.9405
계약 위반 0건
경계 입력 7가지 × 2시나리오 통과
컨텍스트 최댓값 6,830자
test_offline OK
```

노트북에 들어간 코드가 로컬에서 시험한 코드와 같은지도 확인했습니다.

```
retrieval.py: 노트북에 그대로 포함
generation.py: 노트북에 그대로 포함
terms.json 그대로 포함
노트북에서 되읽은 조항 수 72 | 원본과 동일 True
```

제가 직접 찾은 결함 2건도 고쳤습니다.
- 문서명 부스트가 `max(scores)` 기준이라 코사인 유사도가 음수면 가산점이 감점으로 뒤집힘 → 점수 폭 기준으로 변경
- `・`(U+30FB)가 한자 제거 정규식에 걸려 `변경・제한・중지`를 한 토큰으로 뭉침 → 채점 토큰 3개 손실
- 4bit 로드 실패 시 1번 셀이 죽음 → 3B fp16 자동 폴백

---

[사용자]

'(다운로드 폴더)/코덱스_하이브리드(2)_answers_public_9.json'
3번셀에 (다운로드 폴더)/message_linda.txt 넣었고 돌렸어.

```
[공개 P — 공식 gold_questions_public10 기반] 10문항
  검색  Hit@4 1.000   MRR 1.000
  생성  키팩트 토큰 F1 프록시 0.712
  지연  p50 12.4s   최대 38.5s   평균 길이 152자
[검증 H — 자체 제작, 비공식] 18문항
  검색  Hit@4 1.000   MRR 0.847
  생성  키팩트 토큰 F1 프록시 0.552
공개 정량 프록시: MRR 20.000/20 + 키팩트 F1 21.357/30 = 41.357/50
```

'~/ai-test-codex/클로드_2차(1)_result_generator_9.ipynb' 도 만들어서 돌려봤어.
결과는 '(다운로드 폴더)/클로드_2차(1)_answers_public_9.json' 와

```
!! P03 [basic] 키팩트 F1 0.262 |  15.0s |   84자
      약한 키팩트 33%: 통합로그인은 카카오계정이 적용된 개별 서비스에서 ...
!! P06 [trap ] 키팩트 F1 0.300 |   6.9s |   52자
!! P07 [basic] 키팩트 F1 0.462 |  15.7s |   60자
[공개 P] 검색 MRR 1.000 / 키팩트 프록시 0.597 / 평균 길이 76자
[검증 H] 검색 MRR 0.972 / 키팩트 프록시 0.625 / 평균 길이 92자
공개 정량 프록시: MRR 20.000/20 + 키팩트 F1 17.920/30 = 37.920/50

[P03] 1. 통합로그인
2. SSO(Single Sign On)
3. 카카오계정 정보 통합 관리
4. 사업자/단체 카카오계정
5. 기타 회사가 제공하는 서비스

[P06] 카카오계정을 먼저 생성해야 합니다.
통합서비스 이용계약은 동의→확인→승낙의 순서로 체결됩니다.
```

로 나왔어. 코덱스_하이브리드(2)_answers_public_9.json 와 이것중에 뭐가 더 좋은지 알려줘

---

[Claude]

두 파일을 같은 보정 채점기로 직접 채점했습니다.

```
$ python3 (보정 채점기 비교)
파일                                      MRR     키팩트F1     객관50점
클로드_1차_answers_public_9.json         1.0000    0.6905    40.715
코덱스_하이브리드(2)_answers_public_9.json   1.0000    0.6894    40.683
클로드_2차(1)_answers_public_9.json      1.0000    0.5777    37.331
eric_결과기(1)_answers_public_9.json    1.0000    0.5484    36.453

문항          하이브리드(2)      클로드2차(1)       차이
P01           0.8824        0.9697  +0.0873
P02           0.5366        0.6471  +0.1105
P03           0.8366        0.3500  -0.4866
P04           0.8387        0.8387  +0.0000
P05           0.7442        0.8095  +0.0653
P06           0.7037        0.2778  -0.4259
P07           0.5769        0.3500  -0.2269
P08           0.3590        0.4211  +0.0621
P09           0.7742        0.5161  -0.2581
P10           0.6420        0.5970  -0.0450
```

**답: 코덱스_하이브리드(2)가 더 낫습니다 — 객관 50점에서 40.683 vs 37.331 (+3.35점).**

다만 갈립니다. 클로드_2차는 짧은 사실 문항(P01·P02·P05·P08)에선 전부 이기고 열거·다항에서만
무너집니다. 반대로 검색은 클로드가 확실히 낫습니다(검증 18문항 MRR 0.972 vs 0.847).
비공개 30문항은 공개 10문항이 아니라 검증셋에 가까우므로 20점 배점에서 약 +2.5점이 걸려 있습니다.

실행본이 제 빌드와 같은 것인지 확인했습니다.

```
1번 셀 동일: True
ANSWER_MODE = "generate" / USE_7B = True / DOC_NAME_BOOST = 0.5
```

원인은 프롬프트 한 곳입니다. 시스템 프롬프트 3번 규칙("군더더기 없이 곧바로 답만")이
4·5번 규칙(모든 하위 질문·항목 설명)을 눌러버렸습니다. 모델이 "짧게"를 "내용을 줄여라"로
읽었습니다. 유형별 지시를 질문 바로 옆에 붙이는 방식으로 고쳤습니다.

- 열거형 → "항목 이름만 쓰지 말고, 각 항목마다 조항에 적힌 설명까지 함께 옮기세요"
- 다항 → "각각에 대해 한 문장씩 모두 답하세요"
- 가부 → "예/아니오로 시작한 뒤 근거 문장을 원문 그대로"

**codex 라운드 3 (실행 결함 검토, 원문은 `debate/round3_codex.md`)**

> **P1 — BGE failure leaks GPU state.** A partial BGE model can remain on GPU and cause the
> 7B/fallback load to OOM. Add deletion/`gc.collect()`/`torch.cuda.empty_cache()` in that `except`.
>
> **P1 — `accelerate` is not guaranteed.** The cell installs `bitsandbytes` but not `accelerate`;
> Transformers requires Accelerate for `device_map` loads, so both the 7B and 3B attempts can fail.
>
> **P1 — 7B fallback does not clear failed-load allocations.** Clear CUDA memory before retrying.
>
> **P1 — Dense query failures are not covered.** A BGE CPU/device error raises through
> `answer_question` instead of disabling dense retrieval and falling back to BM25.
>
> **P2 — The 7000-character budget is applied incorrectly.** It counts ranks 1–2 against the budget
> before considering ranks 3–4. With the four longest articles `(3482, 2560, 2361, 1968)`,
> ranks 3–4 become header-only. Track a separate lower-rank budget.
>
> **No hard-rule violation found.** The artifact has exactly two code cells, cell 2 matches the
> common runner after only the team-number substitution.

5건 전부 고치고 재빌드했습니다.

```
$ python3 v2/test_offline.py
계약 위반 0건
경계 입력 7가지 × 2시나리오 통과
컨텍스트 최댓값 7,398자      ← 예산 수정 반영 (이전 6,830자)
test_offline OK

$ python3 v2/build_notebook.py
  1번 셀 79,338자 / 631줄
  2번 셀 23,375자 / 565줄  (_SP_TEAM="9")
```

로컬에서 한 번도 실행되지 않은 `search()` 를 세 상태로 직접 돌렸습니다.

```
DENSE=None:        [('카카오 위치정보 이용약관', 8), ('카카오 통합 약관', 16), ...]
DENSE 정상:         [('카카오 위치정보 이용약관', 8), ('카카오 위치정보 이용약관', 5)]
[검색] 임베딩 질의 실패 (RuntimeError: CUDA error) — BM25 단독으로 전환
DENSE 질의 실패 후:  [('카카오 위치정보 이용약관', 8), ...]
DENSE 비활성화됨: True
폴백 결과가 BM25 단독과 동일: True
```

---

## 2026-08-11

[사용자]

(자료 정리 요청 — 대화 원문, 결과 파일, 파일 설명, 시간순 작업 보고서를 codex에 전달할 형태로 생성)

[Claude]

`codex_인계/` 에 `claude_01.md`, `claude_02.md`, `파일_설명.md`, `작업_보고서.md` 를 작성.
전체 답변 파일 27종을 보정 채점기로 일괄 재채점해 순위표를 만들고, `ai-test` 갈래(설계_근거.md,
개발과정.md, git log 47커밋)와 `제출본/decision_log.md` 갈래를 합쳐 시간순으로 정리.
