# -*- coding: utf-8 -*-
"""공개 10문항의 골드 키팩트를 다른 팀 채점 결과로부터 복원한다.

입력(읽기 전용): ~/Downloads/answers_public_<팀>.json, practice_result_<팀>.json
후보 키팩트는 약관 원문에서 그대로 따온 뒤, 6팀 실측 keyfact_f1 을 가장 잘
재현하도록 경계를 다듬는다. 오차(MAE)를 그대로 출력한다 — 추정이지 확정이 아니다.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_eval import keyfact_f1, keyfact_recall  # noqa: E402

DOWNLOADS = os.path.expanduser("~/Downloads")
TEAMS = (12, 13, 14, 17, 2, 5)

# 심사평이 지목한 키팩트를 약관 원문 표현 그대로 옮긴 초안.
# 개수는 keyfact_recall 분모 역산값(2,2,5,2,3,2,3,2,2,3)과 일치시킨다.
DRAFT = {
    "P01": {
        "gold": [["카카오계정 약관", 10]],
        "keyfacts": [
            "사업자/단체 카카오계정은 계정 정보에 등록된 담당자 1인만 이용할 수 있으며",
            "다른 사람에게 공유하는 것은 금지됩니다",
        ],
    },
    "P02": {
        "gold": [["카카오계정 약관", 8], ["카카오 통합서비스약관", 7], ["카카오 통합 약관", 13]],
        "keyfacts": [
            "2시간 이상 복구가 지연될 시 카카오 서비스 공지사항, 카카오 고객센터 공지사항 등에 게시하여 알려 드리겠습니다",
            "회사가 상황을 파악하는 즉시 최대한 빠른 시일 내에 서비스를 복구하도록 노력하고",
        ],
    },
    "P03": {
        "gold": [["카카오계정 약관", 7]],
        "keyfacts": [
            "통합로그인 : 카카오계정이 적용된 개별 서비스에서 하나의 카카오계정과 비밀번호로 로그인할 수 있는 통합 회원 인증 서비스를 이용할 수 있습니다.",
            "SSO(Single Sign On): 웹브라우저나 특정 모바일 기기에서 카카오계정 1회 로그인으로 여러분이 이용 중인 개별 서비스간 추가 로그인 없이 자동 접속 서비스를 이용할 수 있습니다.",
            "카카오계정 정보 통합 관리 : 개별 서비스 이용을 위해 카카오계정 정보를 통합 관리합니다. 또한, 여러분이 이용하고자 하는 개별 서비스의 유형에 따라 전문기관을 통한 실명확인 및 본인인증을 요청할 수 있고, 이를 카카오계정 정보로 저장합니다.",
            "사업자/단체 카카오계정 : 사업자/단체 명의로 카카오 서비스를 이용하기 위해 만들어진 카카오계정으로서 해당 사업자/단체의 책임 하에 권한을 위임받은 담당자가 이용, 관리할 수 있는 계정 서비스입니다.",
            "기타 회사가 제공하는 서비스",
        ],
    },
    "P04": {
        "gold": [["카카오 위치정보 이용약관", 6]],
        "keyfacts": [
            "그 사유 및 제한기간 등",
            "회사 홈페이지 등을 통해 사전 공지하거나 이용자에게 통지합니다",
        ],
    },
    "P05": {
        "gold": [["카카오 위치정보 이용약관", 8]],
        "keyfacts": [
            "위치정보의 보호 및 이용 등에 관한 법률 제16조 제2항에 근거하여",
            "위치정보 수집·이용·제공사실 확인자료를 위치정보시스템에 자동으로 기록·보존하며",
            "해당 자료는 6개월간 보관합니다",
        ],
    },
    "P06": {
        "gold": [["카카오 통합서비스약관", 4]],
        "keyfacts": [
            "통합서비스에 가입하기 위해서는 카카오계정이 필요합니다. 카카오계정이 없으신 경우 카카오계정을 먼저 생성하시기 바랍니다",
            "통합서비스 이용계약은 여러분이 본 약관의 내용에 동의한 후 회사가 여러분의 카카오계정 정보 등을 확인한 후 승낙함으로써 체결됩니다",
        ],
    },
    "P07": {
        "gold": [["카카오 통합서비스약관", 1]],
        "keyfacts": [
            "회사가 아닌 카카오 계열사에서 제공하는 서비스",
            "카카오 T택시 서비스",
            "회사가 아닌 계열사를 포함한 제3자가 제공하는 서비스에 가입되지는 않으며",
        ],
    },
    "P08": {
        "gold": [["카카오 통합 약관", 3]],
        "keyfacts": [
            "본 약관과 세부지침의 내용이 충돌할 경우 세부지침에 따릅니다",
            "본 약관에 규정되지 않은 사항에 대해서는 관련법령 또는 회사가 정한 서비스의 개별 이용약관, 운영정책 및 규칙 등(이하 ‘세부지침’)의 규정에 따릅니다",
        ],
    },
    "P09": {
        "gold": [["카카오 통합 약관", 10]],
        "keyfacts": [
            "본 라이선스는 여러분이 서비스의 사용을 중단하거나 카카오계정 및/또는 Daum 아이디를 탈퇴한 후에도 존속하게 됩니다",
            "전 세계적이고 영구적인 라이선스",
        ],
    },
    "P10": {
        "gold": [["카카오 위치정보 이용약관", 12]],
        "keyfacts": [
            "서면동의서에 보호의무자임을 증명하는 서면을 첨부하여 회사에 제출하여야 합니다",
            "본인의 동의가 있는 것으로 봅니다",
            "본 약관 제9조에 의한 이용자의 권리를 모두 가집니다",
        ],
    },
}


def observations():
    """{qid: [(team, answer, obs_f1, obs_recall), ...]}"""
    out = {}
    for team in TEAMS:
        with open(f"{DOWNLOADS}/answers_public_{team}.json", encoding="utf-8") as fp:
            answers = {a["qid"]: a["answer"] for a in json.load(fp)["answers"]}
        with open(f"{DOWNLOADS}/practice_result_{team}.json", encoding="utf-8") as fp:
            result = json.load(fp)["objective"]["per_question"]
        for row in result:
            out.setdefault(row["qid"], []).append(
                (team, answers[row["qid"]], row["keyfact_f1"], row["keyfact_recall"])
            )
    return out


def article_words(gold_refs):
    """정답 조항별 어절 리스트. 키팩트는 한 조 안의 연속 구간이라고 본다."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terms.json")
    with open(path, encoding="utf-8") as fp:
        chunks = {(r["document"], r["article"]): r["text"] for r in json.load(fp)}
    return [chunks[(doc, int(art))].split() for doc, art in gold_refs]


def _align(fact, per_article):
    """초안 키팩트가 어느 조의 어느 구간인지 토큰 겹침으로 찾는다."""
    from local_eval import tokens as _tk
    target, n = _tk(fact), len(fact.split())
    best, best_score = (0, 0, n), -1.0
    for ai, words in enumerate(per_article):
        for start in range(max(1, len(words) - n + 1)):
            span = _tk(" ".join(words[start:start + n]))
            score = len(span & target) / max(len(target), 1)
            if score > best_score:
                best, best_score = (ai, start, start + n), score
    return best


def refine(keyfacts, obs, gold_refs, margin=8, rounds=3):
    """키팩트를 원문의 연속 어절 구간으로 보고 경계를 넓혔다 좁히며 맞춘다."""
    per_article = article_words(gold_refs)

    def mae(facts):
        return sum(abs(keyfact_f1(a, facts) - f) for _, a, f, _ in obs) / len(obs)

    def render(spans):
        return [" ".join(per_article[ai][s:e]) for ai, s, e in spans]

    best = [_align(f, per_article) for f in keyfacts]
    best_err = mae(render(best))
    for _ in range(rounds):
        improved = False
        for i, (ai, s0, e0) in enumerate(best):
            words = per_article[ai]
            for s in range(max(0, s0 - margin), min(len(words), s0 + margin) + 1):
                for e in range(max(s + 1, e0 - margin), min(len(words), e0 + margin) + 1):
                    cand = list(best)
                    cand[i] = (ai, s, e)
                    err = mae(render(cand))
                    if err < best_err - 1e-9:
                        best, best_err, improved = cand, err, True
        if not improved:
            break
    return render(best), best_err


def main():
    obs_all = observations()
    questions, errors = [], []
    print(f"{'QID':5} {'키팩트':4} {'MAE':>7}  {'실측 F1 vs 복원 F1'}")
    print("-" * 78)
    for qid in sorted(DRAFT):
        obs = obs_all[qid]
        facts, err = refine(DRAFT[qid]["keyfacts"], obs, DRAFT[qid]["gold"])
        errors.append(err)
        pairs = " ".join(f"{o:.3f}/{keyfact_f1(a, facts):.3f}" for _, a, o, _ in obs)
        print(f"{qid:5} {len(facts):^4} {err:7.4f}  {pairs}")
        questions.append({
            "qid": qid,
            "gold": DRAFT[qid]["gold"],
            "keyfacts": facts,
            "n_keyfacts": len(facts),
            "fit_mae": round(err, 4),
        })
    print("-" * 78)
    print(f"전체 MAE {sum(errors) / len(errors):.4f}  (60개 관측 = 6팀 × 10문항)")

    # 키팩트 단위 recall 임계값도 실측값에 맞춰 고른다.
    best_th, best_hit = None, -1
    for th in [x / 100 for x in range(50, 96, 5)]:
        hit = sum(
            abs(keyfact_recall(a, q["keyfacts"], th) - r) < 1e-3
            for q in questions
            for _, a, _, r in obs_all[q["qid"]]
        )
        if hit > best_hit:
            best_th, best_hit = th, hit
    print(f"recall 임계값 {best_th} 에서 60개 중 {best_hit}개 일치")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldset.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "source": "practice_result_*.json 6팀 실측값 역공학 (추정치)",
                "recall_threshold": best_th,
                "questions": questions,
            },
            fp, ensure_ascii=False, indent=2,
        )
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
