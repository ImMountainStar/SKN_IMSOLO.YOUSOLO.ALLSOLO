import os
import random
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st


st.set_page_config(page_title="IM SOLO Streamlit", page_icon="💘", layout="wide")

MODEL_NAME = "kakaocorp/kanana-nano-2.1b-instruct"


CHARACTERS = [
    {
        "id": "M1",
        "name": "영호",
        "gender": "M",
        "age": 1980,
        "persona": "감성·자기서사·예술적, 성악가 출신. 낭만형. 감정적.",
        "style": "예술가처럼 은유 섞은 말투. 감정을 크게 표현.",
    },
    {
        "id": "M2",
        "name": "영식",
        "gender": "M",
        "age": 1989,
        "persona": "경쟁·도발·자신감, 외국계 세일즈맨.",
        "style": "느끼하고 직설적인 말투. 유머를 비꼬듯 섞음.",
    },
    {
        "id": "M3",
        "name": "상철",
        "gender": "M",
        "age": 1982,
        "persona": "원칙·고집·건조, 방산회사 직원. 가부장적.",
        "style": "무뚝뚝하고 단답형. 고집스러운 말투.",
    },
    {
        "id": "F1",
        "name": "정숙",
        "gender": "F",
        "age": 1978,
        "persona": "직설·기싸움, 요식업 사업가.",
        "style": "부산 사투리를 쓰며, 말꼬투리 잡고 논쟁 유발. 털털한스타일.",
    },
    {
        "id": "F2",
        "name": "영숙",
        "gender": "F",
        "age": 1990,
        "persona": "리액션 과장·감정기복, 무용강사.",
        "style": "부산 사투리를 섞어 과장된 리액션을 함.",
    },
    {
        "id": "F3",
        "name": "옥순",
        "gender": "F",
        "age": 1995,
        "persona": "돌직구·애교/의존, 공주병 경향.",
        "style": "애교 많고 남자에게 의존적인 말투. 도도함.",
    },
]

CHAR_DICT = {c["id"]: c for c in CHARACTERS}

GAME_CONFIG = {
    "initial_favor_base": 25,
    "initial_favor_rand": 10,
    "initial_trust_base": 35,
    "initial_spark_base": 30,
    "initial_jealousy_base": 15,
    "talk_base_bonus": 2,
    "favor_min": 0,
    "favor_max": 100,
    "stat_min": 0,
    "stat_max": 100,
    "day_chat_rounds": 3,
    "passive_jealousy_per_date": 3,
}

KEYWORDS = [
    (["뮤즈", "로맨틱", "노래", "예술", "스윗", "잘생", "오빠"], {"M1": {"favor": 3, "spark": 3}}),
    (["인정", "팩트", "스테이블", "성과"], {"M2": {"favor": 3, "trust": 2}}),
    (["계획", "원칙", "기준", "결론"], {"M3": {"favor": 2, "trust": 3}}),
    (["직설", "팩폭", "솔직"], {"F1": {"favor": 3, "trust": 2}}),
    (["응원", "고마워", "힘내"], {"F2": {"favor": 2, "trust": 3, "spark": 1}}),
    (["오빠", "챙겨줘", "귀여워", "애교"], {"F3": {"favor": 3, "spark": 3}}),
]

EVENT_CARDS = [
    {
        "id": "secret_date",
        "title": "비밀 데이트 제안",
        "target": "top",
        "prompt": "오늘 밤, {name}이(가) 남몰래 산책을 제안합니다.",
        "choices": {
            "A": {
                "label": "조용히 수락한다",
                "target": {"favor": 6, "trust": 3, "spark": 5, "jealousy": -1},
                "others": {"jealousy": 4},
            },
            "B": {
                "label": "공개적으로 함께 간다",
                "target": {"favor": 4, "trust": 5, "spark": 2},
                "others": {"jealousy": 2},
            },
        },
    },
    {
        "id": "truth_game",
        "title": "진실게임 폭탄 질문",
        "target": "random",
        "prompt": "단체 진실게임에서 {name}의 질문이 날카롭습니다.",
        "choices": {
            "A": {
                "label": "솔직하게 답한다",
                "target": {"favor": 2, "trust": 6, "spark": 1},
                "others": {"jealousy": 1},
            },
            "B": {
                "label": "재치로 넘긴다",
                "target": {"favor": 3, "trust": -2, "spark": 4},
                "others": {"jealousy": 0},
            },
        },
    },
    {
        "id": "cooking_mission",
        "title": "요리 미션",
        "target": "random",
        "prompt": "요리 미션 파트너로 {name}이(가) 배정됐습니다.",
        "choices": {
            "A": {
                "label": "리드해서 완성한다",
                "target": {"favor": 4, "trust": 4, "spark": 1},
                "others": {"jealousy": 2},
            },
            "B": {
                "label": "상대 리듬에 맞춘다",
                "target": {"favor": 3, "trust": 2, "spark": 3},
                "others": {"jealousy": 1},
            },
        },
    },
    {
        "id": "anonymous_letter",
        "title": "익명 편지",
        "target": "top",
        "prompt": "새벽에 {name}에게 익명 편지를 보낼 기회가 생겼습니다.",
        "choices": {
            "A": {
                "label": "진심 고백 편지",
                "target": {"favor": 5, "trust": 2, "spark": 5},
                "others": {"jealousy": 3},
            },
            "B": {
                "label": "가벼운 응원 편지",
                "target": {"favor": 3, "trust": 4, "spark": 2},
                "others": {"jealousy": 1},
            },
        },
    },
]


@st.cache_resource(show_spinner=True)
def load_local_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        token=hf_token,
    )
    return tokenizer, model


def clamp_favor(score: int) -> int:
    return max(GAME_CONFIG["favor_min"], min(GAME_CONFIG["favor_max"], int(score)))


def clamp_stat(score: int) -> int:
    return max(GAME_CONFIG["stat_min"], min(GAME_CONFIG["stat_max"], int(score)))


def target_gender(characters: List[Dict], player_gender: str) -> List[Dict]:
    opp = "F" if player_gender == "M" else "M"
    return [c for c in characters if c["gender"] == opp]


def first_love(pool: List[Dict], seed: Optional[int] = None) -> Dict[str, int]:
    rng = random.Random(seed) if seed is not None else random
    base = GAME_CONFIG["initial_favor_base"]
    spread = GAME_CONFIG["initial_favor_rand"]
    return {c["id"]: clamp_favor(base + rng.randint(0, spread)) for c in pool}


def init_states(pool: List[Dict], seed: Optional[int] = None) -> Dict[str, Dict[str, int]]:
    rng = random.Random((seed + 999) if seed is not None else None)
    states = {}
    for c in pool:
        states[c["id"]] = {
            "trust": clamp_stat(GAME_CONFIG["initial_trust_base"] + rng.randint(-5, 5)),
            "spark": clamp_stat(GAME_CONFIG["initial_spark_base"] + rng.randint(-5, 5)),
            "jealousy": clamp_stat(GAME_CONFIG["initial_jealousy_base"] + rng.randint(-5, 5)),
        }
    return states


def apply_favor_delta(favor: Dict[str, int], cid: str, delta: int) -> int:
    before = favor.get(cid, 0)
    favor[cid] = clamp_favor(before + int(delta))
    return favor[cid] - before


def apply_state_delta(states: Dict[str, Dict[str, int]], cid: str, key: str, delta: int) -> int:
    before = states[cid].get(key, 0)
    states[cid][key] = clamp_stat(before + int(delta))
    return states[cid][key] - before


def merge_effect(dst: Dict[str, int], src: Dict[str, int]) -> Dict[str, int]:
    for k in ("favor", "trust", "spark", "jealousy"):
        dst[k] += int(src.get(k, 0))
    return dst


def tipping_chat(favor: Dict[str, int], states: Dict[str, Dict[str, int]], text: str, target_id: str):
    effect = {"favor": 0, "trust": 0, "spark": 0, "jealousy": 0}
    matched_keywords = []

    effect["favor"] += apply_favor_delta(favor, target_id, GAME_CONFIG["talk_base_bonus"])
    effect["spark"] += apply_state_delta(states, target_id, "spark", 1)

    t = (text or "").strip()
    for words, char_effects in KEYWORDS:
        hit = [w for w in words if w in t]
        if not hit:
            continue
        matched_keywords.extend(hit)

        for cid, delta_map in char_effects.items():
            if cid not in favor:
                continue
            for key, delta in delta_map.items():
                if key == "favor":
                    effect["favor"] += apply_favor_delta(favor, cid, delta)
                else:
                    effect[key] += apply_state_delta(states, cid, key, delta)

    return effect, sorted(set(matched_keywords))


def memory_context(log: List[str], limit: int = 4) -> str:
    if not log:
        return ""
    return " / ".join(log[-limit:])


def clean_generated_text(text: str) -> str:
    text = text.split("답변:", 1)[-1]
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    cleaned_lines = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t:
            continue
        if any(x in t for x in ("시스템", "사용자", "플레이어", "캐릭터", "가이드", "지침", "최근 기억", "현재 감정상태")):
            continue
        cleaned_lines.append(t)
    text = " ".join(cleaned_lines)
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r"(?<=[\.!?！？…])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    if len(sents) > 2:
        sents = sents[:2]
    return " ".join(sents) if sents else "네, 말씀 고맙습니다."


def fallback_reply(char: Dict, user_text: str) -> str:
    base = [
        f"{user_text}라고 말해주셔서 고마워요.",
        "저도 지금 분위기 진지하게 보고 있어요.",
        "오늘은 조금 더 솔직하게 대화해보고 싶네요.",
    ]
    tone = {
        "M1": "그 말이 마음에 오래 남을 것 같아요.",
        "M2": "그 자신감, 저는 꽤 높게 평가해요.",
        "M3": "핵심을 짚는 대화라서 좋습니다.",
        "F1": "돌려 말하지 않는 점, 저는 좋게 봐요.",
        "F2": "리액션이 좋아서 저도 기분이 올라가요.",
        "F3": "지금 느낌, 꽤 설레는데요?",
    }
    return f"{random.choice(base)} {tone.get(char['id'], '계속 이야기해요.')}"


def llm_reply(char: Dict, user_text: str, memory_summary: str, favor_score: int, relation_state: Dict[str, int]) -> str:
    model_bundle = st.session_state.get("model_bundle")
    if model_bundle is None:
        return fallback_reply(char, user_text)

    tokenizer, model = model_bundle
    system = (
        "너는 한국 예능 '나는 솔로'의 해당 출연자 그 자체다. "
        "오직 캐릭터 1인칭으로만 말한다. "
        "해설/메타/역할표시/괄호 금지. 플레이어 대사 생성 금지. "
        "짧고 자연스럽게. 모든 답변은 존댓말로."
    )
    prompt = (
        f"{system}\n\n"
        f"캐릭터: {char['name']} / 성향:{char.get('persona','')} / 말투:{char.get('style','')}\n"
        f"현재 호감도: {favor_score}\n"
        f"현재 감정상태(내 기준): 신뢰 {relation_state.get('trust',0)}, 설렘 {relation_state.get('spark',0)}, 질투 {relation_state.get('jealousy',0)}\n"
        f"최근 기억: {memory_summary}\n"
        f"플레이어: {user_text}\n\n답변:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            no_repeat_ngram_size=3,
            repetition_penalty=1.08,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        )
        raw = tokenizer.decode(out[0], skip_special_tokens=True)
        return clean_generated_text(raw)
    except Exception:
        return fallback_reply(char, user_text)


def heuristic_score(transcript: List[Dict[str, str]]) -> int:
    pos = ["고마", "좋", "설레", "응원", "웃", "행복", "멋", "대단", "귀여"]
    neg = ["싫", "별로", "부담", "짜증", "불편", "최악", "화나", "실망"]
    score = 0
    for turn in transcript:
        if turn["role"] != "user":
            continue
        t = turn["text"]
        score += sum(2 for p in pos if p in t)
        score -= sum(2 for n in neg if n in t)
        if len(t) >= 15:
            score += 1
    return max(-20, min(20, score))


def get_score(s: str):
    s = s.strip()
    if not s:
        return None
    out = ""
    for i, ch in enumerate(s):
        if i == 0 and ch in "+-":
            out += ch
        elif ch.isdigit():
            out += ch
        else:
            break
    if out in ("+", "-", ""):
        return None
    try:
        return int(out)
    except Exception:
        return None


def score_llm(transcript: List[Dict[str, str]], char: Dict) -> int:
    model_bundle = st.session_state.get("model_bundle")
    if model_bundle is None:
        return heuristic_score(transcript)

    tokenizer, model = model_bundle
    lines = []
    for turn in transcript:
        who = "플레이어" if turn["role"] == "user" else char["name"]
        lines.append(f"{who}: {turn['text']}")
    all_chat = "\n".join(lines)

    prompt = (
        "너는 연애 예능의 심사위원이다. 아래 대화를 보고 "
        "상대 캐릭터가 사용자에게 줄 호감도 변화 점수(정수)를 판단하라. "
        "반드시 -20부터 20 사이의 정수 한 개만 출력하라. 숫자 외에는 아무 것도 출력하지 마라.\n\n"
        f"캐릭터: {char['name']} / 성향:{char.get('persona','')} / 말투:{char.get('style','')}\n"
        f"대화:\n{all_chat}\n\n"
        "예시: 15\n예시: -10\n예시: 0\n"
        "지금 점수만 출력:"
    )

    try:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_len = enc["input_ids"].shape[1]
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        out_ids = model.generate(
            **enc,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id),
        )
        gen_only_ids = out_ids[0, input_len:]
        text = tokenizer.decode(gen_only_ids, skip_special_tokens=True).strip()
        val = get_score(text)
        if val is None:
            val = 0
        return max(-20, min(20, val))
    except Exception:
        return heuristic_score(transcript)


def apply_llm_effect(favor: Dict[str, int], states: Dict[str, Dict[str, int]], cid: str, llm_delta: int):
    effect = {"favor": 0, "trust": 0, "spark": 0, "jealousy": 0}
    effect["favor"] += apply_favor_delta(favor, cid, llm_delta)

    if llm_delta >= 0:
        effect["trust"] += apply_state_delta(states, cid, "trust", max(1, llm_delta // 3))
        effect["spark"] += apply_state_delta(states, cid, "spark", max(1, llm_delta // 2))
        effect["jealousy"] += apply_state_delta(states, cid, "jealousy", -max(1, llm_delta // 4))
    else:
        amount = abs(llm_delta)
        effect["trust"] += apply_state_delta(states, cid, "trust", -max(1, amount // 2))
        effect["spark"] += apply_state_delta(states, cid, "spark", -max(1, amount // 3))
        effect["jealousy"] += apply_state_delta(states, cid, "jealousy", max(1, amount // 2))
    return effect


def apply_passive_jealousy(states: Dict[str, Dict[str, int]], pool: List[Dict], target_id: str, amount: int):
    for c in pool:
        cid = c["id"]
        if cid == target_id:
            continue
        apply_state_delta(states, cid, "jealousy", amount)


def pick_event_target(card: Dict, pool: List[Dict], favor: Dict[str, int]) -> Dict:
    if card["target"] == "top":
        return max(pool, key=lambda c: favor[c["id"]])
    return random.choice(pool)


def apply_card_effects(favor: Dict[str, int], states: Dict[str, Dict[str, int]], pool: List[Dict], target_id: str, card_effect: Dict):
    total = {"favor": 0, "trust": 0, "spark": 0, "jealousy": 0}
    for key, delta in card_effect.get("target", {}).items():
        if key == "favor":
            total["favor"] += apply_favor_delta(favor, target_id, delta)
        else:
            total[key] += apply_state_delta(states, target_id, key, delta)

    for c in pool:
        cid = c["id"]
        if cid == target_id:
            continue
        for key, delta in card_effect.get("others", {}).items():
            if key == "favor":
                total["favor"] += apply_favor_delta(favor, cid, delta)
            else:
                total[key] += apply_state_delta(states, cid, key, delta)
    return total


def relation_power(favor: int, state: Dict[str, int]) -> float:
    return round(favor + state["trust"] * 0.6 + state["spark"] * 0.8 - state["jealousy"] * 0.5, 1)


def build_ending(favor: Dict[str, int], states: Dict[str, Dict[str, int]], pool: List[Dict]):
    ranking = sorted(
        [(c["id"], relation_power(favor[c["id"]], states[c["id"]])) for c in pool],
        key=lambda x: x[1],
        reverse=True,
    )
    t1_id, t1_score = ranking[0]
    t2_id, t2_score = ranking[1] if len(ranking) > 1 else (None, -999)

    t1 = states[t1_id]
    t1_name = CHAR_DICT[t1_id]["name"]
    gap = t1_score - t2_score

    if t1_score >= 95 and t1["trust"] >= 65 and t1["spark"] >= 65 and t1["jealousy"] <= 45:
        ending_type = "운명 커플 엔딩"
        line = f"[대성공] {t1_name}와(과) 서로 확신한 공식 커플이 되었습니다."
    elif t1_score >= 82 and gap <= 8 and t2_id is not None:
        t2_name = CHAR_DICT[t2_id]["name"]
        ending_type = "삼각관계 엔딩"
        line = f"[혼돈] {t1_name} vs {t2_name} 감정선이 충돌해 마지막 선택이 엇갈렸습니다."
    elif t1_score >= 80 and t1["jealousy"] >= 70:
        ending_type = "불꽃 집착 엔딩"
        line = f"[고자극] {t1_name}와 강하게 끌렸지만 질투가 커져 불안한 관계가 됐습니다."
    elif t1_score >= 72:
        ending_type = "현실 커플 엔딩"
        line = f"[성공] {t1_name}와 천천히 맞춰가는 안정적인 썸-연인 루트입니다."
    elif t1_score >= 60:
        ending_type = "우정 보류 엔딩"
        line = f"[보류] {t1_name}와는 호감이 남았지만 이번 시즌에서는 친구로 마무리됐습니다."
    else:
        ending_type = "솔로 성장 엔딩"
        line = "[노매칭] 이번엔 솔로로 끝났지만, 다음 시즌을 위한 데이터는 충분히 쌓였습니다."

    return ending_type, line, ranking


def init_game(player_name: str, player_age: str, player_job: str, player_gender: str, seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
    pool = target_gender(CHARACTERS, player_gender)
    st.session_state.player = {
        "name": player_name or "플레이어",
        "age": player_age,
        "job": player_job,
        "gender": player_gender,
    }
    st.session_state.pool = pool
    st.session_state.favor = first_love(pool, seed=seed)
    st.session_state.states = init_states(pool, seed=seed)
    st.session_state.memories = {c["id"]: [] for c in pool}
    st.session_state.logs = []
    st.session_state.phase = "day1_choice"
    st.session_state.chat = None
    st.session_state.event = None


def start_chat(label: str, cid: str, next_phase: str):
    st.session_state.chat = {
        "label": label,
        "cid": cid,
        "next_phase": next_phase,
        "turn": 1,
        "transcript": [],
        "rule_effect_total": {"favor": 0, "trust": 0, "spark": 0, "jealousy": 0},
    }
    st.session_state.phase = "chat"


def finalize_chat():
    chat = st.session_state.chat
    cid = chat["cid"]
    char = CHAR_DICT[cid]
    llm_delta = score_llm(chat["transcript"], char)
    llm_effect = apply_llm_effect(st.session_state.favor, st.session_state.states, cid, llm_delta)
    apply_passive_jealousy(
        st.session_state.states,
        st.session_state.pool,
        target_id=cid,
        amount=GAME_CONFIG["passive_jealousy_per_date"],
    )

    total_effect = {"favor": 0, "trust": 0, "spark": 0, "jealousy": 0}
    merge_effect(total_effect, chat["rule_effect_total"])
    merge_effect(total_effect, llm_effect)

    st.session_state.logs.append(
        f"[{char['name']} 데이트] 호감 {total_effect['favor']:+d}, 신뢰 {total_effect['trust']:+d}, "
        f"설렘 {total_effect['spark']:+d}, 질투 {total_effect['jealousy']:+d} (LLM {llm_delta:+d})"
    )

    st.session_state.chat = None
    st.session_state.phase = chat["next_phase"]


def start_event(day_label: str, next_phase: str):
    card = random.choice(EVENT_CARDS)
    target = pick_event_target(card, st.session_state.pool, st.session_state.favor)
    st.session_state.event = {
        "day_label": day_label,
        "card": card,
        "target": target,
        "next_phase": next_phase,
    }
    st.session_state.phase = "event"


def apply_event_choice(choice: str):
    event = st.session_state.event
    card = event["card"]
    target = event["target"]
    eff = apply_card_effects(
        st.session_state.favor,
        st.session_state.states,
        st.session_state.pool,
        target["id"],
        card["choices"][choice],
    )
    st.session_state.logs.append(
        f"[이벤트:{card['title']}] {target['name']} 대상 | 호감 {eff['favor']:+d}, 신뢰 {eff['trust']:+d}, 설렘 {eff['spark']:+d}, 질투 {eff['jealousy']:+d}"
    )
    st.session_state.event = None
    st.session_state.phase = event["next_phase"]


def render_scoreboard():
    st.subheader("관계 현황")
    rows = []
    for c in st.session_state.pool:
        cid = c["id"]
        stt = st.session_state.states[cid]
        rows.append(
            {
                "이름": c["name"],
                "호감": st.session_state.favor[cid],
                "신뢰": stt["trust"],
                "설렘": stt["spark"],
                "질투": stt["jealousy"],
                "관계력": relation_power(st.session_state.favor[cid], stt),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_setup():
    st.title("IM SOLO YOU SOLO ALL SOLO - Streamlit")
    st.caption("노트북 기반 채팅형 연애 게임 웹 버전")

    with st.form("setup_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("이름", value="플레이어")
            age = st.text_input("나이", value="29")
        with c2:
            job = st.text_input("직업", value="개발자")
            gender = st.radio("성별", ["M", "F"], horizontal=True)
        seed_txt = st.text_input("랜덤 시드(선택)", value="")
        use_model = st.checkbox("로컬 HuggingFace 모델 사용", value=False)
        submitted = st.form_submit_button("게임 시작")

    if submitted:
        seed = int(seed_txt) if seed_txt.strip().isdigit() else None
        if use_model:
            try:
                with st.spinner("모델 로딩 중..."):
                    st.session_state.model_bundle = load_local_model(MODEL_NAME)
                st.success("모델 로딩 완료")
            except Exception as exc:
                st.session_state.model_bundle = None
                st.warning(f"모델 로딩 실패, fallback 모드로 진행합니다: {exc}")
        else:
            st.session_state.model_bundle = None

        init_game(name, age, job, gender, seed)
        st.rerun()


def render_day_choice(title: str, next_chat_label: str, next_after_chat: str, pass_next: str):
    st.header(title)
    names = ["패스"] + [c["name"] for c in st.session_state.pool]
    choice = st.selectbox("상대 선택", names, key=f"sel_{title}")
    if st.button("선택 확정", key=f"btn_{title}"):
        if choice == "패스":
            st.session_state.logs.append(f"[{title}] 패스")
            st.session_state.phase = pass_next
        else:
            cid = next(c["id"] for c in st.session_state.pool if c["name"] == choice)
            start_chat(next_chat_label, cid, next_after_chat)
        st.rerun()


def render_chat_phase():
    chat = st.session_state.chat
    cid = chat["cid"]
    char = CHAR_DICT[cid]

    st.header(f"{chat['label']} - {char['name']} ({chat['turn']}/{GAME_CONFIG['day_chat_rounds']})")
    for turn in chat["transcript"]:
        speaker = st.session_state.player["name"] if turn["role"] == "user" else char["name"]
        st.write(f"**{speaker}**: {turn['text']}")

    with st.form("chat_turn_form", clear_on_submit=True):
        user_text = st.text_input("당신의 한 마디")
        submitted = st.form_submit_button("말하기")

    if submitted and user_text.strip():
        chat["transcript"].append({"role": "user", "text": user_text.strip()})
        eff, matched = tipping_chat(st.session_state.favor, st.session_state.states, user_text, cid)
        merge_effect(chat["rule_effect_total"], eff)

        mem = memory_context(st.session_state.memories[cid], limit=4)
        reply = llm_reply(char, user_text, mem, st.session_state.favor[cid], st.session_state.states[cid])
        chat["transcript"].append({"role": "char", "text": reply})

        tag = f"키워드:{','.join(matched)}" if matched else "키워드:없음"
        st.session_state.memories[cid].append(f"U:{user_text} | C:{reply} | {tag}")
        st.session_state.memories[cid] = st.session_state.memories[cid][-12:]

        chat["turn"] += 1
        if chat["turn"] > GAME_CONFIG["day_chat_rounds"]:
            finalize_chat()
        st.rerun()


def render_event_phase():
    event = st.session_state.event
    card = event["card"]
    target = event["target"]
    st.header(f"🎴 EVENT CARD - {event['day_label']}")
    st.subheader(card["title"])
    st.write(card["prompt"].format(name=target["name"]))

    c1, c2 = st.columns(2)
    if c1.button(f"A) {card['choices']['A']['label']}"):
        apply_event_choice("A")
        st.rerun()
    if c2.button(f"B) {card['choices']['B']['label']}"):
        apply_event_choice("B")
        st.rerun()


def render_final_phase():
    st.header("파이널")
    ending_type, line, ranking = build_ending(st.session_state.favor, st.session_state.states, st.session_state.pool)
    st.success(f"엔딩 타입: {ending_type}")
    st.write(line)

    rows = []
    for cid, power in ranking:
        stt = st.session_state.states[cid]
        rows.append(
            {
                "이름": CHAR_DICT[cid]["name"],
                "관계력": power,
                "호감": st.session_state.favor[cid],
                "신뢰": stt["trust"],
                "설렘": stt["spark"],
                "질투": stt["jealousy"],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def game_router():
    phase = st.session_state.get("phase", "setup")

    if phase == "setup":
        render_setup()
        return

    st.title("IM SOLO - 진행 중")
    c1, c2 = st.columns([2, 1])
    with c2:
        if st.button("처음부터 다시 시작"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    with c1:
        render_scoreboard()

    with st.expander("진행 로그", expanded=True):
        if st.session_state.logs:
            for log in st.session_state.logs[-20:]:
                st.write(f"- {log}")
        else:
            st.write("아직 로그가 없습니다.")

    if phase == "day1_choice":
        render_day_choice(
            title="Day 1: 단체 저녁 (첫인상 선택)",
            next_chat_label="첫 talk",
            next_after_chat="day1_event",
            pass_next="day1_event",
        )
    elif phase == "day1_event":
        if st.session_state.event is None:
            start_event("Day 1 밤", "day2_intro")
            st.rerun()
        render_event_phase()
    elif phase == "day2_intro":
        st.header("Day 2: 랜덤 1:1 데이트")
        if st.button("랜덤 데이트 시작"):
            cid = random.choice([c["id"] for c in st.session_state.pool])
            st.session_state.logs.append(f"[Day2] 랜덤 데이트 상대: {CHAR_DICT[cid]['name']}")
            start_chat("1:1 talk", cid, "day2_event")
            st.rerun()
    elif phase == "day2_event":
        if st.session_state.event is None:
            start_event("Day 2 밤", "day3_choice")
            st.rerun()
        render_event_phase()
    elif phase == "day3_choice":
        render_day_choice(
            title="Day 3: 지목 데이트",
            next_chat_label="지목 데이트",
            next_after_chat="day3_event",
            pass_next="day3_event",
        )
    elif phase == "day3_event":
        if st.session_state.event is None:
            start_event("Day 3 밤", "final")
            st.rerun()
        render_event_phase()
    elif phase == "chat":
        render_chat_phase()
    elif phase == "final":
        render_final_phase()


if "phase" not in st.session_state:
    st.session_state.phase = "setup"
if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None

game_router()
