import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


DEFAULT_MODEL = "qwen3-vl-8b"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "single"

POSITIONS = {
    "左前方",
    "前方",
    "右前方",
    "左侧",
    "右侧",
    "左后方",
    "后方",
    "右后方",
    "当前车道",
}

CATEGORIES = {
    "盲区",
    "障碍物",
    "交通信号",
    "车辆行为",
    "路况",
    "交通管制",
    "跟车",
    "换道条件",
}

DETAIL_SCORE_VALUES = {0.0, 0.5, 1.0}

EXTRACTION_SYSTEM_PROMPT = """
你是一个自动驾驶场景结构化标注助手。任务是把驾驶场景描述转换为 JSON。

核心原则：
- 只允许抽取原文中明确表达的信息。
- 严禁补充、推理、常识扩展或新增原文不存在的因素。
- 一个因素必须只对应一个位置、一个大类、一个细节、一个原文片段。
- 如果一句话中存在多个因素，必须拆分，禁止合并。

必须严格只输出 JSON，不要输出 markdown，不要输出 ```json。

输出结构：
{
  "因素": [
    {
      "位置": "",
      "大类": "",
      "细节": "",
      "原文片段": ""
    }
  ],
  "动作": {
    "横向决策": [],
    "纵向决策": [],
    "执行策略": ""
  }
}

位置只能从以下值中选择：
左前方、前方、右前方、左侧、右侧、左后方、后方、右后方、当前车道

大类只能从以下值中选择：
盲区、障碍物、交通信号、车辆行为、路况、交通管制、跟车、换道条件

横向决策只能从以下值中选择：
换道、避让、保持、转弯
归一化规则：
- 左换道 / 右换道 / 变道 / 变更车道 / 并线 -> 换道
- 左避障 / 右避障 / 绕行避让 / 避障 -> 避让
- 左转 / 右转 / 掉头 / 转弯 -> 转弯
- 未明确描述横向动作时，默认保持

纵向决策只能从以下值中选择：
保持、加速、减速、停车
归一化规则：
- 减速 / 慢行 -> 减速
- 起步 / 跟车起步 / 加速 -> 加速
- 刹停 / 停车等待 / 停止 -> 停车
- 未明确描述纵向动作时，默认保持
- 连续动作，例如“减速后停车”，输出为 ["减速", "停车"]

执行策略只能从以下值中选择：
直接执行、条件满足后执行
判定规则：
- 直接执行：当前可以立即执行动作，例如减速通过盲区、积水、弯道或正常转弯。
- 条件满足后执行：需要等待车辆、行人、信号、交警指令或安全条件满足后再执行。

示例：
输入：右前方车辆切入自车道，自车减速让行。
输出：
{
  "因素": [
    {
      "位置": "右前方",
      "大类": "车辆行为",
      "细节": "车辆切入",
      "原文片段": "右前方车辆切入自车道"
    }
  ],
  "动作": {
    "横向决策": ["保持"],
    "纵向决策": ["减速"],
    "执行策略": "条件满足后执行"
  }
}

输入：左前方存在车辆遮挡盲区，右前方存在豁口盲区，自车应减速行驶通过。
输出：
{
  "因素": [
    {
      "位置": "左前方",
      "大类": "盲区",
      "细节": "车辆遮挡盲区",
      "原文片段": "左前方存在车辆遮挡盲区"
    },
    {
      "位置": "右前方",
      "大类": "盲区",
      "细节": "豁口盲区",
      "原文片段": "右前方存在豁口盲区"
    }
  ],
  "动作": {
    "横向决策": ["保持"],
    "纵向决策": ["减速"],
    "执行策略": "直接执行"
  }
}

输入：前方事故车占道，右侧空闲，右后无车，应右变道。
输出：
{
  "因素": [
    {
      "位置": "前方",
      "大类": "障碍物",
      "细节": "事故车占道",
      "原文片段": "前方事故车占道"
    },
    {
      "位置": "右侧",
      "大类": "换道条件",
      "细节": "车道空闲",
      "原文片段": "右侧空闲"
    },
    {
      "位置": "右后方",
      "大类": "换道条件",
      "细节": "无来车",
      "原文片段": "右后无车"
    }
  ],
  "动作": {
    "横向决策": ["换道"],
    "纵向决策": ["保持"],
    "执行策略": "条件满足后执行"
  }
}
"""

DETAIL_SCORING_SYSTEM_PROMPT = """
你是一个自动驾驶场景语义细节评分模型。
只根据输入的 detail pair 给出相似度分数 score 和简短关系描述。
score 只能是 0.0、0.5、1.0。
必须严格只输出 JSON，不要输出 markdown，不要输出 ```json。
"""


def create_openai_client(base_url: str = DEFAULT_BASE_URL, api_key: str = "EMPTY") -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def normalize_action_list(value: Any, default: str = "保持") -> List[str]:
    if not value:
        return [default]
    if isinstance(value, str):
        items = [value]
    else:
        items = list(value)
    result = [safe_str(item) for item in items if safe_str(item)]
    return result or [default]


def normalize_extracted_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    factors = []
    for factor in scene.get("因素", []) or []:
        factors.append(
            {
                "位置": safe_str(factor.get("位置")),
                "大类": safe_str(factor.get("大类")),
                "细节": safe_str(factor.get("细节")),
                "原文片段": safe_str(factor.get("原文片段")),
            }
        )

    action = scene.get("动作", {}) or {}
    return {
        "因素": factors,
        "动作": {
            "横向决策": normalize_action_list(action.get("横向决策"), default="保持"),
            "纵向决策": normalize_action_list(action.get("纵向决策"), default="保持"),
            "执行策略": safe_str(action.get("执行策略")) or "直接执行",
        },
    }


def infer_category(fragment: str) -> str:
    if any(keyword in fragment for keyword in ("盲区", "遮挡", "豁口")):
        return "盲区"
    if any(keyword in fragment for keyword in ("事故车", "障碍", "占道", "锥桶", "施工", "大货车")):
        return "障碍物"
    if any(keyword in fragment for keyword in ("红灯", "绿灯", "黄灯", "信号灯")):
        return "交通信号"
    if any(keyword in fragment for keyword in ("切入", "变道", "行人", "车辆", "SUV", "车")):
        return "车辆行为"
    if any(keyword in fragment for keyword in ("空闲", "无车", "无来车", "安全")):
        return "换道条件"
    if any(keyword in fragment for keyword in ("拥堵", "积水", "弯道", "路面")):
        return "路况"
    return "车辆行为"


def extract_detail(fragment: str, position: str) -> str:
    detail = fragment.replace(position, "", 1)
    detail = re.sub(r"^(存在|有|一辆|一名|一个|的)", "", detail)
    detail = re.sub(r"(自车|应|应该|需要).*$", "", detail)
    detail = re.sub(r"[，。；、\s]+", "", detail)
    return detail or fragment


def infer_actions(summary: str) -> Dict[str, Any]:
    lateral = []
    if any(keyword in summary for keyword in ("变道", "换道", "并线")):
        lateral.append("换道")
    if any(keyword in summary for keyword in ("避让", "避障", "绕行")):
        lateral.append("避让")
    if any(keyword in summary for keyword in ("转弯", "左转", "右转", "掉头")):
        lateral.append("转弯")
    if not lateral:
        lateral.append("保持")

    longitudinal = []
    if any(keyword in summary for keyword in ("减速", "慢行", "让行")):
        longitudinal.append("减速")
    if any(keyword in summary for keyword in ("停车", "刹停", "停止")):
        longitudinal.append("停车")
    if any(keyword in summary for keyword in ("加速", "起步")):
        longitudinal.append("加速")
    if not longitudinal:
        longitudinal.append("保持")

    conditional_keywords = ("等待", "让行", "信号", "行人", "无车", "安全", "条件", "切入", "变道")
    strategy = "条件满足后执行" if any(keyword in summary for keyword in conditional_keywords) else "直接执行"
    return {"横向决策": lateral, "纵向决策": longitudinal, "执行策略": strategy}


def extract_coc_summary_locally(summary: str) -> Dict[str, Any]:
    factors = []
    position_pattern = "|".join(sorted((re.escape(item) for item in POSITIONS), key=len, reverse=True))
    for fragment in re.split(r"[，。；;]", summary):
        fragment = safe_str(fragment)
        if not fragment or "自车" in fragment:
            continue
        match = re.search(position_pattern, fragment)
        if match is None:
            continue
        position = match.group(0)
        factors.append(
            {
                "位置": position,
                "大类": infer_category(fragment),
                "细节": extract_detail(fragment, position),
                "原文片段": fragment,
            }
        )

    if not factors:
        factors.append(
            {
                "位置": "前方",
                "大类": infer_category(summary),
                "细节": extract_detail(summary, "前方"),
                "原文片段": summary,
            }
        )

    return normalize_extracted_scene({"因素": factors, "动作": infer_actions(summary)})


def extract_coc_summary(summary: str, client: Any, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": summary},
        ],
        temperature=0.0,
    )
    return normalize_extracted_scene(parse_json_object(response.choices[0].message.content))


def position_score(reference_pos: str, candidate_pos: str) -> float:
    reference_pos = safe_str(reference_pos)
    candidate_pos = safe_str(candidate_pos)
    if not reference_pos or not candidate_pos:
        return 0.0
    if reference_pos == candidate_pos:
        return 1.0

    direction_groups = [
        {"前方", "左前方", "右前方", "正前方"},
        {"左方", "左侧", "左前方", "左后方"},
        {"右方", "右侧", "右前方", "右后方"},
        {"后方", "左后方", "右后方", "正后方"},
    ]
    for group in direction_groups:
        if reference_pos in group and candidate_pos in group:
            return 0.5
    return 0.0


def category_score(reference_category: str, candidate_category: str) -> float:
    return 1.0 if safe_str(reference_category) == safe_str(candidate_category) else 0.0


def coarse_anchor_score(reference_factor: Dict[str, Any], candidate_factor: Dict[str, Any]) -> float:
    if category_score(reference_factor.get("大类"), candidate_factor.get("大类")) <= 0:
        return 0.0
    return position_score(reference_factor.get("位置"), candidate_factor.get("位置"))


def detail_sort_score(reference_detail: str, candidate_detail: str) -> float:
    reference_detail = safe_str(reference_detail)
    candidate_detail = safe_str(candidate_detail)
    if not reference_detail or not candidate_detail:
        return 0.0
    if reference_detail == candidate_detail:
        return 1.0
    if reference_detail in candidate_detail or candidate_detail in reference_detail:
        return 0.7

    reference_chars = set(reference_detail)
    candidate_chars = set(candidate_detail)
    union = reference_chars | candidate_chars
    if not union:
        return 0.0
    return 0.3 if len(reference_chars & candidate_chars) / len(union) >= 0.4 else 0.0


def match_one_factor(
    reference_factor: Dict[str, Any],
    candidate_factors: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ranked_candidates = []
    for candidate_factor in candidate_factors:
        coarse_score = coarse_anchor_score(reference_factor, candidate_factor)
        if coarse_score <= 0:
            continue
        sort_score = detail_sort_score(reference_factor.get("细节"), candidate_factor.get("细节"))
        ranked_candidates.append(
            {
                "candidate_factor": candidate_factor,
                "coarse_score": coarse_score,
                "sort_score": sort_score,
                "rank": coarse_score * 10 + sort_score,
            }
        )

    if not ranked_candidates:
        return {
            "reference_factor": reference_factor,
            "candidate_factor": None,
            "coarse_score": 0.0,
            "candidate_rank": [],
        }

    ranked_candidates.sort(key=lambda item: item["rank"], reverse=True)
    best = ranked_candidates[0]
    return {
        "reference_factor": reference_factor,
        "candidate_factor": best["candidate_factor"],
        "coarse_score": best["coarse_score"],
        "candidate_rank": ranked_candidates,
    }


def unmatched_factor_pair(reference_factor: Dict[str, Any], candidate_rank: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "reference_factor": reference_factor,
        "candidate_factor": None,
        "coarse_score": 0.0,
        "candidate_rank": candidate_rank or [],
    }


def match_factor_pairs_one_to_one(
    reference_factors: Sequence[Dict[str, Any]],
    candidate_factors: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ranked_by_reference: List[List[Dict[str, Any]]] = [[] for _ in reference_factors]
    all_ranked_pairs = []
    for reference_index, reference_factor in enumerate(reference_factors):
        for candidate_index, candidate_factor in enumerate(candidate_factors):
            coarse_score = coarse_anchor_score(reference_factor, candidate_factor)
            if coarse_score <= 0:
                continue
            sort_score = detail_sort_score(reference_factor.get("细节"), candidate_factor.get("细节"))
            ranked_item = {
                "candidate_factor": candidate_factor,
                "coarse_score": coarse_score,
                "sort_score": sort_score,
                "rank": coarse_score * 10 + sort_score,
            }
            ranked_by_reference[reference_index].append(ranked_item)
            all_ranked_pairs.append(
                {
                    "reference_index": reference_index,
                    "candidate_index": candidate_index,
                    "reference_factor": reference_factor,
                    "candidate_factor": candidate_factor,
                    "coarse_score": coarse_score,
                    "rank": ranked_item["rank"],
                }
            )

    for ranked_candidates in ranked_by_reference:
        ranked_candidates.sort(key=lambda item: item["rank"], reverse=True)

    matched_reference_indexes = set()
    matched_candidate_indexes = set()
    matched_pairs: Dict[int, Dict[str, Any]] = {}
    all_ranked_pairs.sort(key=lambda item: item["rank"], reverse=True)
    for pair in all_ranked_pairs:
        reference_index = pair["reference_index"]
        candidate_index = pair["candidate_index"]
        if reference_index in matched_reference_indexes or candidate_index in matched_candidate_indexes:
            continue
        matched_reference_indexes.add(reference_index)
        matched_candidate_indexes.add(candidate_index)
        matched_pairs[reference_index] = {
            "reference_factor": pair["reference_factor"],
            "candidate_factor": pair["candidate_factor"],
            "coarse_score": pair["coarse_score"],
            "candidate_rank": ranked_by_reference[reference_index],
        }

    factor_pairs = []
    for reference_index, reference_factor in enumerate(reference_factors):
        factor_pairs.append(
            matched_pairs.get(
                reference_index,
                unmatched_factor_pair(reference_factor, ranked_by_reference[reference_index]),
            )
        )
    return factor_pairs


def score_action_list(reference_list: Any, candidate_list: Any) -> float:
    reference_set = {safe_str(item) for item in (reference_list or []) if safe_str(item)}
    candidate_set = {safe_str(item) for item in (candidate_list or []) if safe_str(item)}
    if not reference_set or not candidate_set:
        return 0.0
    if reference_set == candidate_set:
        return 1.0
    if reference_set & candidate_set:
        return 0.5
    return 0.0


def score_action_text(reference_text: Any, candidate_text: Any) -> float:
    reference_text = safe_str(reference_text)
    candidate_text = safe_str(candidate_text)
    if not reference_text or not candidate_text:
        return 0.0
    if reference_text == candidate_text:
        return 1.0
    if reference_text in candidate_text or candidate_text in reference_text:
        return 0.5
    return 0.0


def score_actions(reference_action: Dict[str, Any], candidate_action: Dict[str, Any]) -> Dict[str, float]:
    return {
        "lat": score_action_list(reference_action.get("横向决策"), candidate_action.get("横向决策")),
        "lon": score_action_list(reference_action.get("纵向决策"), candidate_action.get("纵向决策")),
        "strategy": score_action_text(reference_action.get("执行策略"), candidate_action.get("执行策略")),
    }


def coarse_match(reference_scene: Dict[str, Any], candidate_scene: Dict[str, Any]) -> Dict[str, Any]:
    factor_pairs = match_factor_pairs_one_to_one(
        reference_scene.get("因素", []),
        candidate_scene.get("因素", []),
    )
    return {
        "factor_pairs": factor_pairs,
        "action_score": score_actions(reference_scene.get("动作", {}), candidate_scene.get("动作", {})),
    }


def build_detail_pairs(factor_pairs: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    detail_pairs = []
    for pair in factor_pairs:
        candidate_factor = pair.get("candidate_factor")
        if candidate_factor is None:
            continue
        detail_pairs.append(
            (
                safe_str(pair.get("reference_factor", {}).get("细节")),
                safe_str(candidate_factor.get("细节")),
            )
        )
    return detail_pairs


def score_detail_pairs(
    detail_pairs: Sequence[Tuple[str, str]],
    client: Any,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if not detail_pairs:
        return {"results": []}

    pairs_text = "\n".join(
        f'{index + 1}. ("{reference_detail}", "{candidate_detail}")'
        for index, (reference_detail, candidate_detail) in enumerate(detail_pairs)
    )
    user_prompt = f"""
现在输入如下 detail pair：
{pairs_text}

评分规则：
1.0 = 描述的是同一个细节、同一种风险来源或同一个对象状态
0.5 = 同类相关但关键细节不完全一致；或一个泛化、一个具体；或数量不同
0.0 = 不是同一件事，关键对象、机制或风险来源不同

严格只输出 JSON：
{{
  "results": [
    {{
      "reference_detail": "...",
      "candidate_detail": "...",
      "relation": "...",
      "score": 0.0
    }}
  ]
}}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DETAIL_SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return parse_json_object(response.choices[0].message.content)


def score_detail_pairs_locally(detail_pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    results = []
    for reference_detail, candidate_detail in detail_pairs:
        score = detail_sort_score(reference_detail, candidate_detail)
        if score >= 0.75:
            normalized_score = 1.0
            relation = "本地规则判断为同一细节"
        elif score >= 0.25:
            normalized_score = 0.5
            relation = "本地规则判断为同类相关"
        else:
            normalized_score = 0.0
            relation = "本地规则判断为不匹配"
        results.append(
            {
                "reference_detail": reference_detail,
                "candidate_detail": candidate_detail,
                "relation": relation,
                "score": normalized_score,
            }
        )
    return {"results": results}


def extract_detail_scores(detail_score_result: Dict[str, Any], expected_len: int) -> List[float]:
    scores = []
    for item in detail_score_result.get("results", []) or []:
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if score not in DETAIL_SCORE_VALUES:
            if score >= 0.75:
                score = 1.0
            elif score >= 0.25:
                score = 0.5
            else:
                score = 0.0
        scores.append(score)
    if len(scores) < expected_len:
        scores.extend([0.0] * (expected_len - len(scores)))
    return scores[:expected_len]


def repair_detail_scores_with_local_rules(
    detail_pairs: Sequence[Tuple[str, str]],
    detail_score_result: Dict[str, Any],
    detail_scores: Sequence[float],
) -> Tuple[Dict[str, Any], List[float]]:
    local_result = score_detail_pairs_locally(detail_pairs)
    local_scores = extract_detail_scores(local_result, len(detail_pairs))
    llm_results = list(detail_score_result.get("results", []) or [])
    repaired_scores = list(detail_scores)
    repaired_results = llm_results[: len(detail_pairs)]
    changed = False

    while len(repaired_scores) < len(detail_pairs):
        repaired_scores.append(local_scores[len(repaired_scores)])
        changed = True

    while len(repaired_results) < len(detail_pairs):
        repaired_results.append(local_result["results"][len(repaired_results)])
        changed = True

    for index, (reference_detail, candidate_detail) in enumerate(detail_pairs):
        if safe_str(reference_detail) == safe_str(candidate_detail) and repaired_scores[index] < 1.0:
            repaired_scores[index] = 1.0
            repaired_results[index] = {
                "reference_detail": reference_detail,
                "candidate_detail": candidate_detail,
                "relation": "相同细节，本地规则修正为满分",
                "score": 1.0,
            }
            changed = True

    if changed:
        repaired_result = dict(detail_score_result)
        repaired_result["results"] = repaired_results
        repaired_result["repaired_by_local_rules"] = True
        return repaired_result, repaired_scores

    return detail_score_result, list(detail_scores)


def merge_detail_scores(
    factor_pairs: Sequence[Dict[str, Any]],
    detail_scores: Sequence[float],
) -> List[Dict[str, Any]]:
    merged = []
    score_index = 0
    for pair in factor_pairs:
        item = dict(pair)
        if item.get("candidate_factor") is None:
            item["fine_score"] = 0.0
        else:
            item["fine_score"] = float(detail_scores[score_index])
            score_index += 1
        merged.append(item)
    return merged


def build_fine_comparisons(
    detail_pairs: Sequence[Tuple[str, str]],
    detail_score_result: Dict[str, Any],
    detail_scores: Sequence[float],
) -> List[Dict[str, Any]]:
    llm_items = detail_score_result.get("results", []) or []
    comparisons = []
    for index, (reference_detail, candidate_detail) in enumerate(detail_pairs):
        llm_item = llm_items[index] if index < len(llm_items) and isinstance(llm_items[index], dict) else {}
        comparisons.append(
            {
                "index": index + 1,
                "reference_detail": reference_detail,
                "candidate_detail": candidate_detail,
                "relation": safe_str(llm_item.get("relation")),
                "fine_score": float(detail_scores[index]) if index < len(detail_scores) else 0.0,
            }
        )
    return comparisons


def compact_factor(factor: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if factor is None:
        return None
    return {
        "位置": safe_str(factor.get("位置")),
        "大类": safe_str(factor.get("大类")),
        "细节": safe_str(factor.get("细节")),
    }


def compact_action_score(action_score: Dict[str, float]) -> Dict[str, Any]:
    values = [
        float(action_score.get("lat", 0.0)),
        float(action_score.get("lon", 0.0)),
        float(action_score.get("strategy", 0.0)),
    ]
    return {
        "横向决策": values[0],
        "纵向决策": values[1],
        "执行策略": values[2],
        "平均分": round(sum(values) / len(values), 4),
    }


def build_compact_result(
    factor_pairs: Sequence[Dict[str, Any]],
    action_score: Dict[str, float],
    summary_score: Dict[str, float],
    fine_comparisons: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    fine_by_detail = {
        (item["reference_detail"], item["candidate_detail"]): item
        for item in fine_comparisons
    }
    matches = []
    for index, pair in enumerate(factor_pairs):
        reference_factor = pair.get("reference_factor")
        candidate_factor = pair.get("candidate_factor")
        fine_item = {}
        if candidate_factor is not None:
            fine_key = (
                safe_str(reference_factor.get("细节")) if reference_factor else "",
                safe_str(candidate_factor.get("细节")),
            )
            fine_item = fine_by_detail.get(fine_key, {})

        matches.append(
            {
                "index": index + 1,
                "status": "matched" if candidate_factor is not None else "unmatched",
                "reference": compact_factor(reference_factor),
                "candidate": compact_factor(candidate_factor),
                "relation": safe_str(fine_item.get("relation")),
                "scores": {
                    "coarse": round(float(pair.get("coarse_score", 0.0)), 4),
                    "fine": round(float(pair.get("fine_score", 0.0)), 4),
                    "combined": round(float(pair.get("combined_score", 0.0)), 4),
                },
            }
        )

    return {
        "summary_score": summary_score,
        "factor_matches": matches,
        "fine_comparisons": list(fine_comparisons),
        "action_score": compact_action_score(action_score),
    }


def calculate_final_result(
    factor_pairs: Sequence[Dict[str, Any]],
    action_score: Dict[str, float],
    coarse_weight: float = 0.5,
    fine_weight: float = 0.5,
) -> Dict[str, Any]:
    scored_pairs = []
    factor_scores = []
    for pair in factor_pairs:
        coarse_score = float(pair.get("coarse_score", 0.0))
        fine_score = float(pair.get("fine_score", 0.0))
        if pair.get("candidate_factor") is None:
            fine_score = 0.0
        factor_score = coarse_weight * coarse_score + fine_weight * fine_score
        item = dict(pair)
        item["combined_score"] = round(factor_score, 4)
        item["factor_score"] = round(factor_score, 4)
        scored_pairs.append(item)
        factor_scores.append(factor_score)

    action_detail = {
        "lat": float(action_score.get("lat", 0.0)),
        "lon": float(action_score.get("lon", 0.0)),
        "strategy": float(action_score.get("strategy", 0.0)),
    }
    factor_total = sum(factor_scores)
    action_total = sum(action_detail.values())
    raw_total = factor_total + action_total
    max_score = len(factor_scores) + 3
    normalized_total = raw_total / max_score if max_score else 0.0

    return {
        "factor_pairs": scored_pairs,
        "action_score": action_score,
        "summary_score": {
            "factor_scores": [round(score, 4) for score in factor_scores],
            "factor_total": round(factor_total, 4),
            "action_scores": action_detail,
            "action_total": round(action_total, 4),
            "raw_total_score": round(raw_total, 4),
            "max_score": max_score,
            "normalized_total_score": round(normalized_total, 4),
        },
        "score_formula": {
            "factor": "factor_score = 0.5 * coarse_score + 0.5 * fine_score; unmatched fine_score = 0",
            "action": "lat/lon/strategy each has max score 1.0",
            "total": "raw_total_score = sum(factor_scores) + lat + lon + strategy; normalized_total_score = raw_total_score / (factor_count + 3)",
        },
    }


def evaluate_extracted_scenes(
    reference_scene: Dict[str, Any],
    candidate_scene: Dict[str, Any],
    client: Optional[Any] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    reference_scene = normalize_extracted_scene(reference_scene)
    candidate_scene = normalize_extracted_scene(candidate_scene)
    coarse_result = coarse_match(reference_scene, candidate_scene)
    detail_pairs = build_detail_pairs(coarse_result["factor_pairs"])

    if not detail_pairs:
        detail_score_result = {"results": []}
        detail_scores = []
    elif client is None:
        detail_score_result = score_detail_pairs_locally(detail_pairs)
        detail_scores = extract_detail_scores(detail_score_result, len(detail_pairs))
    else:
        try:
            detail_score_result = score_detail_pairs(detail_pairs, client=client, model=model)
            detail_scores = extract_detail_scores(detail_score_result, len(detail_pairs))
        except Exception as exc:
            detail_score_result = score_detail_pairs_locally(detail_pairs)
            detail_score_result["detail_score_fallback_error"] = f"{type(exc).__name__}: {exc}"
            detail_scores = extract_detail_scores(detail_score_result, len(detail_pairs))
        else:
            detail_score_result, detail_scores = repair_detail_scores_with_local_rules(
                detail_pairs,
                detail_score_result,
                detail_scores,
            )

    merged_pairs = merge_detail_scores(coarse_result["factor_pairs"], detail_scores)
    final_result = calculate_final_result(merged_pairs, coarse_result["action_score"])
    fine_comparisons = build_fine_comparisons(detail_pairs, detail_score_result, detail_scores)
    final_result["detail_pairs"] = detail_pairs
    final_result["fine_comparisons"] = fine_comparisons
    final_result["llm_detail_score"] = detail_score_result
    final_result["compact_result"] = build_compact_result(
        factor_pairs=final_result["factor_pairs"],
        action_score=final_result["action_score"],
        summary_score=final_result["summary_score"],
        fine_comparisons=fine_comparisons,
    )
    return final_result


def evaluate_coc_pair(
    reference_summary: str,
    candidate_summary: str,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    save_details: bool = True,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    try:
        client = create_openai_client(base_url=base_url, api_key=api_key)
        reference_scene = extract_coc_summary(reference_summary, client=client, model=model)
        candidate_scene = extract_coc_summary(candidate_summary, client=client, model=model)
        extraction_mode = "llm"
    except Exception as exc:
        client = None
        reference_scene = extract_coc_summary_locally(reference_summary)
        candidate_scene = extract_coc_summary_locally(candidate_summary)
        extraction_mode = "local_fallback"
        fallback_error = f"{type(exc).__name__}: {exc}"

    if save_details:
        save_json(reference_scene, output_dir / "reference.json")
        save_json(candidate_scene, output_dir / "candidate.json")

    final_result = evaluate_extracted_scenes(
        reference_scene=reference_scene,
        candidate_scene=candidate_scene,
        client=client,
        model=model,
    )
    final_result["json_files"] = {}
    if save_details:
        final_result["json_files"] = {
            "reference": str(output_dir / "reference.json"),
            "candidate": str(output_dir / "candidate.json"),
        }
    final_result["extraction_mode"] = extraction_mode
    if extraction_mode == "local_fallback":
        final_result["fallback_error"] = fallback_error
        final_result["compact_result"]["extraction_mode"] = extraction_mode

    if save_details:
        save_json({"factor_pairs": final_result["factor_pairs"], "action_score": final_result["action_score"]}, output_dir / "coarse_and_merged.json")
        save_json({"fine_comparisons": final_result["fine_comparisons"], "llm_result": final_result["llm_detail_score"]}, output_dir / "fine_score.json")
        save_json(final_result["compact_result"], output_dir / "compact_result.json")
        save_json(final_result, output_dir / "final_result.json")
    return final_result
