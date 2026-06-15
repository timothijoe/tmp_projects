#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

from formal_reward_core import score_actions


DIRECTION_SWAP = {
    "左前方": "右前方",
    "右前方": "左前方",
    "左侧": "右侧",
    "右侧": "左侧",
    "左后方": "右后方",
    "右后方": "左后方",
    "前方": "右前方",
}

SAME_CATEGORY_DETAIL = {
    "盲区": ["车辆遮挡盲区", "豁口盲区", "弯道盲区", "无灯路口盲区", "环岛盲区", "匝道盲区", "坡道盲区", "施工盲区"],
    "障碍物": ["锥桶占道", "施工区域占道", "事故车占道", "车辆占道停车", "行人占道", "两轮骑行者占道", "三轮骑行者占道"],
    "车辆行为": ["车辆切入", "车辆缓行", "车辆逆向行驶", "行人通行", "两轮骑行者通行", "三轮骑行者通行"],
    "路况": ["坑洼路面", "积水路面", "井盖凹陷", "减速带", "湿滑路面", "大曲率弯道"],
    "交通信号": ["红灯", "绿灯", "黄灯", "左转箭头灯红灯", "直行箭头灯黄灯"],
    "换道条件": ["目标车道空闲", "目标车道拥堵", "后方无来车", "后方有快速来车", "存在安全距离"],
    "交通管制": ["交警示意停车", "交警示意通行", "交警示意左转"],
    "跟车": ["前车起步", "前车缓行", "前车等待红绿灯"],
}

CROSS_CATEGORY = [
    ("盲区", "障碍物", "锥桶占道"),
    ("障碍物", "盲区", "车辆遮挡盲区"),
    ("车辆行为", "路况", "坑洼路面"),
    ("路况", "车辆行为", "车辆切入"),
    ("交通信号", "障碍物", "施工区域占道"),
    ("换道条件", "盲区", "豁口盲区"),
    ("交通管制", "交通信号", "红灯"),
    ("跟车", "障碍物", "事故车占道"),
]


def choose_other_detail(category: str, current: str) -> str:
    options = SAME_CATEGORY_DETAIL.get(category, [])
    for option in options:
        if option and option not in current and current not in option:
            return option
    return f"{current}变化"


def choose_cross_category(category: str) -> Dict[str, str]:
    for old_category, new_category, new_detail in CROSS_CATEGORY:
        if category == old_category:
            return {"category": new_category, "detail": new_detail}
    return {"category": "盲区", "detail": "车辆遮挡盲区"}


def render_factor(factor: Dict[str, str]) -> str:
    position = factor.get("position") or "前方"
    detail = factor.get("detail") or factor.get("category") or "风险"
    category = factor.get("category", "")
    if category == "换道条件":
        return f"{position}{detail}"
    return f"{position}存在{detail}"


def render_candidate(factors: List[Dict[str, str]], action_text: str) -> str:
    descriptions = [render_factor(factor) for factor in factors]
    if descriptions:
        return "，".join(descriptions) + f"，{action_text}。"
    return f"{action_text}。"


def gt_avg(scores: List[float]) -> float:
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def formal_action(action_schema: Dict) -> Dict:
    raw = action_schema.get("raw") if isinstance(action_schema, dict) else None
    if raw:
        return raw
    if isinstance(action_schema, dict) and any(key in action_schema for key in ["横向决策", "纵向决策", "执行策略"]):
        return {
            "横向决策": action_schema.get("横向决策", ["保持"]),
            "纵向决策": action_schema.get("纵向决策", ["保持"]),
            "执行策略": action_schema.get("执行策略", "直接执行"),
        }
    return {
        "横向决策": action_schema.get("lat", ["保持"]) if isinstance(action_schema, dict) else ["保持"],
        "纵向决策": action_schema.get("lon", ["保持"]) if isinstance(action_schema, dict) else ["保持"],
        "执行策略": action_schema.get("strategy", "直接执行") if isinstance(action_schema, dict) else "直接执行",
    }


def gt_action_scores(reference_action_schema: Dict, candidate_action_schema: Dict | None = None) -> Dict[str, float]:
    candidate_action_schema = candidate_action_schema or reference_action_schema
    return score_actions(formal_action(reference_action_schema), formal_action(candidate_action_schema))


def base_record(index: int, schema: Dict, candidate: str, error_type: str, scores: List[float], edits: List[Dict], subtype: str = "") -> Dict:
    reference_action_schema = schema.get("action_schema", {})
    candidate_action_schema = reference_action_schema
    return {
        "index": index,
        "source_id": schema["source_id"],
        "reference": schema["source"],
        "candidate": candidate,
        "category": primary_category(schema),
        "error_type": error_type,
        "factor_error_subtype": subtype,
        "gt_factor_scores": scores,
        "gt_factor_avg": gt_avg(scores),
        "gt_action_scores": gt_action_scores(reference_action_schema, candidate_action_schema),
        "factor_edits": edits,
        "reference_factors": schema["factors"],
        "reference_action_schema": reference_action_schema,
        "candidate_action_schema": candidate_action_schema,
    }


def primary_category(schema: Dict) -> str:
    factors = schema.get("factors", [])
    return factors[0].get("category", "unknown") if factors else "unknown"


def copy_factors(schema: Dict) -> List[Dict]:
    return [dict(factor) for factor in schema.get("factors", [])]


def generate_for_schema(schema: Dict, start_index: int) -> List[Dict]:
    records = []
    index = start_index
    factors = copy_factors(schema)
    n = len(factors)
    if n == 0:
        return records

    full_scores = [1.0] * n
    records.append(
        base_record(index, schema, schema["source"], "complete", full_scores, [])
    )
    index += 1

    # Missing one factor. Generate up to two samples for multi-factor scenes.
    for factor_pos in range(min(n, 2)):
        candidate_factors = [dict(f) for i, f in enumerate(factors) if i != factor_pos]
        scores = [0.0 if i == factor_pos else 1.0 for i in range(n)]
        edits = [
            {
                "factor_index": factors[factor_pos]["factor_index"],
                "edit_type": "missing_factor",
                "field": "factor",
                "from": factors[factor_pos],
                "to": None,
                "gt_factor_score": 0.0,
            }
        ]
        records.append(
            base_record(
                index,
                schema,
                render_candidate(candidate_factors, schema["action_text"]),
                "missing_factor",
                scores,
                edits,
            )
        )
        index += 1

    # Direction swap.
    for factor_pos, factor in enumerate(factors):
        if factor.get("position") not in DIRECTION_SWAP:
            continue
        candidate_factors = copy_factors(schema)
        old = candidate_factors[factor_pos]["position"]
        new = DIRECTION_SWAP[old]
        candidate_factors[factor_pos]["position"] = new
        scores = [1.0] * n
        scores[factor_pos] = 0.5
        edits = [
            {
                "factor_index": factor["factor_index"],
                "edit_type": "direction_swap",
                "field": "position",
                "from": old,
                "to": new,
                "gt_factor_score": 0.5,
            }
        ]
        records.append(
            base_record(
                index,
                schema,
                render_candidate(candidate_factors, schema["action_text"]),
                "wrong_factor",
                scores,
                edits,
                "direction_swap",
            )
        )
        index += 1
        break

    # Same category, wrong subtype/detail.
    for factor_pos, factor in enumerate(factors):
        new_detail = choose_other_detail(factor.get("category", ""), factor.get("detail", ""))
        if not new_detail:
            continue
        candidate_factors = copy_factors(schema)
        old = candidate_factors[factor_pos]["detail"]
        candidate_factors[factor_pos]["detail"] = new_detail
        scores = [1.0] * n
        scores[factor_pos] = 0.5
        edits = [
            {
                "factor_index": factor["factor_index"],
                "edit_type": "same_category_subtype_swap",
                "field": "detail",
                "from": old,
                "to": new_detail,
                "gt_factor_score": 0.5,
            }
        ]
        records.append(
            base_record(
                index,
                schema,
                render_candidate(candidate_factors, schema["action_text"]),
                "wrong_factor",
                scores,
                edits,
                "same_category_subtype_swap",
            )
        )
        index += 1
        break

    # Cross category.
    for factor_pos, factor in enumerate(factors):
        replacement = choose_cross_category(factor.get("category", ""))
        candidate_factors = copy_factors(schema)
        old = {
            "category": candidate_factors[factor_pos]["category"],
            "detail": candidate_factors[factor_pos]["detail"],
        }
        candidate_factors[factor_pos]["category"] = replacement["category"]
        candidate_factors[factor_pos]["detail"] = replacement["detail"]
        scores = [1.0] * n
        scores[factor_pos] = 0.0
        edits = [
            {
                "factor_index": factor["factor_index"],
                "edit_type": "cross_category_swap",
                "field": "category_detail",
                "from": old,
                "to": replacement,
                "gt_factor_score": 0.0,
            }
        ]
        records.append(
            base_record(
                index,
                schema,
                render_candidate(candidate_factors, schema["action_text"]),
                "wrong_factor",
                scores,
                edits,
                "cross_category_swap",
            )
        )
        index += 1
        break

    # Extra hallucinated factor. Reference factors remain correct, but an extra penalty is recorded.
    extra_factor = {"factor_index": n + 1, "position": "前方", "category": "障碍物", "detail": "未提及施工人员", "text_span": ""}
    candidate_factors = copy_factors(schema) + [extra_factor]
    edits = [
        {
            "factor_index": n + 1,
            "edit_type": "extra_factor",
            "field": "factor",
            "from": None,
            "to": extra_factor,
            "gt_factor_score": "penalty_only",
        }
    ]
    records.append(
        base_record(
            index,
            schema,
            render_candidate(candidate_factors, schema["action_text"]),
            "extra_factor",
            full_scores,
            edits,
            "hallucinated_factor",
        )
    )
    return records


def export(records: List[Dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates_with_gt.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pairs = [
        {"index": record["index"], "reference": record["reference"], "candidate": record["candidate"]}
        for record in records
    ]
    (output_dir / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "record_count": len(records),
        "source_count": len({record["source_id"] for record in records}),
        "files": {
            "pairs": "pairs.json",
            "gt": "candidates_with_gt.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas", default="reward_model_v2/data/gt_schema/source_schemas.json")
    parser.add_argument("--output-dir", default="reward_model_v2/data/gt_candidates")
    args = parser.parse_args()

    schemas = json.loads(Path(args.schemas).read_text(encoding="utf-8"))
    records = []
    next_index = 1
    for schema in schemas:
        generated = generate_for_schema(schema, next_index)
        records.extend(generated)
        next_index += len(generated)
    export(records, Path(args.output_dir))
    print(f"wrote {len(records)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
