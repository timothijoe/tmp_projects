#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from formal_reward_core import score_actions


DIRECTION_SWAP = {
    "左前方": "右前方",
    "右前方": "左前方",
    "左侧": "右侧",
    "右侧": "左侧",
    "左后方": "右后方",
    "右后方": "左后方",
    "前方": "右前方",
    "后方": "前方",
    "当前车道": "右侧",
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

CATEGORIES = list(SAME_CATEGORY_DETAIL)

MUTATION_FACTOR_SCORE = {
    "KEEP_FACTOR": 1.0,
    "REPLACE_FACTOR_VALUE": 0.85,
    "REPLACE_DIRECTION": 0.65,
    "REPLACE_SUB_CATEGORY": 0.70,
    "REPLACE_SUPER_CATEGORY": 0.45,
    "CROSS_CATEGORY": 0.25,
    "REMOVE_FACTOR": 0.0,
}

MUTATION_WEIGHTS = {
    "mild": {
        "KEEP_FACTOR": 0.62,
        "REPLACE_FACTOR_VALUE": 0.12,
        "REPLACE_DIRECTION": 0.09,
        "REPLACE_SUB_CATEGORY": 0.11,
        "REPLACE_SUPER_CATEGORY": 0.03,
        "CROSS_CATEGORY": 0.01,
        "REMOVE_FACTOR": 0.02,
    },
    "medium": {
        "KEEP_FACTOR": 0.42,
        "REPLACE_FACTOR_VALUE": 0.10,
        "REPLACE_DIRECTION": 0.12,
        "REPLACE_SUB_CATEGORY": 0.13,
        "REPLACE_SUPER_CATEGORY": 0.09,
        "CROSS_CATEGORY": 0.07,
        "REMOVE_FACTOR": 0.07,
    },
    "severe": {
        "KEEP_FACTOR": 0.22,
        "REPLACE_FACTOR_VALUE": 0.06,
        "REPLACE_DIRECTION": 0.12,
        "REPLACE_SUB_CATEGORY": 0.12,
        "REPLACE_SUPER_CATEGORY": 0.18,
        "CROSS_CATEGORY": 0.17,
        "REMOVE_FACTOR": 0.13,
    },
}

ADD_FACTOR_WEIGHTS = {
    "mild": [0.78, 0.20, 0.02],
    "medium": [0.55, 0.32, 0.13],
    "severe": [0.35, 0.38, 0.20, 0.07],
}

ACTION_MUTATION_WEIGHTS = {
    "mild": {"KEEP_ACTION": 0.82, "REPLACE_ACTION": 0.13, "CONFLICT_ACTION": 0.03, "REMOVE_ACTION": 0.02},
    "medium": {"KEEP_ACTION": 0.64, "REPLACE_ACTION": 0.19, "CONFLICT_ACTION": 0.10, "REMOVE_ACTION": 0.07},
    "severe": {"KEEP_ACTION": 0.45, "REPLACE_ACTION": 0.22, "CONFLICT_ACTION": 0.20, "REMOVE_ACTION": 0.13},
}

ACTION_OPTIONS = {
    "lat": ["保持", "换道", "避让", "转弯"],
    "lon": ["保持", "加速", "减速", "停车"],
    "strategy": ["直接执行", "条件满足后执行"],
}


def weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    names = list(weights)
    return rng.choices(names, weights=[weights[name] for name in names], k=1)[0]


def choose_other(rng: random.Random, options: List[str], current: str) -> str:
    candidates = [item for item in options if item != current and item not in current and current not in item]
    if not candidates:
        candidates = [item for item in options if item != current]
    return rng.choice(candidates) if candidates else f"{current}变化"


def choose_other_category(rng: random.Random, current: str) -> str:
    options = [category for category in CATEGORIES if category != current]
    return rng.choice(options) if options else "障碍物"


def value_variant(detail: str) -> str:
    if "两个" in detail:
        return detail.replace("两个", "一个", 1)
    if "一辆" in detail:
        return detail.replace("一辆", "多辆", 1)
    if "缓慢" in detail:
        return detail.replace("缓慢", "快速", 1)
    if "快速" in detail:
        return detail.replace("快速", "缓慢", 1)
    if "严重" in detail:
        return detail.replace("严重", "轻微", 1)
    if "轻微" in detail:
        return detail.replace("轻微", "严重", 1)
    return f"轻微{detail}" if detail else "轻微风险"


def render_factor(factor: Dict[str, Any]) -> str:
    position = factor.get("position") or "前方"
    detail = factor.get("detail") or factor.get("category") or "风险"
    category = factor.get("category", "")
    if category == "换道条件":
        return f"{position}{detail}"
    return f"{position}存在{detail}"


def render_action(action_schema: Dict[str, Any]) -> str:
    lat = action_schema.get("lat") or ["保持"]
    lon = action_schema.get("lon") or ["保持"]
    strategy = action_schema.get("strategy") or "直接执行"
    parts = []
    if lat and lat != ["保持"]:
        parts.extend(lat)
    if lon and lon != ["保持"]:
        parts.extend(lon)
    if not parts:
        parts.append("保持行驶")
    action = "并".join(parts)
    if strategy == "条件满足后执行":
        return f"自车应在条件满足后{action}"
    return f"自车应{action}"


def render_candidate(factors: List[Dict[str, Any]], action_schema: Dict[str, Any]) -> str:
    descriptions = [render_factor(factor) for factor in factors]
    action_text = render_action(action_schema)
    if descriptions:
        return "，".join(descriptions) + f"，{action_text}。"
    return f"{action_text}。"


def gt_avg(scores: List[float]) -> float:
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def formal_action(action_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "横向决策": action_schema.get("lat", ["保持"]),
        "纵向决策": action_schema.get("lon", ["保持"]),
        "执行策略": action_schema.get("strategy", "直接执行"),
    }


def normalize_action_schema(action_schema: Dict[str, Any]) -> Dict[str, Any]:
    raw = action_schema.get("raw", {}) if isinstance(action_schema, dict) else {}
    lat = action_schema.get("lat") or raw.get("横向决策") or ["保持"]
    lon = action_schema.get("lon") or raw.get("纵向决策") or ["保持"]
    strategy = action_schema.get("strategy") or raw.get("执行策略") or "直接执行"
    return {"lat": list(lat), "lon": list(lon), "strategy": strategy}


def with_raw_action(action_schema: Dict[str, Any]) -> Dict[str, Any]:
    result = normalize_action_schema(action_schema)
    result["raw"] = formal_action(result)
    return result


def action_scores(reference_action_schema: Dict[str, Any], candidate_action_schema: Dict[str, Any]) -> Dict[str, float]:
    return score_actions(formal_action(normalize_action_schema(reference_action_schema)), formal_action(candidate_action_schema))


def primary_category(schema: Dict[str, Any]) -> str:
    factors = schema.get("factors", [])
    return factors[0].get("category", "unknown") if factors else "unknown"


def copy_factors(schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(factor) for factor in schema.get("factors", [])]


def mutate_factor(factor: Dict[str, Any], mutation_type: str, rng: random.Random) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    mutated = dict(factor)
    original = dict(factor)
    if mutation_type == "KEEP_FACTOR":
        to_factor = dict(mutated)
    elif mutation_type == "REMOVE_FACTOR":
        to_factor = None
    elif mutation_type == "REPLACE_DIRECTION":
        mutated["position"] = DIRECTION_SWAP.get(mutated.get("position"), choose_other(rng, list(DIRECTION_SWAP), mutated.get("position", "")))
        to_factor = dict(mutated)
    elif mutation_type == "REPLACE_SUB_CATEGORY":
        category = mutated.get("category", "")
        mutated["detail"] = choose_other(rng, SAME_CATEGORY_DETAIL.get(category, []), mutated.get("detail", ""))
        to_factor = dict(mutated)
    elif mutation_type == "REPLACE_SUPER_CATEGORY":
        category = choose_other_category(rng, mutated.get("category", ""))
        mutated["category"] = category
        mutated["detail"] = rng.choice(SAME_CATEGORY_DETAIL.get(category, [category]))
        to_factor = dict(mutated)
    elif mutation_type == "CROSS_CATEGORY":
        category = choose_other_category(rng, mutated.get("category", ""))
        mutated["position"] = DIRECTION_SWAP.get(mutated.get("position"), "前方")
        mutated["category"] = category
        mutated["detail"] = rng.choice(SAME_CATEGORY_DETAIL.get(category, [category]))
        to_factor = dict(mutated)
    elif mutation_type == "REPLACE_FACTOR_VALUE":
        mutated["detail"] = value_variant(mutated.get("detail", ""))
        to_factor = dict(mutated)
    else:
        to_factor = dict(mutated)

    score = MUTATION_FACTOR_SCORE.get(mutation_type, 1.0)
    record = {
        "factor_index": factor.get("factor_index"),
        "action": "keep" if mutation_type == "KEEP_FACTOR" else ("remove" if mutation_type == "REMOVE_FACTOR" else "replace"),
        "mutation_type": mutation_type,
        "from": original,
        "to": to_factor,
        "gt_factor_score": score,
    }
    return to_factor, record


def make_added_factor(next_index: int, rng: random.Random) -> Dict[str, Any]:
    category = rng.choice(CATEGORIES)
    return {
        "factor_index": next_index,
        "position": rng.choice(list(DIRECTION_SWAP)),
        "category": category,
        "detail": rng.choice(SAME_CATEGORY_DETAIL.get(category, [category])),
        "text_span": "",
        "source": "added_by_mutation",
    }


def mutate_action_dimension(value: Any, dimension: str, mutation_type: str, rng: random.Random) -> tuple[Any, Any]:
    options = ACTION_OPTIONS[dimension]
    if dimension == "strategy":
        old = value or "直接执行"
        if mutation_type == "KEEP_ACTION":
            return old, old
        if mutation_type == "REMOVE_ACTION":
            return old, "直接执行"
        return old, choose_other(rng, options, old)

    old_list = list(value or ["保持"])
    old_primary = old_list[0] if old_list else "保持"
    if mutation_type == "KEEP_ACTION":
        return old_list, old_list
    if mutation_type == "REMOVE_ACTION":
        return old_list, ["保持"]
    if mutation_type == "CONFLICT_ACTION":
        conflict = {"加速": "停车", "停车": "加速", "减速": "加速", "换道": "保持", "避让": "保持", "转弯": "保持", "保持": options[-1]}
        return old_list, [conflict.get(old_primary, choose_other(rng, options, old_primary))]
    return old_list, [choose_other(rng, options, old_primary)]


def mutate_action(action_schema: Dict[str, Any], severity: str, rng: random.Random) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reference = normalize_action_schema(action_schema)
    candidate = normalize_action_schema(action_schema)
    mutations = []
    for dimension in ["lat", "lon", "strategy"]:
        mutation_type = weighted_choice(rng, ACTION_MUTATION_WEIGHTS[severity])
        old, new = mutate_action_dimension(reference[dimension], dimension, mutation_type, rng)
        candidate[dimension] = new
        mutations.append(
            {
                "dimension": dimension,
                "action": "keep" if mutation_type == "KEEP_ACTION" else ("remove" if mutation_type == "REMOVE_ACTION" else "replace"),
                "mutation_type": mutation_type,
                "from": old,
                "to": new,
                "gt_action_score": 1.0 if mutation_type == "KEEP_ACTION" else None,
            }
        )
    return with_raw_action(candidate), mutations


def summarize_mutations(factor_mutations: List[Dict[str, Any]], action_mutations: List[Dict[str, Any]]) -> Dict[str, Any]:
    factor_counts = Counter(item["mutation_type"] for item in factor_mutations)
    action_counts = Counter(item["mutation_type"] for item in action_mutations)
    return {
        "num_gt_factors": sum(1 for item in factor_mutations if item["action"] != "add"),
        "num_kept_factors": factor_counts["KEEP_FACTOR"],
        "num_removed_factors": factor_counts["REMOVE_FACTOR"],
        "num_replaced_factors": sum(factor_counts[name] for name in MUTATION_FACTOR_SCORE if name not in {"KEEP_FACTOR", "REMOVE_FACTOR"}),
        "num_added_factors": factor_counts["ADD_FACTOR"],
        "num_direction_changes": factor_counts["REPLACE_DIRECTION"],
        "num_factor_value_changes": factor_counts["REPLACE_FACTOR_VALUE"],
        "num_sub_category_replacements": factor_counts["REPLACE_SUB_CATEGORY"],
        "num_super_category_replacements": factor_counts["REPLACE_SUPER_CATEGORY"],
        "num_cross_category_replacements": factor_counts["CROSS_CATEGORY"],
        "num_total_factor_mutations": sum(1 for item in factor_mutations if item["mutation_type"] not in {"KEEP_FACTOR"}),
        "num_kept_action_dimensions": action_counts["KEEP_ACTION"],
        "num_removed_action_dimensions": action_counts["REMOVE_ACTION"],
        "num_replaced_action_dimensions": action_counts["REPLACE_ACTION"],
        "num_conflict_action_dimensions": action_counts["CONFLICT_ACTION"],
        "num_total_action_mutations": sum(1 for item in action_mutations if item["mutation_type"] != "KEEP_ACTION"),
    }


def provisional_scores(
    gt_factor_scores: List[float],
    add_count: int,
    reference_action_schema: Dict[str, Any],
    candidate_action_schema: Dict[str, Any],
) -> Dict[str, Any]:
    factor_avg = gt_avg(gt_factor_scores)
    add_penalty = min(0.25, 0.06 * add_count)
    factor_after_add_penalty = round(max(0.0, factor_avg - add_penalty), 4)
    actions = action_scores(reference_action_schema, candidate_action_schema)
    action_avg = round(sum(actions.values()) / len(actions), 4)
    final = round(0.75 * factor_after_add_penalty + 0.25 * action_avg, 4)
    return {
        "gt_factor_avg": factor_avg,
        "add_factor_penalty": add_penalty,
        "gt_factor_avg_after_add_penalty": factor_after_add_penalty,
        "gt_action_scores": actions,
        "gt_action_avg": action_avg,
        "provisional_gt_score": final,
    }


def error_labels(summary: Dict[str, Any]) -> tuple[str, str]:
    if summary["num_total_factor_mutations"] == 0 and summary["num_total_action_mutations"] == 0:
        return "complete", ""
    if summary["num_total_action_mutations"] and summary["num_total_factor_mutations"] == 0:
        return "wrong_action", "action_mutation"
    if summary["num_added_factors"] and summary["num_total_factor_mutations"] == summary["num_added_factors"]:
        return "extra_factor", "hallucinated_factor"
    if summary["num_removed_factors"] and summary["num_replaced_factors"] == 0:
        return "missing_factor", "remove_factor"
    if summary["num_total_factor_mutations"] > 1 or summary["num_total_action_mutations"]:
        return "mixed_mutation", "factor_action_mixed"
    for name, subtype in [
        ("num_direction_changes", "direction_swap"),
        ("num_factor_value_changes", "factor_value_change"),
        ("num_sub_category_replacements", "same_category_subtype_swap"),
        ("num_super_category_replacements", "super_category_swap"),
        ("num_cross_category_replacements", "cross_category_swap"),
    ]:
        if summary[name]:
            return "wrong_factor", subtype
    return "mixed_mutation", "unknown"


def build_record(index: int, schema: Dict[str, Any], severity: str, rng: random.Random, force_complete: bool = False) -> Dict[str, Any]:
    reference_factors = copy_factors(schema)
    reference_action_schema = with_raw_action(schema.get("action_schema", {}))
    candidate_factors = []
    factor_mutations = []
    gt_factor_scores = []

    for factor in reference_factors:
        mutation_type = "KEEP_FACTOR" if force_complete else weighted_choice(rng, MUTATION_WEIGHTS[severity])
        mutated_factor, mutation = mutate_factor(factor, mutation_type, rng)
        factor_mutations.append(mutation)
        gt_factor_scores.append(float(mutation["gt_factor_score"]))
        if mutated_factor is not None:
            candidate_factors.append(mutated_factor)

    add_count = 0 if force_complete else rng.choices(range(len(ADD_FACTOR_WEIGHTS[severity])), weights=ADD_FACTOR_WEIGHTS[severity], k=1)[0]
    next_factor_index = max([factor.get("factor_index", 0) for factor in reference_factors] or [0]) + 1
    for offset in range(add_count):
        added_factor = make_added_factor(next_factor_index + offset, rng)
        candidate_factors.append(added_factor)
        factor_mutations.append(
            {
                "factor_index": added_factor["factor_index"],
                "action": "add",
                "mutation_type": "ADD_FACTOR",
                "from": None,
                "to": added_factor,
                "gt_factor_score": "penalty_only",
            }
        )

    if force_complete:
        candidate_action_schema = reference_action_schema
        action_mutations = [
            {"dimension": "lat", "action": "keep", "mutation_type": "KEEP_ACTION", "from": reference_action_schema["lat"], "to": reference_action_schema["lat"], "gt_action_score": 1.0},
            {"dimension": "lon", "action": "keep", "mutation_type": "KEEP_ACTION", "from": reference_action_schema["lon"], "to": reference_action_schema["lon"], "gt_action_score": 1.0},
            {"dimension": "strategy", "action": "keep", "mutation_type": "KEEP_ACTION", "from": reference_action_schema["strategy"], "to": reference_action_schema["strategy"], "gt_action_score": 1.0},
        ]
    else:
        candidate_action_schema, action_mutations = mutate_action(reference_action_schema, severity, rng)

    summary = summarize_mutations(factor_mutations, action_mutations)
    error_type, factor_error_subtype = error_labels(summary)
    scores = provisional_scores(gt_factor_scores, add_count, reference_action_schema, candidate_action_schema)
    candidate = schema["source"] if force_complete else render_candidate(candidate_factors, candidate_action_schema)
    factor_edits = [item for item in factor_mutations if item["mutation_type"] != "KEEP_FACTOR"]
    return {
        "index": index,
        "source_id": schema["source_id"],
        "reference": schema["source"],
        "candidate": candidate,
        "category": primary_category(schema),
        "severity_level": "complete" if force_complete else severity,
        "error_type": error_type,
        "factor_error_subtype": factor_error_subtype,
        "gt_factor_scores": gt_factor_scores,
        "gt_factor_avg": scores["gt_factor_avg"],
        "gt_action_scores": scores["gt_action_scores"],
        "gt_action_avg": scores["gt_action_avg"],
        "provisional_gt_score": scores["provisional_gt_score"],
        "add_factor_penalty": scores["add_factor_penalty"],
        "gt_factor_avg_after_add_penalty": scores["gt_factor_avg_after_add_penalty"],
        "factor_edits": factor_edits,
        "factor_mutations": factor_mutations,
        "action_mutations": action_mutations,
        "mutation_summary": summary,
        "reference_factors": reference_factors,
        "candidate_factors": candidate_factors,
        "reference_action_schema": reference_action_schema,
        "candidate_action_schema": candidate_action_schema,
    }


def generate_for_schema(schema: Dict[str, Any], start_index: int, variants_per_source: int, rng: random.Random) -> List[Dict[str, Any]]:
    if not schema.get("factors"):
        return []
    records = [build_record(start_index, schema, "mild", rng, force_complete=True)]
    severities = ["mild", "medium", "severe"]
    for offset in range(1, variants_per_source + 1):
        severity = severities[(offset - 1) % len(severities)]
        records.append(build_record(start_index + offset, schema, severity, rng))
    return records


def export(records: List[Dict[str, Any]], output_dir: Path) -> None:
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
        "severity_counts": dict(Counter(record["severity_level"] for record in records)),
        "error_type_counts": dict(Counter(record["error_type"] for record in records)),
        "files": {
            "pairs": "pairs.json",
            "gt": "candidates_with_gt.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemas", default="reward_model_v3/data/gt_schema/source_schemas.json")
    parser.add_argument("--output-dir", default="reward_model_v3/data/gt_candidates")
    parser.add_argument("--variants-per-source", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    schemas = json.loads(Path(args.schemas).read_text(encoding="utf-8"))
    records = []
    next_index = 1
    for schema in schemas:
        generated = generate_for_schema(schema, next_index, args.variants_per_source, rng)
        records.extend(generated)
        next_index += len(generated)
    export(records, Path(args.output_dir))
    print(f"wrote {len(records)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
