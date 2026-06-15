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

DETAILS_BY_CATEGORY = {
    "盲区": ["车辆遮挡盲区", "豁口盲区", "弯道盲区", "无灯路口盲区", "环岛盲区", "匝道盲区", "坡道盲区", "施工盲区"],
    "障碍物": ["锥桶占道", "施工区域占道", "事故车占道", "车辆占道停车", "行人占道", "两轮骑行者占道", "三轮骑行者占道"],
    "车辆行为": ["车辆切入", "车辆缓行", "车辆逆向行驶", "行人通行", "两轮骑行者通行", "三轮骑行者通行"],
    "路况": ["坑洼路面", "积水路面", "井盖凹陷", "减速带", "湿滑路面", "大曲率弯道"],
    "交通信号": ["红灯", "绿灯", "黄灯", "左转箭头灯红灯", "直行箭头灯黄灯"],
    "换道条件": ["目标车道空闲", "目标车道拥堵", "后方无来车", "后方有快速来车", "存在安全距离"],
    "交通管制": ["交警示意停车", "交警示意通行", "交警示意左转"],
    "跟车": ["前车起步", "前车缓行", "前车等待红绿灯"],
}

CATEGORIES = list(DETAILS_BY_CATEGORY)

CHANGE_PENALTY = {
    "VALUE": 0.15,
    "DIRECTION": 0.35,
    "SUB_CATEGORY": 0.30,
    "SUPER_CATEGORY": 0.55,
    "CROSS_CATEGORY": 0.75,
}

FACTOR_OPERATION_WEIGHTS = {
    "mild": {"keep": 0.62, "modify": 0.30, "remove": 0.03, "add_only": 0.05},
    "medium": {"keep": 0.42, "modify": 0.43, "remove": 0.08, "add_only": 0.07},
    "severe": {"keep": 0.22, "modify": 0.55, "remove": 0.13, "add_only": 0.10},
}

CHANGE_TYPE_WEIGHTS = {
    "mild": {"VALUE": 0.25, "DIRECTION": 0.20, "SUB_CATEGORY": 0.42, "SUPER_CATEGORY": 0.10, "CROSS_CATEGORY": 0.03},
    "medium": {"VALUE": 0.16, "DIRECTION": 0.22, "SUB_CATEGORY": 0.28, "SUPER_CATEGORY": 0.19, "CROSS_CATEGORY": 0.15},
    "severe": {"VALUE": 0.08, "DIRECTION": 0.18, "SUB_CATEGORY": 0.20, "SUPER_CATEGORY": 0.28, "CROSS_CATEGORY": 0.26},
}

STACKED_CHANGE_WEIGHTS = {
    "mild": [0.86, 0.14],
    "medium": [0.62, 0.32, 0.06],
    "severe": [0.43, 0.40, 0.17],
}

ADD_COUNT_WEIGHTS = {
    "mild": [0.78, 0.20, 0.02],
    "medium": [0.55, 0.32, 0.13],
    "severe": [0.35, 0.38, 0.20, 0.07],
}

ACTION_OPERATION_WEIGHTS = {
    "mild": {"keep": 0.82, "replace": 0.13, "conflict": 0.03, "remove": 0.02},
    "medium": {"keep": 0.64, "replace": 0.19, "conflict": 0.10, "remove": 0.07},
    "severe": {"keep": 0.45, "replace": 0.22, "conflict": 0.20, "remove": 0.13},
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


def compact_factor(factor: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if factor is None:
        return None
    return {
        "position": factor.get("position", ""),
        "category": factor.get("category", ""),
        "detail": factor.get("detail", ""),
    }


def normalize_action_schema(action_schema: Dict[str, Any]) -> Dict[str, Any]:
    raw = action_schema.get("raw", {}) if isinstance(action_schema, dict) else {}
    lat = action_schema.get("lat") or raw.get("横向决策") or ["保持"]
    lon = action_schema.get("lon") or raw.get("纵向决策") or ["保持"]
    strategy = action_schema.get("strategy") or raw.get("执行策略") or "直接执行"
    return {"lat": list(lat), "lon": list(lon), "strategy": strategy}


def formal_action(action_schema: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_action_schema(action_schema)
    return {
        "横向决策": normalized["lat"],
        "纵向决策": normalized["lon"],
        "执行策略": normalized["strategy"],
    }


def render_factor(factor: Dict[str, Any]) -> str:
    position = factor.get("position") or "前方"
    detail = factor.get("detail") or factor.get("category") or "风险"
    if factor.get("category") == "换道条件":
        return f"{position}{detail}"
    return f"{position}存在{detail}"


def render_action(action_schema: Dict[str, Any]) -> str:
    action_schema = normalize_action_schema(action_schema)
    parts = []
    if action_schema["lat"] != ["保持"]:
        parts.extend(action_schema["lat"])
    if action_schema["lon"] != ["保持"]:
        parts.extend(action_schema["lon"])
    if not parts:
        parts.append("保持行驶")
    action = "并".join(parts)
    if action_schema["strategy"] == "条件满足后执行":
        return f"自车应在条件满足后{action}"
    return f"自车应{action}"


def render_candidate(factors: List[Dict[str, Any]], action_schema: Dict[str, Any]) -> str:
    factor_text = "，".join(render_factor(factor) for factor in factors)
    action_text = render_action(action_schema)
    return f"{factor_text}，{action_text}。" if factor_text else f"{action_text}。"


def value_variant(detail: str) -> str:
    replacements = [
        ("两个", "一个"),
        ("一辆", "多辆"),
        ("缓慢", "快速"),
        ("快速", "缓慢"),
        ("严重", "轻微"),
        ("轻微", "严重"),
    ]
    for old, new in replacements:
        if old in detail:
            return detail.replace(old, new, 1)
    return f"轻微{detail}" if detail else "轻微风险"


def choose_change_types(severity: str, rng: random.Random) -> List[str]:
    max_count = len(STACKED_CHANGE_WEIGHTS[severity])
    count = rng.choices(range(1, max_count + 1), weights=STACKED_CHANGE_WEIGHTS[severity], k=1)[0]
    changes = []
    while len(changes) < count:
        change = weighted_choice(rng, CHANGE_TYPE_WEIGHTS[severity])
        if change not in changes:
            changes.append(change)
        if change == "CROSS_CATEGORY":
            break
        if change == "SUPER_CATEGORY":
            changes = [item for item in changes if item != "SUB_CATEGORY"]
    return changes


def apply_factor_changes(factor: Dict[str, Any], change_types: List[str], rng: random.Random) -> Dict[str, Any]:
    result = dict(factor)
    if "DIRECTION" in change_types:
        result["position"] = DIRECTION_SWAP.get(result.get("position"), choose_other(rng, list(DIRECTION_SWAP), result.get("position", "")))
    if "CROSS_CATEGORY" in change_types:
        category = choose_other(rng, CATEGORIES, result.get("category", ""))
        result["category"] = category
        result["detail"] = rng.choice(DETAILS_BY_CATEGORY.get(category, [category]))
        return result
    if "SUPER_CATEGORY" in change_types:
        category = choose_other(rng, CATEGORIES, result.get("category", ""))
        result["category"] = category
        result["detail"] = rng.choice(DETAILS_BY_CATEGORY.get(category, [category]))
    elif "SUB_CATEGORY" in change_types:
        category = result.get("category", "")
        result["detail"] = choose_other(rng, DETAILS_BY_CATEGORY.get(category, []), result.get("detail", ""))
    if "VALUE" in change_types:
        result["detail"] = value_variant(result.get("detail", ""))
    return result


def score_factor_change(operation: str, change_types: List[str]) -> float:
    if operation == "keep":
        return 1.0
    if operation == "remove":
        return 0.0
    penalty = sum(CHANGE_PENALTY[item] for item in change_types)
    return round(max(0.0, 1.0 - penalty), 4)


def factor_labels(operation: str, change_types: List[str]) -> Dict[str, Any]:
    if operation == "add":
        return {
            "category_change_label": "not_applicable",
            "detail_change_label": "not_applicable",
            "direction_change_label": "not_applicable",
            "delete_label": "not_deleted",
            "add_label": "added_factor",
            "is_category_changed": False,
            "is_detail_changed": False,
            "is_direction_changed": False,
            "is_deleted": False,
            "is_added": True,
        }
    if operation == "remove":
        return {
            "category_change_label": "not_applicable",
            "detail_change_label": "not_applicable",
            "direction_change_label": "not_applicable",
            "delete_label": "deleted_factor",
            "add_label": "not_added",
            "is_category_changed": False,
            "is_detail_changed": False,
            "is_direction_changed": False,
            "is_deleted": True,
            "is_added": False,
        }

    category_changed = "SUPER_CATEGORY" in change_types or "CROSS_CATEGORY" in change_types
    detail_changed = any(item in change_types for item in ["VALUE", "SUB_CATEGORY", "SUPER_CATEGORY", "CROSS_CATEGORY"])
    direction_changed = "DIRECTION" in change_types
    if "CROSS_CATEGORY" in change_types:
        category_label = "cross_category_changed"
    elif "SUPER_CATEGORY" in change_types:
        category_label = "super_category_changed"
    else:
        category_label = "category_unchanged"
    return {
        "category_change_label": category_label,
        "detail_change_label": "detail_changed" if detail_changed else "detail_unchanged",
        "direction_change_label": "direction_changed" if direction_changed else "direction_unchanged",
        "delete_label": "not_deleted",
        "add_label": "not_added",
        "is_category_changed": category_changed,
        "is_detail_changed": detail_changed,
        "is_direction_changed": direction_changed,
        "is_deleted": False,
        "is_added": False,
    }


def make_added_factor(rng: random.Random) -> Dict[str, str]:
    category = rng.choice(CATEGORIES)
    return {
        "position": rng.choice(list(DIRECTION_SWAP)),
        "category": category,
        "detail": rng.choice(DETAILS_BY_CATEGORY.get(category, [category])),
    }


def mutate_factor(factor: Dict[str, Any], severity: str, rng: random.Random, force_keep: bool = False) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    operation = "keep" if force_keep else weighted_choice(rng, FACTOR_OPERATION_WEIGHTS[severity])
    if operation == "add_only":
        operation = "keep"
    if operation == "keep":
        to_factor = dict(factor)
        change_types: List[str] = []
    elif operation == "remove":
        to_factor = None
        change_types = ["REMOVE"]
    else:
        change_types = choose_change_types(severity, rng)
        to_factor = apply_factor_changes(factor, change_types, rng)

    labels = factor_labels(operation, change_types)
    return to_factor, {
        "factor_id": f"gt_{factor.get('factor_index')}",
        "operation": operation,
        **labels,
        "from": compact_factor(factor),
        "to": compact_factor(to_factor),
        "score": score_factor_change(operation, [item for item in change_types if item != "REMOVE"]),
    }


def add_factor_change(rng: random.Random, add_index: int) -> tuple[Dict[str, Any], Dict[str, Any]]:
    factor = make_added_factor(rng)
    return factor, {
        "factor_id": f"add_{add_index}",
        "operation": "add",
        **factor_labels("add", ["ADD"]),
        "from": None,
        "to": compact_factor(factor),
        "score": "extra_penalty",
    }


def mutate_action_dimension(value: Any, dimension: str, operation: str, rng: random.Random) -> Any:
    options = ACTION_OPTIONS[dimension]
    if dimension == "strategy":
        old = value or "直接执行"
        if operation == "keep":
            return old
        if operation == "remove":
            return "直接执行"
        return choose_other(rng, options, old)

    old_list = list(value or ["保持"])
    old_primary = old_list[0] if old_list else "保持"
    if operation == "keep":
        return old_list
    if operation == "remove":
        return ["保持"]
    if operation == "conflict":
        conflicts = {"加速": "停车", "停车": "加速", "减速": "加速", "换道": "保持", "避让": "保持", "转弯": "保持", "保持": options[-1]}
        return [conflicts.get(old_primary, choose_other(rng, options, old_primary))]
    return [choose_other(rng, options, old_primary)]


def mutate_action(action_schema: Dict[str, Any], severity: str, rng: random.Random, force_keep: bool = False) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    reference = normalize_action_schema(action_schema)
    candidate = normalize_action_schema(action_schema)
    changes = []
    for dimension in ["lat", "lon", "strategy"]:
        operation = "keep" if force_keep else weighted_choice(rng, ACTION_OPERATION_WEIGHTS[severity])
        new_value = mutate_action_dimension(reference[dimension], dimension, operation, rng)
        candidate[dimension] = new_value
        changes.append(
            {
                "dimension": dimension,
                "operation": operation,
                "change_label": "unchanged" if operation == "keep" else f"{dimension}_{operation}",
                "is_changed": operation != "keep",
                "from": reference[dimension],
                "to": new_value,
            }
        )
    return candidate, changes


def action_scores(reference_action: Dict[str, Any], candidate_action: Dict[str, Any]) -> Dict[str, float]:
    return score_actions(formal_action(reference_action), formal_action(candidate_action))


def summarize_factor_changes(factor_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    operation_counts = Counter(item["operation"] for item in factor_changes)
    return {
        "gt_factor_count": sum(1 for item in factor_changes if item["factor_id"].startswith("gt_")),
        "kept": operation_counts["keep"],
        "modified": operation_counts["modify"],
        "removed": operation_counts["remove"],
        "added": operation_counts["add"],
        "category_changed": sum(1 for item in factor_changes if item["is_category_changed"]),
        "detail_changed": sum(1 for item in factor_changes if item["is_detail_changed"]),
        "direction_changed": sum(1 for item in factor_changes if item["is_direction_changed"]),
        "deleted": sum(1 for item in factor_changes if item["is_deleted"]),
        "added_label_count": sum(1 for item in factor_changes if item["is_added"]),
        "stacked_changed": sum(
            1
            for item in factor_changes
            if sum([item["is_category_changed"], item["is_detail_changed"], item["is_direction_changed"]]) > 1
        ),
    }


def summarize_action_changes(action_changes: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(item["operation"] for item in action_changes)
    return {
        "kept": counts["keep"],
        "replaced": counts["replace"],
        "conflicted": counts["conflict"],
        "removed": counts["remove"],
    }


def sample_factor_labels(factor_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "category_change_label": "has_category_change" if factor_summary["category_changed"] else "no_category_change",
        "detail_change_label": "has_detail_change" if factor_summary["detail_changed"] else "no_detail_change",
        "direction_change_label": "has_direction_change" if factor_summary["direction_changed"] else "no_direction_change",
        "delete_label": "has_deleted_factor" if factor_summary["deleted"] else "no_deleted_factor",
        "add_label": "has_added_factor" if factor_summary["added"] else "no_added_factor",
        "num_category_changed_factors": factor_summary["category_changed"],
        "num_detail_changed_factors": factor_summary["detail_changed"],
        "num_direction_changed_factors": factor_summary["direction_changed"],
        "num_deleted_factors": factor_summary["deleted"],
        "num_added_factors": factor_summary["added"],
    }


def sample_action_labels(action_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    changed = [item for item in action_changes if item["is_changed"]]
    count = len(changed)
    labels = {
        0: "no_action_change",
        1: "one_action_dimension_changed",
        2: "two_action_dimensions_changed",
        3: "three_action_dimensions_changed",
    }
    return {
        "action_change_label": labels.get(count, f"{count}_action_dimensions_changed"),
        "num_changed_action_dimensions": count,
        "changed_action_dimensions": [item["dimension"] for item in changed],
        "changed_action_operations": [item["operation"] for item in changed],
    }


def primary_category(schema: Dict[str, Any]) -> str:
    factors = schema.get("factors", [])
    return factors[0].get("category", "unknown") if factors else "unknown"


def provisional_score(factor_changes: List[Dict[str, Any]], reference_action: Dict[str, Any], candidate_action: Dict[str, Any]) -> Dict[str, Any]:
    gt_factor_scores = [
        float(item["score"])
        for item in factor_changes
        if item["factor_id"].startswith("gt_")
    ]
    factor_avg = round(sum(gt_factor_scores) / len(gt_factor_scores), 4) if gt_factor_scores else 0.0
    add_count = sum(1 for item in factor_changes if item["operation"] == "add")
    add_penalty = min(0.25, 0.06 * add_count)
    factor_score = round(max(0.0, factor_avg - add_penalty), 4)
    actions = action_scores(reference_action, candidate_action)
    action_avg = round(sum(actions.values()) / len(actions), 4)
    total = round(0.75 * factor_score + 0.25 * action_avg, 4)
    return {
        "factor_scores": gt_factor_scores,
        "factor_avg": factor_avg,
        "add_penalty": add_penalty,
        "factor_score_after_add_penalty": factor_score,
        "action_scores": actions,
        "action_avg": action_avg,
        "provisional_score": total,
    }


def error_type(factor_summary: Dict[str, Any], action_summary: Dict[str, int]) -> str:
    factor_mutations = factor_summary["modified"] + factor_summary["removed"] + factor_summary["added"]
    action_mutations = action_summary["replaced"] + action_summary["conflicted"] + action_summary["removed"]
    if factor_mutations == 0 and action_mutations == 0:
        return "complete"
    if factor_mutations == 0:
        return "action_only"
    if action_mutations == 0:
        return "factor_only"
    return "factor_and_action"


def build_record(index: int, schema: Dict[str, Any], severity: str, rng: random.Random, force_complete: bool = False) -> Dict[str, Any]:
    reference_action = normalize_action_schema(schema.get("action_schema", {}))
    candidate_action, action_changes = mutate_action(reference_action, severity, rng, force_keep=force_complete)

    candidate_factors = []
    per_factor_labels = []
    for factor in schema.get("factors", []):
        changed_factor, change = mutate_factor(factor, severity, rng, force_keep=force_complete)
        per_factor_labels.append(change)
        if changed_factor is not None:
            candidate_factors.append(changed_factor)

    add_count = 0 if force_complete else rng.choices(range(len(ADD_COUNT_WEIGHTS[severity])), weights=ADD_COUNT_WEIGHTS[severity], k=1)[0]
    for add_index in range(1, add_count + 1):
        added_factor, change = add_factor_change(rng, add_index)
        candidate_factors.append(added_factor)
        per_factor_labels.append(change)

    factor_summary = summarize_factor_changes(per_factor_labels)
    action_summary = summarize_action_changes(action_changes)
    record_factor_labels = sample_factor_labels(factor_summary)
    record_action_labels = sample_action_labels(action_changes)
    scores = provisional_score(per_factor_labels, reference_action, candidate_action)
    return {
        "index": index,
        "source_id": schema["source_id"],
        "reference": schema["source"],
        "candidate": schema["source"] if force_complete else render_candidate(candidate_factors, candidate_action),
        "category": primary_category(schema),
        "severity_level": "complete" if force_complete else severity,
        "error_type": error_type(factor_summary, action_summary),
        "sample_factor_labels": record_factor_labels,
        "per_factor_labels": per_factor_labels,
        "factor_summary": factor_summary,
        "sample_action_labels": record_action_labels,
        "per_action_labels": action_changes,
        "action_summary": action_summary,
        "scores": scores,
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
