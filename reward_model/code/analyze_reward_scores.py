#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def score_from_result_item(item: Dict[str, Any]) -> Dict[str, float]:
    result = item.get("result") or {}
    final_reward = result.get("final_reward") or {}
    summary_score = result.get("summary_score") or {}
    if final_reward:
        return {
            "model_score": float(final_reward.get("final_reward", 0.0)),
            "task_score": float(final_reward.get("task_score", 0.0)),
            "text_quality_score": float(final_reward.get("text_quality_score", 0.0)),
        }
    if summary_score:
        task_score = float(summary_score.get("normalized_total_score", 0.0))
        return {"model_score": task_score, "task_score": task_score, "text_quality_score": 1.0}
    return score_from_score_item(item)


def score_from_score_item(item: Dict[str, Any]) -> Dict[str, float]:
    factor_scores = [float(x) for x in item.get("factor_scores", [])]
    action_scores = item.get("action_scores", {}) or {}
    action_values = [
        float(action_scores.get("lat", 0.0)),
        float(action_scores.get("lon", 0.0)),
        float(action_scores.get("strategy", 0.0)),
    ]
    denom = len(factor_scores) + len(action_values)
    task_score = (sum(factor_scores) + sum(action_values)) / denom if denom else 0.0
    return {"model_score": round(task_score, 4), "task_score": round(task_score, 4), "text_quality_score": ""}


def load_score_rows(batch_result_path: str) -> List[Dict[str, Any]]:
    batch = load_json(batch_result_path)
    if "results" in batch:
        raw_rows = batch["results"]
        return [
            {
                "index": item["index"],
                "reference": item["reference"],
                "candidate": item["candidate"],
                **score_from_result_item(item),
            }
            for item in raw_rows
        ]
    raw_rows = batch.get("score_items", [])
    return [
        {
            "index": item["index"],
            "reference": item["reference"],
            "candidate": item["candidate"],
            **score_from_score_item(item),
        }
        for item in raw_rows
    ]


def merge_with_labels(score_rows: List[Dict[str, Any]], labels_path: str) -> List[Dict[str, Any]]:
    labels = {int(item["index"]): item for item in load_json(labels_path)}
    merged = []
    for row in score_rows:
        index = int(row["index"])
        label = labels.get(index, {})
        expected_action_scores = label.get("expected_action_scores", {}) or {}
        expected_factor_scores = label.get("expected_factor_scores", "")
        expected_normalized = label.get("expected_normalized_total_score", "")
        model_score = float(row.get("model_score", 0.0))
        expected_score_delta = (
            round(model_score - float(expected_normalized), 4)
            if expected_normalized not in ["", None]
            else ""
        )
        merged.append(
            {
                **row,
                "source_id": label.get("source_id", ""),
                "category": label.get("category", ""),
                "subcategory": label.get("subcategory", ""),
                "target_score": label.get("target_score", ""),
                "error_type": label.get("error_type", ""),
                "factor_error_subtype": label.get("factor_error_subtype", ""),
                "expected_factor_scores": json.dumps(expected_factor_scores, ensure_ascii=False)
                if isinstance(expected_factor_scores, list)
                else expected_factor_scores,
                "expected_action_lat": expected_action_scores.get("lat", ""),
                "expected_action_lon": expected_action_scores.get("lon", ""),
                "expected_action_strategy": expected_action_scores.get("strategy", ""),
                "expected_normalized_total_score": expected_normalized,
                "expected_score_delta": expected_score_delta,
            }
        )
    return merged


def summarize(rows: List[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(group_key, "")].append(row)
    summary = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scores = [float(item["model_score"]) for item in items]
        target_scores = [
            float(item["target_score"])
            for item in items
            if item.get("target_score") not in ["", None]
        ]
        summary.append(
            {
                group_key: key,
                "count": len(items),
                "avg_model_score": round(mean(scores), 4) if scores else 0.0,
                "min_model_score": round(min(scores), 4) if scores else 0.0,
                "max_model_score": round(max(scores), 4) if scores else 0.0,
                "avg_target_score": round(mean(target_scores), 4) if target_scores else "",
            }
        )
    return summary


def pairwise_rank_violations(rows: List[Dict[str, Any]], tolerance: float = 0.0) -> List[Dict[str, Any]]:
    by_source = defaultdict(list)
    for row in rows:
        if row.get("source_id") != "" and row.get("target_score") != "":
            by_source[int(row["source_id"])].append(row)

    violations = []
    for source_id, items in by_source.items():
        for better in items:
            for worse in items:
                if float(better["target_score"]) <= float(worse["target_score"]):
                    continue
                if float(better["model_score"]) + tolerance < float(worse["model_score"]):
                    violations.append(
                        {
                            "source_id": source_id,
                            "category": better.get("category", ""),
                            "better_index": better["index"],
                            "better_target_score": better["target_score"],
                            "better_model_score": better["model_score"],
                            "better_error_type": better.get("error_type", ""),
                            "better_candidate": better["candidate"],
                            "worse_index": worse["index"],
                            "worse_target_score": worse["target_score"],
                            "worse_model_score": worse["model_score"],
                            "worse_error_type": worse.get("error_type", ""),
                            "worse_candidate": worse["candidate"],
                            "model_score_gap": round(
                                float(worse["model_score"]) - float(better["model_score"]), 4
                            ),
                        }
                    )
    violations.sort(key=lambda item: item["model_score_gap"], reverse=True)
    return violations


def source_level_summary(rows: List[Dict[str, Any]], violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    violation_counts = defaultdict(int)
    for item in violations:
        violation_counts[item["source_id"]] += 1

    by_source = defaultdict(list)
    for row in rows:
        if row.get("source_id") != "":
            by_source[int(row["source_id"])].append(row)

    result = []
    for source_id, items in sorted(by_source.items()):
        scores = [float(item["model_score"]) for item in items]
        result.append(
            {
                "source_id": source_id,
                "category": items[0].get("category", ""),
                "candidate_count": len(items),
                "avg_model_score": round(mean(scores), 4) if scores else 0.0,
                "rank_violation_count": violation_counts[source_id],
                "reference": items[0].get("reference", ""),
            }
        )
    result.sort(key=lambda item: item["rank_violation_count"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-result", required=True, help="formal_batch_reward.py 输出的 batch_result.json")
    parser.add_argument(
        "--labels-json",
        default="reward_model/data/reward_candidates/reward_candidates_pairs_with_expected_scores.json",
        help="带 target_score/error_type/expected_scores 的 pairs 文件",
    )
    parser.add_argument("--output-dir", required=True, help="分析报表输出目录")
    parser.add_argument("--tolerance", type=float, default=0.0, help="排序违例容忍分差")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    score_rows = load_score_rows(args.batch_result)
    merged = merge_with_labels(score_rows, args.labels_json)
    violations = pairwise_rank_violations(merged, tolerance=args.tolerance)
    source_summary = source_level_summary(merged, violations)

    write_json(output_dir / "scored_with_labels.json", merged)
    write_json(output_dir / "rank_violations.json", violations)
    write_json(output_dir / "source_summary.json", source_summary)

    write_csv(
        output_dir / "scored_with_labels.csv",
        merged,
        [
            "index",
            "source_id",
            "category",
            "subcategory",
            "target_score",
            "error_type",
            "factor_error_subtype",
            "model_score",
            "task_score",
            "text_quality_score",
            "expected_factor_scores",
            "expected_action_lat",
            "expected_action_lon",
            "expected_action_strategy",
            "expected_normalized_total_score",
            "expected_score_delta",
            "reference",
            "candidate",
        ],
    )

    for key in ["target_score", "error_type", "factor_error_subtype", "category", "subcategory"]:
        rows = summarize(merged, key)
        write_csv(
            output_dir / f"summary_by_{key}.csv",
            rows,
            [key, "count", "avg_model_score", "min_model_score", "max_model_score", "avg_target_score"],
        )

    write_csv(
        output_dir / "rank_violations_top.csv",
        violations[:500],
        [
            "source_id",
            "category",
            "better_index",
            "better_target_score",
            "better_model_score",
            "better_error_type",
            "better_candidate",
            "worse_index",
            "worse_target_score",
            "worse_model_score",
            "worse_error_type",
            "worse_candidate",
            "model_score_gap",
        ],
    )
    write_csv(
        output_dir / "source_summary.csv",
        source_summary,
        ["source_id", "category", "candidate_count", "avg_model_score", "rank_violation_count", "reference"],
    )

    report = {
        "count": len(merged),
        "source_count": len({row.get("source_id") for row in merged if row.get("source_id") != ""}),
        "rank_violation_count": len(violations),
        "avg_model_score": round(mean(float(row["model_score"]) for row in merged), 4) if merged else 0.0,
        "outputs": [
            "scored_with_labels.csv",
            "summary_by_target_score.csv",
            "summary_by_error_type.csv",
            "summary_by_factor_error_subtype.csv",
            "summary_by_category.csv",
            "rank_violations_top.csv",
            "source_summary.csv",
        ],
    }
    write_json(output_dir / "analysis_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
