#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


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


def load_actual_scores(batch_result_path: str) -> Dict[int, Dict[str, Any]]:
    data = load_json(batch_result_path)
    rows = data.get("results") or data.get("score_items") or []
    actual = {}
    for row in rows:
        index = int(row["index"])
        result = row.get("result", {})
        summary = result.get("summary_score", {}) if result else {}
        if not summary:
            summary = row
        factor_scores = summary.get("factor_scores", [])
        action_scores = summary.get("action_scores", {})
        actual[index] = {
            "actual_factor_scores": factor_scores,
            "actual_factor_avg": round(sum(factor_scores) / len(factor_scores), 4) if factor_scores else 0.0,
            "actual_action_scores": action_scores,
            "actual_normalized_total_score": summary.get("normalized_total_score", ""),
        }
    return actual


def align_scores(gt_scores: List[float], actual_scores: List[float]) -> Dict[str, Any]:
    max_len = max(len(gt_scores), len(actual_scores))
    padded_gt = list(gt_scores) + [0.0] * (max_len - len(gt_scores))
    padded_actual = list(actual_scores) + [0.0] * (max_len - len(actual_scores))
    diffs = [round(a - g, 4) for g, a in zip(padded_gt, padded_actual)]
    abs_diffs = [abs(item) for item in diffs]
    return {
        "aligned_gt_factor_scores": padded_gt,
        "aligned_actual_factor_scores": padded_actual,
        "factor_score_diffs": diffs,
        "max_abs_factor_diff": round(max(abs_diffs), 4) if abs_diffs else 0.0,
        "avg_abs_factor_diff": round(mean(abs_diffs), 4) if abs_diffs else 0.0,
    }


def build_comparison_rows(gt_rows: List[Dict[str, Any]], actual_by_index: Dict[int, Dict[str, Any]], tolerance: float) -> List[Dict[str, Any]]:
    rows = []
    for gt in gt_rows:
        index = int(gt["index"])
        actual = actual_by_index.get(index, {})
        actual_found = index in actual_by_index
        actual_scores = actual.get("actual_factor_scores", [])
        actual_action_scores = actual.get("actual_action_scores", {})
        gt_scores = gt.get("gt_factor_scores", [])
        gt_action_scores = gt.get("gt_action_scores", {})
        aligned = align_scores(gt_scores, actual_scores)
        gt_avg = float(gt.get("gt_factor_avg", 0.0))
        actual_avg = float(actual.get("actual_factor_avg", 0.0))
        avg_delta = round(actual_avg - gt_avg, 4)
        action_diffs = {
            key: round(float(actual_action_scores.get(key, 0.0)) - float(gt_action_scores.get(key, 0.0)), 4)
            for key in ["lat", "lon", "strategy"]
        }
        success = actual_found and aligned["max_abs_factor_diff"] <= tolerance
        rows.append(
            {
                "index": index,
                "source_id": gt.get("source_id"),
                "category": gt.get("category"),
                "error_type": gt.get("error_type"),
                "factor_error_subtype": gt.get("factor_error_subtype"),
                "gt_factor_scores": json.dumps(gt_scores, ensure_ascii=False),
                "actual_factor_scores": json.dumps(actual_scores, ensure_ascii=False),
                "gt_action_scores": json.dumps(gt_action_scores, ensure_ascii=False),
                "actual_action_scores": json.dumps(actual_action_scores, ensure_ascii=False),
                "action_score_diffs": json.dumps(action_diffs, ensure_ascii=False),
                "gt_factor_avg": gt_avg,
                "actual_factor_avg": actual_avg,
                "factor_avg_delta": avg_delta,
                "max_abs_factor_diff": aligned["max_abs_factor_diff"],
                "avg_abs_factor_diff": aligned["avg_abs_factor_diff"],
                "success": success,
                "actual_found": actual_found,
                "factor_edits": json.dumps(gt.get("factor_edits", []), ensure_ascii=False),
                "reference": gt.get("reference"),
                "candidate": gt.get("candidate"),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key, "")].append(row)
    summary = []
    for value, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scored_items = [item for item in items if item.get("actual_found")]
        summary.append(
            {
                key: value,
                "count": len(items),
                "actual_count": len(scored_items),
                "success_count": sum(1 for item in scored_items if item["success"]),
                "success_rate": round(sum(1 for item in scored_items if item["success"]) / len(scored_items), 4) if scored_items else 0.0,
                "avg_gt_factor_avg": round(mean(float(item["gt_factor_avg"]) for item in scored_items), 4) if scored_items else 0.0,
                "avg_actual_factor_avg": round(mean(float(item["actual_factor_avg"]) for item in scored_items), 4) if scored_items else 0.0,
                "avg_abs_factor_diff": round(mean(float(item["avg_abs_factor_diff"]) for item in scored_items), 4) if scored_items else 0.0,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-result", required=True)
    parser.add_argument("--gt-json", default="reward_model_v2/data/gt_candidates/candidates_with_gt.json")
    parser.add_argument("--output-dir", default="reward_model_v2/outputs/gt_comparison")
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()

    gt_rows = load_json(args.gt_json)
    actual_by_index = load_actual_scores(args.batch_result)
    rows = build_comparison_rows(gt_rows, actual_by_index, args.tolerance)
    output_dir = Path(args.output_dir)

    write_json(output_dir / "gt_vs_formal.json", rows)
    write_csv(
        output_dir / "gt_vs_formal.csv",
        rows,
        [
            "index",
            "source_id",
            "category",
            "error_type",
            "factor_error_subtype",
            "gt_factor_scores",
            "actual_factor_scores",
            "gt_action_scores",
            "actual_action_scores",
            "action_score_diffs",
            "gt_factor_avg",
            "actual_factor_avg",
            "factor_avg_delta",
            "max_abs_factor_diff",
            "avg_abs_factor_diff",
            "success",
            "actual_found",
            "factor_edits",
            "reference",
            "candidate",
        ],
    )

    failures = [row for row in rows if row["actual_found"] and not row["success"]]
    failures.sort(key=lambda item: item["max_abs_factor_diff"], reverse=True)
    write_csv(
        output_dir / "failures_top.csv",
        failures[:500],
        [
            "index",
            "source_id",
            "category",
            "error_type",
            "factor_error_subtype",
            "gt_factor_scores",
            "actual_factor_scores",
            "gt_action_scores",
            "actual_action_scores",
            "action_score_diffs",
            "gt_factor_avg",
            "actual_factor_avg",
            "max_abs_factor_diff",
            "actual_found",
            "factor_edits",
            "reference",
            "candidate",
        ],
    )

    scored_rows = [row for row in rows if row["actual_found"]]
    summary = {
        "count": len(rows),
        "actual_count": len(scored_rows),
        "missing_actual_count": len(rows) - len(scored_rows),
        "success_count": sum(1 for row in scored_rows if row["success"]),
        "success_rate": round(sum(1 for row in scored_rows if row["success"]) / len(scored_rows), 4) if scored_rows else 0.0,
        "tolerance": args.tolerance,
        "avg_abs_factor_diff": round(mean(float(row["avg_abs_factor_diff"]) for row in scored_rows), 4) if scored_rows else 0.0,
    }
    write_json(output_dir / "summary.json", summary)

    for key in ["error_type", "factor_error_subtype", "category"]:
        summary_rows = summarize(rows, key)
        write_csv(
            output_dir / f"summary_by_{key}.csv",
            summary_rows,
            [key, "count", "actual_count", "success_count", "success_rate", "avg_gt_factor_avg", "avg_actual_factor_avg", "avg_abs_factor_diff"],
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
