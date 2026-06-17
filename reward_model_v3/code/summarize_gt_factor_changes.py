#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FACTOR_FIELDS = ["position", "category", "detail"]


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


def factor_diff(before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
    changes = []
    for field in FACTOR_FIELDS:
        old_value = before.get(field, "") if before else ""
        new_value = after.get(field, "") if after else ""
        if old_value != new_value:
            changes.append({"field": field, "from": old_value, "to": new_value})
    return changes


def compact_factor(factor: Dict[str, Any]) -> Dict[str, Any]:
    return {field: factor.get(field, "") for field in FACTOR_FIELDS}


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    factor_changes = []
    operation_counts = {"add": 0, "replace": 0, "delete": 0, "keep": 0}

    for factor_label in record.get("per_factor_labels", []):
        operation = factor_label.get("operation", "")
        factor_id = factor_label.get("factor_id", "")
        before = factor_label.get("from") or {}
        after = factor_label.get("to") or {}

        if operation == "add":
            operation_counts["add"] += 1
            factor_changes.append(
                {
                    "change_type": "add",
                    "factor_id": factor_id,
                    "from": None,
                    "to": compact_factor(after),
                    "note": "added_factor",
                }
            )
        elif operation == "modify":
            operation_counts["replace"] += 1
            factor_changes.append(
                {
                    "change_type": "replace",
                    "factor_id": factor_id,
                    "from": compact_factor(before),
                    "to": compact_factor(after),
                    "changed_fields": factor_diff(before, after),
                    "category_change_label": factor_label.get("category_change_label", ""),
                    "detail_change_label": factor_label.get("detail_change_label", ""),
                    "direction_change_label": factor_label.get("direction_change_label", ""),
                    "note": "replaced_factor",
                }
            )
        elif operation == "remove":
            operation_counts["delete"] += 1
            factor_changes.append(
                {
                    "change_type": "delete",
                    "factor_id": factor_id,
                    "from": compact_factor(before),
                    "to": None,
                    "note": "deleted_factor",
                }
            )
        elif operation == "keep":
            operation_counts["keep"] += 1

    return {
        "index": record["index"],
        "source_id": record.get("source_id", ""),
        "category": record.get("category", ""),
        "severity_level": record.get("severity_level", ""),
        "error_type": record.get("error_type", ""),
        "reference": record.get("reference", ""),
        "candidate": record.get("candidate", ""),
        "num_added_factors": operation_counts["add"],
        "num_replaced_factors": operation_counts["replace"],
        "num_deleted_factors": operation_counts["delete"],
        "num_kept_factors": operation_counts["keep"],
        "factor_changes": factor_changes,
        "factor_score": record.get("scores", {}).get("factor_score_after_add_penalty", ""),
        "provisional_score": record.get("scores", {}).get("provisional_score", ""),
    }


def compact_csv_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "index": row["index"],
                "source_id": row["source_id"],
                "category": row["category"],
                "severity_level": row["severity_level"],
                "error_type": row["error_type"],
                "num_added_factors": row["num_added_factors"],
                "num_replaced_factors": row["num_replaced_factors"],
                "num_deleted_factors": row["num_deleted_factors"],
                "num_kept_factors": row["num_kept_factors"],
                "factor_changes": json.dumps(row["factor_changes"], ensure_ascii=False),
                "factor_score": row["factor_score"],
                "provisional_score": row["provisional_score"],
                "reference": row["reference"],
                "candidate": row["candidate"],
            }
        )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reward_model_v3/data/gt_candidates/candidates_with_gt.json",
        help="V3 candidates_with_gt.json",
    )
    parser.add_argument(
        "--output-json",
        default="reward_model_v3/data/gt_candidates/factor_change_summary.json",
        help="逐 pair 的 factor 变化明细 JSON",
    )
    parser.add_argument(
        "--output-csv",
        default="reward_model_v3/data/gt_candidates/factor_change_summary.csv",
        help="逐 pair 的 factor 变化简表 CSV",
    )
    args = parser.parse_args()

    rows = [summarize_record(record) for record in load_json(args.input)]
    write_json(Path(args.output_json), rows)
    write_csv(
        Path(args.output_csv),
        compact_csv_rows(rows),
        [
            "index",
            "source_id",
            "category",
            "severity_level",
            "error_type",
            "num_added_factors",
            "num_replaced_factors",
            "num_deleted_factors",
            "num_kept_factors",
            "factor_changes",
            "factor_score",
            "provisional_score",
            "reference",
            "candidate",
        ],
    )
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
