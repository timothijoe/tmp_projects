#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from formal_reward_core import evaluate_extracted_scenes, extract_coc_summary_locally


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


def expected_scores(reference: str, candidate: str) -> Dict[str, Any]:
    reference_scene = extract_coc_summary_locally(reference)
    candidate_scene = extract_coc_summary_locally(candidate)
    result = evaluate_extracted_scenes(reference_scene, candidate_scene, client=None)
    summary_score = result["summary_score"]
    return {
        "expected_factor_scores": summary_score["factor_scores"],
        "expected_action_scores": summary_score["action_scores"],
        "expected_factor_total": summary_score["factor_total"],
        "expected_action_total": summary_score["action_total"],
        "expected_raw_total_score": summary_score["raw_total_score"],
        "expected_normalized_total_score": summary_score["normalized_total_score"],
        "expected_factor_pairs": result["factor_pairs"],
    }


def build_rows(items: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
    rows = []
    total = len(items)
    for index, item in enumerate(items, 1):
        row = dict(item)
        row.update(expected_scores(item["reference"], item["candidate"]))
        rows.append(row)
        if show_progress:
            sys.stderr.write(f"\rProgress: {index}/{total}")
            if index == total:
                sys.stderr.write("\n")
            sys.stderr.flush()
    return rows


def compact_csv_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "index": row["index"],
                "source_id": row.get("source_id", ""),
                "category": row.get("category", ""),
                "subcategory": row.get("subcategory", ""),
                "target_score": row.get("target_score", ""),
                "error_type": row.get("error_type", ""),
                "factor_error_subtype": row.get("factor_error_subtype", ""),
                "expected_factor_scores": json.dumps(
                    row["expected_factor_scores"], ensure_ascii=False
                ),
                "expected_action_lat": row["expected_action_scores"].get("lat", ""),
                "expected_action_lon": row["expected_action_scores"].get("lon", ""),
                "expected_action_strategy": row["expected_action_scores"].get("strategy", ""),
                "expected_normalized_total_score": row["expected_normalized_total_score"],
                "reference": row["reference"],
                "candidate": row["candidate"],
            }
        )
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="reward_model_v3/data/reward_candidates/reward_candidates_pairs_with_labels.json",
        help="带 target_score/error_type 的候选 pairs 文件",
    )
    parser.add_argument(
        "--output-json",
        default="reward_model_v3/data/reward_candidates/reward_candidates_pairs_with_expected_scores.json",
    )
    parser.add_argument(
        "--output-csv",
        default="reward_model_v3/data/reward_candidates/reward_candidates_expected_scores.csv",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = build_rows(load_json(args.input), show_progress=not args.no_progress)
    write_json(Path(args.output_json), rows)
    write_csv(
        Path(args.output_csv),
        compact_csv_rows(rows),
        [
            "index",
            "source_id",
            "category",
            "subcategory",
            "target_score",
            "error_type",
            "factor_error_subtype",
            "expected_factor_scores",
            "expected_action_lat",
            "expected_action_lon",
            "expected_action_strategy",
            "expected_normalized_total_score",
            "reference",
            "candidate",
        ],
    )
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
