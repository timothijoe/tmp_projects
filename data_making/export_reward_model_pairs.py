#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pair_item(index, row, with_labels=False):
    item = {
        "index": index,
        "reference": row["source"],
        "candidate": row["text"],
    }
    if with_labels:
        item.update(
            {
                "source_id": row["source_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "target_score": row["score"],
                "error_type": row["error_type"],
            }
        )
        if "factor_error_subtype" in row:
            item["factor_error_subtype"] = row["factor_error_subtype"]
    return item


def export(rows, output_dir):
    output_dir = Path(output_dir)

    compatible_pairs = [pair_item(index, row) for index, row in enumerate(rows, 1)]
    labeled_pairs = [pair_item(index, row, with_labels=True) for index, row in enumerate(rows, 1)]

    write_json(output_dir / "reward_candidates_pairs.json", compatible_pairs)
    write_json(output_dir / "reward_candidates_pairs_with_labels.json", labeled_pairs)

    by_error_type = defaultdict(list)
    by_wrong_subtype = defaultdict(list)
    by_category = defaultdict(list)

    for row in rows:
        by_error_type[row["error_type"]].append(row)
        by_category[row["category"]].append(row)
        if row["error_type"] == "wrong_factor":
            by_wrong_subtype[row.get("factor_error_subtype", "unknown")].append(row)

    for error_type, subset in sorted(by_error_type.items()):
        pairs = [pair_item(index, row) for index, row in enumerate(subset, 1)]
        write_json(output_dir / f"reward_candidates_pairs_{error_type}.json", pairs)

    for subtype, subset in sorted(by_wrong_subtype.items()):
        pairs = [pair_item(index, row) for index, row in enumerate(subset, 1)]
        write_json(output_dir / f"reward_candidates_pairs_wrong_{subtype}.json", pairs)

    for category, subset in sorted(by_category.items()):
        pairs = [pair_item(index, row) for index, row in enumerate(subset, 1)]
        write_json(output_dir / f"reward_candidates_pairs_category_{category}.json", pairs)

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)

    groups = []
    for source_id in sorted(by_source):
        source_rows = by_source[source_id]
        source_rows.sort(key=lambda item: item["score"], reverse=True)
        groups.append(
            {
                "source_id": source_id,
                "reference": source_rows[0]["source"],
                "category": source_rows[0]["category"],
                "subcategory": source_rows[0]["subcategory"],
                "candidates": [
                    {
                        "candidate": row["text"],
                        "target_score": row["score"],
                        "error_type": row["error_type"],
                        **(
                            {"factor_error_subtype": row["factor_error_subtype"]}
                            if "factor_error_subtype" in row
                            else {}
                        ),
                    }
                    for row in source_rows
                ],
            }
        )
    write_json(output_dir / "reward_candidates_groups_with_labels.json", groups)

    summary = {
        "total_records": len(rows),
        "total_sources": len(by_source),
        "files": {
            "compatible_all": "reward_candidates_pairs.json",
            "labeled_all": "reward_candidates_pairs_with_labels.json",
            "grouped_labeled": "reward_candidates_groups_with_labels.json",
        },
        "counts_by_error_type": {key: len(value) for key, value in sorted(by_error_type.items())},
        "counts_by_wrong_factor_subtype": {
            key: len(value) for key, value in sorted(by_wrong_subtype.items())
        },
        "counts_by_category": {key: len(value) for key, value in sorted(by_category.items())},
    }
    write_json(output_dir / "reward_candidates_export_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_making/reward_candidates.jsonl")
    parser.add_argument("--output-dir", default="reward_model/data/reward_candidates")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    summary = export(rows, args.output_dir)
    print(
        f"exported {summary['total_records']} records "
        f"from {summary['total_sources']} sources to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
