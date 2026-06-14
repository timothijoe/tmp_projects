#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from formal_reward_core import extract_coc_summary_locally


def clean_sentence(text: str) -> str:
    text = "".join(str(text).split())
    if text and text[-1] not in "。！？；":
        text += "。"
    return text


def action_text(sentence: str) -> str:
    sentence = clean_sentence(sentence).rstrip("。")
    marker = "自车应"
    if marker in sentence:
        return marker + sentence.split(marker, 1)[1]
    return "自车应谨慎通行"


def build_schema(source_id: int, source: str) -> dict:
    source = clean_sentence(source)
    scene = extract_coc_summary_locally(source)
    factors = []
    for index, factor in enumerate(scene.get("因素", []), 1):
        factors.append(
            {
                "factor_index": index,
                "position": factor.get("位置", ""),
                "category": factor.get("大类", ""),
                "detail": factor.get("细节", ""),
                "text_span": factor.get("原文片段", ""),
            }
        )
    return {
        "source_id": source_id,
        "source": source,
        "action_text": action_text(source),
        "factors": factors,
        "action_schema": scene.get("动作", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_making/test_sentences.txt")
    parser.add_argument("--output", default="reward_model_v2/data/gt_schema/source_schemas.json")
    args = parser.parse_args()

    lines = [
        clean_sentence(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schemas = [build_schema(index, source) for index, source in enumerate(lines, 1)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(schemas)} schemas to {output}")


if __name__ == "__main__":
    main()
