#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from formal_reward_core import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    create_openai_client,
    extract_coc_summary,
    extract_coc_summary_locally,
)


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


def build_action_schema(scene: dict) -> dict:
    action = scene.get("动作", {}) or {}
    return {
        "lat": action.get("横向决策", ["保持"]),
        "lon": action.get("纵向决策", ["保持"]),
        "strategy": action.get("执行策略", "直接执行"),
        "raw": action,
    }


def extract_scene_with_fallback(
    source: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    local_only: bool = False,
) -> tuple[dict, str, str]:
    fallback_error = ""
    if not local_only:
        try:
            client = create_openai_client(base_url=base_url, api_key=api_key)
            scene = extract_coc_summary(source, client=client, model=model)
            return scene, "llm", fallback_error
        except Exception as exc:
            fallback_error = f"{type(exc).__name__}: {exc}"

    scene = extract_coc_summary_locally(source)
    return scene, "local_fallback", fallback_error


def build_schema(
    source_id: int,
    source: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
    local_only: bool = False,
) -> dict:
    source = clean_sentence(source)
    scene, extraction_mode, fallback_error = extract_scene_with_fallback(
        source=source,
        model=model,
        base_url=base_url,
        api_key=api_key,
        local_only=local_only,
    )
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
    action_schema = build_action_schema(scene)
    return {
        "source_id": source_id,
        "source": source,
        "action_text": action_text(source),
        "factors": factors,
        "action_schema": action_schema,
        "extraction_mode": extraction_mode,
        "fallback_error": fallback_error,
        "raw_scene": scene,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_making/test_sentences.txt")
    parser.add_argument("--output", default="reward_model_v3/data/gt_schema/source_schemas.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--local-only", action="store_true", help="跳过 LLM，直接使用本地规则抽取")
    args = parser.parse_args()

    lines = [
        clean_sentence(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    schemas = [
        build_schema(
            index,
            source,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            local_only=args.local_only,
        )
        for index, source in enumerate(lines, 1)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(schemas)} schemas to {output}")


if __name__ == "__main__":
    main()
