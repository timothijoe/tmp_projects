import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from formal_reward_core import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    evaluate_coc_pair,
    save_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS_JSON = PROJECT_ROOT / "data" / "pairs.json"
DEFAULT_BATCH_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "batch"

DEFAULT_PAIRS = [
    (
        "左前方存在车辆遮挡盲区，右前方存在豁口盲区，自车应减速行驶通过",
        "左前方存在车辆遮挡盲区，右前方存在豁口盲区，自车应向左避让减速通过。",
    ),
    (
        "前方事故车占道，右侧空闲，右后无车，应右变道",
        "前方有事故车占道，右侧车道空闲且右后方无来车，自车应向右变道",
    ),
]


def count_answer_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def length_reward(
    text: str,
    min_chars: int = 20,
    target_chars: int = 80,
    max_chars: int = 180,
) -> Dict[str, Any]:
    char_count = count_answer_chars(text)
    if char_count <= 0:
        score = 0.0
    elif char_count < min_chars:
        score = char_count / min_chars
    elif char_count <= target_chars:
        score = 1.0
    elif char_count <= max_chars:
        score = 1.0 - 0.5 * ((char_count - target_chars) / (max_chars - target_chars))
    else:
        score = 0.2

    return {
        "char_count": char_count,
        "min_chars": min_chars,
        "target_chars": target_chars,
        "max_chars": max_chars,
        "score": round(max(0.0, min(1.0, score)), 4),
    }


def char_ngrams(text: str, n: int) -> List[str]:
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < n:
        return []
    return [compact[index : index + n] for index in range(len(compact) - n + 1)]


def max_consecutive_repeat(text: str, min_unit: int = 2, max_unit: int = 12) -> Dict[str, Any]:
    compact = re.sub(r"\s+", "", text or "")
    best = {"phrase": "", "times": 1}
    for unit_len in range(min_unit, max_unit + 1):
        index = 0
        while index + unit_len <= len(compact):
            phrase = compact[index : index + unit_len]
            times = 1
            cursor = index + unit_len
            while compact[cursor : cursor + unit_len] == phrase:
                times += 1
                cursor += unit_len
            if times > best["times"]:
                best = {"phrase": phrase, "times": times}
            index += 1
    return best


def repetition_reward(
    text: str,
    ngram_size: int = 6,
    bad_repeat_ratio: float = 0.35,
    terrible_repeat_ratio: float = 0.55,
    bad_consecutive_times: int = 3,
) -> Dict[str, Any]:
    ngrams = char_ngrams(text, ngram_size)
    if not ngrams:
        return {
            "score": 1.0,
            "repeat_ratio": 0.0,
            "unique_ngram_count": len(ngrams),
            "total_ngram_count": len(ngrams),
            "max_consecutive_repeat": {"phrase": "", "times": 1},
            "flag": "ok",
        }

    unique_count = len(set(ngrams))
    repeat_ratio = 1.0 - unique_count / len(ngrams)
    consecutive = max_consecutive_repeat(text)

    if consecutive["times"] >= bad_consecutive_times or repeat_ratio >= terrible_repeat_ratio:
        score = 0.0
        flag = "severe_repetition"
    elif repeat_ratio >= bad_repeat_ratio:
        score = 0.3
        flag = "repetition"
    else:
        score = 1.0
        flag = "ok"

    return {
        "score": score,
        "repeat_ratio": round(repeat_ratio, 4),
        "unique_ngram_count": unique_count,
        "total_ngram_count": len(ngrams),
        "max_consecutive_repeat": consecutive,
        "flag": flag,
    }


def text_quality_reward(candidate: str) -> Dict[str, Any]:
    length = length_reward(candidate)
    repetition = repetition_reward(candidate)
    score = length["score"] * repetition["score"]
    return {
        "score": round(score, 4),
        "length_reward": length,
        "repetition_reward": repetition,
        "formula": "text_quality_score = length_reward.score * repetition_reward.score",
    }


def combine_task_and_text_reward(
    task_score: float,
    text_score: float,
    task_weight: float = 0.85,
    text_weight: float = 0.15,
) -> Dict[str, Any]:
    final_score = task_weight * task_score + text_weight * text_score
    return {
        "task_score": round(task_score, 4),
        "text_quality_score": round(text_score, 4),
        "final_reward": round(final_score, 4),
        "formula": f"final_reward = {task_weight} * task_score + {text_weight} * text_quality_score",
    }


def evaluate_one_pair_with_extra_reward(
    reference: str,
    candidate: str,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
) -> Dict[str, Any]:
    result = evaluate_coc_pair(
        reference_summary=reference,
        candidate_summary=candidate,
        output_dir=output_dir,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    compact = result["compact_result"]
    task_score = float(compact["summary_score"]["normalized_total_score"])
    text_reward = text_quality_reward(candidate)
    compact["text_quality_reward"] = text_reward
    compact["final_reward"] = combine_task_and_text_reward(task_score, text_reward["score"])
    save_json(compact, Path(output_dir) / "compact_result_with_extra_reward.json")
    return compact


def evaluate_pair_list(
    pairs: Sequence[Tuple[str, str]],
    output_dir: str | Path = DEFAULT_BATCH_OUTPUT_DIR,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "EMPTY",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    results = []
    for index, (reference, candidate) in enumerate(pairs, start=1):
        item_output_dir = output_dir / f"case_{index:03d}"
        compact = evaluate_one_pair_with_extra_reward(
            reference=reference,
            candidate=candidate,
            output_dir=item_output_dir,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        results.append(
            {
                "index": index,
                "reference": reference,
                "candidate": candidate,
                "result": compact,
                "output_dir": str(item_output_dir),
            }
        )

    final_rewards = [item["result"]["final_reward"]["final_reward"] for item in results]
    task_scores = [item["result"]["final_reward"]["task_score"] for item in results]
    text_scores = [item["result"]["final_reward"]["text_quality_score"] for item in results]
    summary = {
        "case_count": len(results),
        "avg_final_reward": round(sum(final_rewards) / len(final_rewards), 4) if final_rewards else 0.0,
        "avg_task_score": round(sum(task_scores) / len(task_scores), 4) if task_scores else 0.0,
        "avg_text_quality_score": round(sum(text_scores) / len(text_scores), 4) if text_scores else 0.0,
    }
    batch_result = {"summary": summary, "results": results}
    save_json(batch_result, output_dir / "batch_result.json")
    return batch_result


def load_pairs(path: str | Path) -> List[Tuple[str, str]]:
    data_path = Path(path).expanduser()
    data = json.loads(data_path.read_text(encoding="utf-8"))
    pairs = []
    for item in data:
        if isinstance(item, dict):
            pairs.append((item["reference"], item["candidate"]))
        else:
            pairs.append((item[0], item[1]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch reward evaluation for CoC summary pairs.")
    parser.add_argument(
        "--pairs-json",
        default=str(DEFAULT_PAIRS_JSON),
        help="JSON 文件，格式为 [[ref, candidate], ...] 或 [{'reference': ..., 'candidate': ...}]",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_BATCH_OUTPUT_DIR), help="批量结果输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_path = Path(args.pairs_json)
    pairs = load_pairs(pairs_path) if pairs_path.exists() else DEFAULT_PAIRS
    result = evaluate_pair_list(
        pairs=pairs,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
