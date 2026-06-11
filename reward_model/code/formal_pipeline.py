import argparse
import json
from pathlib import Path

from formal_reward_core import DEFAULT_BASE_URL, DEFAULT_MODEL, evaluate_coc_pair

DEFAULT_REFERENCE = "左前方存在车辆遮挡盲区，右前方存在豁口盲区，自车应减速行驶通过"
DEFAULT_CANDIDATE = "左前方存在车辆遮挡盲区，右前方存在豁口盲区，自车应向左避让减速通过。"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "single"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete formal reward model pipeline.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="第一句/标准 CoC 总结")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE, help="第二句/候选 CoC 总结")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="JSON 输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI-compatible model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_coc_pair(
        reference_summary=args.reference,
        candidate_summary=args.candidate,
        output_dir=args.output_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    print(json.dumps(result["compact_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
