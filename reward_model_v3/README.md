# Reward Model V3

V3 是基于 V2 的 factor/action 粒度候选生成版本。主要入口文档见：

```text
reward_model_v3/README_V3.md
```

常用流程：

```bash
python3 reward_model_v3/code/v1_build_gt_schema.py \
  --input data_making/test_sentences.txt \
  --output reward_model_v3/data/gt_schema/source_schemas.json \
  --local-only

python3 reward_model_v3/code/v2_generate_gt_candidates.py \
  --schemas reward_model_v3/data/gt_schema/source_schemas.json \
  --output-dir reward_model_v3/data/gt_candidates \
  --variants-per-source 12 \
  --seed 13

python3 reward_model_v3/code/formal_batch_reward.py \
  --pairs-json reward_model_v3/data/gt_candidates/pairs.json \
  --output-dir reward_model_v3/outputs/gt_batch

python3 reward_model_v3/code/v3_compare_gt_with_formal.py \
  --batch-result reward_model_v3/outputs/gt_batch/batch_result.json \
  --gt-json reward_model_v3/data/gt_candidates/candidates_with_gt.json \
  --output-dir reward_model_v3/outputs/gt_comparison
```
