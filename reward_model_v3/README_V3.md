# Reward Model V3

V3 继续沿用 V2 的 source schema、formal reward 和 comparison 流程，但候选生成改成 factor/action 粒度：

1. 对每个 GT factor 独立决定 `keep/remove/replace`。
2. 对每个 action 维度独立决定 `keep/remove/replace/conflict`。
3. 对新增 factor 单独记录 `ADD_FACTOR`，作为 precision penalty。
4. 每条 candidate 同时输出 mutation log、summary 和 provisional GT 分数。

## Step 1: 生成 Source Schema

```bash
python3 reward_model_v3/code/v1_build_gt_schema.py \
  --input data_making/test_sentences.txt \
  --output reward_model_v3/data/gt_schema/source_schemas.json \
  --local-only
```

## Step 2: 基于 Schema 生成 V3 Candidates

```bash
python3 reward_model_v3/code/v2_generate_gt_candidates.py \
  --schemas reward_model_v3/data/gt_schema/source_schemas.json \
  --output-dir reward_model_v3/data/gt_candidates \
  --variants-per-source 12 \
  --seed 13
```

输出：

```text
reward_model_v3/data/gt_candidates/pairs.json
reward_model_v3/data/gt_candidates/candidates_with_gt.json
reward_model_v3/data/gt_candidates/summary.json
```

`pairs.json` 仍然兼容 `formal_batch_reward.py`。

`candidates_with_gt.json` 新增：

- `factor_mutations`: 每个 GT factor 的去向，以及新增 factor。
- `action_mutations`: 每个 action 维度的变化。
- `mutation_summary`: 从 mutation log 聚合出来的计数。
- `provisional_gt_score`: 基于 factor/action mutation 的暂定总分。
- `candidate_factors`: candidate 的结构化 factor 列表。
- `candidate_action_schema`: candidate 的结构化 action。

同时保留 V2 兼容字段：

- `factor_edits`
- `gt_factor_scores`
- `gt_factor_avg`
- `gt_action_scores`
- `reference_action_schema`
- `candidate_action_schema`

## Factor Mutation Types

每个 GT factor 都会被独立处理：

```text
KEEP_FACTOR
REPLACE_FACTOR_VALUE
REPLACE_DIRECTION
REPLACE_SUB_CATEGORY
REPLACE_SUPER_CATEGORY
CROSS_CATEGORY
REMOVE_FACTOR
```

额外新增的 factor 记录为：

```text
ADD_FACTOR
```

当前 provisional factor 分：

```text
KEEP_FACTOR: 1.00
REPLACE_FACTOR_VALUE: 0.85
REPLACE_SUB_CATEGORY: 0.70
REPLACE_DIRECTION: 0.65
REPLACE_SUPER_CATEGORY: 0.45
CROSS_CATEGORY: 0.25
REMOVE_FACTOR: 0.00
ADD_FACTOR: 不进入 recall average，额外扣 add_factor_penalty
```

## Action Mutation Types

action 按 `lat/lon/strategy` 三个维度独立处理：

```text
KEEP_ACTION
REPLACE_ACTION
CONFLICT_ACTION
REMOVE_ACTION
```

action 分数由 `formal_reward_core.score_actions()` 计算，输出到 `gt_action_scores`。

## Provisional Score

当前暂定规则：

```text
factor_avg = avg(per_gt_factor_score)
add_factor_penalty = min(0.25, 0.06 * num_added_factors)
factor_after_add_penalty = max(0, factor_avg - add_factor_penalty)
action_avg = avg(lat/lon/strategy action scores)
provisional_gt_score = 0.75 * factor_after_add_penalty + 0.25 * action_avg
```

这个分数是后续标定的起点，不是最终 rubric。

## Step 3: 跑 Formal Reward

```bash
python3 reward_model_v3/code/formal_batch_reward.py \
  --pairs-json reward_model_v3/data/gt_candidates/pairs.json \
  --output-dir reward_model_v3/outputs/gt_batch
```

## Step 4: 比较 GT 和 Formal 输出

```bash
python3 reward_model_v3/code/v3_compare_gt_with_formal.py \
  --batch-result reward_model_v3/outputs/gt_batch/batch_result.json \
  --gt-json reward_model_v3/data/gt_candidates/candidates_with_gt.json \
  --output-dir reward_model_v3/outputs/gt_comparison
```
