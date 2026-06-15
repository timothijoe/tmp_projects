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

`candidates_with_gt.json` 的核心字段：

- `per_factor_labels`: 每个 factor 的变化标签；一句话有多个 factor，就有多条 `gt_*` 记录。
- `sample_factor_labels`: 整条样本的 factor 聚合标签。
- `factor_summary`: factor 变化计数。
- `per_action_labels`: 每个 action 维度的变化标签。
- `sample_action_labels`: 整条样本的 action 聚合标签。
- `action_summary`: action 变化计数。
- `scores`: 基于变化记录计算的暂定分数。

V3 不再保留 V2 兼容字段，避免同一件事出现多套名字。

## Factor Mutation Types

每个 GT factor 都会被独立处理，并在 `per_factor_labels` 里输出一条记录。多个 factor 就会有多条 `gt_*` 记录：

```json
{
  "factor_id": "gt_1",
  "operation": "modify",
  "category_change_label": "category_unchanged",
  "detail_change_label": "detail_changed",
  "direction_change_label": "direction_changed",
  "delete_label": "not_deleted",
  "add_label": "not_added",
  "is_category_changed": false,
  "is_detail_changed": true,
  "is_direction_changed": true,
  "is_deleted": false,
  "is_added": false,
  "from": {
    "position": "前方",
    "category": "车辆行为",
    "detail": "两个行人缓慢移动"
  },
  "to": {
    "position": "右前方",
    "category": "车辆行为",
    "detail": "车辆缓行"
  },
  "score": 0.35
}
```

`operation` 只表示元素层面的动作：

```text
keep
modify
remove
add
```

新增 factor：

```json
{
  "factor_id": "add_1",
  "operation": "add",
  "category_change_label": "not_applicable",
  "detail_change_label": "not_applicable",
  "direction_change_label": "not_applicable",
  "delete_label": "not_deleted",
  "add_label": "added_factor",
  "is_category_changed": false,
  "is_detail_changed": false,
  "is_direction_changed": false,
  "is_deleted": false,
  "is_added": true,
  "from": null,
  "to": {
    "position": "左侧",
    "category": "车辆行为",
    "detail": "车辆切入"
  },
  "score": "extra_penalty"
}
```

整条样本会额外在 `sample_factor_labels` 里给一组聚合 label：

```json
{
  "category_change_label": "has_category_change",
  "detail_change_label": "has_detail_change",
  "direction_change_label": "no_direction_change",
  "delete_label": "no_deleted_factor",
  "add_label": "has_added_factor",
  "num_category_changed_factors": 1,
  "num_detail_changed_factors": 1,
  "num_direction_changed_factors": 0,
  "num_deleted_factors": 0,
  "num_added_factors": 1
}
```

当前 provisional factor 分由变化类型扣分得到：

```text
VALUE: -0.15
DIRECTION: -0.35
SUB_CATEGORY: -0.30
SUPER_CATEGORY: -0.55
CROSS_CATEGORY: -0.75
remove: 0.00
add: 不进入 GT factor 平均分，额外扣 add_penalty
```

## Action Changes

action 按 `lat/lon/strategy` 三个维度独立处理，输出到 `per_action_labels`：

```json
{
  "dimension": "lat",
  "operation": "replace",
  "change_label": "lat_replace",
  "is_changed": true,
  "from": ["避让"],
  "to": ["换道"]
}
```

整条样本的 action 聚合 label 输出到 `sample_action_labels`：

```json
{
  "action_change_label": "two_action_dimensions_changed",
  "num_changed_action_dimensions": 2,
  "changed_action_dimensions": ["lat", "lon"],
  "changed_action_operations": ["replace", "conflict"]
}
```

## Provisional Score

当前暂定规则：

```text
factor_avg = avg(scores for gt_* factors)
add_penalty = min(0.25, 0.06 * num_added_factors)
factor_score_after_add_penalty = max(0, factor_avg - add_penalty)
action_avg = avg(lat/lon/strategy action scores)
provisional_score = 0.75 * factor_score_after_add_penalty + 0.25 * action_avg
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
