# Reward Model V2

V2 的目标是把数据生成和 GT factor 分数绑定起来：

1. 固化 reference 的结构化 schema。
2. 生成 candidate 时记录改了哪个 factor。
3. 根据编辑类型直接生成 GT factor scores。
4. 用 `formal_batch_reward.py` 跑实际分数。
5. 比较 GT factor scores 和 formal 实际 factor scores，计算成功率。

## Step 1: 生成 Source Schema

```bash
python3 reward_model_v2/code/v1_build_gt_schema.py \
  --input data_making/test_sentences.txt \
  --output reward_model_v2/data/gt_schema/source_schemas.json
```

输出：

```text
reward_model_v2/data/gt_schema/source_schemas.json
```

## Step 2: 基于 Schema 生成带 GT 的 Candidates

```bash
python3 reward_model_v2/code/v2_generate_gt_candidates.py \
  --schemas reward_model_v2/data/gt_schema/source_schemas.json \
  --output-dir reward_model_v2/data/gt_candidates
```

输出：

```text
reward_model_v2/data/gt_candidates/pairs.json
reward_model_v2/data/gt_candidates/candidates_with_gt.json
reward_model_v2/data/gt_candidates/summary.json
```

`pairs.json` 给 formal reward 跑分。

`candidates_with_gt.json` 保留：

- `factor_edits`
- `gt_factor_scores`
- `gt_factor_avg`

## Step 3: 跑 Formal Reward

```bash
python3 reward_model_v2/code/formal_batch_reward.py \
  --pairs-json reward_model_v2/data/gt_candidates/pairs.json \
  --output-dir reward_model_v2/outputs/gt_batch
```

输出：

```text
reward_model_v2/outputs/gt_batch/batch_result.json
```

## Step 4: 比较 GT 和 Formal 输出

```bash
python3 reward_model_v2/code/v3_compare_gt_with_formal.py \
  --batch-result reward_model_v2/outputs/gt_batch/batch_result.json \
  --gt-json reward_model_v2/data/gt_candidates/candidates_with_gt.json \
  --output-dir reward_model_v2/outputs/gt_comparison
```

输出：

```text
reward_model_v2/outputs/gt_comparison/summary.json
reward_model_v2/outputs/gt_comparison/gt_vs_formal.csv
reward_model_v2/outputs/gt_comparison/failures_top.csv
reward_model_v2/outputs/gt_comparison/summary_by_error_type.csv
reward_model_v2/outputs/gt_comparison/summary_by_factor_error_subtype.csv
reward_model_v2/outputs/gt_comparison/summary_by_category.csv
```

## 当前 GT Factor Rubric

每个 reference factor 独立打分：

- `complete`: 1.0
- `missing_factor`: 0.0
- `direction_swap`: 0.5
- `same_category_subtype_swap`: 0.5
- `cross_category_swap`: 0.0
- `extra_factor`: 原 reference factor 不扣分，额外记录 hallucination edit

`gt_factor_avg = sum(gt_factor_scores) / reference_factor_count`

V2 暂时只评估 factor，不评估 action。
