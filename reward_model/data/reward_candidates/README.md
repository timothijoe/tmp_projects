# Reward Candidates Data

这个目录存放基于 `data_making/test_sentences.txt` 生成的 reward model 测试数据。

## 最核心文件

- `reward_candidates_pairs.json`
  - 用途：直接给 `formal_batch_reward.py` 跑分。
  - 格式：和 `reward_model/data/pairs.json` 一致，只包含 `index/reference/candidate`。

- `reward_candidates_pairs_with_labels.json`
  - 用途：跑完模型后做分析。
  - 相比 `reward_candidates_pairs.json`，额外包含 `source_id/category/subcategory/target_score/error_type/factor_error_subtype`。

- `reward_candidates_pairs_with_expected_scores.json`
  - 用途：把生成时的标签进一步映射到 formal reward 的预期输出。
  - 额外包含 `expected_factor_scores`、`expected_action_scores`、`expected_normalized_total_score`。
  - 注意：当前 `formal_batch_reward.py` 的 `action_scores.lat` 只区分 `换道/避让/保持/转弯`，不区分左/右方向。因此“向左换道”改成“向右换道”时，formal action 分数可能仍然较高；这种方向错误主要要看 factor 或后续扩展 action schema。

- `reward_candidates_groups_with_labels.json`
  - 用途：人工检查梯度质量。
  - 按 `source_id` 分组，把同一原句下的所有候选放在一起。

- `reward_candidates_export_summary.json`
  - 用途：查看导出统计。

- `reward_candidates_expected_scores.csv`
  - 用途：表格化查看每条候选的预期 `factor_scores/action_scores`。

## 拆分文件

按错误类型拆分：

- `reward_candidates_pairs_complete.json`
- `reward_candidates_pairs_generalized.json`
- `reward_candidates_pairs_missing_factor.json`
- `reward_candidates_pairs_wrong_factor.json`
- `reward_candidates_pairs_unsafe_action.json`

按 wrong factor 子类型拆分：

- `reward_candidates_pairs_wrong_direction_swap.json`
- `reward_candidates_pairs_wrong_same_category_subtype_swap.json`
- `reward_candidates_pairs_wrong_cross_category_swap.json`

按场景类别拆分：

- `reward_candidates_pairs_category_*.json`

这些拆分文件不是必须输入，主要用于单独测试某一类错误或某一类场景。

## 生成关系

```text
data_making/test_sentences.txt
  -> data_making/generate_reward_candidates.py
  -> data_making/reward_candidates.jsonl
  -> data_making/export_reward_model_pairs.py
  -> reward_model/data/reward_candidates/*.json
  -> reward_model/code/build_expected_scores.py
  -> reward_candidates_pairs_with_expected_scores.json
```

## 使用命令

跑全量：

```bash
python3 reward_model/code/formal_batch_reward.py \
  --pairs-json reward_model/data/reward_candidates/reward_candidates_pairs.json \
  --output-dir reward_model/outputs/reward_candidates_batch
```

分析结果：

```bash
python3 reward_model/code/analyze_reward_scores.py \
  --batch-result reward_model/outputs/reward_candidates_batch/batch_result.json \
  --output-dir reward_model/outputs/reward_candidates_analysis
```
