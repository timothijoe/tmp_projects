# Reward Model Usage

这个目录里主要有两个入口脚本：

- `code/formal_batch_reward.py`: 批量评分，默认读取 `data/pairs.json`
- `code/formal_pipeline.py`: 单条 reference/candidate 评分

## 批量运行

在仓库根目录执行：

```bash
python3 reward_model/code/formal_batch_reward.py
```

运行时会在命令行显示进度条，例如当前处理到第几条、总共多少条。

默认行为：

- 输入数据：`reward_model/data/pairs.json`
- 输出目录：`reward_model/outputs/batch`
- 汇总结果：`reward_model/outputs/batch/batch_result.json`
- 关键字段 JSON：`reward_model/outputs/batch/score_items.json`
- 关键字段 CSV：`reward_model/outputs/batch/score_items.csv`

`score_items.json` 和 `score_items.csv` 会单独保存这些字段：

- `index`: 索引值
- `reference`: reference 原句
- `candidate`: candidate 原句
- `factor_scores`: `summary_score.factor_scores`
- `action_scores`: `summary_score.action_scores`

如需指定其他数据或输出目录：

```bash
python3 reward_model/code/formal_batch_reward.py \
  --pairs-json reward_model/data/pairs.json \
  --output-dir reward_model/outputs/batch
```

默认不会保存每条样本的完整中间结果子文件夹。如果需要 `case_001/`、`case_002/` 这类详细结果，增加：

```bash
python3 reward_model/code/formal_batch_reward.py --save-case-details
```

如果不想显示进度条，增加：

```bash
python3 reward_model/code/formal_batch_reward.py --no-progress
```

## 因素匹配规则

`summary_score.factor_scores` 使用一对一匹配：一个 candidate 因素最多只能匹配一个 reference 因素。这样可以避免候选句只说了一个因素时，被重复借给多个 reference 因素打分。

`pairs.json` 支持两种格式：

```json
[
  {
    "reference": "标准答案",
    "candidate": "候选答案"
  }
]
```

或：

```json
[
  ["标准答案", "候选答案"]
]
```

## 单条运行

```bash
python3 reward_model/code/formal_pipeline.py
```

默认会用脚本里的样例 reference/candidate，输出到：

```text
reward_model/outputs/single
```

如需指定输入：

```bash
python3 reward_model/code/formal_pipeline.py \
  --reference "左前方存在车辆遮挡盲区，自车应减速通过。" \
  --candidate "左前方有遮挡盲区，自车减速通过。"
```

## LLM 服务参数

脚本默认会尝试连接 OpenAI-compatible 服务：

- `--model`: 默认 `qwen3-vl-8b`
- `--base-url`: 默认 `http://127.0.0.1:8000/v1`
- `--api-key`: 默认 `EMPTY`

如果当前环境没有安装 `openai` 包，或者本地服务不可用，代码会自动切到本地规则 fallback，仍然可以跑完整流程。结果里出现 `"extraction_mode": "local_fallback"` 表示使用的是本地规则。

如果已经启动了本地模型服务，可以这样运行：

```bash
python3 reward_model/code/formal_batch_reward.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-vl-8b \
  --api-key EMPTY
```

## 查看参数

```bash
python3 reward_model/code/formal_batch_reward.py --help
python3 reward_model/code/formal_pipeline.py --help
```
