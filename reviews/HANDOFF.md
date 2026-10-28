# v1 实现交接说明

日期：2026-10-28。
对应设计合同：`docs/README_CODE_DESIGN.md`（最高优先级）。
当前等级：**D0**；代码已实现，验证未达到 D1。计划外实现与已知偏离见 `reviews/EXTRA_WORK.md`。

## 1. 已实现范围

| 层 | 文件 | 说明 |
|---|---|---|
| 合同 | `src/sbona/contracts.py` | `ModelInput` / `TargetBatch` / `GraphContext` / 输出与 scaler 数据类；形状、dtype、设备检查 |
| 配置 | `src/sbona/config.py` | 严格 YAML schema，未知 key fail-fast；跨段校验；CLI > env > config > 默认根目录 |
| 数据 | `src/sbona/data/` | PEMS08 `.npz` / 合成数据、无泄漏窗口、train-only scaler、二值无向图与行归一化稀疏扩散 |
| Backbone | `src/sbona/backbones/` | STAEformer v1（按本地 pristine commit 重实现，未直接 vendor）；STID / Graph WaveNet / AGCRN 明确 pending |
| 模型 | `src/sbona/models/` | 动作生成、Sobol anchor、结构化共享残差解码、Router、`SBoN.forward_candidates` / `predict_top1` |
| 目标 | `src/sbona/objectives/` | 候选 cost、consistency 约束、soft target、candidate/router/separation loss |
| 训练 | `src/sbona/training/` | Stage A/B/C/D、F/R/Q probe、checkpoint/resume、hash 校验、真实 optimizer step 计数 |
| 评测 | `src/sbona/evaluation/` | 流式指标、三种 oracle、regime、多样性、bootstrap、run artifacts、profiling、aggregation |
| Baselines | `src/sbona/baselines/` | single、independent heads、unstructured、perturbation；`build_adapter` |
| CLI | `src/sbona/cli.py` | `prepare-data`、`train-base`、`train-single`、`train-candidates`、`train-router`、`probe-route`、`evaluate`、`profile`、`aggregate` |
| 配置样例 | `configs/experiments/` | PEMS08 base/single/K8/probe 与 synthetic smoke |

### 关键合同落实

- `SBoN.forward_candidates(inp, graph)` 与 `SBoN.predict_top1(inp, graph)` 不接收 target 或任意 kwargs；测试检查签名。
- `predict_top1` 只解码 Router 选中的一条动作，不调用 `forward_all`。
- `m=0` 时残差精确为 0（`W_out` 无 bias，`tanh(0)=0`）。
- scaler 的 `mu/sigma` 与 envelope 只用训练段；`sigma = max(pop_std, 1e-6)`。
- oracle 按整条 `[T, N, 1]` 样本选候选；MAE / MSE / cost oracle 分开命名。
- K=1 时报告为 `single`，不写伪 `top1_*` 文件；K>1 的部署评测必须找到已训练 Router checkpoint。
- 评测写入 sibling `<source-run>-eval-<split>`，不覆盖训练 run 的 `metrics.json`。
- 每次运行写 `resolved_config.yaml`、`manifest.json`、`train.jsonl`、`metrics.json`、checkpoint；失败写 `failure.json`。

## 2. 本次实际执行的验证

环境（临时 venv，未提交到仓库）：

- Python 3.11.11
- torch 2.2.2（CPU）
- numpy 1.26.4
- scipy 1.15.2
- PyYAML 6.0.2
- pytest 9.1.1

命令与结果：

```bash
cd /Users/ljw/Documents/code/SBoN-Adapter
python -m pytest -q
# 65 passed, 1 skipped
# skipped: tests/gpu/test_gpu_smoke.py (CUDA unavailable)

ruff check src tests scripts
# All checks passed

python -m compileall -q src/sbona
# OK
```

### Synthetic 端到端 smoke

`tests/integration/test_cli_synthetic.py` 已通过，覆盖：

1. `prepare-data`（synthetic fixture）
2. `train-base`
3. `train-single`
4. `train-candidates`
5. `train-router`
6. `evaluate` base 与 K=2 candidates
7. 检查 `candidates` 训练 run 的 `metrics.json` 未被覆盖
8. 检查 `top1_test.npz` 与根目录 `experiments.csv` 生成

此外手工执行了同类的 `profile`、`aggregate`、K=1 `train-single` + `evaluate`，以及 1200 步 synthetic 的 `probe-route`，均成功写出 artifact。以上数值均为 synthetic/CPU smoke，不是科学结果。

### PEMS08 真实数据 smoke

使用本地只读数据与图文件，在临时目录执行：

```bash
python -m sbona.cli prepare-data \
  --config configs/experiments/pems08.yaml \
  --data-root <tmp>/data \
  --output-root <tmp>/out
```

结果：`shape=(17856, 170)`，`nodes=170`，无向边 `274`，窗口 `train=10690, val=3560, test=3561`。

另外用 test split 的 2 个窗口做了 CPU forward：

- `history: (2, 12, 170, 1)`
- `candidates: (2, 8, 12, 170, 1)`
- `top1: (2, 12, 170, 1)`
- 输出有限

**未执行**真实 PEMS08 训练、Router 训练或真实 test 指标评测。

## 3. 首次运行中发现并修复的问题

1. `TrafficWindowDataset.__getitem__` 返回 3D `ModelInput`，违反 batched `[B, L, N, 1]` 合同。
   - 现在每个样本是 batch=1；`collate_samples` 用 `torch.cat` 沿 batch 轴拼接。
2. `_masked_mean_over_nonbatch` 把候选轴一起归约，得到 `[B]` 而非 `[B, K]`，导致 K=1 训练报 `values must be [B, K]`。
   - 现在固定 batch/candidate 轴，归约 `dim>=2`；新增形状回归测试。
3. `CandidateHealthAccumulator.result()` 使用未定义变量 `sigma`。
   - 现在在 `update()` 保存 `self.sigma`，结果使用保存值。
4. `validate_scheme` 先报“target leaves its split”而不是泄漏。
   - 现在先检查跨 split 重叠，再检查 split 内边界。
5. 测试自身问题：`build_subrange_scheme` 关键字参数后跟位置参数（SyntaxError）、tie-aware 期望值与定义不一致、浮点精确比较过严。
   - 已修正测试数据与容差；未放宽实现语义。

## 4. 明确未完成 / 阻塞

- **GPU smoke 未执行**：当前机器无 CUDA；不能声称 A100 显存、吞吐、bf16 或 DDP 行为。
- **真实数据训练未执行**：没有任何真实数据集的 checkpoint、真实 MAE、Router 收益或 oracle 分析。
- **STID / Graph WaveNet / AGCRN 未实现**：缺 pinned commit/许可证/等价验证，配置会 fail-fast，不冒充支持。
- **依赖 lock 与 CI 未提供**：`pyproject.toml` 有依赖声明，但无锁文件/CI workflow。
- **日历来源是数据自带 tod/dow 通道 + 有记录的起始星期**：`scripts/build_calendar.py` 的通道模式保留原始 tod（包括 PEMS-BAY 的 1 小时缺口），只施加常量星期相位偏移；均匀模式则从记录在案的起始时间生成。两种模式都用周末/工作日比值做一致性检查，但不是发布方原始时间戳证明。若拿到原始时间戳应替换 `data/<NAME>/calendar.npz` 并重跑 `prepare-data`。
- **METR-LA / PEMS-BAY 已接入配置但未训练**：`derived_binary_undirected` 与 `speed (mph)` 协议已实现，`prepare-data` 与 CPU 前向通过；没有 checkpoint 或指标，不能声称可复现论文结果。
- **probe 对数据长度有要求**：小型 synthetic fixture 会因 R/Q 窗口不足而 fail-fast；1200 步 synthetic 已验证可运行。
- **上游 STAEformer 许可**：本地 pin commit `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f` 无 LICENSE，因此本仓按其架构重实现，未直接复制源码。

## 4.1 数据与日历（2026-10-28 更新）

- `data/` 已放入 PEMS03/04/07/08 的 `data.npz`、图 CSV、PEMS03 节点 ID 与 `calendar.npz`；全部经 `prepare-data` 验证。
- 新增 `data.timestamps_file` 配置项：支持 `.npy`/`.npz` 的 `datetime64 [M]`、整数 `[M]` 或 `(tod, dow) [M,2]` 表；文件 sha256 进入 prepare signature，改动即重建。
- `build_calendar` 新增 `[M,2]` 分支，要求 tod ∈ [0,1)、dow 为整数 0..6；不再默默把全零当周一。
- 四个 PEMS 配置已启用 `day_of_week: true` 与 `dow_embedding_dim: 24`（上游默认值）；PEMS08 CPU 前向已验证嵌入路径。
- METR-LA / PEMS-BAY 的邻接取自 Zenodo 5724362（CC-BY-4.0），已核验节点顺序与起始时间；转换后 METR-LA 1313 条无向边、PEMS-BAY 2079 条。配置使用 `derived_binary_undirected`（上游矩阵非对称）、`data.unit: speed (mph)` 与 `official_compat_nonzero`；已完成 `prepare-data` 与 CPU 前向，未训练。详见 `reviews/DATA_PROVENANCE.md`。
- 计划外实现（外部日历、派生拓扑、单位协议、额外数据集、`.gitignore` 修复、已知偏离）汇总在 `reviews/EXTRA_WORK.md`。

## 5. 建议下一步

1. 在 A100 环境执行 `tests/gpu/test_gpu_smoke.py`，记录驱动/CUDA/显存/耗时。
2. 补齐依赖 lock 与最小 CPU CI。
3. 按 `docs/README_EXPERIMENTS.md` 的 Gate 顺序跑 PEMS08 Stage A → B → C → D，并保存完整 run 目录；用 PEMS04/07 做跨数据集复核。
4. 先看 Router 是否真的优于 uniform / single；若否，按失败结果交付，不改测试集或指标。
5. 再评估第二个 backbone；没有 pin/许可证/等价测试前不要合并。

## 6. 非声明

- 本仓当前没有任何论文级结果。
- synthetic smoke 的数值只证明管线可运行，不能用于比较方法。
- CPU 测试通过不等于 GPU 训练正确，也不等于科学假设成立。
