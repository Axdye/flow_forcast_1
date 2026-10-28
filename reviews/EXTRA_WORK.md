# 计划外工作记录

日期：2026-10-28。

本文件记录**已经实现、但不在原计划明确要求范围内**的工作，避免这些改动在交接时被误认为设计合同的一部分，也避免后来者重复踩坑。

“原计划”指：

- `docs/README_CODE_DESIGN.md`（最高优先级合同）
- `docs/README_EXPERIMENTS.md`
- `tasks/TASK_A.md` … `tasks/TASK_D.md`
- `configs/design_defaults.yaml`

本文件不改变任何合同，也不替代 `reviews/STATUS.md` 与 `reviews/HANDOFF.md`。凡是与合同有张力的事项，单列在 §9。

## 1. 一览

| ID | 内容 | 原计划状态 | 主要文件 | 当前验证 |
|---|---|---|---|---|
| X1 | 外部日历文件 `data.timestamps_file` | 未规定 | `src/sbona/config.py`、`data/manifest.py`、`data/dataset.py` | 单测 + 六个数据集 `prepare-data` |
| X2 | `scripts/build_calendar.py`（均匀 / 通道两种模式） | 未规定 | `scripts/build_calendar.py` | 单测；生成六个 `calendar.npz` |
| X3 | 星期语义：相位偏移与 PEMS-BAY 缺口 | 只规定“不能猜” | `scripts/build_calendar.py`、`reviews/DATA_PROVENANCE.md` | 起始星期与周末/工作日比值交叉检查 |
| X4 | PEMS03/04/07 实验配置 | 只点名数据集，未要求配置 | `configs/experiments/pems0{3,4,7}.yaml` | `prepare-data` 全部通过 |
| X5 | METR-LA / PEMS-BAY 数据落地 | 计划外数据集 | `data/METRLA/`、`data/PEMSBAY/`、`configs/experiments/{metrla,pemsbay}.yaml` | `prepare-data` + CPU 前向；**未训练** |
| X6 | `derived_binary_undirected` 拓扑标签 | 只允许 published 二值图 | `src/sbona/data/graph.py`、`config.py` | 单测：同扩散、不同 `graph_hash` |
| X7 | `data.unit` 与 `metric_units(unit)` | 只要求 manifest 记单位 | `config.py`、`evaluation/runner.py`、`cli.py` | 单测；prepared manifest 与 `metric_units` 均为 mph |
| X8 | `data/<NAME>/SOURCE.json` 与 `reviews/DATA_PROVENANCE.md` | 只要求记录来源 | `data/*/SOURCE.json`、`reviews/DATA_PROVENANCE.md` | 文件 hash 与上游下载比对 |
| X9 | 额外测试 `test_calendar.py`、`test_units_topology.py` | 未规定 | `tests/unit/` | 当前 65 passed, 1 skipped |
| F1 | `.gitignore` 误伤 `src/sbona/data/` | —（缺陷） | `.gitignore` | 修复后源码包进入版本控制 |

## 2. 数据与日历

### X1 外部日历文件 `data.timestamps_file`

**做了什么**

- `DataConfig` 新增可选键 `timestamps_file`，路径相对 `data_root` 解析。
- 支持 `.npz` / `.npy`，内容可以是：
  - `[M]` `datetime64`；
  - `[M]` 整数（步索引，或大于 `1e5` 时按“分钟数”解释）；
  - `[M, 2]` 浮点表：第 0 列 time-of-day ∈ `[0, 1)`，第 1 列 day-of-week（整数 0..6，Monday=0）。
- 文件长度必须等于原始步数 `M`；`[M, 2]` 表严格校验 tod 范围与 dow 整数性。
- 文件 sha256 进入 `compute_prepare_signature`。换日历会触发 prepared 缓存重建，不会静默复用旧特征。
- manifest 的 `timestamps_source` 写入 `external calendar file <path> sha256=<前16位>`，并把文件登记到 `files`（role=`timestamps`）。
- 配置层拒绝 `timestamps_file` 与 `timestamps_available: false` 同时出现。
- `build_calendar` 在没有日历时仍退化为相对 time-of-day 且 `calendar_available=false`；`day_of_week: true` 会继续 fail-fast，不会猜星期。

**为什么算计划外**

设计 §4.1 只要求 manifest 记录“时间戳来源”，§4.1 又禁止在没有可靠时间戳时猜起始星期，但没有规定日历如何进入系统。外部文件 + hash 是本次选择的实现方式。

**验证**

- `tests/unit/test_calendar.py` 覆盖 `[M, 2]`、tod/dow 非法值、长度不匹配、配置冲突、prepare 集成与签名失效。
- 六个数据集均通过 `prepare-data`，`calendar_available=true`。

### X2 `scripts/build_calendar.py`

**做了什么**

两种互斥模式：

1. `--start/--steps`（均匀模式）：从记录在案的起始时间生成 `datetime64[s]` 轴，适合数据本身没有时间通道的情况。
2. `--from-data --start-weekday`（通道模式）：读取 `data.npz` 自带的 tod/dow 通道，输出 `[M, 2]` 表。

通道模式的两个关键约束：

- `tod` **原样保留**，因此数据内部的缺口不会被“均匀时间轴”抹平。
- `--start-weekday` 只用于计算一个**常量相位偏移** `(start_weekday - phase[0]) % 7`，不做任何逐点插值或拟合。这保证把“day_index % 7”这类相位通道映射成真实星期时，变化规则是显式且可复核的。

生成的 `.npz` 同时写入 `note` 与 `mode`，便于事后判断日历来源。该脚本属于数据来源工具，不参与模型计算。

**为什么不直接用均匀模式**

PEMS-BAY 在原始第 20183 步附近缺了 12 个 5 分钟样本（相邻 tod 差为 13 步，即 1 小时）。均匀时间轴会在缺口后整体偏移一小时；通道模式保留 `tod`，缺口位置不变。

### X3 星期语义：相位 vs 真实星期

**做了什么**

| 数据集 | dow 通道含义 | 起始 | 起始星期 | 偏移 |
|---|---|---|---|---|
| PEMS03 | `day_index % 7` 相位 | 2018-09-01 | 周六（5） | +5 |
| PEMS04 | 相位 | 2018-01-01 | 周一（0） | 0 |
| PEMS07 | 相位 | 2017-05-01 | 周一（0） | 0 |
| PEMS08 | 相位 | 2016-07-01 | 周五（4） | +4 |
| METR-LA | 真实星期 | 2012-03-01 | 周四（3） | 0 |
| PEMS-BAY | 真实星期 | 2017-01-01 | 周日（6） | 0 |

- PEMS 四个数据集的 dow 通道都满足 `dow == day_index // 288 % 7`，只是 `day_index % 7` 的相位，不能当作绝对星期。
- METR-LA / PEMS-BAY 的 dow 通道不满足该关系，但首行值与真实起始星期一致，视为真实星期，偏移为 0。

**交叉检查**（不是独立证明）

| 数据集 | 周末/工作日平均原始值 | 解释 |
|---|---|---|
| PEMS03 | 0.878 | 流量，周末更低 |
| PEMS04 | 0.876 | 同上 |
| PEMS07 | 0.928 | 同上 |
| PEMS08 | 0.915 | 同上 |
| METR-LA | 1.051 | 速度，周末更高 |
| PEMS-BAY | 1.064 | 速度，周末更高 |

这只能说明星期相位与数据形态一致，**不能替代发布方原始时间戳**。若拿到原始时间戳，应替换 `data/<NAME>/calendar.npz` 并重跑 `prepare-data`（签名会变）。

## 3. 数据集扩展

### X4 PEMS03/04/07 配置

计划把 PEMS03/04/07/08 列为默认数据集，但初始仓库只有 PEMS08 的实验配置。本次补齐：

- `configs/experiments/pems03.yaml`、`pems04.yaml`、`pems07.yaml`，与 PEMS08 同构。
- PEMS03 的图 CSV 使用真实传感器 ID，因此配置 `node_ids_file: PEMS03/PEMS03.txt`。已实测：用 `PEMS03.txt` 映射后重建的邻接与 `adj_PEMS03.pkl` 完全一致（1093 个非零 = 2×546 条无向边 + 1 个自环）。
- PEMS04/07/08 的图 CSV 端点已是列索引，不需要 `node_ids_file`。
- 四套 PEMS 配置都打开 `day_of_week: true` 与 `dow_embedding_dim: 24`（上游 STAEformer 默认值），前提是有 `calendar.npz`。

**验证结果**

| 数据集 | shape | 图边数 | train/val/test 窗口 | graph_hash 前缀 |
|---|---|---|---|---|
| PEMS03 | (26208, 358) | 546 | 15701 / 5231 / 5231 | `917b4c31fde1` |
| PEMS04 | (16992, 307) | 340 | 10172 / 3387 / 3388 | `e236abc1e69b` |
| PEMS07 | (28224, 883) | 866 | 16911 / 5634 / 5634 | `ee67865b7c4d` |
| PEMS08 | (17856, 170) | 274 | 10690 / 3560 / 3561 | `443c287a2748` |

### X5 METR-LA / PEMS-BAY

**为什么是计划外**

计划默认数据集只有 PEMS 四件套，METR-LA / PEMS-BAY 没有出现在合同里。本机原先只有这两个数据集的时间序列，没有路网。

**做了什么**

- 时间序列沿用本地 `Flow-RL 2/STAEformer/data/{METRLA,PEMSBAY}/data.npz`，只读复制到 `data/`。
- 邻接从 Zenodo 记录 `10.5281/zenodo.5724362` 下载（CC-BY-4.0，引用 DCRNN ICLR'18）：
  - METR-LA `adj_mx`，sha256 `a35687c6e15aa228…`
  - PEMS-BAY `adj_mx_bay`，sha256 `116275f5704d0d49…`
  - 与 DCRNN GitHub `data/sensor_graph/` 的文件字节一致。
- 原始 pickle 用只允许 numpy 类型的受限 `Unpickler` 读取，禁止任意可执行对象。
- 节点顺序已核验：Zenodo 同名 CSV 的列顺序与邻接 `ids` 完全一致；前 571 / 728 行数值与本地序列最大差约 3e-6。
- 上游邻接本身**非对称**（`gen_adj_mx.py` 的对称化那行被注释掉），因此按 `adj > 0` → 去对角 → 双向并集 → 只留 `i < j` 转成派生二值边表：

| 数据集 | 节点 | 原始非零 | 无向边 | 孤立节点 |
|---|---|---|---|---|
| METR-LA | 207 | 1722 | 1313 | 1 |
| PEMS-BAY | 325 | 2694 | 2079 | 6 |

- 单位是速度 mph，不是流量；缺失口径按 DCRNN 惯例“零值视为缺失”，即 `missing_policy: official_compat_nonzero`、`zeros_are_valid: false`。METR-LA 约 8.11% 为零，PEMS-BAY 为 0%。

**当前状态**

已完成 `prepare-data` 与两样本 CPU 前向：

| 数据集 | shape | 图边数 | train/val/test 窗口 | graph_hash 前缀 |
|---|---|---|---|---|
| METRLA | (34272, 207) | 1313 | 20540 / 6843 / 6844 | `378d072270c1` |
| PEMSBAY | (52116, 325) | 2079 | 31246 / 10412 / 10413 | `d7e851440748` |

**没有训练、没有评测指标、没有 GPU 结果**，不能作为任何论文表格的数据。

## 4. 图与单位协议

### X6 `derived_binary_undirected`

设计 §5 只允许“发布方道路连接关系”的二值拓扑，另允许 `identity` 诊断。METR-LA / PEMS-BAY 的发布矩阵非对称，若直接标成 `published_binary_undirected` 会错误宣称“发布方原图”。

本次新增拓扑标签：

- 计算路径与 `published_binary_undirected` 完全相同：无向、去重、自环、`A = D^-1 (W + I)`、稀疏 COO。
- 拓扑字符串进入 `graph_hash`，所以两种标签的 hash 必然不同，不能互相冒充。
- 配置必须显式写 `graph.topology: derived_binary_undirected` 才启用；未知拓扑仍然 fail-fast。
- 数据来源与转换规则写在 `reviews/DATA_PROVENANCE.md` 与 `data/*/SOURCE.json`。

单测断言：同边集下两者扩散矩阵一致、`graph_hash` 不同；未知拓扑报错。

### X7 `data.unit` 与 `metric_units(unit)`

**问题**

原实现把 `METRIC_UNITS` 写成全局常量，MAE/RMSE 固定标注 “flow per sampling interval”。直接跑速度数据集会把 mph 写成 flow，属于错误的计量口径。

**做了什么**

- `DataConfig.unit` 默认值保持旧的 flow 文案，因此旧配置输出不变。
- `metric_units(unit)` 生成标签：`mae/rmse/regret` 跟随 `data.unit`；`mape=percent`；`nvr/euvr=fraction of valid elements`；`retention=dimensionless`。
- `METRIC_UNITS` 保留为 `metric_units()` 的默认构造，兼容既有导入。
- `evaluation/runner.py` 与 `cli.py` 改为使用 `metric_units(cfg.data.unit)`；`experiments.csv` 的 `units` 列同样跟随。
- manifest 记录 `unit`，speed 数据集写入 `"speed (mph)"`。
- `unit` 已加入 prepare signature：改单位会重建 prepared 缓存，避免 manifest 文案与产物不一致。

**验证**

- 单测覆盖 `data.unit` 读取、空字符串拒绝、MAE/RMSE 跟随单位、MAPE 保持 percent。
- METRLA/PEMSBAY 的 prepared manifest 为 `speed (mph)`；`metric_units("speed (mph)")` 返回 mph。当前**没有真实 eval 产物**，因为尚未训练。

## 5. 来源与可复现

### X8 `SOURCE.json` 与 `DATA_PROVENANCE.md`

- 每个 `data/<NAME>/SOURCE.json` 记录：上游文件、下载 URL 或本地只读路径、sha256、字节数、许可、节点数、边数、日历规则、转换规则。
- `reviews/DATA_PROVENANCE.md` 是这些信息的汇总，并明确写出：
  - PEMS 四件套来自 Caltrans PeMS 的镜像，镜像未提供再许可声明；
  - METR-LA / PEMS-BAY 邻接来自 Zenodo CC-BY-4.0；
  - 哪些验证做了、哪些没做。
- `data/` 与 `outputs/` 被 gitignore，来源文件随仓库保留。

## 6. 测试

新增两个测试文件：

- `tests/unit/test_calendar.py`（7 项）：`[M, 2]` 表、tod/dow 范围、文件读写、长度不匹配、配置冲突、prepare 集成、签名随日历变化。
- `tests/unit/test_units_topology.py`（6 项）：派生拓扑被接受、同扩散不同 hash、未知拓扑拒绝、`data.unit`、空 unit 拒绝、指标单位跟随。

当前总结果：

```text
pytest -q  -> 65 passed, 1 skipped   (跳过项：无 CUDA，GPU smoke 不声称)
ruff check src tests scripts -> All checks passed
```

## 7. 顺带修复：`.gitignore` 误伤源码包

**发现**

`.gitignore` 里的 `data/` 没有锚定根目录，匹配了任意层级的 `data` 目录，导致整个 `src/sbona/data/` 包（manifest、dataset、graph、scaler、windows）从未进入版本控制。`git show 7388459 --name-only` 可以确认该包缺失。

**修复**

- 把数据/输出目录改为根锚定：`/data/`、`/datasets/`、`/outputs/`、`/checkpoints/`、`/logs/` 等。
- `src/sbona/data/` 现在可以正常跟踪；`data/` 与 `outputs/` 仍然被忽略。

**影响**

早期提交如果被单独 clone，会缺少数据模块；本提交补上。此后 `data/` 语义只指仓库根的数据目录。

## 8. 已知偏离 / 需要审查确认

1. **配置 `schema_version` 未从 1 升到 2。**
   新增的 `data.unit`、`data.timestamps_file` 都是可选键且有默认值，旧配置无需修改；但设计 §14 写的是“配置/输出 schema 升级必须增加版本号”。本轮判断“加可选键、不改既有键语义”不构成强制升级，因此保留 1。
   如果审查认为任何加键都必须升版本，应在一次提交内把 `Config.schema_version` 提到 2，并同步所有 `configs/experiments/*.yaml`、测试 fixture 与文档；不要只改代码默认值。
2. **manifest 的 `license` / `source` 仍是通用文案。**
   目前 `license` 写的是 “see dataset manifest / original publisher”，`source` 是数据文件路径；具体许可与上游 URL 在 `SOURCE.json` 和 `DATA_PROVENANCE.md` 里。若要让 manifest 自包含，需要新增 `data.license` / `data.source_url` 配置并进入签名。
3. **日历是“有记录的起始日期 + 相位偏移”，不是发布方原始时间戳。**
   已做周末/工作日一致性检查，但这只是交叉检查。拿到原始时间戳后应替换并重跑。
4. **`derived_binary_undirected` 是派生拓扑。**
   它不声称等价于发布方原始加权/有向图；论文表格必须同时报告拓扑标签与转换规则。

## 9. 未执行 / 不声称

- GPU smoke、A100 显存/吞吐/bf16、DDP 行为：**未执行**。
- 任何真实数据训练、Router 训练、test MAE/RMSE/MAPE：**未执行**。
- METR-LA / PEMS-BAY：只完成数据准备与 CPU 前向，**没有训练或评测结果**。
- STID / Graph WaveNet / AGCRN：仍 pending，配置 fail-fast，不冒充支持。
- `data/` 下的原始数据：不入库（体积与许可原因），只入库来源记录与 hash。
- 依赖 lock 文件与 CI：仍未提供。

## 10. 复现命令

```bash
# 1) 生成日历（通道模式）
python scripts/build_calendar.py --from-data data/PEMS08/data.npz \
  --start-weekday 4 --note "PEMS08 embedded tod/dow, Friday start" \
  --out data/PEMS08/calendar.npz

# 2) 准备六个数据集
for d in pems03 pems04 pems07 pems08 metrla pemsbay; do
  PYTHONPATH=src python -m sbona.cli prepare-data \
    --config "configs/experiments/$d.yaml"
done

# 3) 质量门
python -m pytest -q
python -m ruff check src tests scripts
```

## 11. 维护规则

- 以后任何“计划没写但做了”的改动，都先在本文件加一行，再在 `reviews/STATUS.md` / `reviews/HANDOFF.md` 留链接。
- 如果改动触及合同（信息边界、损失、尺度、图定义、checkpoint 选择、oracle 粒度），不能只写在这里，必须走设计变更并在 `docs/` 留记录。
- 本文件不构成“已完成”的证明；完成标准仍以 `docs/README_REVIEW.md` 为准。
