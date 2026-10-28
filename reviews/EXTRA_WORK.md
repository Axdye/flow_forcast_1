# 计划外工作记录

日期：2026-10-28；2026-10-29 追加 §12（九个官方 backbone 与 MCL-WTA baseline）。

本文件记录**已经实现、但不在原计划明确要求范围内**的工作，避免这些改动在交接时被误认为设计合同的一部分，也避免后来者重复踩坑。

“原计划”指：

- `docs/README_CODE_DESIGN.md`（最高优先级合同）
- `docs/README_EXPERIMENTS.md`
- `tasks/TASK_A.md` … `tasks/TASK_D.md`
- `configs/design_defaults.yaml`

本文件不改变任何合同，也不替代 `reviews/STATUS.md` 与 `reviews/HANDOFF.md`。凡是与合同有张力的事项，单列在 §8。

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
| X9 | 额外测试 `test_calendar.py`、`test_units_topology.py` | 未规定 | `tests/unit/` | 当前 158 passed, 8 skipped |
| X10 | `scripts/fetch_data.py` + `provenance/raw_sources.json` | 只要求“记录来源”，未要求可下载/可校验 | `scripts/fetch_data.py`、`provenance/raw_sources.json` | 全量下载六个数据集并逐字节校验 |
| X11 | `scripts/import_raw.py`（上游格式 → prepare 布局） | 未规定 | `scripts/import_raw.py` | 六数据集导入并与审计 hash 对齐 |
| X12 | `tests/data/golden_prepared.json` + `test_raw_import_golden.py` | 未规定 | `tests/data/`、`tests/integration/` | 7 passed（真实上游文件） |
| X13 | 单 seed 化 + split 列 + 身份哈希选 run | 原计划要求 ≥3 seeds、mean±std | `src/sbona/cli.py`、`training/trainer.py`、`evaluation/artifacts.py`、`tests/unit/test_run_selection.py` | 当时 89 passed；4 类“拿错 checkpoint / 算错结果”缺陷已复现并修复 |
| X14 | `backbone.params`：逐架构超参白名单 + 官方 PEMS08 默认值 | 合同只固定 STAEformer 的扁平字段 | `src/sbona/config.py`、`tests/unit/test_config.py` | 21 项 config 测试；未知键/跨架构键/错类型均 fail-fast |
| X15 | STAEformer `input_dim` 1→3，并修正 weekday embedding 索引 | 旧实现强制 `input_dim=1` | `src/sbona/config.py`、`backbones/staeformer.py`、全部 `configs/**/*.yaml` | 单测覆盖 1/3 与拒绝 2；九配置均可构造 |
| X16 | 八个官方 backbone 接入（Graph WaveNet、AGCRN、STID、D2STGNN、PDFormer、HimNet、ST-SSDL、HyperD） | 只要求 STAEformer 主线 | `src/sbona/backbones/*.py`、`backbones/common.py` | `tests/unit/test_backbones.py` 45 项：九模型 × 三 mode + 梯度 + 图依赖 |
| X17 | 统一的日历通道 / 稠密图转换与缓存 / 未来日历外推 | 合同未规定传参约定 | `src/sbona/backbones/common.py` | 由 X16 间接覆盖；每模型显式声明 `FEATURE_SPECS` |
| X18 | MCL-WTA baseline（独立头 + 硬 WTA + score head）+ `wta_loss` / `wta_score_loss` + `CandidateOutput.candidate_logits` | 合同无 WTA baseline | `baselines/mcl_wta.py`、`objectives/losses.py`、`contracts.py`、`training/trainer.py`、`cli.py` | `test_baselines.py` 8 项 + objectives 6 项 + CLI 端到端 1 项 |
| X19 | 逐 backbone PEMS08 实验配置 + `pems08_mcl_wta.yaml` | 只要求 STAEformer 配置 | `configs/experiments/pems08_*.yaml` | `test_every_backbone_has_a_pems08_experiment_config`；N=170 CPU 实构建全部通过 |
| X20 | `source_manifest.json` schema 2：九仓库 commit/许可/文件哈希 | 只有 STAEformer 一项 | `third_party/source_manifest.json`、`LOCK.md`、`BACKBONE_CONTRACTS.md` | 9 模型 28 文件 sha256；4 个仓库无 LICENSE 记为 pending |
| X21 | `scripts/smoke_backbones.py` 九模型 × 三 mode 前向诊断 | 未规定 | `scripts/smoke_backbones.py` | 27 行 `ok`；不是测试，不替代 pytest |
| F1 | `.gitignore` 误伤 `src/sbona/data/` | —（缺陷） | `.gitignore` | 修复后源码包进入版本控制 |
| F2 | `.gitignore` 补 `/raw/` | —（配套） | `.gitignore` | 原始下载目录不入库 |
| F3 | HyperD 周期表零初始化导致 `linear1` 无梯度 | —（缺陷） | `backbones/hyperd.py`、`tests/unit/test_backbones.py` | `set_periodic_prior` 注入后梯度恢复，单测断言 |

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

### X10 `scripts/fetch_data.py` 与 `provenance/raw_sources.json`

**做了什么**

- 仓库不 vendor 数据；`provenance/raw_sources.json` 登记每个**上游原始文件**的下载 URL（可多个回退）、字节数、sha256、来源仓库与许可。
- `fetch_data.py` 按顺序尝试 URL，下载到 `.part`，校验大小与 sha256 后才原子改名；失败删除临时文件并返回非零。
- 支持 `archive:<tarball>#<member>` 形式：先下载一次 tarball 并缓存到 `<dest>/.cache/`，再从归档中取单个成员。用于 raw.githubusercontent 不稳定时的回退。
- `--verify-only` 只校验已下载文件，不联网；`--force` 强制重下。

**为什么算计划外**

设计合同只要求“记录来源”，没有要求“可下载、可校验”。机器上只会提供上游原始文件，所以把来源记录升级成可执行校验链是必要的，但不是原计划内容。

**验证**

六个数据集共 13 个文件全部通过真实下载与 sha256 校验。六个 `data.npz`、四个 PEMS `*.csv`、`PEMS03.txt`、两个 Zenodo `adj_mx*.pkl` 与本地审计副本**逐字节相同**。

**已知限制**

URL 指向仓库默认分支，上游若强行改历史，只能靠 sha256 拦截（会失败而不是静默接受）；没有为每个文件固定 commit。

### X11 `scripts/import_raw.py`

**做了什么**

把上游原始格式转换成 `prepare-data` 期望的布局，转换规则全部显式：

| 输入 | 处理 |
|---|---|
| `.npz`（key `data`/`values`） | 已是规范形态则**逐字节复制**；否则重写 `data.npz` |
| `.npy` | 写 `data.npz` |
| `.h5`（可选 `h5py`） | 读数值与 `axis0` 时间戳 |
| `.csv`（带时间列） | 拆出数值与 `datetime64[s]` |
| `.pkl` 邻接（`(ids, id_to_ind, adj)` / `(ids, adj)` / ndarray / scipy sparse） | 受限 `Unpickler`，仅允许 numpy/scipy；`adj>0` → 去对角 → 双向并集 → `i<j` |
| `.npz`/`.npy`/`.csv`/`.txt` 边表 | 映射节点 id、去自环、去重 |

关键点：

- **受限反序列化**：`pickle` 只允许 `numpy`/`scipy` 顶层模块，其他类直接拒绝，防止数据文件执行任意代码。
- **日历通道检测**：不能只看通道数。ASTGNN 的 PEMS04/08 `data.npz` 也是 3 通道，但通道 1/2 是 occupancy/speed。检测要求 dow 为 0..6 整数、逐日 +1、tod 每日回绕；PEMS-BAY 的 1 小时缺口仍能通过，因为判据是“每日回绕次数”而不是“严格周期”。
- **不覆盖已审计文件**：目标已存在时默认失败，必须显式 `--force`。
- 每次导入写 `data/<NAME>/RAW_SOURCE.json`：上游文件名、sha256、转换动作、边数、日历模式与假设、输出文件 hash。

**验证**

六数据集导入后 `prepare-data` 的 `data_hash / valid_hash / graph_hash / split_hash / scaler_hash / node_order_hash / 窗口数 / calendar_hash` 全部与审计副本一致（见 X12）。

### X12 Golden hash 回归测试

**做了什么**

- `tests/data/golden_prepared.json` 记录六数据集审计副本的 prepared hash 与窗口数。
- `tests/integration/test_raw_import_golden.py` 在临时 `data_root` 中执行 `import_raw.py` → `prepare_dataset` → 逐项断言。
- 默认跳过；设置 `SBONA_RAW_DIR` 后启用，约 30 s。

**为什么算计划外**

合同要求“数据/接口正确”，但没有规定机器可验证的判定方式。只写“已核验”无法在干净机器上复现；golden hash 把“核验过”变成一条可执行断言。

**验证**

```
SBONA_RAW_DIR=raw python -m pytest tests/integration/test_raw_import_golden.py -q
7 passed
```

这七个断言覆盖：序列数值、缺失/零值掩码、图上边集、窗口切分、train-only 标定、节点顺序、日历表。任一上游文件被替换或转换逻辑被改动都会失败。

### X13 单 seed 化、split 列与身份哈希选 run

**做了什么**

1. **seed 只有一个来源**：`configs/experiments/*.yaml` 的 `seed`。删除了 `train-*` / `evaluate` 的 `--seed` 覆盖，避免训练完再用 CLI 改标 seed。
2. **`experiments.csv` 增加 `split` 列**，写入处共 5 个（Base/single/C/D 训练期为 `val`，`evaluate` 用实际 `--split`）。`aggregate --split {train,val,test}`（默认 `test`）只建该 split 的表，val/train 不再混进主表。
3. **`aggregate` 不再按 seed 求均值**：删除 `aggregate_seeds`。每组 `(dataset, backbone, method, metric)` 只允许一行；同组重复行、CSV 含多个 seed、缺 `split` 列都直接报错，而不是被当成独立重复。
4. **自动选 run 改为身份哈希**：
   - `backbone_identity_hash` = seed + protocol + backbone 架构（**不含 method**，否则 `pems08_staeformer.yaml` 训出的 Base 无法被 `pems08_single.yaml` 复用）；
   - `adapter_identity_hash` = seed + protocol + method_id + adapter 参数 + K；
   - Stage A 只比 backbone 身份，B/C/D/P 还比 adapter 身份；
   - 同一身份命中多个 run 时报错并列出 run 名，不再按 mtime 取“最新”；
   - `--kind` 非默认（如 `independent_heads`）时，Stage C 的 `method_id` 自动带后缀，避免与主方法共用身份；
   - 显式 `--base-checkpoint` / `--candidate-checkpoint` 也校验身份：Stage B/C 拒绝别的 seed/架构的 Base，Stage D 拒绝别的 seed/method/K 的候选，`evaluate` 读取后重新核对。

**复现过的四个缺陷**（修复前均可用脚本稳定复现）

| 缺陷 | 修复前 | 修复后 |
|---|---|---|
| 请求 seed 11 的 Base | 返回 seed 22（目录更新） | 只返回 seed 11；无匹配则 `FileNotFoundError` |
| 请求 structured 候选 | 返回 independent_heads | 按 adapter 身份区分；同身份多命中报错 |
| 请求 seed 11 的 Router | 返回 seed 22 | 按 seed 区分；无匹配返回 `None` |
| 单 seed 的 val+test 两行 | `seeds=2, mean=1.0, std=0.283` | 只取 `split=test` 一行；重复行报错 |

验证：`tests/unit/test_run_selection.py`（8 项）、`tests/unit/test_artifacts.py`（单 seed / split / 重复行），以及 `scripts/verify_selection.py` 端到端脚本（13 项全部 PASS，可用 `PYTHONPATH=src python scripts/verify_selection.py` 复跑）。全量 `pytest -q`：89 passed, 8 skipped。

**对旧产物的处理（fail closed）**

- 旧 checkpoint 的 `hashes` 没有 `backbone_identity_hash` / `adapter_identity_hash`，`resume` 与自动发现都会拒绝它（而不是退回“差不多就用”）；必须重训或显式传 `--*-checkpoint`（显式路径仍会走 hash 校验，因此同样会被拒）。
- 旧 `outputs/experiments.csv` 没有 `split` 列，继续 append 会把旧行的 split 写空；现在遇到旧表头直接报错，要求先移走旧表或重跑对应评测。
- `aggregate` 遇到旧 CSV 给出“missing columns: ['split']”，`--split` 不会去猜。

**为什么算计划外**

合同写了“必须校验数据/图/split/scaler/K/架构”，但没有规定“怎么在多 run 目录里找唯一那个 checkpoint”，也没有规定 `experiments.csv` 的 split 语义。原实现按目录 mtime 取最新、按 seed 求均值，在多 seed / 多 split 下会静默拿错模型或算出不存在的方差。

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
- `tests/unit/test_units_topology.py`（7 项）：派生拓扑被接受、同扩散不同 hash、未知拓扑拒绝、`data.unit`、空 unit 拒绝、指标单位跟随。

当前总结果：

```text
pytest -q  -> 158 passed, 8 skipped
              (8 = GPU smoke 无 CUDA + 7 个 opt-in 原始数据 golden 测试)
SBONA_RAW_DIR=raw pytest tests/integration/test_raw_import_golden.py -q -> 7 passed
ruff check src tests scripts -> All checks passed
```

测试环境（本轮）：`/opt/miniconda3/envs/pytorch` 的 torch 2.2.2，叠加一个 numpy 1.26.4 overlay（该 env 自带的 numpy 2.3.3 与 torch 2.2.2 的 `torch.from_numpy` ABI 冲突）。命令见 `reviews/STATUS.md`。这只是让 CPU 测试能跑，不构成可复现环境。

## 7. 顺带修复：`.gitignore` 误伤源码包

**发现**

`.gitignore` 里的 `data/` 没有锚定根目录，匹配了任意层级的 `data` 目录，导致整个 `src/sbona/data/` 包（manifest、dataset、graph、scaler、windows）从未进入版本控制。`git show 7388459 --name-only` 可以确认该包缺失。

**修复**

- 把数据/输出目录改为根锚定：`/data/`、`/datasets/`、`/outputs/`、`/checkpoints/`、`/logs/` 等。
- `src/sbona/data/` 现在可以正常跟踪；`data/` 与 `outputs/` 仍然被忽略。

**影响**

早期提交如果被单独 clone，会缺少数据模块；本提交补上。此后 `data/` 语义只指仓库根的数据目录。

## 8. 已知偏离 / 需要审查确认

0. **当前只有单 seed，尚未满足 `docs/README_EXPERIMENTS.md` §12 的统计合同。**
   该节要求每个种子单独展示、主表 mean±sample std（ddof=1）、至少 3 seeds。当前实现按用户决定**先固定一个 seed**（seed 只来自配置，CLI 不提供 `--seed`，因此无法事后改标），`aggregate` 每组只输出一条记录并拒绝多 seed / 重复行，**不计算也不声称 seed 间方差**。这属于“暂缓”，不是“满足”：在拿到 Router 相对 single 的初步收益并确认值得扩展之前，任何表格都不得写 mean±std 或显著性结论。
   恢复多 seed 时的最小改动：配置层引入 seed 列表、`aggregate_experiments` 按 `(dataset, backbone, method, metric, split)` 分组重算 mean/std、并恢复 `--seed` 只作为“选择已有 run”而非改标手段。
1. **配置 `schema_version` 未从 1 升到 2。**
   新增的 `data.unit`、`data.timestamps_file` 都是可选键且有默认值，旧配置无需修改；但设计 §14 写的是“配置/输出 schema 升级必须增加版本号”。本轮判断“加可选键、不改既有键语义”不构成强制升级，因此保留 1。
   如果审查认为任何加键都必须升版本，应在一次提交内把 `Config.schema_version` 提到 2，并同步所有 `configs/experiments/*.yaml`、测试 fixture 与文档；不要只改代码默认值。
2. **manifest 的 `license` / `source` 仍是通用文案。**
   目前 `license` 写的是 “see dataset manifest / original publisher”，`source` 是数据文件路径；具体许可与上游 URL 在 `SOURCE.json` 和 `DATA_PROVENANCE.md` 里。若要让 manifest 自包含，需要新增 `data.license` / `data.source_url` 配置并进入签名。
3. **日历是“有记录的起始日期 + 相位偏移”，不是发布方原始时间戳。**
   已做周末/工作日一致性检查，但这只是交叉检查。拿到原始时间戳后应替换并重跑。
4. **`derived_binary_undirected` 是派生拓扑。**
   它不声称等价于发布方原始加权/有向图；论文表格必须同时报告拓扑标签与转换规则。
5. **STAEformer 的 weekday embedding 索引与官方不同。**
   官方（BasicTS v0.5.8）把 `dow/7` 分数直接 `.long()`，对任何一天都截断到 0，等于永远索引星期 0。本仓保留官方“分数进入输入投影”的做法（`input_dim=3`），但 embedding 用 `round(dow*7)` 取整数索引。这是**修正**，不是扩展；论文必须披露。
6. **PDFormer 的 `sem_mask` 被禁用。** DTW 聚类先验需要完整训练序列，契约不提供；语义头退化为 dense attention，`pattern_keys` 改为可学习 per-node 参数（正弦初始化）。这不是官方配置。
7. **HyperD 的周期表从零开始。** 官方加载 `daily_init.npy` / `weekly_init.npy`（训练集日均/周均剖面），adapter 看不到训练集；提供 `set_periodic_prior(daily, weekly)` 注入。零初始化时 `linear1` 权重无梯度（已列入允许清单），注入后恢复。
8. **MCL-WTA 不是 TimeMCL 官方复现。** 它是“独立头 + 硬 WTA + score head”的移植：丢掉了概率头 / Router / Stage D，score head 在 Stage C 与头一起训练并直接部署。与 TimeMCL 的差异必须在论文中写明。
9. **`input_dim` 默认值改为 3 会改变已有 checkpoint 的身份哈希。**
   `backbone_identity_payload` 含 `asdict(cfg.backbone)`，旧 Base checkpoint 不能直接复用；必须按新配置重训。这是有意的，不是回归。

## 9. 未执行 / 不声称

- GPU smoke、A100 显存/吞吐/bf16、DDP 行为：**未执行**。
- 任何真实数据训练、Router 训练、test MAE/RMSE/MAPE：**未执行**。
- METR-LA / PEMS-BAY：只完成数据准备与 CPU 前向，**没有训练或评测结果**。
- 九个 backbone 只完成 CPU 前向/反向与形状/梯度契约；**没有**加载官方权重做逐元素等价比较，**没有** GPU smoke，**没有**任何真实数据指标。
- D2STGNN / HimNet / STAEformer / HyperD 上游无 LICENSE 文件：可作内部复现，**不能**声称已获再发布许可。
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

## 12. 2026-10-29 追加：九个官方 backbone 与 MCL-WTA baseline

### X16 九个 backbone 的接入口径

每个模型都实现在**本仓自己的** `BackboneWrapper` 上（`encode_and_predict(ModelInput, GraphContext) -> (node_hidden [B,N,D], base_pred_z [B,T,N,1])`），BasicTS 只在核对模型定义与官方超参时作参考，**不是 runtime 依赖**。

| backbone | 上游 commit | node hidden | Partial 白名单 | 是否读道路图 |
|---|---|---|---|---|
| staeformer | `fc49d39b` | 最后空间 attention 后最后历史步 | 最后 N 个空间层 + 输出投影 | 否（学习型 all-to-all） |
| graph_wavenet | `6b162e80` | `end_conv_1` 之后 | `end_conv_1/2` | 是 |
| agcrn | `7fbbf2ae` | encoder 最后一步 | `end_conv` | 否（自适应图） |
| stid | `e8b313bc` | `regression_layer` 之前 | `encoder` + `regression_layer` | 否（无空间路径） |
| d2stgnn | `82c2d38b` | 最后 decouple 的 forecast 输入 | `out_fc_1/2` | 是 |
| pdformer | `f8c8f6ad` | 最后 encoder block 输出 | 最后 block + `end_conv2` | 是 |
| himnet | `2a9aa44d` | 最后 `HimGCN` 输出 | `decoder` + `out_proj` | 否（学习支撑） |
| stssdl | `a98331d3` | decoder 最后状态 | `decoder`/`proj`/`hypernet` | 是 |
| hyperd | `5bd93a82` | `stfe` 的 token 表示 | `stfe.fc` + 两个周期线性层 | 是（经周期表） |

逐模型的补丁、输入通道与方向约定见 `third_party/BACKBONE_CONTRACTS.md` §2；commit/许可/文件哈希见 `third_party/source_manifest.json`。

**为什么算计划外**：合同只把 STAEformer 定为主线，其余架构在旧 `BACKBONE_CONTRACTS.md` 里是“待锁/待接入”。本轮把八个补齐，`PENDING` 清空。

**验证**：`tests/unit/test_backbones.py`（45 项）

- 九模型 × `frozen/partial/full` = 27 组形状/有限性/`requires_grad`/`node_feature_dim` 契约；
- 全模式反向：断言“除上游确有的死参数外，每个可训练参数都拿到非零梯度”，并要求 ≥80% 张量有梯度；
- 图依赖：只对真正消费道路图的 5 个模型断言换图会改变预测（HyperD 需先注入周期先验）；
- 死参数白名单（`UNUSED_PARAMETER_PREFIXES`）是显式的：GWNet `residual_convs.*`/`bn.7.*`/`gconv.7.*`；D2STGNN backcast 分支与 distance function 的 time-slot embedding；HyperD 零先验下的 `linear1`。

另外做了一次 **N=170（PEMS08 规模）CPU 实构建**，九个模型都能在真实配置下前向出 `[1,12,170,1]`；其中 HyperD 参数量最大（13.65M），STID 最小（0.12M）。这不是指标，只是形状与内存可行性检查。`scripts/smoke_backbones.py` 是同一检查的可重复版本（9 模型 × 3 mode，极小图）。

### X14 `backbone.params` 与官方 PEMS08 默认值

- `BACKBONE_PARAM_DEFAULTS` 是每架构的**官方 PEMS08 配置**白名单；`backbone.params` 只允许本架构的键。
- 跨架构键、未知键、错类型（含 `bool` 当 `int`）全部 fail-fast，避免“写了但没生效”。
- 不写 `params` 得到官方默认值；写一个键只覆盖该键。
- `schema_version` 保持 1：纯增量、每键有默认值，旧配置无需改。
- 每个 `configs/experiments/pems08_<backbone>.yaml` 都**显式写全**官方默认值，并由 `test_every_backbone_has_a_pems08_experiment_config` 断言与 `BACKBONE_PARAM_DEFAULTS` 一致，防止以后改动默认值悄悄改掉已记录的实验。

### X15 STAEformer `input_dim` 与 weekday 修正

旧实现强制 `input_dim=1`（只投影 flow），与官方 `nn.Linear(3, ...)` 投影 `[flow, tod, dow]` 冲突。本轮：

- `validate()` 接受 `input_dim ∈ {1, 3}`，拒绝 2；dataclass 默认与全部配置改为 3。
- `use_mixed_proj=True` 仍是硬约束。
- `dow` 以**分数**进入输入投影（官方值），但 weekday embedding 用 `round(dow*7)` 索引：官方对分数 `.long()` 永远得 0。

**影响**：`backbone_identity_payload` 包含 `asdict(cfg.backbone)`，所以旧 Base checkpoint 的身份哈希变了，必须重训。这是刻意的。

### X18 MCL-WTA baseline

`src/sbona/baselines/mcl_wta.py`：

- `K` 个独立残差头（`ResidualHead`），与 `independent_heads` 同构但**不带 Router**；
- 硬 WTA：`wta_loss(error, valid)` 用 `argmin` 选出每个样本的赢家，只用赢家的误差回传（loser 得到零梯度，不是软加权）；
- score head：`ScoreHead(h_global, pred_z)` 输出 `[B,K]` 分数，用 `wta_score_loss(logits, error.detach(), valid)` 做交叉熵，权重 `loss.wta_score`（默认 0.1）；
- `predict_top1` 用 score head 的 `argmax`，不是 Router；
- `set_stage("D")` 直接报错：本 baseline 不训练 Stage-D Router，score head 在 Stage C 与头联合训练并直接部署；
- `supports_separation = False`（WTA 与 separation loss 语义冲突）。

配套改动：

| 位置 | 改动 |
|---|---|
| `contracts.py` | `CandidateOutput.candidate_logits: Tensor \| None`，形状校验 `[B,K]` |
| `objectives/losses.py` | 新增 `wta_loss`、`wta_score_loss`，形状/设备校验与既有 loss 一致 |
| `training/trainer.py` | `_train_epoch` 按 `model.loss_kind` 分派（`weighted` / `wta`），WTA 分支加 `wta_score_loss` |
| `baselines/__init__.py` | `ADAPTER_KINDS` 增加 `mcl_wta`；`build_adapter` 不构造 Router |
| `cli.py` | `--kind mcl_wta`；`K>=2` 守卫；`train-router` 在身份校验**之前**拒绝；`evaluate`/`profile` 跳过 Router checkpoint、`set_stage("C")`、`selector="score_head"`、表格行用 `<method>_score_top1` |

**为什么算计划外**：合同里没有 WTA baseline，也没有 `candidate_logits` 字段。这是用户明确要求的对照方法，因此单列。

**验证**：`tests/unit/test_baselines.py`（8 项，含“只有赢家头拿到梯度”“score head 只由自己的交叉熵训练”“没有 Router 属性”）、`tests/unit/test_objectives.py`（6 项 WTA 数学/掩码/形状）、`tests/integration/test_cli_synthetic.py::test_cli_synthetic_mcl_wta_end_to_end`（训练 → `train-router` 必须失败 → evaluate 产出 `_score_top1` 行）。

### X17 `backbones/common.py`

三块共享胶水，**只**做这三件事：

1. **日历通道**：`FEATURE_SPECS`（`flow / tod / dow_index / dow_fraction / dow_onehot`）+ `stack_features`。每个 adapter 自己声明要哪几路，不共享“默认约定”——传错不会报错，只会学错 embedding。
2. **稠密图转换与缓存**：`GraphContext` 存的是稀疏行归一化算子，上游要稠密 `[N,N]` 邻接/跳数/拉普拉斯特征向量；`GraphCache` 按 `(graph_hash, device)` 缓存，并显式处理方向（`A` vs `A.T`）。
3. **`future_calendar` 与 `TorchBackboneAdapter`**：前者给 HimNet / ST-SSDL 外推**未来 tod/dow**（确定性日历，绝不读 target）；后者做形状/节点对齐检查后委派给 `forward_features`，保证 hidden 与 prediction 来自**一次** forward。

### X19 实验配置

新增 `configs/experiments/pems08_{graph_wavenet,agcrn,stid,d2stgnn,pdformer,himnet,stssdl,hyperd}.yaml` 与 `pems08_mcl_wta.yaml`。每个文件头注释列出该模型相对官方的偏差；`pems08_mcl_wta.yaml` 的 `method_id: sbona`，配合 `--kind mcl_wta` 落成 `sbona_mcl_wta`，不会与 structured candidates 混淆。

### X20 来源登记

`third_party/source_manifest.json` 升到 schema 2：九个仓库的 `repository/commit/license/files[{path,sha256,bytes}]/patches/input_features/validation`。`validation.native_equivalence` 与 `gpu_smoke` 全部为 `false`，这是事实而不是遗漏。

- 5 个仓库有 MIT / Apache-2.0 LICENSE（Graph WaveNet、AGCRN、STID、PDFormer、ST-SSDL）。
- 4 个仓库**没有** LICENSE / COPYING（D2STGNN、HimNet、STAEformer、HyperD）：本仓不 vendor 其源码，但仍属衍生重写，`pending` 状态不解除。
- BasicTS pin 到 `63fa7192…`（v0.5.8）并注明“仅参考、非 runtime 依赖”：更晚的版本不含这些模型。

### 本轮的“不能声称”

- 不能声称与官方实现**数值等价**：没有官方 checkpoint，因此没有逐元素比较。
- 不能声称任一 backbone 在 PEMS08 上有效：没有任何真实数据训练或指标。
- 不能声称 GPU / bf16 可用：只跑了 CPU。
- 不能声称 4 个无 LICENSE 仓库可再发布。
