# 数据来源与本地布局

日期：2026-10-28。本文件记录 `data/` 下每个文件的可追溯来源、hash、转换规则与已做/未做的验证。`data/` 本身被 `.gitignore` 排除，所以来源信息必须放在这里。

## 1. 目录布局

```
data/
├── PEMS03/{data.npz, PEMS03.csv, PEMS03.txt, calendar.npz, SOURCE.json}
├── PEMS04/{data.npz, PEMS04.csv, calendar.npz, SOURCE.json}
├── PEMS07/{data.npz, PEMS07.csv, calendar.npz, SOURCE.json}
├── PEMS08/{data.npz, PEMS08.csv, calendar.npz, SOURCE.json}
├── METRLA/{source_adj_mx.pkl, adj_edges.npz, calendar.npz, SOURCE.json}
└── PEMSBAY/{source_adj_mx.pkl, adj_edges.npz, calendar.npz, SOURCE.json}
```

`data_root` 的解析优先级仍是 CLI `--data-root` > `SBONA_DATA_ROOT` > `paths.data_root` > 默认 `data`。

## 1.1 上游原始文件与机器复现链

`data/` 里的内容分两类：**上游原样文件**和**本仓转换/生成的派生文件**。只有前者需要从网络获取，后者必须由脚本重建。`provenance/raw_sources.json` 只登记上游文件，并逐个校验 sha256 与字节数。

```bash
# 1) 下载并校验上游原始文件（约 156 MB）
python scripts/fetch_data.py --all --dest raw

# 2) 归一化成 prepare-data 期望的布局
python scripts/import_raw.py --all --raw-dir raw --data-root data

# 3) 构建 prepared 产物
for d in pems03 pems04 pems07 pems08 metrla pemsbay; do
  PYTHONPATH=src python -m sbona.cli prepare-data --config configs/experiments/$d.yaml
done
```

| 数据集 | 序列（上游文件） | 图（上游文件） | node ids |
|---|---|---|---|
| PEMS03 | STAEformer `data/PEMS03/data.npz` | ASTGNN `PEMS03.csv` | ASTGNN `PEMS03.txt` |
| PEMS04 | STAEformer `data/PEMS04/data.npz` | ASTGNN `PEMS04.csv` | 列索引 |
| PEMS07 | STAEformer `data/PEMS07/data.npz` | ASTGNN `PEMS07.csv` | 列索引 |
| PEMS08 | STAEformer `data/PEMS08/data.npz` | ASTGNN `PEMS08.csv` | 列索引 |
| METR-LA | STAEformer `data/METRLA/data.npz` | Zenodo 5724362 `adj_mx_METR-LA.pkl` | 列索引 |
| PEMS-BAY | STAEformer `data/PEMSBAY/data.npz` | Zenodo 5724362 `adj_mx_PEMS-BAY.pkl` | 列索引 |

上游文件与本地审计副本的 sha256 **逐字节相同**（不是“数值接近”）：六个 `data.npz`、四个 PEMS `*.csv`、`PEMS03.txt`、两个 Zenodo `adj_mx*.pkl`。

### 可执行的校验

`tests/data/golden_prepared.json` 记录了审计副本经 `prepare-data` 后的
`data_hash / valid_hash / graph_hash / split_hash / scaler_hash / node_order_hash / 窗口数 / calendar_hash`。
`tests/integration/test_raw_import_golden.py` 从上游文件重新导入并 `prepare`，逐项断言与 golden 相等：

```bash
SBONA_RAW_DIR=raw python -m pytest tests/integration/test_raw_import_golden.py -q
```

这不只是“能跑”：只要上游文件被替换、节点顺序被改、图被改、切分被改或标定统计被改，对应 hash 立刻不等。

`import_raw.py` 还会在 `data/<NAME>/RAW_SOURCE.json` 记录本次使用的上游文件名、sha256、转换规则和输出文件 hash。`SOURCE.json` 保留为审计摘要，两者不互相替代。

### 派生文件（不可下载，必须由脚本生成）

| 文件 | 生成方式 |
|---|---|
| `calendar.npz` | `scripts/import_raw.py`（优先数据自带 tod/dow，其次嵌入时间戳，最后文档起始日期均匀采样）；等价于 `scripts/build_calendar.py` |
| `adj_edges.npz` | `scripts/import_raw.py` 从 `adj_mx*.pkl` 二值化：`adj>0` → 去对角 → 双向并集 → 保留 `i<j` |
| `SOURCE.json` | 人工审计摘要，记录起始日期、节点数、边数与上游 hash |

## 2. PEMS 四件套

| 数据集 | 序列 shape | 节点 | 时间步 | 起始（已核验） | 图边数（去自环/去重） |
|---|---|---|---|---|---|
| PEMS03 | (26208, 358, 3) | 358 | 91 天 | 2018-09-01（周六） | 546 |
| PEMS04 | (16992, 307, 3) | 307 | 59 天 | 2018-01-01（周一） | 340 |
| PEMS07 | (28224, 883, 3) | 883 | 98 天 | 2017-05-01（周一） | 866 |
| PEMS08 | (17856, 170, 3) | 170 | 62 天 | 2016-07-01（周五） | 274 |

- 序列文件的原发布位置是 STAEformer 仓库 `data/<NAME>/data.npz`（唯一 key `data`，第 0 通道为 flow；PEMS 另含 tod/dow 通道）。本地审计副本与公开文件 sha256 相同。
- 图文件的原发布位置是 ASTGNN 仓库 `data/<NAME>/<NAME>.csv`，本地副本与公开文件 sha256 相同。PEMS03 的 CSV 使用真实传感器 ID，需配合 `PEMS03.txt`（358 个 ID）映射到列序；已实测该映射重建的邻接与 `adj_PEMS03.pkl` 完全一致（1093 非零 = 2×546 + 1 自环）。
- 上游版权：PeMS 数据由 Caltrans 发布，经 STAEformer / ASTGNN 镜像获得。镜像仓库未提供再许可声明，本仓只做研究用途引用，不声称拥有数据权利。

### 文件 sha256（前 16 位）

| 数据集 | data.npz | 图 CSV | node ids | calendar.npz |
|---|---|---|---|---|
| PEMS03 | `9a47e9044ead4159` | `b978bc1fee8172ac` | `553bd79618c1e7eb` | `0cccf3d62e26b046` |
| PEMS04 | `6e46317aa8c87ac3` | `68ffaa80447eb344` | — | `2ad07b99267f229a` |
| PEMS07 | `f0523989cc048d98` | `f7d127ebf976f1e2` | — | `03e833f6c413573f` |
| PEMS08 | `709383c68dd591f5` | `859d68e7583ed4e6` | — | `dc6d11bec2b4406c` |

完整 hash 与字节数见各目录的 `SOURCE.json`，`prepare-data` 会把它们写入 dataset manifest 的 `files`。

## 3. 日历文件

`calendar.npz` 用 `scripts/build_calendar.py` 生成，再经 `data.timestamps_file` 进入 `build_calendar`。脚本有两种模式：

1. **通道模式**（当前实际使用）：读取 `data.npz` 自带的 tod/dow 通道，`tod` 原样保留（包括 PEMS-BAY 的 1 小时缺口），dow 只施加常量星期相位偏移。
2. **均匀模式**：从记录在案的起始时间生成 `datetime64[s]` 轴，适合没有时间通道的数据。

通道模式示例：

```bash
python scripts/build_calendar.py --from-data data/PEMS08/data.npz \
  --start-weekday 4 --note "PEMS08 embedded tod/dow, Friday start" \
  --out data/PEMS08/calendar.npz
```

| 数据集 | 起始 | 起始星期 | dow 通道含义 | 周末/工作日平均原始值 |
|---|---|---|---|---|
| PEMS03 | 2018-09-01 | 周六 | `day_index % 7` 相位 | 0.878（流量） |
| PEMS04 | 2018-01-01 | 周一 | 相位 | 0.876（流量） |
| PEMS07 | 2017-05-01 | 周一 | 相位 | 0.928（流量） |
| PEMS08 | 2016-07-01 | 周五 | 相位 | 0.915（流量） |
| METR-LA | 2012-03-01 | 周四 | 真实星期 | 1.051（速度） |
| PEMS-BAY | 2017-01-01 | 周日 | 真实星期 | 1.064（速度） |

PEMS 四件套的 dow 通道满足 `dow == day_index // 288 % 7`，只是相位；METR-LA / PEMS-BAY 的通道本身已是真实星期，偏移为 0。周末/工作日比值与数据形态一致（流量周末更低、速度周末更高），但这是**一致性检查**，不是独立时间戳证明；若后续拿到发布方原始时间戳，应替换并重跑 `prepare-data`（签名会变，缓存自动失效）。

## 4. METR-LA / PEMS-BAY（已接入配置，未训练）

这是本机原先缺失的部分：只有 `Flow-RL 2/STAEformer/data/{METRLA,PEMSBAY}/data.npz` 时间序列，没有任何路网。

### 新增文件

| 数据集 | 文件 | 来源 | sha256（前 16 位） |
|---|---|---|---|
| METR-LA | `source_adj_mx.pkl` | Zenodo 10.5281/zenodo.5724362 → `adj_mx_METR-LA.pkl` | `a35687c6e15aa228` |
| PEMS-BAY | `source_adj_mx.pkl` | Zenodo 10.5281/zenodo.5724362 → `adj_mx_PEMS-BAY.pkl` | `116275f5704d0d49` |

- Zenodo 记录许可为 **CC-BY-4.0**，引用 DCRNN（Li et al., ICLR 2018）。
- 这两个文件与 DCRNN GitHub `data/sensor_graph/adj_mx*.pkl` **字节完全一致**（下载后比对 sha256），Zenodo 版本有 DOI 与许可，故以 Zenodo 为登记来源。
- 原始 pickle 是 Python 2 协议，读取时使用只允许 numpy 类型的受限 `Unpickler`（禁止任意可执行对象）。

### 节点顺序已核验

对两个数据集分别做了部分 CSV 下载（Zenodo 同名 CSV），逐列比对本地 STAEformer 时间序列：

| 数据集 | CSV 列数 | 本地节点数 | 列顺序 == 邻接 `ids` | 前 N 行数值 |
|---|---|---|---|---|
| METR-LA | 207 | 207 | 是 | 571 行，最大绝对差 3.4e-06 |
| PEMS-BAY | 325 | 325 | 是 | 728 行，最大绝对差 3.1e-06 |

起始时间也由 CSV 首行直接核验：METR-LA `2012-03-01 00:00:00`，PEMS-BAY `2017-01-01 00:00:00`。

### 邻接转换规则

`SOURCE.json` / `adj_edges.npz` 记录的转换：`adj > 0` → 去掉对角 → 双向取并集 → 只保留 `i < j`。

| 数据集 | 节点 | 非零（含对角） | 无向边 | 孤立节点 |
|---|---|---|---|---|
| METR-LA | 207 | 1722 | 1313 | 1 |
| PEMS-BAY | 325 | 2694 | 2079 | 6 |

注意：上游 `gen_adj_mx.py` 的对称化一行被注释掉，所以发布的矩阵本身是非对称的；本仓按图合同做双向并集，属于**派生二值拓扑**，不是发布方原始二值图。配置使用独立的 `graph.topology: derived_binary_undirected` 标签写入 `graph_hash`，不能冒充 `published_binary_undirected`。

### 单位与缺失口径

- 第 0 通道是**速度 mph**，不是流量。配置写 `data.unit: "speed (mph)"`，评测产物 `metric_units` 的 MAE/RMSE/regret 跟随该单位。
- 零值按 DCRNN 惯例视为缺失：`missing_policy: official_compat_nonzero`、`zeros_are_valid: false`。METR-LA 约 8.11% 为零，PEMS-BAY 为 0%。

### 当前状态

已生成 `configs/experiments/metrla.yaml`、`pemsbay.yaml`，并完成 `prepare-data` 与两样本 CPU 前向：

| 数据集 | shape | 图边数 | train/val/test 窗口 | graph_hash 前缀 | 单位 |
|---|---|---|---|---|---|
| METRLA | (34272, 207) | 1313 | 20540 / 6843 / 6844 | `378d072270c1` | speed (mph) |
| PEMSBAY | (52116, 325) | 2079 | 31246 / 10412 / 10413 | `d7e851440748` | speed (mph) |

**没有训练、没有评测指标、没有 GPU 结果。**

## 5. 仍未完成（不要当成已支持）

1. METR-LA / PEMS-BAY 只做了数据准备与 CPU 前向，没有训练或论文指标。
2. PEMS03/04/07/08 只做了 `prepare-data` 与 PEMS08 CPU 前向 smoke，没有训练。
3. 日历是数据自带 tod/dow + 有记录的起始星期，不是发布方原始时间戳。上游 `index.npz` 只有窗口索引，没有 datetime。
4. 真实 GPU smoke、依赖 lock、CI 仍未提供。
5. `fetch_data.py` 的上游 URL 指向会移动的默认分支，靠 sha256 拦截漂移；没有为每个文件固定 git commit。

## 6. 复现命令

从零开始（干净机器，只需要网络）：

```bash
# 下载 + 校验上游原始文件
python scripts/fetch_data.py --all --dest raw

# 归一化布局（写入 data/<NAME>/，并记录 RAW_SOURCE.json）
python scripts/import_raw.py --all --raw-dir raw --data-root data

# 准备六个数据集
for d in pems03 pems04 pems07 pems08 metrla pemsbay; do
  PYTHONPATH=src python -m sbona.cli prepare-data --config configs/experiments/$d.yaml
done

# 与审计副本的 hash 对账
SBONA_RAW_DIR=raw python -m pytest tests/integration/test_raw_import_golden.py -q
```

只重建日历（`import_raw.py` 已包含该步骤，单独使用时）：

```bash
python scripts/build_calendar.py --from-data data/<NAME>/data.npz \
  --start-weekday <0..6> --note "<source note>" --out data/<NAME>/calendar.npz
```
