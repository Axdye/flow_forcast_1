# 实现与审查状态

日期：2026-10-29（上一版 2026-10-28）。

当前等级：**D0（设计就绪）**。v1 代码已实现并通过 CPU 测试，但 GPU smoke、真实数据训练和科学有效性验证尚未执行，因此不能宣称 D1–D3。计划外实现与已知偏离见 `reviews/EXTRA_WORK.md`（§12 为本轮追加）。

| 项目 | 状态 | 证据 |
|---|---|---|
| D0 研究/开发合同 | 已编写；交付时做链接与结构自检 | docs/ 与 tasks/ |
| A 共享合同与数据实现 | v1 已实现，CPU 测试覆盖 | `src/sbona/contracts.py`、`config.py`、`data/`；`tests/unit/test_config.py`、`test_scaler_and_graph.py`、`test_window_scheme.py` |
| B 模型实现 | v1 已实现：九个 backbone adapter + `models/` | `src/sbona/backbones/`（9 个 module + `common.py`）；`tests/unit/test_backbones.py` 45 项 |
| B′ backbone 接入合同 | 九个全部接入，`PENDING` 为空 | `third_party/BACKBONE_CONTRACTS.md`、`src/sbona/backbones/__init__.py` |
| B″ 架构超参 | `backbone.params` 逐架构白名单 + 官方 PEMS08 默认值 | `config.py::BACKBONE_PARAM_DEFAULTS`；`tests/unit/test_config.py` |
| C 训练实现 | v1 已实现（A/B/C/D + probe + WTA 分支） | `src/sbona/training/`；`tests/integration/test_synthetic_pipeline.py`、`test_cli_synthetic.py` |
| C′ MCL-WTA baseline | 已实现（独立头 + 硬 WTA + score head，**非** TimeMCL 官方复现） | `src/sbona/baselines/mcl_wta.py`、`objectives/losses.py`；`tests/unit/test_baselines.py` 8 项 |
| D 评测实现 | v1 已实现（metrics/oracle/regimes/artifacts/profile） | `src/sbona/evaluation/`；`tests/unit/test_metrics.py`、`test_objectives.py`、`test_regimes.py`、`test_artifacts.py` |
| CPU 单测/集成测试 | 已执行：158 passed, 8 skipped（不含 opt-in 数据测试） | `tests/`；环境 Python 3.11.11、torch 2.2.2、numpy 1.26.4 overlay、scipy 1.15.2 |
| 静态检查 | 已执行：ruff 无错误 | `ruff check src tests scripts`（ruff 0.17.3） |
| GPU smoke | 未执行 | `tests/gpu/test_gpu_smoke.py` 因无 CUDA 跳过；不声称 GPU 结果 |
| 真实数据审计 | 部分执行，未训练 | PEMS03/04/07/08 与 METR-LA/PEMS-BAY 均完成 `prepare-data`（含日历与 day-of-week）；PEMS08 与 METR-LA 各 2 样本 CPU forward 通过；见 `reviews/DATA_PROVENANCE.md` |
| 上游数据可复现 | 已执行：13 个上游文件真实下载并逐字节校验 | `scripts/fetch_data.py --all` + `provenance/raw_sources.json` |
| 原始文件 → prepared 对账 | 已执行：7 passed | `SBONA_RAW_DIR=raw pytest tests/integration/test_raw_import_golden.py`；`tests/data/golden_prepared.json` |
| 真实数据训练/评测 | 未执行 | 无任何真实数据集 checkpoint 或 MAE 结果 |
| 可路由性与科学有效性 | 未验证 | 尚无真实实验；synthetic 数值不是科学证据 |
| 实验表格自洽性 | 已修复并回归 | `experiments.csv` 带 `split`；`aggregate --split` 单 seed 建表，拒绝多 seed / 重复行；run 自动发现按身份哈希（seed+method/K），多命中报错。见 `tests/unit/test_run_selection.py`、`test_artifacts.py` |
| 外部 backbone runtime | 九个 CPU 前向/反向通过；**无官方权重等价测试** | `tests/unit/test_backbones.py`（9×3 mode、梯度、图依赖）；N=170 实构建前向通过 |
| 上游来源登记 | 已执行：9 仓库 commit + 28 文件 sha256，schema 2 | `third_party/source_manifest.json`、`third_party/LOCK.md` |
| 上游许可证 | **5 个已核验，4 个缺失** | MIT：Graph WaveNet、AGCRN、PDFormer、ST-SSDL；Apache-2.0：STID；**无 LICENSE**：D2STGNN、HimNet、STAEformer、HyperD |
| METR-LA / PEMS-BAY | 已接入配置，未训练 | 邻接取自 Zenodo 5724362（CC-BY-4.0）；`derived_binary_undirected` + `data.unit: speed (mph)` + `official_compat_nonzero`；`prepare-data` 与 CPU forward 通过，无指标 |
| 依赖 lock / CI | 未提供 | `pyproject.toml` 已提供；锁文件与 CI 配置待补 |
| Codex 最终代码审查 | 等待 | 按 README_REVIEW 执行 |

## 本次验证命令

```bash
# 临时 overlay：torch 2.2.2 (env) + numpy 1.26.4（env 自带 numpy 2.3.3 与 torch 2.2.2 的
# torch.from_numpy ABI 冲突）。这不是可复现环境，只是让 CPU 测试能跑。
PYTHONPATH="src:$TMPDIR/np126/extracted/lib/python3.11/site-packages:\
/opt/miniconda3/envs/pytorch/lib/python3.11/site-packages" \
  /opt/miniconda3/bin/python -m pytest tests -q
# 158 passed, 8 skipped (CUDA unavailable + 7 opt-in raw-data tests)

# 九个 backbone 的 PEMS08 规模（N=170）CPU 实构建 + 前向（一次性检查，不是测试）
#   staeformer 152 / graph_wavenet 512 / agcrn 64 / stid 128 / d2stgnn 256
#   pdformer 64 / himnet 96 / stssdl 76 / hyperd 768  （node_feature_dim）

# 九个 backbone × 三种 mode 的极小图前向诊断（27 行 ok 即通过）
PYTHONPATH="src:..." /opt/miniconda3/bin/python scripts/smoke_backbones.py

ruff check src tests scripts
# All checks passed (ruff 0.17.3)

python -m compileall -q src/sbona
# OK

# opt-in：真实上游文件 → prepared 的 golden hash 对账
SBONA_RAW_DIR=raw python -m pytest tests/integration/test_raw_import_golden.py -q
# 7 passed

# 端到端复现/验证：拿错 checkpoint 与算错聚合结果
PYTHONPATH=src python scripts/verify_selection.py
# A/B/C/D 共 13 项全部 PASS
```

## 不能从当前状态推出的结论

- 不能推出 SBoN 在 PEMS08 上有效或优于 baseline。
- 不能推出 seed 间方差或任何 `mean±std` 结论：当前只跑固定单 seed，聚合不计算方差，`docs/README_EXPERIMENTS.md` §12 的多 seed 合同仍是暂缓项。
- 不能推出 Router 可学习、oracle 收益可部署。
- 不能推出九个 backbone 与官方实现**数值等价**：没有官方 checkpoint，因此没有逐元素比较；只验证了形状、有限性、梯度路径与图依赖。
- 不能推出某个 backbone 更强：九个模型都没有真实数据指标，参数量差异（0.12M–13.65M）不是性能证据。
- 不能推出 D2STGNN / HimNet / STAEformer / HyperD 可再发布：上游无 LICENSE 文件，状态保持 pending。
- 不能推出 GPU 显存、吞吐或 bf16 行为符合验收要求。
- 不能推出 METR-LA / PEMS-BAY 已训练或有效：只完成数据准备与 CPU 前向，没有 checkpoint 或指标。
- 不能宣称日历是发布方原始时间戳：PEMS 的日历由记录在案的起始日期与自带 tod/dow 通道生成，并做周末一致性校验，不是独立时间戳证明。
- 不能把旧的 STAEformer checkpoint 直接复用：`input_dim` 默认值改为 3 改变了 `backbone_identity_hash`，必须按新配置重训。

请后续每次更新附 commit/命令/日志路径，不只把“未开始”改为“完成”。
