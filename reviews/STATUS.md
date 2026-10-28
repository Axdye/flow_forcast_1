# 实现与审查状态

日期：2026-10-28。

当前等级：**D0（设计就绪）**。v1 代码已实现并通过 CPU 测试，但 GPU smoke、真实数据训练和科学有效性验证尚未执行，因此不能宣称 D1–D3。计划外实现与已知偏离见 `reviews/EXTRA_WORK.md`。

| 项目 | 状态 | 证据 |
|---|---|---|
| D0 研究/开发合同 | 已编写；交付时做链接与结构自检 | docs/ 与 tasks/ |
| A 共享合同与数据实现 | v1 已实现，CPU 测试覆盖 | `src/sbona/contracts.py`、`config.py`、`data/`；`tests/unit/test_config.py`、`test_scaler_and_graph.py`、`test_window_scheme.py` |
| B 模型实现 | v1 已实现（STAEformer 主线） | `src/sbona/backbones/staeformer.py`、`models/`；`tests/unit/test_actions_decoder.py` |
| C 训练实现 | v1 已实现（A/B/C/D + probe） | `src/sbona/training/`；`tests/integration/test_synthetic_pipeline.py`、`test_cli_synthetic.py` |
| D 评测实现 | v1 已实现（metrics/oracle/regimes/artifacts/profile） | `src/sbona/evaluation/`；`tests/unit/test_metrics.py`、`test_objectives.py`、`test_regimes.py`、`test_artifacts.py` |
| CPU 单测/集成测试 | 已执行：65 passed, 1 skipped | `python -m pytest -q`；环境 Python 3.11.11、torch 2.2.2、numpy 1.26.4、scipy 1.15.2 |
| 静态检查 | 已执行：ruff 无错误 | `ruff check src tests scripts` |
| GPU smoke | 未执行 | `tests/gpu/test_gpu_smoke.py` 因无 CUDA 跳过；不声称 GPU 结果 |
| 真实数据审计 | 部分执行，未训练 | PEMS03/04/07/08 与 METR-LA/PEMS-BAY 均完成 `prepare-data`（含日历与 day-of-week）；PEMS08 与 METR-LA 各 2 样本 CPU forward 通过；见 `reviews/DATA_PROVENANCE.md` |
| 真实数据训练/评测 | 未执行 | 无任何真实数据集 checkpoint 或 MAE 结果 |
| 可路由性与科学有效性 | 未验证 | 尚无真实实验；synthetic 数值不是科学证据 |
| 外部 backbone runtime | 未验证 | STID / Graph WaveNet / AGCRN 仍为 pending |
| METR-LA / PEMS-BAY | 已接入配置，未训练 | 邻接取自 Zenodo 5724362（CC-BY-4.0）；`derived_binary_undirected` + `data.unit: speed (mph)` + `official_compat_nonzero`；`prepare-data` 与 CPU forward 通过，无指标 |
| 依赖 lock / CI | 未提供 | `pyproject.toml` 已提供；锁文件与 CI 配置待补 |
| Codex 最终代码审查 | 等待 | 按 README_REVIEW 执行 |

## 本次验证命令

```bash
# 临时 Python 3.11 环境（torch 2.2.2，CPU）
python -m pytest -q
# 65 passed, 1 skipped (CUDA unavailable)

ruff check src tests scripts
# All checks passed

python -m compileall -q src/sbona
# OK
```

## 不能从当前状态推出的结论

- 不能推出 SBoN 在 PEMS08 上有效或优于 baseline。
- 不能推出 Router 可学习、oracle 收益可部署。
- 不能推出 STAEformer 与其他 backbone 等价或兼容。
- 不能推出 GPU 显存、吞吐或 bf16 行为符合验收要求。
- 不能推出 METR-LA / PEMS-BAY 已训练或有效：只完成数据准备与 CPU 前向，没有 checkpoint 或指标。
- 不能宣称日历是发布方原始时间戳：PEMS 的日历由记录在案的起始日期与自带 tod/dow 通道生成，并做周末一致性校验，不是独立时间戳证明。

请后续每次更新附 commit/命令/日志路径，不只把“未开始”改为“完成”。
