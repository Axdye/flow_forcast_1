# 实现与审查状态

日期：2026-10-28。

当前等级：**D0（设计就绪）**。v1 代码已实现并通过 CPU 测试，但 GPU smoke、真实数据训练和科学有效性验证尚未执行，因此不能宣称 D1–D3。

| 项目 | 状态 | 证据 |
|---|---|---|
| D0 研究/开发合同 | 已编写；交付时做链接与结构自检 | docs/ 与 tasks/ |
| A 共享合同与数据实现 | v1 已实现，CPU 测试覆盖 | `src/sbona/contracts.py`、`config.py`、`data/`；`tests/unit/test_config.py`、`test_scaler_and_graph.py`、`test_window_scheme.py` |
| B 模型实现 | v1 已实现（STAEformer 主线） | `src/sbona/backbones/staeformer.py`、`models/`；`tests/unit/test_actions_decoder.py` |
| C 训练实现 | v1 已实现（A/B/C/D + probe） | `src/sbona/training/`；`tests/integration/test_synthetic_pipeline.py`、`test_cli_synthetic.py` |
| D 评测实现 | v1 已实现（metrics/oracle/regimes/artifacts/profile） | `src/sbona/evaluation/`；`tests/unit/test_metrics.py`、`test_objectives.py`、`test_regimes.py`、`test_artifacts.py` |
| CPU 单测/集成测试 | 已执行：52 passed, 1 skipped | `python -m pytest -q`；环境 Python 3.11.11、torch 2.2.2、numpy 1.26.4、scipy 1.15.2 |
| 静态检查 | 已执行：ruff 无错误 | `ruff check src tests` |
| GPU smoke | 未执行 | `tests/gpu/test_gpu_smoke.py` 因无 CUDA 跳过；不声称 GPU 结果 |
| 真实数据审计 | 部分执行，未训练 | PEMS08 `prepare-data` 得到 `(17856, 170)`、274 条无向边；2 样本 CPU forward 通过 |
| 真实数据训练/评测 | 未执行 | 无真实 PEMS08 checkpoint 或 MAE 结果 |
| 可路由性与科学有效性 | 未验证 | 尚无真实实验；synthetic 数值不是科学证据 |
| 外部 backbone runtime | 未验证 | STID / Graph WaveNet / AGCRN 仍为 pending |
| 依赖 lock / CI | 未提供 | `pyproject.toml` 已提供；锁文件与 CI 配置待补 |
| Codex 最终代码审查 | 等待 | 按 README_REVIEW 执行 |

## 本次验证命令

```bash
# 临时 Python 3.11 环境（torch 2.2.2，CPU）
python -m pytest -q
# 52 passed, 1 skipped (CUDA unavailable)

ruff check src tests
# All checks passed

python -m compileall -q src/sbona
# OK
```

## 不能从当前状态推出的结论

- 不能推出 SBoN 在 PEMS08 上有效或优于 baseline。
- 不能推出 Router 可学习、oracle 收益可部署。
- 不能推出 STAEformer 与其他 backbone 等价或兼容。
- 不能推出 GPU 显存、吞吐或 bf16 行为符合验收要求。

请后续每次更新附 commit/命令/日志路径，不只把“未开始”改为“完成”。
