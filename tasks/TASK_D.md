# DeepSeek D：独立评测、反例测试与实验工具

## 任务正文（可直接复制）

你负责 D 任务：src/sbona/evaluation、scripts、tests/integration、tests/gpu 和 CI。在独立分支工作。你的职责包含找出 A/B/C 的科学或工程错误；不能为了让测试通过修改预期公式。

读完 README_CODE_DESIGN.md、README_EXPERIMENTS.md、README_REVIEW.md 后，先手算小数组的 masked MAE/RMSE/MAPE、oracle 选择、sum/count aggregation、空分母、负 gain/retention。expected value 不能来自被测函数。

实现 metric-specific oracle，整条样本选一个候选；区分 cost/MAE/MSE oracle。实现候选健康、violation rate+magnitude、训练阈值的状态分组。无 speed 标签时不得把流量下降直接叫拥堵。

独立测试未来泄漏（更换/清空/打乱 TargetBatch 后推理不变）、single-decode（让 forward_all 抛错，selected 仍能运行，检查大张量形状）、all-vs-selected、冻结 buffer、masked NaN、切分跨界、恢复一致性。

实现 paired time-block bootstrap、seed 级汇总、Holm 校正与预登记 comparisons；不要按重叠窗口做独立 t-test。实现 CUDA profiling（warm-up、同步、p50/p95、memory），不伪造 GPU 数字。

实现 experiments.csv 展开/聚合和结果 artifact 校验：缺 run、混用 protocol、数据 hash 不一致、oracle 冒充 deployable 必须报错。聚合只读结构化日志，所有失败保留。

CI 用 synthetic CPU fixture，不下载私有数据、不启动正式训练。GPU smoke 单独人工/作业触发；CI 不可见 GPU 时输出未执行，不写 passed。

交付 reviews/HANDOFF_D_01.md：独立发现的问题、手算 fixture、命令/日志、尚待真实 GPU 验证项。最终由 Codex 复核，不自行将所有 Gate 标为通过。
