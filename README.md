# SBoN-Adapter：从头重建的研究与开发合同

版本：v1.0 · 2026-10-28（Asia/Shanghai） · 计划硬件：1–4 × NVIDIA A100 80GB。

**当前交付已包含 v1 代码实现与 CPU/synthetic 验证，但尚未完成 GPU smoke、真实数据训练或任何论文结果。** 下文 CLI 已实现；所有科学主张仍需实验检验。实现与验证状态见 `reviews/STATUS.md` 和 `reviews/HANDOFF.md`。目标安装路径为 `/Users/ljw/Documents/code/SBoN-Adapter`。

研究对象是**交通流多步预测**，不是交通信号控制，不训练大语言模型。DeepSeek 是代码实现协作者；A100 用于交通预测模型训练和实验。

## 阅读顺序

| 文件 | 解决的问题 | 必读人 |
|---|---|---|
| [故事线](docs/README_STORY.md) | 为什么做、假设是什么、什么结果足以支持论文 | 全员 |
| [代码设计](docs/README_CODE_DESIGN.md) | 数据、模型、公式、梯度、API、训练状态机的精确定义 | 全员 |
| [实验与算力规划](docs/README_EXPERIMENTS.md) | 先跑什么、1/2/4 卡怎么排、何时停止 | 实验负责人 |
| [DeepSeek 并行开发](docs/README_PARALLEL.md) | 文件所有权、依赖、交付、合并顺序 | 全员 |
| [审查与验收](docs/README_REVIEW.md) | Codex 最终审查依据，什么算完成 | 全员 |
| [来源与设计变更](docs/README_PROVENANCE.md) | 前情、旧仓资产、已核验与待核验项 | 负责人 |
| [任务包](tasks/README.md) | 可直接复制给四个 DeepSeek 会话的任务 | 用户 |
| [依赖登记](third_party/LOCK.md) | 上游源码固定与许可审计 | Backbone 负责人 |

文档优先级：代码设计中的 MUST 合同 > 实验协议 > 分工任务 > 故事线。发生冲突时提交设计变更，不允许各实现者自行猜测。v1.0 的默认项不能被旧故事线覆盖。

## 已固定的决策

1. 新建独立 Git 仓库；旧 `Flow-RL` 和 `Flow-RL 2` 仅作只读资产来源。
2. 自有轻量 PyTorch runner 统一数据、训练和评测；**不依赖整套 BasicTS runner**，避免旧框架与新模型接口交叉。
3. 首个闭环：PEMS08 + STAEformer，12 步历史预测 12 步未来，flow-only，Frozen 模式。
4. 主方法 v1 使用无旁路的结构化共享残差解码器；候选模型冻结后再训练 Router，联合训练是后续消融。
5. 先验证可路由性，再扩展四数据集/多 backbone；Oracle 收益不能替代部署收益。
6. 采用 1 GPU/实验作业，2–4 卡优先并行不同实验；DDP 不是首版交付要求。
7. 论文主指标为原始单位全预测窗 MAE；oracle 的选择准则必须显式命名。

## 完成程度分级

| 级别 | 必须有的交付 | 不代表什么 |
|---|---|---|
| D0 设计就绪 | 本仓 README、接口、任务、审查合同 | 不代表代码可运行 |
| D1 工程闭环 | CPU 测试、GPU smoke、无泄漏、单解码、可恢复训练 | 不代表论文有效 |
| D2 科学 pilot | 两数据集多种子候选与路由证据、失败分析 | 不代表跨 backbone 泛化 |
| D3 论文证据 | 锁定协议、主表/消融/成本/置信区间自动生成 | 不自动构成 SOTA |

当前：D0（v1 代码已实现，但 GPU smoke 与真实数据训练未执行，不能宣称 D1）。`reviews/STATUS.md` 中不得预先勾选 D1–D3。

## 开发启动顺序

先由 DeepSeek A 完成数据/API 基础合同，再让 B/C/D 在独立分支开发；第一轮 B/C/D 可用严格符合合同的 synthetic fixture 并行，不等待真实数据。每个任务提交可审查 commit 和交付说明，由 Codex 检查接口、实现、测试和证据后集成。不要让四个会话同时编辑一个工作目录。

工程闭环后，按实验文档的 Gate 顺序开跑；发现科学假设不成立可以交付正确的失败结果，不能为通过验收更改测试集或指标。

## CLI 合同（已实现）

```bash
python -m sbona.cli prepare-data --config configs/experiments/pems08.yaml
python -m sbona.cli train-base --config configs/experiments/pems08_staeformer.yaml
python -m sbona.cli train-candidates --config configs/experiments/pems08_staeformer_k8.yaml
python -m sbona.cli probe-route --config configs/experiments/pems08_probe.yaml
python -m sbona.cli train-router --config configs/experiments/pems08_staeformer_k8.yaml
python -m sbona.cli evaluate --run outputs/RUN_ID --split test --protocol-lock protocol.lock.json
python -m sbona.cli profile --run outputs/RUN_ID --batch-sizes 1 64
python -m sbona.cli aggregate --manifest experiments.csv
```

当前仓库已包含 `pyproject.toml`、可运行 CLI、CPU 测试与 GPU smoke 脚本；尚缺真实依赖 lock 与 CI 配置，GPU smoke 未执行。空函数/占位模块不会被当作已实现。
