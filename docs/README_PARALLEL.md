# DeepSeek 并行开发与集成合同

本次不自动调用 DeepSeek、不上传私有材料、不新建用户聊天。用户将任务包分配给独立 DeepSeek 会话；Codex 负责后续审查和集成。任何“通过最终审查”必须对应真实代码、命令输出与具体 commit。

## 1. 四个角色，不按整篇论文复制四次

| 任务 | 角色 | 独占编辑目录/文件 | 第一轮交付 |
|---|---|---|---|
| A | 数据、backbone、基础工程 | contracts.py/config.py/data/backbones/third_party/pyproject 与依赖锁 | 合同、PEMS08 数据、STAEformer 等价前向 |
| B | 候选模型与预解码 Router | models/、baselines/（模型部分） | 结构解码器、动作、single、Top1 双路径 |
| C | 训练目标与状态机 | objectives/、training/、cli.py | 正确 loss、可恢复 stages、probe |
| D | 独立评测与 QA | evaluation/、scripts/、tests/integration、tests/gpu、CI | 手算指标、泄漏/单解码/恢复测试、统计与 profiling |

A/B/C 各自拥有 `tests/unit/test_data_*`、`test_models_*`、`test_objectives_*` 等对应前缀；D 不修改这些测试来掩盖失败，而是补独立交叉测试。配置文件由 C 维护，A/B/D 通过 schema 或 patch 提案请求新增字段。

Codex 可修改所有模块以解决审查发现，但应说明原因和重测范围；实施者不能把自己的自测声明当作最终验收。

## 2. 共享文件先冻结

第零步由 A 根据代码设计实现 contracts.py 和配置 schema，附 synthetic fixture（B=2,L=T=12,N=5,K∈{1,4}，带孤立节点与缺失 mask），只依赖最少环境。

Codex 审核合同后给出共同 base commit。B/C/D 从这个 commit 开始集成。第零步未完成时可以阅读设计、开发各自文件或本地 fixture，但不得各自提交不兼容 contracts.py。

共享接口变更流程：提出问题 → 在 `reviews/ADR-XXX.md` 描述旧/新接口、原因和影响 → Codex 确认并改共享合同 → 各分支同步 → 更新版本与兼容测试。不得使用 `try/except` 静默兼容两套含义不同的 tensor。

## 3. Git 并行工作约定

每个会话使用独立 checkout/worktree 与分支，建议：

```text
codex/data-backbones
codex/candidate-models
codex/training-objectives
codex/evaluation-qa
```

若 DeepSeek 工具不支持 worktree，使用独立本地 clone。不要只在同一个目录里切换分支后继续让其他会话写文件。worktree/clone 的实际创建由安排开发时执行，本次仅初始化文档仓库。

禁止提交数据、checkpoint、log、API key；不要复制旧仓 `.git`，不要把整个第三方仓变成无 `.gitmodules` 的 gitlink。首次 vendor 仅复制用到的源码，保留许可证与 patch 清单。

每个功能一个可审查 commit，commit message 描述行为。不能 force-push 共享分支，不 reset 他人的工作。不要求“所有功能一个巨型提交”。

## 4. 依赖与集成顺序

```text
A0 合同/fixture
  ├── A1 数据 + STAEformer
  ├── B1 projector/actions/decoder/router
  ├── C1 oracle/loss + toy trainer
  └── D1 hand-computed metrics + adversarial tests
          ↓
I1 合成数据端到端集成
          ↓
I2 PEMS08 Base → Single → Candidates → Router
          ↓
G1 GPU/恢复/单解码验收
          ↓
G2/G3 pilot（通过才扩展 A2 其他 backbone 与完整消融）
```

B 不必等 A 下载完整数据；C 不必等真实 backbone，可以使用合同 fixture。D 独立构造 expected value，不能直接调用被测实现来生成答案。实际模型合并后必须重跑集成测试，不能仅依赖各分支单测。

## 5. 每次交付模板

每个任务提交 `reviews/HANDOFF_<TASK>_<ROUND>.md`：

```text
任务 / 分支 / base commit / head commit：
实现范围及本次未完成项：
修改文件列表：
接口/配置/schema 变化（没有也写无）：
公式到函数的映射：
实际执行的命令、exit code、环境、耗时、日志路径：
未执行测试及原因（CPU / CUDA / 数据不足）：
关键数值样例与预期：
风险、已知缺陷、需要负责人决定的事项：
外部源码/许可证/依赖变化：
建议 Codex 重点审查点：
```

没有 GPU 就写“GPU 验收未执行”，不能根据 CPU shape 测试推断 A100 运行通过。没实际下载数据就写 synthetic。没有运行不能生成看似真实的训练曲线或 metrics.json。

## 6. 审查节奏

第一轮审合同和纯函数；第二轮审模型双路径与梯度；第三轮审真实数据端到端；第四轮审实验统计与最终结果。Codex 返回逐项 P0/P1/P2 问题、文件定位、复现步骤及重测范围。

- P0：未来泄漏、数据损坏、训练/测试混用、伪造实验或 destructive 操作。
- P1：错误 loss/metric、输出与合同不符、selected 解码 K 次、恢复错误、不可复现依赖。
- P2：非阻断性能/可维护性问题，明确是否须在 G5 前修。

P0/P1 未清零不得启动大规模正式实验。代码正确但假设失败，审查结论可为“工程通过、研究不支持”，这是合法结果。

## 7. 如何把实现交回给 Codex

下一次提供仓库路径、四个 head commit、HANDOFF 文件和已运行的测试/训练日志路径，指明审查范围。Codex 先检查 diff 与源代码，再复跑必要测试，最后更新 `reviews/STATUS.md`。不必手动把整个仓库复制到聊天里。

当前仓库不会自动监听未来 DeepSeek 的修改，也没有设置周期性自动审查。用户把实现交回本聊天后继续审查即可。
