# Codex 审查与验收标准

本文是后续代码审查合同。当前未有实现，不能将下面的测试视为已通过。工程验收与科学成功分开记录。

## 1. 审查顺序

1. 核对分支/base/head commit、diff、第三方来源与 HANDOFF。
2. 阅读公共接口与数据路径，先排查目标泄漏和尺度错误。
3. 按公式逐项核对 decoder、loss、gradient flow 与 checkpoint 选择。
4. 执行独立的 CPU 测试，再进行 GPU smoke/恢复/性能测试。
5. 检查训练产物和聚合脚本，核对实际证据与声明。
6. 返回问题清单和允许进入的 Gate，不因为代码量大或测试数量多就签收。

## 2. 阻断性测试矩阵

| ID | 测试 | 独立预期/反例 | 严重级别 |
|---|---|---|---|
| DATA-01 | split target 区间 | train/val/test 的未来时间集合不相交；跨边界窗口拒绝 | P0 |
| DATA-02 | train-only 统计 | 任意改写 val/test 后 scaler/envelope/hash 逻辑不变 | P0 |
| DATA-03 | 缺失/零值 | 真零参与 MAE；NaN 先安全替换；全无效不算零误差 | P1 |
| DATA-04 | 节点映射 | 同时置换节点/图后可逆；仅错换图应 fail | P1 |
| BB-01 | adapter 等价 | eval 模式 base 与相应 pristine/兼容补丁版本前向相同 | P1 |
| BB-02 | Frozen | parent.train 后 backbone 仍 eval，参数/BN buffers 不变 | P1 |
| BB-03 | Partial | 白名单参数变化，其他无梯度/无 buffer 漂移 | P1 |
| MOD-01 | selected 等价 | eval fp32 下 all.gather 与 selected 接近 | P1 |
| MOD-02 | 单解码 | forward_all 被替换为抛异常后 predict_top1 仍可运行 | P1 |
| MOD-03 | 隐藏 K 维计算 | profiler/张量 hook 无 [B,K,T,N,*] selected 激活 | P1 |
| MOD-04 | 结构不被旁路 | m=0 残差严格 0，无 local/FiLM bias | P1 |
| MOD-05 | 图传播 | 小图 sparse 递推=显式 dense；孤立节点保持自身 | P1 |
| LEAK-01 | 真值扰动 | 相同 X，任意替换 Y/mask/sample_id，Top1 完全不变 | P0 |
| LEAK-02 | API/导入 | inference 不接受 TargetBatch；models 无 oracle 导入 | P0 |
| LOSS-01 | 正权重 | w>=epsilon/K 且 sum=1；误差方向梯度手算正确 | P1 |
| LOSS-02 | 梯度隔离 | Router loss 不更新 candidate/projector/backbone | P1 |
| LOSS-03 | K=1/相同成本 | 无 NaN、w=1/uniform、sep=0 | P1 |
| LOSS-04 | 缩放 | 复制相同候选 K 次不会多乘/少乘 K | P1 |
| LOSS-05 | 单位 | raw 负流量才算 violation，非 z<0 | P1 |
| MET-01 | 聚合 | 不同大小 batch 的总 sum/count 与一次性计算相同 | P1 |
| MET-02 | oracle 粒度 | 不许逐节点拼接；cost oracle 与 MAE oracle 可不同 | P0 |
| MET-03 | retention | 无/负 oracle gain 返回 null 或有效负值，不裁剪 | P1 |
| RUN-01 | checkpoint 恢复 | epoch 边界恢复的状态/数据顺序/曲线符合协议 | P1 |
| RUN-02 | 错配拒绝 | K/图/scaler/split 不同不能 resume | P1 |
| RUN-03 | test 禁调参 | best selection 不读取 test metric | P0 |
| PROBE-01 | 时间留出 | F/R/Q 拟合范围独立，scaler 不读 R/Q | P0 |
| ART-01 | 结果诚信 | mock/synthetic/未运行明确标注；失败不静默丢弃 | P0 |
| ART-02 | 来源一致 | aggregate 拒绝不同 protocol/hash/units 的主表混合 | P1 |

MOD-01 初始容差 fp32 CPU `atol=1e-6,rtol=1e-5`，GPU 同精度使用可解释容差；bf16 容差必须先经真实数值校验。不能看到任意大差异就放宽到 1%。同一环境严格 CPU 确定性恢复应近乎精确；CUDA 非确定性应披露最大差异、算子来源和重复范围。

## 3. 必须手算的数值 fixture

- y=[0,2,4]，预测 [1,1,8]，全有效：MAE=2，MSE=6，RMSE=sqrt(6)。MAPE 阈值 1，仅后二项，MAPE=75%。
- 两候选的误差分布跨节点相反，确认 oracle 选完整候选而不是逐元素最小。
- 成本 [0,0] → w=[.5,.5]；K=1 → w=[1]；同候选复制不能改变 L_cand。
- 原始 μ=100、s=20，z=-1 对应 raw=80，不应受非负惩罚；z=-6 对应 raw=-20，应有无量纲负惩罚 1。
- empty mask → 无有效监督、optimizer 不 step、metrics=null 且 count=0。
- 真实图变化不为零，即使所有节点都在 [0,C] 也不能期待 spatial penalty 为零。

fixture 可更详细，但 expected value 必须独立推导并写注释。公式截图/口头说“看起来对”不算验收。

## 4. 工程完成的分层要求

### D1：工程闭环

- `pip install -e '.[dev]'` 后 package/CLI 可用；Linux 环境 lock 真实存在。
- PEMS08 数据准备、Base→single→candidates→router→eval→aggregate 全链条执行；synthetic 与真实数据结果清楚分开。
- 上述 P0/P1 测试通过；K=1、B=1、不同 N、缺失 mask 通过。
- A100 至少完成真实数据两个 epoch 或预登记步数的 smoke；含 optimizer 更新、checkpoint、恢复、最终 eval。
- 保存全部必要 artifacts；外部运行不依赖某个开发者机器上的绝对路径。

### D2：研究 pilot

- PEMS08/04 的至少 3 seeds 配对结果，Base 与 matched single 不弱化。
- 专用时间留出 probe 与负控制；Router/Oracle/Fixed/Random 分离。
- 报告支持和不支持的假设，不用 oracle 替代主部署结果。
- 有实际 GPU-h 和显存 profiling，决定是否投入主矩阵。

### D3：论文证据

- G4 后锁定协议，多骨干/数据主表、消融与效率；不足范围明确缩小结论。
- paired time-block CI、多种子变异、比较家族清单；无法检验时不强造 p 值。
- 所有表图来自脚本并可追溯至 run_id/config/checkpoint/data hash。
- 许可证/引用/迁移补丁齐全，无旧二进制/敏感数据提交。
- README 快速开始由独立干净 checkout 验证；GPU 复现实验记录可读。

## 5. 性能验收不只看 decoder 调用次数

一次 forward_all 也可能一次函数调用却算 K 条轨迹；一次 selected 也可能内部复制 K 份。必须联合检查源码、关键 tensor shape、profiler 和显存随 K 变化。动作 `[B,K,5]` 与 Router `[B,K]` 是允许的。

端到端延迟包括 backbone/projector/router/selected decoder，不能只报 decoder kernel。所有对照同 batch、precision、TF32、hardware、warm-up；吞吐和 batch=1 延迟分别报告。

## 6. 最终审查报告模板

```text
审查日期 / reviewer / base / head：
范围与未审范围：
已核查设计条款：
P0/P1/P2 问题（位置、触发、影响、修复建议）：
实际复跑命令与输出：
CPU / GPU / 真实数据覆盖：
剩余风险与外部依赖：
工程结论：通过 / 有条件通过 / 不通过
科学结论：支持 / 部分支持 / 不支持 / 尚无证据
允许进入的 Gate：
```

“最终审查”不是提前承诺所有实现都能通过；Codex 在拿到代码后按此标准逐项复核。review 作者不伪装独立审稿人；同一会话对自己起草文档的检查只称设计自检。
