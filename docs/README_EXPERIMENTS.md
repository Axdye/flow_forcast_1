# 实验、算力与阶段计划

## Material Passport

- Origin：academic-research-suite / experiment-agent，plan；2026-10-28，v1.0。
- 资源：用户提供约 1–4 张 A100 80GB；PCIe/SXM、互联、CPU、RAM、存储、占用期限未提供。
- Verification Status：PLAN ONLY。没有实测 epoch 时间、显存、统计功效或预测提升。
- 目标：以最少的无效训练验证可路由性，再扩展为可复现论文证据。

## 1. 实验原则

充足显存优先用于可靠 batch、更多独立重复和公平对照。默认每个作业使用一张 GPU，四卡同时跑四个作业；不用单模型四卡训练作为启动前提。模型是交通流预测器，GPU 预算不包含托管 DeepSeek 编码成本。

所有方法统一数据版本、split、mask、scaler、输入输出长度、输入特征、图信息可用性和指标。若官方训练配置与统一协议不同，区分“官方复现”与“统一协议重跑”，不把跨协议数字混入表格。

主要比较预登记为 **Router@1 vs 同参数 single adapter 的 raw 全窗 MAE**。次指标为 RMSE/MAPE、一致性、成本和候选诊断。不能看到结果后改主指标。

## 2. 从工程到科学的 Gate

| Gate | 输入与动作 | 交付 | 通过条件与失败处理 |
|---|---|---|---|
| G0 数据/接口 | 审计 PEMS08，锁源码，原 forward 等价，CPU 测试 | data_audit、测试报告、来源锁 | 节点/切分/尺度/前向正确；失败先修工程 |
| G1 GPU 闭环 | Base、K=1、K=8 各短跑；恢复和 profile | smoke.json、失败日志、吞吐 | 无泄漏/NaN/OOM，selected 等价，checkpoint 可恢复 |
| G2 候选价值 | PEMS08，结构/无结构/独立头，先 1 seed 再 3 seeds | 嵌套 oracle 曲线、分支质量 | 优于 single 的 oracle gain 超出 seed 噪声；没有则最多两轮预登记修正 |
| G3 可路由性 | 时间留出 probe + 固定候选正式 Router，PEMS08/04 | regret/MAE/负控制/retention | Router 优于 fixed/random，且有优于 single 的可重复趋势 |
| G4 配置冻结 | 两数据集 3 seeds、关键负控制、成本 | protocol.lock.json、选择记录 | 主张与误差条一致；若不成立暂停主矩阵 |
| G5 主证据 | 多数据/多 backbone、关键消融、统计 | 自动主表和分析图、完整 manifest | 严格合同通过；结果正负均报告 |

“Oracle 改善约 1%、retention 超过 30%、延迟不高于 Base 的 1.3 倍”只可作为 pilot 的工程观察线，不是理论结论，也不作为选择性删结果的规则。科学通过关注相对同参数 single 的增益、paired CI、跨 seed 一致性与实际成本。提交审查时同时报告绝对变化和相对变化。

G2/G3 若连续两轮合理、已登记的修改均失败，不再扩大超参/数据规模以追逐正结果。保留失败证据，按故事线收缩主张。新方法改动需要版本号及重跑比较。

## 3. 可路由性 probe：防止“只是在训练集上分类成功”

不能在训练过完整 train 的候选模型上，再用同一个 train 的随机 K-fold 宣称标签可泛化。分支标签是未来损失产生的，候选模型也可能过拟合这些未来。

单独建立 probe 协议，仅使用正式训练区间 `[0,a)`：

1. F 区间 `[0,floor(.7*a))`：拟合所有统计、训练 probe 专用 Base 和候选模型。
2. R 区间 `[floor(.7*a),floor(.85*a))`：冻结候选后生成 oracle 标签/成本，训练线性与两层 MLP probe。
3. Q 区间 `[floor(.85*a),a)`：只评估 probe，不训练候选或 probe。

每区间窗口完整历史与目标都位于自己的区间内，边界各舍弃不足 L+T 的窗口；F 的 scaler/envelope 不使用 R/Q。probe 超参在 R 内按时间 80/20 划分，Q 保持未使用。数据不足时报告低样本数，不合并测试数据补量。

所有 split、样本 ID 和弃用边界记录进 probe manifest。若做多个 forward-chaining 折，每折重训自己的 Base/候选并在折内训练 probe；分支身份不跨折直接对齐，不汇总原始 branch id 混淆矩阵。

比较：

- uniform random branch，固定 RNG；
- R 上最常获胜分支；
- R 上平均 raw MAE 最低分支（它不一定等于多数获胜分支）；
- soft-target 线性分类器；
- soft-target MLP；
- OPTIONAL 直接预测每分支成本的回归 probe。

主诊断为 Q 上实际选择 MAE 和 regret；同时报告 top-1/top-2、oracle 软标签熵、tie/近似平局比例、branch 使用率。分类器 log-loss 不是“已测得真实条件熵”，只能作为可预测性诊断。

负控制：在 R 内按连续时间块打乱历史特征与 oracle 标签对应关系，保持标签频率；在 Q 原始配对上评估。负控制接近正常 probe 时，说明可能只是分支先验，不支持历史条件选择。

这个专用 probe 不作为最终模型；通过后正式 Base/Candidates 使用正式训练区间训练，D-stage Router 使用 train 标签和 val 早停。正式 test 仍完全隔离。

## 4. 首轮 pilot 配置与顺序

固定 seeds=[11,22,33]，首轮调试只用 11。初始 K=8，D=64，R=3，alpha=1、lambda_consistency=0。先隔离预测性能，再加一致性代价。

| 批次 | 作业 | 用途 |
|---|---|---|
| P0 | PEMS08 STAEformer Base | 数据与基础模型 sanity |
| P1 | K=1 matched single；K=8 structured；K=8 unstructured；K=8 independent heads | 覆盖与容量归因 |
| P2 | 固定候选上的 random/fixed/Hard CE/Soft KL/Soft KL+regret | 路由能否保留收益 |
| P3 | 专用 F/R/Q probe、负控制；PEMS04 重复 P0–P2 核心项 | 可路由性与数据泛化 |
| P4 | seeds 22/33；准确性 vs 一致性；single compute-matched | 判断是否扩展 |

G1 阶段可用训练子集 debug；G2 起的数据子集/完整数据必须独立 protocol_id，不能混合汇报。Gaussian output noise 只作覆盖诊断，不把它的 oracle 性能写成可部署强基线。

## 5. 最小且充分的对照定义

| 方法 | 精确定义 | 公平性要求 |
|---|---|---|
| Base | 同一训练出的 backbone | 所有配对 adapter 共享对应 seed 的 checkpoint |
| Single | K=1 结构 residual，训练 action；另做宽化版本 | 按实际 trainable 参数调宽到 full 方法 ±5%；未匹配则报告并标为未匹配 |
| Compute-matched single | 同预算 GPU 时间训练/调参的 single | 上限和早停规则预登记，不能刻意训坏 |
| Independent heads | 同 backbone，共享 projector，K 个残差头+同类 Router | 调整 hidden width 匹配参数；输出/路由接口同主方法 |
| Unstructured latent | 五维 latent，经共享低秩时间-节点 decoder | 相同 K、图与输入；参数匹配；没有 o/δ/d/ρ 包络 |
| MCL-style | 同 independent heads，hard WTA+score head | 明确是机制移植，不冒称 TimeMCL 官方完整复现 |
| Gaussian | Base raw/z 输出加训练尺度固定高斯噪声 | std 只在 val 调；不计算测试 oracle 最优 std |
| MC dropout | 固定 backbone，仅 dropout 层 train，其他 eval | 记录 K 次 backbone 成本；无有效 dropout 时报告不可用 |
| Full-decode selector | 输入历史和 K 条预测学习选择，推理不读 Y | 显式成本对照，训练标签与 Router 相同 |

小参数差异不可通过添加未参与 forward 的“假参数”补齐。对所有学习方法给同等 trial 数和近似 GPU-h 两套预算视图；匹配参数不意味着匹配计算量。

## 6. 主实验矩阵与作业数

标准档的 dataset/backbone 单元：STAEformer × PEMS03/04/07/08 共 4；STID、Graph WaveNet、AGCRN 各 × PEMS07/08 共 6，合计 10 单元。

每单元 3 seeds × [Base、matched single、structured(C+D)] = **90 个逻辑训练作业**。structured 的 C/D 分 stage 记录，总预算是两者之和。Base 作业复用，禁止重复记账。已有同配置/同 hash 的 pilot 可以复用，必须在 manifest 指向同一 run，而非复制结果伪装独立重复。

再增加：

- PEMS08/04 × 3 seeds × [unstructured、independent、MCL-style]：18 个作业。
- 核心 6 种训练消融 × 2 数据集 × 3 seeds：36 个作业（no floor、no sep、hard WTA、no graph R=0、no regret、no balance；后两者可复用固定候选，仅重训 Router）。
- 一致性 4 个设置 × PEMS08 × 3 seeds：12 个作业（accuracy-only 若已存在可复用；negative、negative+envelope、再加 spatial）。
- 固定 Kmax=16 的 candidate 模型 2 数据集 × 3 seeds：6 个作业；K 子集离线评估不另计训练。
- independent-trained K={1,2,4,8,16} 扩展是后续档；与嵌套子集曲线分开。

上述约 162 个逻辑训练作业是标准档的**上限粗清单，去重后更少**，不包含超参失败试验、专用 probe、额外宽化调参和可选 Full。实施时由 experiments.csv 逐行展开，不能仅凭此数字作最终资源申请。

至少三个 backbone 才支持相应迁移主张；如果只有单卡且周期有限，先 STAEformer+STID+Graph WaveNet，再 AGCRN。宁可缩小结论范围，也不压成所有结果单 seed。

**当前实现进度（2026-10-29）**：九个官方 backbone（STAEformer、Graph WaveNet、AGCRN、STID、D2STGNN、PDFormer、HimNet、ST-SSDL、HyperD）都已实现在同一 `BackboneWrapper` 合同上，并各有 `configs/experiments/pems08_<backbone>.yaml`（官方 PEMS08 超参显式写全）。这只是**代码可用**，不是上述矩阵已完成：没有任何真实数据训练、没有指标、没有 seed 间方差。加入后四个架构也不改变本节的 10 单元主表——主表仍按预算预先固定，扩展单元须另行登记。

## 7. K 扩展曲线的区别

**Nested-K**：固定一个 Kmax=16 模型和预登记 anchor 顺序，使用前 1/2/4/8/16 个候选；对 raw MAE 最小选择的 oracle 误差必不增加。不能把这一数学必然性本身当作方法有效性证据，应看相同预算对照及增益大小。

**Retrained-K**：各 K 重新训练，各自验证选模型，体现训练预算/容量差异。不同 K 不保证单调。不得把曲线非单调当作代码错误或删掉点。

Router 是固定 K 的分类器，不能仅截断 logits 就声称为另一个 K 正式模型。若做 subset 路由，标记为限制候选的敏感性分析，或训练匹配的 router。

## 8. 调参预算

最多两轮 pilot 搜索，每轮最多 8 个完整 candidate trial；先 PEMS08/seed11，选出不超过 2 个方案在 PEMS04/seed11 验证，再使用 seeds22/33 检查稳定性。不是 K×tau×epsilon×alpha 全笛卡尔积。

第一轮只动 tau∈{.2,.5,1}、epsilon∈{.02,.05,.1} 的预登记子集；第二轮可根据失败诊断动 residual_dim∈{16,32,64} 或 R∈{0,1,3}，一次只解决一个假设。每 trial 保存假设、配置、开始/结束时间、失败原因和 GPU-h。

alpha 的准确性主线固定 1；一致性扩展用 {.8,1} 起步。若报告 alpha sweep，每点训练对应模型与 Router，不把推理时改 alpha 当作条件化路由已经实现。

目标 venue 未确定，当前不承诺覆盖 2026 年全部最新 SOTA。正式投稿前补一次直接竞争者检索；未运行/不同任务的方法不能塞进同协议主表。

## 9. 一至四卡的队列方案

| 可用 GPU | Pilot 排队 | 通过 G4 后 | 不建议 |
|---|---|---|---|
| 1 | 串行 Base→single/candidates→Router→probe | 最小三 backbone、两数据集、保留 3 seeds，再扩展 | 先铺四数据集全部消融 |
| 2 | 卡0 工程主线；卡1 对照/PEMS04 | 两个 seed 或两个数据集独立跑 | 两卡 DDP 调试吞掉时间 |
| 4 | 卡0 主线；卡1 matched single/无结构；卡2 其他 seed；卡3 probe/第二数据集 | 四作业并行，PEMS07 独占卡，性能测量时整卡独占 | 一卡挤多个竞争显存的训练作业 |

调度器最小功能：按 manifest DAG 执行父 checkpoint 依赖，GPU 互斥锁、明确 CUDA_VISIBLE_DEVICES、run_id 独占输出、退出码/心跳记录、失败保留。不要把所有作业 background 后只看最后一个 exit code。

性能 profiling 不与别的作业共享 GPU。训练 CPU workers 从 4 起步实测；4 卡可先准备 16–32 CPU cores、64–128GB RAM、NVMe 200GB 可用空间，这是资源规划值，实际以数据审计和 IO profile 为准，不是最低要求承诺。

## 10. 时间与 GPU-hour 估算方法

每个关键形状（PEMS08/07、Base/K=8、fp32/bf16）先预热 20 batch，再测至少 100 batch 的训练 step 时间、峰值显存和 eval 时间。排除首轮编译/缓存时间但另记启动成本。

```text
job_gpu_hours = gpu_count * (epochs * (train_batches*t_step + t_val)
                            + t_prepare + t_final_eval) / 3600
total_gpu_hours = sum(job_gpu_hours) + 预留 25% 的重跑/调试预算
ideal_wall_hours = total_gpu_hours / 可并行 GPU 数
planned_wall_hours = ideal_wall_hours / 调度利用率
```

调度利用率可先按 .7 规划，但必须标为假设。示例：若最终量得总预算 400 GPU-h，4 卡且 .7 利用率约 143 wall-h；1 卡则约 571 wall-h。**400 GPU-h 是算例，不是本项目测得的预算。**

80GB 显存只是单卡容量，四卡通常不能视为一个 320GB 自动共享内存池。1 GPU/作业时每作业必须独立装入一张卡。OOM 不能无日志自动降精度或 batch。

## 11. 工作周期（按验收而非固定日历承诺）

- 开发第 1 波：合同、数据与 backbone；B/C/D 用 fixture 并行实现核心/损失/测试。
- 第 2 波：集成、GPU smoke、断点恢复、Base 与 single。
- 第 3 波：候选、固定候选 Router、probe；先决定假设是否值得扩展。
- 第 4 波：G4 冻结后主矩阵、跨 backbone、关键消融。
- 第 5 波：统计、profiling、状态分析、论文图表与复核。

可用 4–6 周作为人员排期草案，不能视为性能/实现工期保证。第一轮 100-step profile 和 G1 完成后重估工期；计算资源长期不可用或核心假设失败时应调整范围。

## 12. 统计分析合同

每个种子结果单独展示，主表 mean±sample std（ddof=1），至少 3 seeds；关键比较有预算再增加至 5。不能只保留最好 seed，也不能把日窗口当作独立模型重复。

预测误差以**预测起点所属连续时间块**聚合，所有 horizon 保留。默认 block=288 起点（一天），paired moving-block bootstrap 10,000 次：相同重采样索引作用于比较两方法，以误差 sum 和 valid count 重新计算指标。不得 bootstrap 已平均的 batch RMSE。

窗口相互重叠且跨天仍有关联，额外 block=576/2016 做敏感性分析；测试天数太少需说明 CI 不稳，不伪造统计功效。seed 变异和时间块变异分别报告；可作 seed+block 两层 bootstrap，但三个 seed 的 CI 仍有限，不能将它解释为覆盖所有随机性。

主比较家族预先固定为十个 dataset/backbone 单元的 Router vs matched single。若做检验与显著性标记，提供配对 block sign-permutation 的双侧 p 值（在 block 对称/近似独立假设下），Holm 校正此家族；假设不满足则仅报告 CI/效应量，不强做显著性星号。bootstrap CI 本身不自动提供合法的 p 值。

主 metric lock、comparison family、block 长度、bootstrap seed、重复数在 test 运行前写入 protocol.lock。无效/不完整 run 显式列出，不静默剔除；所有失败占用 GPU-h 计入成本。

## 13. profiling 协议

同 GPU、同输入、同精度、同图缓存、无训练共跑。预热 100 次，测量 1000 次（规模太大允许固定减少并披露）；CUDA event 或同步后的 wall-time，报告 batch=1 p50/p95 与 batch=64 吞吐、峰值 allocated/reserved memory。

分别测 Base、Router@1、全 K 解码，以及 full-decode learned selector。future-aware oracle selection 的开销可离线测，但不称为可部署服务。

报告模型 device-resident latency 与包含 CPU→GPU/反归一化的 end-to-end latency 两套，缓存 backbone hidden 的实验不算在线推理。低维动作生成虽有 K 维开销仍需计入。

## 14. 结果冻结与论文生成

G4 后产生 protocol.lock，主模型配置只由 train/val 决定。test 评测运行完后若发现工程 bug，递增 protocol 版本并重跑所有受影响方法，保留原错误记录；不能只重跑本方法。

所有主表和图从 experiments.csv 指向的 metrics/prediction artifacts 生成；脚本验证 hash、单位、mask 和 split 是否一致。主图可输出 CSV/PDF/SVG，图注说明 oracle、seed、CI 和缺失项。不能手填或复制旧论文数字。
