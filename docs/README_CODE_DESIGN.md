# 代码开发详细合同 v1.0

本文定义必须实现的行为，而非目前已有的代码。MUST 为验收要求；SHOULD 为推荐；OPTIONAL 不得阻塞主线。源码版本及对应接口见 [依赖登记](../third_party/LOCK.md)。架构变更须记录 ADR，不能在实现中静默改变公式。

## 1. 范围与技术决策

- Python 3.11；PyTorch 2.x。准确 wheel、CUDA runtime、NumPy 等版本由第一轮 Linux/A100 smoke 后写入 lock，不伪造已验证环境。
- 自有 runner；核心依赖控制在 torch、numpy、scipy、PyYAML、数据读取所需包；开发依赖 pytest、ruff、类型检查器。第三方源码的额外依赖单独登记。
- CPU 支持数据/指标/小图单元测试；训练和性能验收在 Linux A100。Mac 测试不能代替 CUDA 验收。
- v1 仅单进程单 GPU 训练；多 GPU 通过独立任务并行。以后增加 DDP 时须处理分布式 sum/count、sampler 重复样本和全局 balance。
- 日志默认本地 JSONL；SwanLab/W&B 为显式可选插件，不作为运行必要条件。
- 代码布局采用 `src/sbona`，避免本地目录遮蔽第三方包。

## 2. 目标目录与职责

```text
src/sbona/
  contracts.py                 # 共享 dataclass / Protocol / schema version
  config.py                    # 严格配置解析、互斥参数、路径解析
  cli.py                       # 所有入口；无业务公式
  data/
    manifest.py                # 文件/节点/时间轴/版本与 sha256
    windows.py                 # 按目标区间划分窗口
    scaler.py                  # 训练数据有效值统计
    graph.py                   # 节点对齐、邻接、归一化、稀疏传播
    dataset.py                 # ModelInput + TargetBatch
  backbones/
    base.py                    # 模式切换、精确参数白名单
    staeformer.py
    graph_wavenet.py
    agcrn.py
    stid.py
  models/
    projector.py
    actions.py
    decoder.py
    router.py
    adapter.py                 # forward_candidates / predict_top1
  objectives/
    constraints.py
    oracle.py
    losses.py
  training/
    stages.py                  # 显式状态机
    trainer.py                 # batch、optimizer、early stop
    checkpoint.py              # 原子保存与恢复
    probe.py                   # 时间留出的可路由性诊断
  evaluation/
    metrics.py                 # 流式 sum/count，不平均 batch RMSE
    oracle.py                  # 仅离线、显式接受 TargetBatch
    diversity.py
    regimes.py
    statistics.py
    profiler.py
    artifacts.py
  baselines/
    single.py
    independent_heads.py
    unstructured.py
    perturbation.py
tests/{unit,integration,gpu}/
configs/{datasets,backbones,experiments}/
scripts/                       # 调度/聚合薄封装
third_party/                   # LOCK、许可证、最小 vendor 源码
outputs/                       # ignored
data/                          # ignored，路径也可在仓外
```

`models` 和 `backbones` MUST 不导入 `objectives`、`training` 或评测 oracle。未来真值只能存在于 TargetBatch，不能藏进 `aux`、`**kwargs`、global 变量或模型 cache。

## 3. 共享类型与信息边界

以下是接口合同，不是可直接运行的完整 Python 文件。A 负责落地，其他任务不得另造同名类型。

```python
@dataclass(frozen=True)
class ModelInput:
    history: Tensor          # float32 [B,L,N,1]，标准化 flow
    observed: Tensor         # bool [B,L,N,1]，原始观测是否有效
    time_of_day: Tensor      # float32 [B,L]， [0,1)
    day_of_week: Tensor      # int64 [B,L]，0..6；无可信日历则禁用

@dataclass(frozen=True)
class TargetBatch:
    values_raw: Tensor       # float32 [B,T,N,1]
    valid: Tensor            # bool [B,T,N,1]
    sample_ids: Tensor       # int64 [B]，未来首步绝对索引

@dataclass(frozen=True)
class GraphContext:
    diffusion: Tensor        # [N,N] sparse COO，包含 self-loop
    edge_index: Tensor       # int64 [2,E]，无自环、去重约束边
    edge_weight: Tensor      # float32 [E]，非负
    graph_hash: str
    node_order_hash: str

@dataclass
class BackboneOutput:
    base_pred_z: Tensor      # [B,T,N,1]
    node_hidden: Tensor      # [B,N,D_b]

@dataclass
class ActionOutput:
    physical: Tensor         # [B,K,5]
    normalized: Tensor       # [B,K,5]，各维映射到 [0,1]
    mu: Tensor               # [B,5]
    sigma: Tensor            # [B,5]

@dataclass
class CandidateOutput:
    pred_z: Tensor           # [B,K,T,N,1]
    actions: ActionOutput
    h_global: Tensor         # [B,D]
    base_pred_z: Tensor      # [B,T,N,1]

@dataclass
class Top1Output:
    pred_z: Tensor           # [B,T,N,1]
    index: Tensor            # int64 [B]
    logits: Tensor           # [B,K]
    selected_action: Tensor  # [B,5]

@dataclass
class OracleOutput:
    error: Tensor            # [B,K]，保留候选梯度
    consistency: Tensor      # [B,K]，保留候选梯度
    cost: Tensor             # [B,K]，已 detach
    soft_target: Tensor      # [B,K]，已 detach
    best_index: Tensor       # [B]，argmin cost
    valid_sample: Tensor     # bool [B]
```

Batch 元信息（split、绝对时间）留在 loader/trainer，不通过不透明 `dict` 传入模型。`ModelInput.observed` v1 仅供输入检查/缺失处理，所有 backbone 均不把它作为额外预测特征；若启用 mask 特征，必须作为统一新协议。

公开函数合同：

```python
BackboneAdapter.encode_and_predict(inp: ModelInput, graph: GraphContext) -> BackboneOutput
FeatureProjector.forward(hidden: Tensor) -> tuple[Tensor, Tensor]  # h_node,h_global
ActionGenerator.forward(h_global: Tensor) -> ActionOutput
Decoder.forward_all(h_node: Tensor, actions: Tensor, graph: GraphContext) -> Tensor
Decoder.forward_selected(h_node: Tensor, selected: Tensor, graph: GraphContext) -> Tensor
Router.forward(h_global: Tensor) -> Tensor  # logits
SBoN.forward_candidates(inp: ModelInput, graph: GraphContext) -> CandidateOutput
SBoN.predict_top1(inp: ModelInput, graph: GraphContext) -> Top1Output
build_oracle(out: CandidateOutput, target: TargetBatch, stats, graph, config) -> OracleOutput
```

`predict_top1` MUST 没有 y/target 参数，也不接受任意 kwargs。所有模块的 device 跟随输入或 registered buffers，禁止 `.cuda()` 硬编码。K、N、B 不能写死。断言通道数、形状、节点顺序，拒绝隐式广播错误。

## 4. 数据与时间切分

### 4.1 数据 manifest

每个数据集记录：原始来源、许可/使用条件、每个文件 sha256、真实 shape、flow 通道、单位、采样间隔、起止索引、时间戳来源、节点 ID 顺序、邻接来源、缺失定义及审计统计。

旧仓已下载 PEMS 可只读引用，但必须重新计算 hash；已有 `.npz`、index 和 scaler 不能未经检查整体继承。不要把“名字叫 PEMS08”当作数据版本证明。

默认数据集 PEMS03/04/07/08；节点数与时间点数从实际文件审计生成，不以 README 硬编码代替检查。没有可靠时间戳时允许相对 time-of-day，但 weekday 特征及周末分析 MUST 禁用，不能猜起始星期。

### 4.2 唯一切分规则

长度 M 的原始时间序列，`a=floor(0.6*M)`，`b=floor(0.8*M)`。目标区间分别是 `[0,a)`、`[a,b)`、`[b,M)`。窗口未来起点 s，历史 `[s-L,s)`，目标 `[s,s+T)`，L=T=12。

一个窗口的**整个目标区间**必须落在一个 split 内，且 s>=L。验证/测试历史可以使用此前已经发生的观测，包括前一 split 尾部；训练目标不能跨入验证。不同 split 的目标时间索引集合必须不相交。

无需把相邻测试窗口当作独立样本做显著性检验。额外提供 strict-block 边界敏感性实验时另起协议 ID，不混入默认表。

### 4.3 缺失与零值

优先采用发布方有效值 mask。未说明时，NaN/Inf/明确 sentinel 判无效；零流量默认有效，不能未经证据将真实零流量当缺失。若官方 benchmark 使用 `Y!=0`，另行登记 `official_compat` 协议并全方法一致运行。

历史无效 flow 使用**训练集全局均值**填充，标准化后为零，不利用未来插值。保存原始 mask。scaler 统计不包含填充值。目标无效值在做算术前 `where` 安全替换，不能依赖 `NaN * 0 = 0`。

全无效目标样本不参与监督损失、router、balance 和分离项的 batch 聚合；全 batch 无有效样本则跳过 optimizer/scheduler 更新并计数，禁止当成零误差好样本。metric 分母为零返回 JSON null 和原因，不返回 NaN 或 0。

### 4.4 唯一尺度约定

v1 对训练区间 `[0,a)` 所有有效 flow 拟合**全局标量** μ、s，s=max(std,1e-6)，使用 population std。不是每节点 scaler，避免图差分单位不一致。

- `history`、`base_pred_z`、`residual_z`、`pred_z` 全为 z=(y-μ)/s。
- 逆变换 `y_raw=s*z+μ` 保持可微。
- 候选误差 E 在 z 空间为 masked MSE；等价于 raw MSE / s²。
- 一致性先在 raw 空间判违反，再除 s 或 s² 转为无量纲（见第 8 节）。
- 所有论文 MAE/RMSE/MAPE 在 raw 空间计算。

每节点经验上包络 C_n 为训练有效 raw flow 的 0.995 分位数，使用 linear quantile。有效样本>=1000 用节点值；不足则使用训练全局分位数并记录 fallback 节点清单。完全无训练观测的节点仍保留，但报告数量。

## 5. 图合同

v1 使用数据发布方道路连接关系，二值拓扑。按 manifest 的节点 ID 映射，不默认 CSV 行号等于节点序号。默认无向：每条道路加入双向边、去重；距离权重、方向传播属于另行消融，不能在一部分数据集偷偷换图定义。

令 W 为无自环二值邻接，`A=D^-1(W+I)`，每行和为 1。`A[i,j]` 表示节点 i 聚合节点 j。孤立点添加自环后保持自身。约束边使用无向边一次计数 `(i<j)`，不含 self-loop，权重 1。

图扩散为 `S0=S; S_{r+1}=A*S_r`，使用稀疏 COO 和 `torch.sparse.mm` 对 `[N,B*D_r]` 传播再 reshape；不显式创建 `A^2/A^3`，不创建 `[B,K,T,N,N]`。sparse 运算使用 fp32，避免 AMP 支持差异。

没有道路图时：数据准备阶段报错。允许另起 `graph=identity` 诊断 run，但不得默默作为正式图实验。Backbone 自己的 learned adjacency 保持原算法，SBoN 外接道路图是额外信息，所有残差对照获得相同图；Base 与图增强方法的信息差异必须披露。

## 6. Backbone 接入与模式

官方源码在 `third_party/LOCK.md` 指定版本内做最小 adapter，禁止从旧目录未提交的随机生成头开始。v1 先 STAEformer，再 STID、Graph WaveNet、AGCRN。每个 adapter 要有 original-forward 与 adapted-forward 的 raw/base 等价测试。

### 6.1 STAEformer 精确接口

已检查本地 pristine Git commit `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f` 的 `model/STAEformer.py`；在线固定版本若有差异，以 LOCK 与显式 diff 为准。

- `input_dim=1`，构造 `[flow_z,tod,dow]` 输入；flow 投影只读第 0 通道，日历进入原 embedding。
- 依序经过 `attn_layers_t`、`attn_layers_s` 后得到 `[B,L,N,model_dim]`，取 `x[:, -1, :, :]` 为 node_hidden。
- 基础输出完整沿用原 `use_mixed_proj` 路径：transpose/flatten 后 `output_proj`。提取 hidden 不能改变原 forward 的基础预测。
- v1 固定 `use_mixed_proj=True`；每数据集模型维度与层数来自锁定配置。
- Partial 白名单：`attn_layers_s[-1]` 与 `output_proj`。其余被冻结，固定 dropout；不要把所有 normalization/bias 默认解冻。
- 不在第二次 forward 中重算 hidden，避免算两遍 backbone。

其他 backbone 的准确提取点由 [BACKBONE_CONTRACTS.md](../third_party/BACKBONE_CONTRACTS.md) 固定；对应任务必须核对源文件并通过等价测试后才能标为支持。

### 6.2 Frozen 的含义

所有 backbone 参数 `requires_grad=False`，backbone 始终 eval（包括父模块调用 train 后）；dropout 关闭、BN running stats 不变；编码可用 no_grad。projector 仍可训练。

Partial/Full 则不能包住整个 backbone `no_grad`；Partial 的冻结子模块保持 eval，白名单模块 train。每个 run 输出 trainable 参数名和数量。优化器不包含冻结参数。

不同训练阶段恢复后必须重新应用 mode 规则。不能只在构造函数里冻结一次。

## 7. 模型数学定义与默认结构

### 7.1 投影器与 pooling

`h_node = Linear(D, D)(GELU(Linear(D_b,D)(LayerNorm(hidden))))`，D=64；v1 projector dropout=0。无残差旁路。`h_global=h_node.mean(dim=1)`。不使用未定义的 attention pooling，降低实现差异。

### 7.2 动作生成

两层 MLP `D→64→10`，GELU；前五维为 μ，后五维经 `0.05+softplus` 为 σ。每个候选有 learnable anchor `[K,5]`，固定 seed 的 scrambled Sobol 在 [-1,1] 初始化；禁止全部为零。

`z_bk=mu_b+sigma_b*anchor_k`；各维转换：

| 维度 | 范围 | 映射 | 单位 |
|---|---|---|---|
| o | [0,T-1] | (T-1)*sigmoid(z0) | 预测步 |
| δ | [0.25,3] | .25+2.75*sigmoid(z1) | 步/hop |
| m | [-3,3] | 3*tanh(z2) | z 残差倍数 |
| d | [1,T] | 1+(T-1)*sigmoid(z3) | 步 |
| ρ | [.01,1] | .01+.99*sigmoid(z4) | 每步衰减率 |

normalized 动作按 `(physical-min)/(max-min)` 映射。T=1 时 o 固定为 0，normalized o 固定 0，避免零除。K=1 仍训练动作生成器与唯一 anchor；不是固定中心动作。

### 7.3 无旁路结构解码器 v1

默认 R=3，D_r=32，`S=Linear(64,32)(GELU(Linear(64,32)(h_node)))`。稀疏传播生成 `[B,R+1,N,D_r]` 的 S_r，所有候选共享。输出层 `W_out:32→1`，**bias=False**。

时间 t=0..T-1。固定平滑尺度 `s_p=1.0, s_o=0.5, s_d=0.5`，单位均为步，不设为可学习变量。

```text
q[b,k,t,r] = -((t-o[b,k])-r*delta[b,k])² / (2*s_p²)
pi = softmax(q, dim=r)
e[b,k,t] = sigmoid((t-o)/s_o)
           * sigmoid((o+d-t)/s_d)
           * exp(-rho * relu(t-o-d))
V[b,k,t,n,:] = sum_r pi[b,k,t,r] * S_r[b,r,n,:]
u[b,k,t,n,1] = m[b,k] * e[b,k,t] * W_out(V[b,k,t,n,:])
residual_z = r_scale * tanh(u/r_scale), r_scale=3.0
pred_z = base_pred_z[:,None] + residual_z
```

无 FiLM、无 time embedding、无自由 local branch。这样 m=0 必须严格产生零残差，幅度不能被 bias 绕过。d/ρ 对有限预测窗可能弱可辨识，因此语义通过干预测试，不预先承诺。

训练候选与选中候选共用同一个数学 kernel；selected 路径只增加长度 1 的候选轴再 squeeze，不得先构造 K 份轨迹。单次 selected 允许共享图基准备，但不能调用 forward_all 后 gather。

W_out 默认 N(0,1e-3) 小随机初始化，基础输出变化应小但非严格零。旧稿零初始化会使第一次反传的上游梯度为零；`zero_init` 仅作为消融，测试需要允许第一步退化，不能拿“所有参数首步非零梯度”做不合理断言。

残差 saturation 比例 `abs(u)>3*r_scale` 必须记录。后续扩展若加入 local branch，必须有单独 method_id、显式权重上界、无旁路消融，禁止直接混进主方法。

### 7.4 Router

`LayerNorm(64) → Linear(64,128) → GELU → Dropout(.1) → Linear(128,K)`，输出 logits，loss 使用 fp32 log_softmax。v1 固定 alpha 的独立训练，不接受推理临时 alpha；偏好条件化是另一个实验版本。

Router 输入固定候选模型的 `h_global.detach()`，使路由损失只更新 Router。推理取 argmax，平局选最小索引。随机对照独立 RNG，不改变模型 RNG 状态。

## 8. 误差与一致性代价

M_b 为目标有效 mask；按每样本有效元素数规约，得到 `[B,K]`。记 raw 预测 yhat，训练标量 std 为 s。

```text
E = masked_mean(((yhat-y)/s)²)
P_neg = masked_mean((relu(-yhat)/s)²)
P_env = masked_mean((relu(yhat-C_node)/s)²)
P_spa = weighted_mean_edges( Huber((yhat_u-yhat_v)/s, delta=1) )
P = P_neg + P_env + 0.1*P_spa
```

Huber 精确定义：|x|<=1 时 .5*x²，否则 |x|-.5。空间项仅取两个端点目标均有效的时间/边，分母为对应有效边权之和；无有效边时 `P_spa=0` 且记录 count=0。非负、上包络和图变化使用同一有效目标支持集，保证可比较。

在部署中无需计算训练代价；Y_mask 不属于 Router 输入。图变化对于非恒定合法流量一般不为零，测试只验证已知值和增加差分时的响应，不要求所有“合法”预测图惩罚为零。

## 9. Oracle、soft-WTA 与精确梯度合同

### 9.1 组内归一化

对 E 和 P **先 detach**，沿 K 使用 population std（correction=0），`Z(v)=(v-mean(v))/max(std(v),1e-3)`。这叫普通 z-score，不叫 robust z-score。K=1 或全相等时 Z=0。

`C=alpha*Z(E)+(1-alpha)*Z(P)`，`w=softmax(-C/tau)`。默认 alpha=1.0、tau=.5，先验证准确性主线；alpha=.8、lambda_consistency=.1 为一致性扩展。alpha=1 不强制其他 loss 为零，准确性-only 配置须同时设 lambda_consistency=0。

全无效样本保留占位 C=0、w=uniform，`valid_sample=False`，聚合时剔除。所有组内归一化/softmax/loss reductions 使用 fp32。

### 9.2 Candidate loss

```text
w_tilde = (1-epsilon)*w.detach() + epsilon/K, epsilon=.05
L_cand = mean_valid_b(sum_k(w_tilde * (E + lambda_consistency*P)))
L_sep = mean_valid_b(mean_i<j(relu(.25 - norm(a_i-a_j,2))²))
L_generator = L_cand + .01*L_sep
```

`a` 为 normalized 动作。K=1 时 L_sep 为标量 0。明确是**候选维求和、样本维均值**，不要再乘 K 或再平均 K。权重非负、每样本总和为 1。

优化器 AdamW(weight_decay=1e-4)；不要同时再加手写 L2 惩罚，以免重复 weight decay。候选模型冻结 backbone 时没有 base loss；Full/Partial 可另加 `.2*base_MSE`，属于独立实验。

### 9.3 Router loss

```text
logp = log_softmax(logits.float(), -1)
L_KL = mean_valid_b(sum_k(w*(log(clamp(w,1e-12))-logp)))
L_reg = mean_valid_b(sum_k(exp(logp)*(C-min(C))))
u = mean_valid_b(exp(logp))
L_bal = sum_k((u-1/K)²)
L_router = L_KL + .2*L_reg + beta(epoch)*L_bal
```

beta 最大 .01，前 10% 最大 epoch 线性升到 .01，10%–50% 固定，50%–100% 线性降到 .002。balance 调的是 router soft probability，不保证真实轨迹多样性，不能据此宣称防止 generator 坍缩。

K=1 不训练 Router，返回 logits=0/index=0。所有 route targets/cost/hidden detach；路由训练 optimizer 仅包含 Router 参数。

| loss | backbone | projector | actions | decoder | router |
|---|---|---|---|---|---|
| candidate（Frozen） | 否 | 是 | 是 | 是 | 否 |
| separation | 否 | 是 | 是 | 否 | 否 |
| KL/regret/balance 主设置 | 否 | 否 | 否 | 否 | 是 |
| Full candidate 扩展 | 是 | 是 | 是 | 是 | 否 |

“每个共享 decoder 参数都有非零梯度”不是 loser floor 的正确测试。应直接对独立人工候选张量验证有非零误差时各候选获得正确方向梯度，并测试真实模型累计多个 step 后参数更新。

## 10. 训练状态机与 checkpoint 选择

| Stage | 可训练模块 | 初始状态 | 最佳 checkpoint 的验证指标 |
|---|---|---|---|
| A base | backbone | 官方架构随机初始化 | raw MAE |
| B single | projector/actions/decoder，K=1 | A best，独立初始化 adapter | single raw MAE |
| C candidates | projector/actions/decoder，K=8 | 同一个 A best，独立初始化 adapter | alpha=1 用 MAE-oracle；alpha<1 用原始无量纲 E+lambda*P 最小候选的平均成本 |
| P probe | probe classifier/regressor | 专门时间切分的固定候选 | 仅训练内留出集；见实验文档 |
| D router | router | C best 全部冻结 | Router@1 raw MAE |
| E joint（可选） | 指定模块 | C/D best | Router@1 raw MAE |

Stage B 不是 Stage C 的权重 warm-start，避免 K=1 对称初始化造成分支一致。B 与 C 从同一个 Base checkpoint 开始，公平记录各自预算。E 不属于默认方法；要实现时另写梯度 ADR，特别说明共享 projector 是否接收路由梯度。

C 的中间 oracle 验证用于候选 checkpoint 选择，不能表述为模型从未见过验证标签；最终独立 test 一次性评估冻结的完整流程。

默认上限：Base 100 epoch，single/candidates 80，router 40；patience 分别 15/15/10；同一指标最小 improvement 1e-5，平局保留较早 checkpoint。LR base=1e-3、adapter=1e-3、router=3e-4（是预设起点，验证调参结果必须另存）；gradient clip=5；默认 constant LR，减少恢复状态歧义。

训练 batch 初始 32，GPU pilot 决定最终锁定 batch。若 gradient accumulation，按有效样本数加权，不平均不同大小 microbatch 的均值；batch balance 在所有 microbatch 的逻辑 batch 聚合，做不到则明确改变协议，v1 默认 accumulation=1。

每个 stage checkpoint 保存：schema_version、stage、所有模型 state_dict、optimizer、scheduler（若存在）、AMP scaler（若存在）、epoch/global_step、best_metric、patience_counter、Python/NumPy/Torch CPU/CUDA RNG、sampler RNG、resolved_config、数据/图/scaler/split hash、上游锁 hash、父 checkpoint hash。

checkpoint 临时文件写完后原子 rename，保存 latest 与 best。默认只支持 epoch 边界恢复；中途异常从上一 epoch checkpoint 恢复，不声称逐 batch 精确续训。改变 K/N/图/split/scaler/架构时必须拒绝 resume；warm-start 用独立参数并生成新 run_id。

自动发现 checkpoint（不显式给 `--*-checkpoint` / `--*-run`）只能按**身份哈希**匹配，不能按 mtime 或目录名猜测：backbone 身份 = seed + protocol + backbone 架构；adapter 身份 = seed + protocol + method_id + adapter 参数 + K。Stage A 只用 backbone 身份（Base 必须能被同 seed 的所有 adapter 阶段复用），B/C/D/P 还要求 adapter 身份一致。同一身份匹配到多个 run 时必须报错并列出 run 名，由人决定保留哪个，禁止“取最新”。显式传入 `--*-checkpoint` / `--*-run` 也不能绕过身份校验：Stage B/C 拒绝外来 Base，Stage D 拒绝外来候选，`evaluate` 在读取 checkpoint 后重新核对身份。

Frozen 特征缓存 OPTIONAL：key 必须含 split/sample_id/backbone checkpoint/graph/scaler/feature schema；只有 eval 的冻结模块可缓存。投影器还训练时不能缓存投影后的 hidden。缓存速度不计入在线 inference latency。

## 11. 推理路径

```python
def predict_top1(inp, graph):
    bb = backbone.encode_and_predict(inp, graph)
    h_node, h_global = projector(bb.node_hidden)
    actions = action_generator(h_global).physical
    logits = router(h_global)
    idx = logits.argmax(-1)
    selected = actions[torch.arange(actions.size(0), device=actions.device), idx]
    residual = decoder.forward_selected(h_node, selected, graph)
    return Top1Output(bb.base_pred_z + residual, idx, logits, selected)
```

上层 evaluate 显式调用 model.eval 和 inference_mode。结构路径里不读取 TargetBatch。Top1 和离线 oracle 的入口、日志、表格列名分开。主指标使用未裁剪预测；非负裁剪可以另列 postprocess 版本，必须对所有方法一致应用。

## 12. 指标与 oracle 的无歧义定义

### 12.1 预测误差

全窗 MAE = 所有有效预测元素绝对误差之和 / 有效元素数；RMSE = sqrt(平方误差总和/有效元素数)。不是平均每个 batch 或每个 horizon 的 RMSE。单 horizon 3/6/12 指第 3/6/12 步，不是前 3/6/12 步累计。

MAPE 有效条件：target.valid 且 abs(Y_raw)>=1（单位为每采样间隔 flow，阈值入配置）。值为 `100*mean(abs(error)/abs(Y))`，报告百分数与有效覆盖率。零值仍参与 MAE/RMSE。禁止从预测值决定 mask。

### 12.2 Oracle 选择粒度

每个样本选择**一条覆盖整个 T×N 的候选**，不是每节点/每 horizon/每指标分别拼接。

- oracle_mae：按样本 masked raw MAE 选 k，再用选出的完整预测算所有指标。
- oracle_mse：按样本 masked raw MSE 选 k；对同一候选集可给 RMSE 最佳选择。
- oracle_cost：按训练 C 选 k，不是 MAE 或 RMSE 严格下界。

主文分别展示，不能在同一条“oracle 轨迹”上按指标偷换候选。若需要各指标独立最优下界，用不同命名。

`MAE retention=(MAE_base-MAE_router)/(MAE_base-MAE_oracle_mae)`。分母<=1e-6 raw 单位时输出 null，记录 numerator/denominator/reason；不加 epsilon 制造巨大比值，也不截断到 [0,1]。single-reference retention 可另报但名字区分。

cost regret 使用相同完整 K 候选组的 C；accuracy regret 使用 MAE_router-MAE_oracle_mae。K 子集分析不能跨不同 z-score 标尺混合成本。top-1 tie-aware 统计将与最优成本相差<=1e-6 的候选视为等价；同时报告固定 argmin 标签版本。

### 12.3 候选健康与一致性

分别报告 oracle_cost/oracle_mae/router 的硬使用率、soft probability 均值、熵、exp(entropy)、最大份额。所有候选接近同一轨迹时即使 soft 均匀也可能坍缩。

轨迹距离在有效元素上计算 raw L1 与 /s 的标准化 L1；动作距离用 normalized 动作。记录每分支误差与幅度饱和率，不能只看候选间距离。

NVR 为 yhat<0 的有效元素比例，EUVR 为 yhat>C_n 比例；各自还报告违反幅度均值（含所有有效元素）及条件均值（仅违反元素）；无违反时条件均值为 null。图变化报告同一有效边支持下预测与真值的 Huber TV，真值为 0 时 ratio 为 null。

## 13. 结果、日志与错误处理

每个 run 必须保存：

```text
resolved_config.yaml
manifest.json                # 数据/图/源码/环境/上游版本与状态
train.jsonl                  # epoch、stage、split、sum/count、loss、time、memory
metrics.json                 # schema_version、protocol_id、metric units、counts
checkpoint_best.pt
checkpoint_latest.pt
profile.json
failure.json                 # 仅异常时，包含阶段、错误、最后 checkpoint
```

predictions 以 sample_id 顺序分 shard 存 `[S,T,N,1]` 和有效 mask；候选全集默认不保存，可重算。Top1 logits/index/action 可以独立小文件保存。只存必要 subset 的候选，避免数十 GB 无限制堆积。

JSON 使用 null 表达无定义项，`allow_nan=False`。manifest 包括 GPU 型号/数量/驱动、torch/CUDA、precision、TF32、deterministic、git commit、dirty flag 和 dirty diff hash。正式主表只接受 clean commit；开发 run 可 dirty 但不可冒充正式结果。

OOM/NaN/Inf 立即把 run 标为 failed 并保存诊断，不自动减 batch 或吞掉错误继续。改变 batch/precision 后是新的 run，不能续写旧成功记录。缺 key、未知配置字段、文件不存在和维度错配都 fail-fast。

## 14. 配置 schema

参考 [design_defaults.yaml](../configs/design_defaults.yaml) 是设计默认值，不是已能直接执行的实验配置。实现需提供严格 schema，禁止未知 key，分离逻辑配置与本机路径。路径优先 `--data-root`/`--output-root`，其次环境变量，绝不能硬编码旧服务器 `/root/Flow-RL`。

必须校验 K>=1、tau>0、0<=epsilon<=1、0<=alpha<=1、动作范围、有效 horizon、graph node count、数据 hash、backbone checkpoint 匹配。`alpha<1` 且 lambda_consistency=0 可以作为特定排序消融，但必须显式 method_id，不属于默认一致性方法。

配置/输出 schema 升级必须增加版本号；旧 checkpoint 迁移不能默认 strict=False 加载。每个模块要有类型注解和关于轴/尺度的 docstring，不要求机械地为简单 getter 写测试。

## 15. 性能预算与实现细节

候选输出内存为 B*K*T*N*4 bytes（fp32 单通道），并非全部训练显存；主要激活可能是 B*K*T*N*D_r。实现优先先将每 hop S_r 线性映射为单通道再与 pi 融合，因为 W_out 线性且无 bias，数学等价，避免巨大 D_r 中间张量。

selected 路径不得产生带 K 的高维时间节点张量；允许 `[B,K,5]`、`[B,K]`。candidate chunking 可选，必须保证与完整 K 的归一化、soft target 和梯度等价，不能逐 chunk 分别计算 softmax。

v1 默认 fp32；完成数值对照后可锁定 bf16 autocast，sparse ops、oracle、loss reductions 仍 fp32。不同方法性能比较 precision/TF32 设置完全一致。不能声称 A100 80GB 必然可容纳指定 batch，先 profile PEMS07。

完整测试与最终完成标准见 [审查文档](README_REVIEW.md)。
