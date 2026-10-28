# Backbone 接口核验记录

更新日期：2026-10-29（上一版 2026-10-28，只覆盖 STAEformer/Graph WaveNet/AGCRN/STID 的源码审读）。

本文件记录九个 backbone 的**接入合同**：提取哪一层作为 node hidden、预测如何回到 `[B,T,N,1]`、图的方向约定、日历输入、Partial 白名单、与官方的差异以及验证状态。commit、许可证与被提取文件 SHA256 见 [LOCK.md](LOCK.md) 与 [source_manifest.json](source_manifest.json)。

## 0. 公共输入与输出

所有方法获得同一份历史 flow 与可信历史日历，**不使用未来实际观测、不使用 targets**。

- `ModelInput.history`：`[B,L,N,1]`，已按 train-only scaler 标准化（`flow_z`）。
- `ModelInput.time_of_day`：`[B,L]`，`[0,1)`。
- `ModelInput.day_of_week`：`[B,L]`，整数 `0..6`（Monday=0）。**不是** BasicTS 的 `dow/7` 分数。
- 统一输出：`(node_hidden [B,N,D], base_prediction [B,T,N,1])`，两者来自**一次** forward。

日历通道的差异由 `sbona.backbones.common.FEATURE_SPECS` 显式声明（`flow / tod / dow_index / dow_fraction / dow_onehot`），没有一个“共享约定”被静默套用；传错不会报错，只会学错 embedding，所以每个 adapter 必须自己声明。

`node_hidden` 的宽度（`BackboneWrapper.node_feature_dim`）是 projector 的输入维度，必须与 forward 返回的 hidden 一致；`tests/unit/test_backbones.py::test_forward_contract` 对九个模型逐一断言。

## 1. 统一的“原生 vs adapter 等价”口径

原生 forward 一致性指：**相同输入配置、相同权重、相同精度、相同补丁基线**下的比较。改变输入维度不能拿另一份官方配置做逐元素等价检查。

本轮**没有**做逐元素等价测试（没有加载官方权重）。已完成的是：CPU 前向/反向、形状/有限性、Frozen-Partial-Full 梯度行为、图依赖敏感性。**不能**据此声称与官方数值等价。

## 2. 逐模型

### 2.1 STAEformer

- 源码：`model/STAEformer.py` @ `fc49d39b…`（无 LICENSE）。
- Hidden：最后一个 temporal+spatial attention 块之后，取最后历史步 → `[B,N,model_dim]`；预测沿用官方 mixed projection → `[B,T,N,1]`。
- Partial：最后 `partial_unfreeze_last_n_spatial` 个空间 attention 层 + 输出投影。
- **`input_dim`：官方值为 3**（`nn.Linear(3, input_embedding_dim)` 投影 `[flow, tod, dow]`）。旧实现强制 `input_dim=1`，与本仓“按官方仓库实现”的要求冲突；本轮放宽为 `1` 或 `3`，dataclass 默认与全部实验配置改为 3。`use_mixed_proj` 仍必须为 `True`。
- **dow 约定**：官方 BasicTS 传 `dow/7`，模型再 `dow_embedding(dow.long())` —— 对任何一天的分数都截断成 0，等价于永远索引星期 0。本仓保留官方**分数**进入投影（`input_dim=3` 的第三通道），但 weekday embedding 用整数 `round(dow*7)` 索引。这是对官方 bug 的修正，不是输入扩展，已在 `docs/README_CODE_DESIGN.md` 记录。
- 没有可信 weekday 时禁用 weekday embedding（`dow_embedding_dim=0`），且必须重训对应 Base。

### 2.2 Graph WaveNet

- 源码：`model.py` @ `6b162e80…`（MIT）。
- Hidden：skip 汇总经过 `end_conv_1` + ReLU 之后、`end_conv_2` 之前 → `[B,end_channels,N,1]` → `[B,N,end_channels]`。预测取 `end_conv_2` 输出 `[B,T,N,1]`。
- Partial：两个末端输出卷积（`end_conv_1`、`end_conv_2`）。
- **补丁**：上游 `gate_convs` 声明为 `Conv1d` 却接收 4-D `[B,C,N,L]`，现代 PyTorch 直接报错；官方 BasicTS port 也用 `Conv2d`。本仓按等价 `Conv2d` 实现并记录，不当作方法改进。
- **图方向**：`nconv` 是 `einsum('ncvl,vw->ncwl')`，即 `out[v]=Σ_w A[v,w]x[w]`；契约的 `A[i,j]` 表示“行 i 聚合列 j”，所以 adapter 喂 `(A.T, A)` 这对 `doubletransition`。
- **输入**：官方 `in_dim=2`（flow, tod）；本仓保持，不塞 weekday。
- 上游未使用的 `residual_convs`（通道数不变时永远不走）与最后一个 extra block 保持原样，`tests/unit/test_backbones.py` 把它们显式列为“允许无梯度”。

### 2.3 AGCRN

- 源码：`model/AGCRN.py`、`AGCRNCell.py`、`AGCN.py` @ `7fbbf2ae…`（MIT）。
- Hidden：encoder 最后一步 `[B,N,rnn_units]`；预测沿官方 `end_conv` + reshape/permute → `[B,T,N,1]`。
- Partial：`end_conv`（官方 head）。`node_embeddings` 默认冻结。
- **自适应图**：AGCRN 不读道路图，`softmax(relu(E Eᵀ))` 的 Chebyshev 支撑每次调用重算，不做缓存（缓存会改变 autograd 图）。
- 官方 forward 签名含 `targets`，但计算不使用；wrapper 只接受 `ModelInput`。
- hidden 初始化的 device/dtype 跟随输入。
- 官方 PEMS08 配置：`input_dim=1`（flow only）、`embed_dim=2`、`rnn_units=64`、`num_layers=2`、`cheb_order=2` —— 即 `backbone.params` 的默认值。

### 2.4 STID

- 源码：`stid/arch/stid_arch.py`、`stid/arch/mlp.py` @ `e8b313bc…`（Apache-2.0），并与 BasicTS v0.5.8 的 port 对照。
- Hidden：`regression_layer` 之前的 `[B,D,N,1]` → `[B,N,D]`；预测保留 `[B,T,N,1]`。
- Partial：`encoder` + `regression_layer`。
- **无空间路径**：不读图（`learned all-to-all` 也没有，只有 node identity embedding），是“空间效应归因于 decoder”的对照。
- **日历**：官方 `time_in_day_emb[(tod*288)]`、`day_in_week_emb[(dow*7)]` 中的 `dow` 是 BasicTS 的分数；本仓直接用整数星期索引。embedding 仍是 7 行。
- 上游把索引转 CPU LongTensor 的写法改为 device-preserving long（兼容补丁）。

### 2.5 D2STGNN

- 源码：`models/` 全目录 @ `82c2d38b…`（**无 LICENSE**）。
- 结构：解耦层把信号拆成 diffusion 分量（动态 + 静态隐图）与 inherent 分量（GRU + multi-head self-attention），各自有 forecast 分支；分支输出 `T/gap` 个状态再展开 `gap` 步。
- Hidden：最后一层 decouple 的 forecast 输入 → `[B,N,256]`。
- Partial：`out_fc_1`、`out_fc_2`。
- **`use_pre=False`**：上游在 `D2STGNN.__init__` 里硬编码 `model_args['use_pre'] = False`，道路图只通过动态图 mask 进入，支撑数为 `(len(predefined)+1)*k_s+1`。本仓照抄该行为，不提供“打开 predefined”的开关（打开会与官方不一致且需要重新核对支撑数）。
- **图方向**：`STLocalizedConv` 期望 `out[v]=Σ_w A[v,w]x[w]`，adapter 喂 `(A.T, A)`。
- 上游未使用的 backcast 分支与 distance function 的 time-slot embedding 保持原样并列入“允许无梯度”。

### 2.6 PDFormer

- 源码：`libcity/model/traffic_flow_prediction/PDFormer.py` @ `f8c8f6ad…`（MIT）。
- Hidden：最后 encoder block 的输出表示（`skip_dim`）→ `[B,N,D]`；预测经 `end_conv1/end_conv2`。
- Partial：最后一个 encoder block + `end_conv2`。
- **`lap_mx`（Laplacian PE）**：从道路图精确计算，忠实。
- **`sem_mask` / `pattern_keys`（DTW 聚类 + 原型模式）**：需要完整训练序列，adapter 看不到。本仓**禁用语义 mask**（语义头 dense attention，不编造先验），`pattern_keys` 改为可学习 per-node 参数、正弦初始化。两项都在 `reviews/EXTRA_WORK.md` 记录，不是静默替换。
- 日历：官方消费 `[value, time_in_day, one_hot(weekday,7)]`（`feature_dim=1` + `ext_dim=8`）；`time_in_day` 转 1440 分钟索引。

### 2.7 HimNet

- 源码：`model/HimNet.py` @ `2a9aa44d…`（**无 LICENSE**）。
- 结构：两个 GCRU encoder，卷积权重分别由 node embedding（`meta_axis="S"`）与 timestamp embedding（`meta_axis="T"`）生成，相加后构造样本级自适应图，自回归解码。
- Hidden：最后 `HimGCN` 输出 → `[B,N,96]`；预测 `[B,T,N,1]`。
- Partial：`decoder` + `out_proj`。
- **参数初始化 bug**：上游 `nn.init.xavier_normal_(nn.Parameter(...))` 返回 `None`，导致 `weights_pool`/`bias_pool`/`node_embedding` 被静默设为 `None`。本仓正确注册参数并 in-place 初始化。
- **未来协变量**：官方 decoder 输入 `[y_prev, y_cov]` 需要未来 tod/dow。这是确定性日历值，adapter 用 `future_calendar` 外推，**不读 target**。
- **teacher forcing 关闭**：喂 `labels[:,t]` 会把目标泄漏进前向，契约禁止。
- **不读道路图**：支撑从 node embedding 学出来，属自适应图对照。

### 2.8 ST-SSDL（NeurIPS 2025）

- 源码：`model_STSSDL/STSSDL.py` @ `a98331d3…`（MIT）。
- 结构：AGCRN 式 encoder-decoder + 原型自监督分支；encoder 摘要匹配原型，原型与摘要驱动 hyper-network 生成样本级图，decoder 沿该图前推。
- Hidden：decoder 最后状态 → `[B,N,decoder_dim]`；预测 `[B,T,N,1]`。
- Partial：`decoder`、`proj`、`hypernet`。
- **对比损失不返回**：SBoN 自带目标函数，`latent_dis`/`prototype_dis` 只计算不进入 loss。
- **`x_his = x`**：官方还编码一个“上一周期参考序列”用于对比项；adapter 只有当前 history，故令 `x_his = x`。预测路径不受影响（`h_a`/`v_a` 只与 `h_t`/`v_t` 一起进 hyper-network）。
- **`y_cov`**（未来 tod）同样用 `future_calendar` 外推。
- **图归一化**：官方用对称归一化，本仓喂契约的行归一化 `A`；方向与 `AGCN` 的 `einsum("nm,bmc->bnc")` 一致。属已知差异。
- 注：`AGCN` 的直接邻接乘法意味着改变道路图必须改变预测；`tests/unit/test_backbones.py` 对此有断言。

### 2.9 HyperD（AAAI 2026）

- 源码：`baselines/HyperD/arch/{hyperd_arch,STFE,attention,loss}.py` @ `5bd93a82…`（**无 LICENSE**）。
- 结构：`y = stfe(x - S_in) + S_out`，`S` 是“混合周期模式”：可学习日表 + 可学习周表，各自在图上前推并做轴注意力。
- Hidden：`stfe` 的 token 表示 `[B,N,T*embed_size]`（PEMS08 默认 `12*64=768`）；预测 `[B,T,N,1]`。
- Partial：`stfe.fc` + 两个周期表的线性层。
- **周期表零初始化**：官方加载 `daily_init.npy` / `weekly_init.npy`（训练集日均/周均剖面）。adapter 看不到训练集，故 `S` 从零开始，`set_periodic_prior(daily, weekly)` 提供注入口。零初始化时 `linear1` 的权重无梯度（输入恒为 0），测试显式记录该行为；注入先验后恢复。
- **`dual_view_loss` 不返回**：那是上游 runner 的频域辅助损失，依赖预测的分解视图。
- **周索引**：官方用 `tod*288 + dow*288`（`dow` 为分数），本仓用整数 weekday 直接计算 `daily_index + weekday*288`。
- **图方向**：官方 `einsum('ln,nv->lv')` 前推的是 `data @ adj`；adapter 传 `A.T`。

## 3. Partial 模式定义

Partial 是**本仓定义的 head-only 微调**，不是“最后一个完整时空块微调”，也不对应任何官方训练配方。白名单：

| backbone | partial 白名单 |
|---|---|
| staeformer | 最后 N 个 spatial attention 层 + 输出投影 |
| graph_wavenet | `end_conv_1`, `end_conv_2` |
| agcrn | `end_conv` |
| stid | `encoder`, `regression_layer` |
| d2stgnn | `out_fc_1`, `out_fc_2` |
| pdformer | 最后一个 encoder block, `end_conv2` |
| himnet | `decoder`, `out_proj` |
| stssdl | `decoder`, `proj`, `hypernet` |
| hyperd | `stfe.fc`, `daily_emb.linear*`, `weekly_emb.linear*` |

Frozen 模式下 `trainable_parameter_names()` 必须为空且输出 `requires_grad == False`；Full 模式下每个模型的预测路径都必须有梯度 —— 两条都由 `tests/unit/test_backbones.py` 断言。

## 4. 超参数：`backbone.params`

新增的架构专属超参走 `backbone.params`，**不**新增顶层键：

- `config.BACKBONE_PARAM_DEFAULTS` 是每个架构的**官方 PEMS08 默认值**白名单。
- `backbone.params` 只允许出现本架构的键；出现别的架构的键直接报错，避免“看起来设置了其实没生效”。
- 类型按默认值校验（bool/int/float/str），bool 不会被当作 int。
- 未写 `params` 时得到官方默认值；写一个键只覆盖该键。
- `schema_version` 保持 1：这是纯增量且每个键都有默认值，旧配置不需要改。

## 5. 接入前逐项签收

每个 backbone 都需：不可变 source commit + 必要文件哈希 + 许可证 + 补丁清单 + 输入配置 + 原生/adapter 等价测试 + Frozen buffers 测试 + CPU/GPU smoke。缺任一项状态保留 pending。

当前状态（2026-10-29）：

| 项 | 状态 |
|---|---|
| commit + 文件 SHA256 | 九个全部完成（`source_manifest.json`） |
| 许可证 | 5 个 MIT/Apache-2.0 已核验；D2STGNN / HimNet / STAEformer / HyperD 无 LICENSE，pending |
| 补丁清单 | 九个全部记录（本文件 §2） |
| 输入配置 | 九个全部记录 |
| CPU 前向/反向 + 3 种 mode | 九个全部通过（27 组合） |
| 图依赖断言 | 5 个真正的图模型通过；4 个自适应图/无图模型不断言 |
| 原生 vs adapter 逐元素等价 | **未做**（无官方权重） |
| GPU smoke / bf16 | **未做** |

网页可读不是依赖已安装，源码可读不是训练已复现。
