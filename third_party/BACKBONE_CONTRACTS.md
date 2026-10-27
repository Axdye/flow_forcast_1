# Backbone 接口核验记录

日期 2026-10-28。以下为源码审读与本项目接入要求；除 STAEformer 本地 commit 外，其余 Git commit 待锁，均未执行 runtime。不得把本文件当作“已支持四 backbone”的证明。

## 公共输入与输出

所有方法获得同一份历史 flow 与可信历史日历，不使用未来实际观测。STAEformer/STID 使用原日历 embedding；Graph WaveNet/AGCRN 的统一协议可配置输入三通道 `[flow_z,tod,dow/7]`，这是相对 flow-only 官方设置的输入扩展，必须在统一重跑中披露。原生 flow-only 复现另列 protocol。

无可信 weekday 时所有方法禁用 weekday，不能给某个方法伪造星期。如果这样改变 STAEformer/STID embedding 大小，必须重新训练对应 Base，不能直接加载不匹配 checkpoint。

原生 forward 一致性指**相同输入配置、相同权重、相同精度、相同补丁基线**下的比较。改变输入维度不能拿另一官方配置做逐元素等价检查。

## STAEformer

源码：[STAEformer.py](https://github.com/XDZhelheim/STAEformer/blob/fc49d39b2f1a8e3cf37b6289d7240680e1690f3f/model/STAEformer.py)。已从本地 Git 对象读取该版本，未使用脏工作区文件。

提取最终空间 attention 后的 hidden，再取最后历史步得到 `[B,N,model_dim]`；原预测继续沿混合投影输出 `[B,T,N,1]`。具体规则已在代码设计 6.1 固定。Partial 解冻最后空间 attention 层及输出投影。

STAEformer 的 dow 是整数索引 0..6，不能把 STID 的归一化 dow 直接传进去。构造 `[flow_z,tod,dow]`，`input_dim=1`。基础 forward 不调用两遍。

## Graph WaveNet

源码：[model.py](https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py)。2026-10-28 已读，master 为可变引用，正式 vendor 前必须补 commit/hash。

取最后 skip 汇总经第一层输出卷积与激活后的表示，即第二层输出卷积之前。hidden 形状 `[B,end_channels,N,T_reduced]`，取最后时刻并转为 `[B,N,end_channels]`。预测沿原最终输出层得到 `[B,T,N,1]`；默认 L=12 必须验证 reduced time 为 1，不允许 squeeze 掩盖其他长度。

Partial 白名单为两个末端输出卷积；所有时序/图块与 BN 冻结。这是本项目定义的 head-only Partial，不冒称最后完整时空块微调。

源码含在四维激活上使用 Conv1d 且传二维 kernel 的旧写法。A 必须在当前 PyTorch 上确认失败点；如改为等价 Conv2d，记录补丁、权重形状、保存/加载策略，并用修补后的纯 Base 对照 adapter。不能把兼容性修复当成方法改进。图 nconv 的方向约定与 SBoN 的左乘聚合不同，需显式转换 support，不能直接复用名称猜方向。

## AGCRN

源码：[AGCRN.py](https://github.com/LeiBAI/AGCRN/blob/master/model/AGCRN.py)。已读，commit 待锁；还须一起固定 AGCRNCell/AVWGCN 等递归依赖。

encoder 输出 `[B,L,N,rnn_units]`，取最后历史步为 hidden，预测沿原 end_conv 和 reshape/permute 路径。Partial 仅解冻 encoder 最后一层 cell 与 end_conv；node_embeddings 默认冻结。

公开 forward 有 targets 参数，但所读版本的计算不使用它。SBoN wrapper 应直接拆出编码/预测逻辑，只接受 ModelInput；不要因为上游签名含 targets 就把 TargetBatch 传进去。初始化 hidden 的 device 和 dtype 必须跟随输入。三个历史特征的设置需要 `input_dim=3`，原始 flow-only 实验则单独配置与重跑。

## STID

源码：[stid_arch.py](https://github.com/GestaltCogTeam/STID/blob/master/stid/arch/stid_arch.py)。已读，commit 待锁；需要同时固定 `mlp.py`。

取 encoder 后、regression_layer 前的 hidden `[B,D_b,N,1]`，squeeze 最后维并转成 `[B,N,D_b]`；预测保留 `[B,T,N,1]`。Partial 解冻 encoder 最后一个 MLP 和 regression_layer。

该版本日历读取会把星期值乘 day_of_week_size，故 wrapper 传 `[flow_z,tod,dow/7]`，input_dim=1。必须测试星期 0 与 6 不越界。原代码使用转 CPU LongTensor 的写法，迁移到 device-preserving long 索引需记为兼容补丁并验证等价。

原 forward 带 future_data/batch_seen/epoch/train 等框架参数；wrapper 去掉这些未使用输入，不依赖 BasicTS runner。

## 接入前逐项签收

每个 backbone 都需：不可变 source commit + 必要文件哈希 + 许可证 + 补丁清单 + 输入配置 + 原生/adapter 等价测试 + Frozen buffers 测试 + CPU/GPU smoke。缺任一项状态保留 pending。网页可读不是依赖已安装，源码可读不是训练已复现。
