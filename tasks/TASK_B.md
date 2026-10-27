# DeepSeek B：结构化候选与预解码路由

## 任务正文（可直接复制）

你负责 SBoN-Adapter B 任务。阅读共同文档并使用 A 的 contracts.py；在独立分支开发。你拥有 src/sbona/models、baselines 的模型部分及模型 unit tests。不重写数据/指标/训练器，不改共享合同。

按 README_CODE_DESIGN.md 逐式实现 projector、Sobol anchors、bounded 五维动作、共享稀疏扩散基、无 bias 输出、固定包络及 tanh 稳定化。v1 不加 FiLM、local residual、自由预测头，也不加入未要求的新架构。K=1 仍训练动作生成器。

实现 forward_candidates 与 predict_top1，selected 路径不允许调用 forward_all、不构造完整 K 条轨迹。可以生成 K 个五维动作。接入 Frozen backbone 的 train/eval 规则；Router 输入来自历史。公式中每个维度都用 docstring 说明。

首先用 A 的小图 fixture 测试，再接 STAEformer。必须覆盖：

- B=1、K=1/4/8、不同 N、孤立节点；
- selected 输出等于 full 输出按指定 index gather（eval/fp32）；
- m=0 零残差、固定其他动作的幅度干预；
- 稀疏递推与小图 dense 参考一致；
- 很小/很大的动作 logits 不产生 NaN；
- 小随机 W_out 的初始化和多步参数更新；
- profiler/张量路径证据证明 selected 没有隐藏 K 维解码。

第二轮实现 K=1/宽化 single、同参数 independent heads 和五维 unstructured latent。记录实际 trainable 参数和推理 FLOPs/时长，不添加死参数假装匹配。MC dropout 如需新增由 C/D 提供评测接口，保持 BN eval。

交付 reviews/HANDOFF_B_01.md：每个数学式映射的函数、接口、真实测试、性能风险。不要跑未经审查的大规模训练，不宣称模型改善了 PEMS。
