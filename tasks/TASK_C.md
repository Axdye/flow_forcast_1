# DeepSeek C：Oracle、损失与训练状态机

## 任务正文（可直接复制）

你负责 C 任务，在独立分支实现 src/sbona/objectives、training、cli.py、实验配置和对应 unit tests。阅读全部合同，使用 A/B 的接口。不能把 y/target 传入 model forward；不能重新定义 MAE。

首轮实现无量纲 E/P、fp32 detached z-score（population std）、非负 floor 权重、候选维 sum/样本维 mean、separation、KL、detached regret 与 balance schedule。全无效样本必须剔除；K=1 无 NaN。不要重复 AdamW 与手工 L2。

用人工独立候选张量验证每个 loss 的梯度方向/路径。Router loss 只更新 Router；验证候选参数梯度为 None/zero，不只是看 target.requires_grad。

实现 Stage A/B/C/D 状态机、精确 freeze/unfreeze、early stop、独立 checkpoint 选择、原子 latest/best、epoch 边界恢复、RNG/optimizer/patience/config/hash 保存。B/C 从相同 Base 独立初始化，不默认把 K=1 权重扩成多分支。默认不联合训练。

根据 README_EXPERIMENTS.md 实现 F/R/Q 可路由性 probe，包括候选只训练 F、统计只拟合 F、probe 只训练 R、Q holdout、固定分支与 block-shuffle 负控制。拒绝随机窗口 K-fold 造成时间泄漏。

实现根 README 中未来 CLI 合同；入口参数校验、退出码、失败日志、run_id 隔离、未知 key 报错。OOM 不自动改 batch/precision。训练过程使用 D 的 metric aggregator，而非私有指标。

验收：手算 oracle；loss 不随 K 重复样本误缩放；梯度隔离；全部 mask 为 false；K=1；checkpoint 中断/恢复对照；冻结参数不变；选择 best 使用 val 而非 test；CLI 正确失败。

交付 reviews/HANDOFF_C_01.md，明确 CUDA/真实数据是否已执行。设计冲突先提交 ADR，不用额外 flag 隐藏不一致。
