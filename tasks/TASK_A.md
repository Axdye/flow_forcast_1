# DeepSeek A：数据、源码与基础工程

## 任务正文（可直接复制）

你负责 SBoN-Adapter 的 A 任务。先阅读根 README、docs/README_CODE_DESIGN.md、README_PARALLEL.md、README_REVIEW.md。只在独立分支/工作目录开发。Codex 做最终审查；你必须提供真实测试日志，不能声称未执行的验证通过。

**第一轮最先交付** contracts.py、严格 config schema、pyproject.toml 和 synthetic fixture。按设计的 dataclass/Protocol 原样落实，禁止不透明 kwargs 携带目标。向其他任务提供 base commit，然后才继续真实数据。

你拥有 src/sbona/data、backbones、contracts.py、config.py、third_party、依赖文件及自己的 unit tests。不修改 models/objectives/training/evaluation 的业务实现。

工作项：

1. 从 third_party/LOCK.md 指定版本获取最小源码，保留许可；读取原模型，记录改动 patch、source hash。不能把旧未提交修改的生成头作为 backbone。
2. PEMS08 数据 manifest、节点 ID 对齐、数据 audit、时间窗口、原始有效 mask、train-only scalar scaler/envelope；所有边界定义依代码设计。
3. 稀疏道路图，去重/self-loop/孤立节点处理；图 hash 与 node_order hash；明确 learned graph 与道路图区别。
4. STAEformer 原始 forward 与 adapter base 输出逐元素等价；hidden 提取点/shape 准确。Frozen/Partial/Full 参数名单与 dropout/BN 状态。
5. 支持从外部 data-root 读取数据；无数据时清楚报错，不能偷偷生成 synthetic 当作 PEMS。
6. 在 Linux CPU 最小环境可安装包；GPU smoke 后锁 wheel/CUDA。未验证组合明确标记。
7. 第二轮再接 STID、Graph WaveNet、AGCRN；未固定源码/许可/前向验证前标为 unsupported，不用“形状相同”假装完成。

验收测试：时间边界手算；修改 val/test 不影响 scaler/envelope；零值/NaN/孤立节点；节点顺序置换一致性；原 forward 等价；Frozen 两次 train step 后参数/运行统计不变；迁移路径不加载旧 GRPO。

交付 reviews/HANDOFF_A_01.md，列 head commit、实际命令、日志、所有待核验依赖。遇到主合同问题提交 ADR，不自行放宽。
