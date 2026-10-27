# 前情来源、设计取舍与待确认项

## 1. 用户授权与交付范围

用户要求：基于约 1–4 张 A100 80GB，从头规划；在 Documents/code 下新建目录、初始化 Git；详细工作故事线与严谨代码设计；准备多个 DeepSeek 并行实现，由 Codex 最终审查。2026-10-28 用户进一步明确允许访问互联网和 GitHub。

本次交付为可供实施的文档合同、任务包、配置参考和审查清单；未被要求现在训练模型。本次不创建 GitHub 远端、不推送、不调用 DeepSeek API，不改变两个旧仓。

## 2. 已读取的前情

指定聊天标题：**搜索一下新论文故事线md 和代码详细设计 md，就在当前项目下**。

聊天 ID：`01a17e34-dff2-70f3-9644-bf5204feed01`。读取了该聊天关于旧设计可信度，以及 `Flow-RL`/`Flow-RL 2` 差异与重建建议的用户问题和最终答复。

本地材料：

- `/Users/ljw/Documents/ChatGPT/毕设/新论文详细设计/01_新论文故事线.md`
- `/Users/ljw/Documents/ChatGPT/毕设/新论文详细设计/02_代码与实验详细设计.md`
- `/Users/ljw/Documents/ChatGPT/毕设/新论文详细设计/03_旧论文改造成新论文.md`

材料指纹见 [source_manifest.json](../third_party/source_manifest.json)。这些是研究草稿，不是实验事实。前情中旧仓大小/运行日志等是此前审查记录；本次没有重新全量验证所有旧二进制。

## 3. 旧资产处理

`Flow-RL` 是旧方法档案；`Flow-RL 2` 含脏工作区及 STAEformer 数据/实验痕迹。新仓不继承两者 Git 历史，避免 checkpoint 历史和不完整 gitlink。保留其原貌，不运行 reset/clean，也不擅自替用户提交旧仓修改。

可选择性复用：原始数据文件、节点映射、官方 pristine backbone 与通用读取思路；先 hash 和 audit，再引用。旧 scaler、mask、index、训练器不能未经核查直接复制。旧预测/权重只可作历史参照，不能变成新主表证据。

## 4. 相对旧设计的明确变更

| 旧设计缺口/选择 | 本合同决定 | 原因 |
|---|---|---|
| 默认多未来能路由 | 把可路由性单独设为 H2/G3 | 不可约随机性不可能被历史预测 |
| “确定性预测=均值”泛化 | 平方损失均值；绝对损失中位数 | 与主 MAE/训练 MSE 区分 |
| BasicTS 或自有 runner 未定 | 自有轻量统一 runner | 避免多套框架/指标混合 |
| oracle group z-score 叫 robust | 普通 z-score，correction=0、方差下界 | 防 K=1 NaN 与术语错误 |
| E/P 混合 raw/z 单位 | E 无量纲、P raw 判违反后除训练 s | loss 权重有一致含义 |
| mixed-cost oracle 被当 MAE 上界 | 分开 cost/MAE/MSE oracle | 选择准则不同，不混用 retention |
| arbitrary FiLM/local 分支 | v1 关闭全部自由旁路 | 先验证结构本身 |
| 最后层零初始化且所有梯度非零 | 小随机初始化；梯度测试针对合理对象 | 零输出层首步会阻断上游梯度 |
| K=1 动作不明 | 正常训练唯一动作 | 完整可运行 single baseline |
| Stage B 后直接展开多分支 | B/C 独立从同 Base 初始化 | 避免对称性、保证比较公平 |
| 候选与 Router 默认联合 | 默认固定候选再训 Router | 标签身份稳定，减少归因歧义 |
| 组内归一化与 regret 可回传 | targets/cost/route hidden detach | route loss 不改 oracle 标尺 |
| balance 被称为候选防坍缩 | 仅路由分布正则；另测真实轨迹 | soft 均匀不等于轨迹多样 |
| 显式 A^r | 稀疏递推 A*S | 避免图幂稠密化 |
| flow 变化当拥堵 | 无独立标签时只说流量阶段 | 流量不能唯一决定交通状态 |
| α=.8 主设置 | accuracy-only 先行，一致性单独扩展 | 分离路由有效性与经验惩罚 |
| 大范围全笛卡尔积 | Gate + 限额 trial + 失败登记 | 有算力也要控制选择偏差 |

这些是新合同的设计选择，不是声称已经被实验验证更优。

## 5. 外部来源核验的边界

已访问 [STAEformer](https://github.com/XDZhelheim/STAEformer)、[Graph WaveNet](https://github.com/nnzhan/Graph-WaveNet)、[AGCRN](https://github.com/LeiBAI/AGCRN)、[STID](https://github.com/GestaltCogTeam/STID) 官方仓库与对应模型源码。具体适配点、问题与链接在 [BACKBONE_CONTRACTS](../third_party/BACKBONE_CONTRACTS.md)。

没有在本次做完整 2026 年文献系统检索，也没有核验旧稿列举的所有近期 baseline 是否可运行。TimeMCL 等仅保留为后续相关工作/直接机制对照线索，不生成未经核验的 bib 条目或 SOTA 结论。

## 6. 下一轮需要实际确定的事项

- 服务器访问方式、卡型 PCIe/SXM、可持续 GPU-h、CPU/RAM/存储；不影响先开发 CPU 合同。
- PEMS 文件真实版本、flow 通道/单位、节点 ID、可靠时间戳；由 A 数据审计确定。
- Python/torch/CUDA 精确环境，其他 backbone commit/许可证；分别在接入前锁定。
- 论文目标 venue 和时间；影响扩展 baseline 范围，不阻塞 MVP。
- 科学 Gate 是否通过；必须等待真实实验，不能在开发文档里预先决定。

凡是未确定项都有明确阶段与负责人；它们不能被实现模型用“合理假设”填成虚构事实。
