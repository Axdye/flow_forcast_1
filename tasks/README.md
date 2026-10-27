# 可直接分发的 DeepSeek 任务

每个会话先提供本仓库路径，再复制对应文件的“任务正文”。四个会话共同阅读根 README、故事线、代码设计、分工和审查文档。

- [A：数据与 backbone](TASK_A.md)
- [B：模型与候选](TASK_B.md)
- [C：目标函数与训练](TASK_C.md)
- [D：独立评测与测试](TASK_D.md)

优先让 A 完成共享合同的第零步；B/C/D 可以并行读设计、准备专属模块和 synthetic 测试。它们不能各自改共享类型。详细 Git 分支与合并规则见 [并行开发](../docs/README_PARALLEL.md)。
