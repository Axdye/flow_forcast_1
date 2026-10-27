# 第三方依赖锁与准入

这是一份诚实的**部分锁定清单**，不是可复现环境 lock 的替代。正式训练前 A 必须补齐每个实际使用组件的不可变 commit、源码文件 SHA256、许可证、补丁及 runtime。未使用的扩展骨干 pending 不阻塞 STAEformer MVP，但阻塞其自身正式实验。

| 名称 | 官方来源 | 当前固定版本 | 状态 |
|---|---|---|---|
| STAEformer | https://github.com/XDZhelheim/STAEformer | fc49d39b2f1a8e3cf37b6289d7240680e1690f3f | 从已有本地 Git 对象核验 model 源码；运行未验证 |
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | pending，禁止用浮动 master 开正式实验 | 主源码在线审读；有 PyTorch 兼容风险 |
| AGCRN | https://github.com/LeiBAI/AGCRN | pending | 主源码在线审读；递归依赖待锁 |
| STID | https://github.com/GestaltCogTeam/STID | pending | 主源码在线审读；星期编码需转换 |
| BasicTS | https://github.com/GestaltCogTeam/BasicTS | 不作为 v1 runtime 依赖 | 仅旧设计参考 |

已知本地 STAEformer 源对象在 `/Users/ljw/Documents/code/Flow-RL 2/STAEformer`。此目录有未提交修改；迁移必须使用 `git show <commit>:<path>` 或 clean checkout，而非复制其当前工作区 model 文件。文件指纹见 [source_manifest.json](source_manifest.json)。

网络记录：2026-10-28 获用户联网授权后，GitHub REST 匿名接口返回 403 rate limit，git ls-remote 超时。浏览工具成功读取官方仓库页及四个主模型源码。未猜测三个缺失 commit，未声称它们已固定。

## 运行依赖必须补齐的清单

Python 精确版本；torch wheel/version/CUDA runtime；GPU driver；numpy/scipy/PyYAML/pytest/ruff 等准确版本；操作系统；pip/uv lock；第三方必要 package；许可复核。优先一个最小 Linux/A100 环境，之后才添加可选日志插件。

源码登记 schema：

```json
{
  "name": "example",
  "repository": "official URL",
  "commit": "40-character verified SHA",
  "files": [{"path": "relative/path.py", "sha256": "verified digest"}],
  "license": {"path": "LICENSE", "verified": false},
  "patches": [],
  "validation": {"source_read": true, "cpu_forward": false, "gpu_smoke": false}
}
```

首次 vendor 必须保留第三方 LICENSE/NOTICE；本仓目前不复制第三方源码，不预先替第三方选择许可证。新项目整体许可证在用户确定发布方式后再选，不默认为所有素材都可同许可发布。
