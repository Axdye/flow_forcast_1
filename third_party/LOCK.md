# 第三方依赖锁与准入

更新日期：2026-10-29（上一版 2026-10-28）。

这是一份诚实的**部分锁定清单**，不是可复现环境 lock 的替代。本轮补齐了九个 backbone 的不可变 commit、许可证状态与被提取文件的 SHA256；运行环境（torch wheel / CUDA / 驱动）仍未 lock，正式训练前必须补齐。

## 1. 上游参考仓库

所有 commit 均来自 2026-10-29 的全新 clone（`git clone` 后 `git rev-parse HEAD`），不是浮动分支，也不是网页快照。被提取文件的 SHA256 记在 [source_manifest.json](source_manifest.json)，schema 见 §5。

| 名称 | 官方来源 | 固定 commit | 许可证 | 本仓状态 |
|---|---|---|---|---|
| STAEformer | https://github.com/XDZhelheim/STAEformer | `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f` | **无 LICENSE 文件** | adapter 已实现；许可证待确认 |
| Graph WaveNet | https://github.com/nnzhan/Graph-WaveNet | `6b162e80c59a1d494809252eca055cff93dc66b1` | MIT | adapter 已实现；CPU smoke 通过 |
| AGCRN | https://github.com/LeiBAI/AGCRN | `7fbbf2aeb099242098a3cf482b55cd45d7295c28` | MIT | adapter 已实现；CPU smoke 通过 |
| STID | https://github.com/zezhishao/STID | `e8b313bc591bdd0101a1619962c9b503e75127c0` | Apache-2.0 | adapter 已实现；CPU smoke 通过 |
| D2STGNN | https://github.com/zezhishao/D2STGNN | `82c2d38ba098a6aa87e8099c1ee8da58480692ec` | **无 LICENSE 文件** | adapter 已实现；许可证待确认 |
| PDFormer | https://github.com/BUAABIGSCity/PDFormer | `f8c8f6ad007a04fad3baee958b89504711852ce9` | MIT | adapter 已实现；CPU smoke 通过 |
| HimNet | https://github.com/XDZhelheim/HimNet | `2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5` | **无 LICENSE 文件** | adapter 已实现；许可证待确认 |
| ST-SSDL | https://github.com/Jimmy-7664/ST-SSDL | `a98331d3a45a081778f3abb608897902a60e301d` | MIT | adapter 已实现；CPU smoke 通过 |
| HyperD | https://github.com/ll121202/HyperD | `5bd93a825955cbe31b2d8dac0041ecb0e1d223d2` | **无 LICENSE 文件** | adapter 已实现；许可证待确认 |
| BasicTS（参考，非运行时依赖） | https://github.com/GestaltCogTeam/BasicTS | `63fa7192a94c7a59519c0a891f0ad9622f3da37c`（v0.5.8） | Apache-2.0 | 只用于核对 instant-traffic 模型定义与官方超参，不作为 runtime 依赖 |

关于 BasicTS 的 pin：master / v1.1.0 **不含** STAEformer / AGCRN / Graph WaveNet / STID 的 instant-traffic 实现，只有 `63fa7192a94c7a59519c0a891f0ad9622f3da37c` 这一版有。任何“BasicTS 官方实现”的说法都必须带这个 commit。

### 许可证待确认（published-code 风险）

D2STGNN、HimNet、STAEformer、HyperD 四个仓库**没有 LICENSE / COPYING 文件**。因此：

- 本仓**不复制**它们的源码文件；`src/sbona/backbones/*.py` 是按论文与源码结构重写的 adapter，并在模块 docstring 中标注来源与 commit。
- 重写仍是衍生关系，**不能据此声称已获得再发布许可**。这四个 backbone 的正式实验与论文附表在取得作者许可或明确引用条款前保持 `license: pending`。
- 若只做内部复现，也应保留本文件与 `source_manifest.json` 的归属记录。

MIT / Apache-2.0 的五个（Graph WaveNet、AGCRN、STID、PDFormer、ST-SSDL）在再发布本仓时应随附其 LICENSE 文本；本仓当前不 vendor 上游源码，故未复制许可证文件。

## 2. 提取点与补丁

每个模型的提取位置、方向约定、输入扩展与兼容补丁都写在 [BACKBONE_CONTRACTS.md](BACKBONE_CONTRACTS.md)。摘要：

| backbone | 提取的 hidden | 需要记录的补丁 |
|---|---|---|
| STAEformer | 最后一个 temporal/spatial attention 块之后、取最后历史步 | `input_dim` 1→3（官方值），dow 以分数进入投影、整数索引 embedding |
| Graph WaveNet | 第二层输出卷积之前 | `nconv` 需要 `A.T`；`Conv2d` 等价改写（上游 Conv1d 旧写法） |
| AGCRN | encoder 最后一步 | 去掉 `targets` 参数；hidden 初始化跟随 device/dtype |
| STID | `regression_layer` 之前 | 日期/星期 embedding 索引取 device-preserving long |
| D2STGNN | 最后一个 decouple layer 的 forecast 输入 | `use_pre=False`（上游硬编码） |
| PDFormer | 最后 encoder 层的输出表示 | DTW 语义先验禁用，改为可学习 per-node pattern key |
| HimNet | 最后 `HimGCN` 输出 | 关闭 teacher forcing；`xavier_normal_` 参数注册 bug |
| ST-SSDL | decoder 最后状态 | `x_his = x`；对比学习输出不计入 loss |
| HyperD | `stfe` 的频域表示 | 周期表零初始化 + `set_periodic_prior` 注入；`dual_view_loss` 不用 |

## 3. 运行依赖必须补齐的清单

Python 精确版本；torch wheel/version/CUDA runtime；GPU driver；numpy/scipy/PyYAML/pytest/ruff 等准确版本；操作系统；pip/uv lock；第三方必要 package；许可复核。

本轮验证环境（**不是**验收环境）：

- `/opt/miniconda3/envs/pytorch`：Python 3.11、torch 2.2.2、numpy 2.3.3（与 torch 的 numpy 1.x 编译 ABI 冲突）。
- 测试用一个 numpy 1.26.4 的临时 overlay 覆盖 `site-packages`，命令见 `reviews/STATUS.md`。这只是让 CPU 测试可跑，**不构成**可复现环境。

## 4. 网络记录

2026-10-28：GitHub REST 匿名接口 403 rate limit，`git ls-remote` 超时；浏览工具只能读网页。
2026-10-29：用户授权联网后，九个官方仓库 `git clone` 成功并逐仓 `git rev-parse HEAD` 固定；`uv`/PyPI 下载除缓存命中外仍超时，故测试环境用本地 conda 包拼装。

## 5. 源码登记 schema

每个上游登记项必须至少有：

```json
{
  "name": "example",
  "repository": "official URL",
  "commit": "40-character verified SHA",
  "license": {"path": "LICENSE", "spdx": "MIT", "verified": true},
  "files": [{"path": "relative/path.py", "sha256": "verified digest", "bytes": 1234}],
  "patches": ["..."],
  "validation": {"source_read": true, "cpu_forward": true, "gpu_smoke": false}
}
```

`third_party/source_manifest.json` 按这个 schema 记录；`validation.cpu_forward` 只表示本仓 adapter 在 CPU 上前向通过（`tests/unit/test_backbones.py`），不代表上游代码在原环境复现成功，也不代表 GPU/bf16 已验证。

首次 vendor 必须保留第三方 LICENSE/NOTICE；本仓目前不复制第三方源码，不预先替第三方选择许可证。新项目整体许可证在用户确定发布方式后再选，不默认为所有素材都可同许可发布。
