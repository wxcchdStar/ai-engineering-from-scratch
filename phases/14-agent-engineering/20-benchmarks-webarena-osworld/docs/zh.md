# 基准测试：WebArena 与 OSWorld (Benchmarks: WebArena and OSWorld)

> WebArena 在四个自托管应用中测试 Web Agent 能力。OSWorld 在 Ubuntu、Windows、macOS 上测试桌面 Agent 能力。在发布时（2023–2024），两者都显示最佳 Agent 与人类之间存在巨大差距。差距正在缩小；失败模式没有改变。

**类型：** 学习 (Learn)
**语言：** Python (标准库)
**前置课程：** Phase 14 · 19 (SWE-bench, GAIA)
**时间：** 约 60 分钟

## 学习目标

- 描述 WebArena 的四个自托管应用以及为什么基于执行的评估很重要。
- 解释为什么 OSWorld 使用真实操作系统截图而非无障碍 API。
- 列举 OSWorld 的两种主要失败模式：GUI 定位和操作知识。
- 总结 OSWorld-G 和 OSWorld-Human 在基础基准之上增加了什么。

## 问题

通用 Agent 可以调用工具。它们能驱动浏览器跨越 20 次点击完成购物结账吗？它们能仅使用键盘和鼠标配置 Linux 机器吗？这些是 WebArena 和 OSWorld 回答的问题。

## 概念

### WebArena (Zhou et al., ICLR 2024)

- 812 个长程任务，跨越四个自托管 Web 应用：购物网站、论坛、类 GitLab 开发工具、商业 CMS。
- 外加实用工具：地图、计算器、草稿本。
- 评估是基于执行的，通过 gym API——订单是否已下、issue 是否已关闭、CMS 页面是否已更新？
- 发布时：最佳 GPT-4 Agent 达到 14.41% 成功率 vs 人类 78.24%。

自托管框架很重要——基准不会不稳定，因为目标应用是固定的和可复现的。

### 扩展

- **VisualWebArena** — 视觉定位任务，成功取决于解释图像（截图作为一等观察）。
- **TheAgentCompany** (2024 年 12 月) — 添加终端 + 编码；更像真实的远程工作环境。

### OSWorld (Xie et al., NeurIPS 2024)

- 369 个真实计算机任务，跨越 Ubuntu、Windows、macOS。
- 自由形式的键盘和鼠标控制真实应用。
- 1920×1080 截图作为观察。
- 发布时：最佳模型 12.24% vs 人类 72.36%。

### 主要失败模式

1. **GUI 定位。** 像素 → 元素映射。模型在 1920×1080 中可靠定位 UI 元素很困难。
2. **操作知识。** 哪个菜单有设置、哪个键盘快捷键、哪个偏好面板。人类多年积累的知识长尾。

### 后续工作

- **OSWorld-G** — 564 样本定位套件 + Jedi 训练集。将定位与规划分解，以便分别测量。
- **OSWorld-Human** — 手动策展的黄金动作轨迹。显示顶级 Agent 使用的步骤比必要多 1.4-2.7 倍（轨迹效率差距）。

### 为什么这很重要

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use (第 21 课) 都在由 WebArena 和 OSWorld 塑造的工作负载上训练。基准是目标；生产模型是交付的答案。

### 基准测试哪里会出错

- **仅截图的评估。** OSWorld 是截图驱动的；在 OSWorld 上评估使用 DOM 或无障碍 API 的 Agent 会错过定位挑战。
- **忽略轨迹长度。** 仅评分成功率会错过 OSWorld-Human 揭示的 1.4-2.7 倍步骤低效。
- **过时的自托管应用。** WebArena 的应用固定特定版本；未经重新策展的更新会破坏可比性。

## 构建

`code/main.py` 实现一个玩具 Web Agent harness：

- 一个最小的"购物应用"状态机：list_items、add_to_cart、checkout。
- 3 个任务的黄金轨迹。
- 一个脚本化 Agent 尝试每个任务。
- 基于执行的评估器（状态检查）和轨迹效率指标（步骤 vs 黄金）。

运行：

```
python3 code/main.py
```

输出：每个任务的成功率和轨迹效率，镜像 OSWorld-Human 的方法论。

## 使用

- **WebArena Verified** 在内部集群上自托管用于持续评估。
- **OSWorld** 在 VM 集群中用于桌面 Agent。
- **计算机使用 Agent** (第 21 课) — Claude、OpenAI CUA、Gemini — 都在类似这些的工作负载上训练。
- **你自己的产品流程** — 为你前 20 个任务捕获黄金轨迹；每周对它们运行 Agent。

## 交付

`outputs/skill-web-desktop-harness.md` 构建一个 Web/桌面 Agent harness，带基于执行的评估和轨迹效率指标。

## 练习

1. 用第二个应用（论坛）扩展玩具 harness。编写 3 个任务加黄金轨迹。
2. 为每个任务添加轨迹效率报告。在你的玩具上，Agent 是黄金的 1 倍、2 倍还是 3 倍？
3. 实现一个"干扰"工具——黄金轨迹从未使用过的工具。脚本化 Agent 会被诱惑吗？
4. 阅读 OSWorld-G。你如何在你自己的评估中将定位失败与规划失败分开？
5. 阅读 WebArena 的应用 README。当你升级其中一个固定应用版本时，什么会坏？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| WebArena | "Web Agent 基准" | 812 个任务，跨越 4 个自托管应用；gym 风格评估 |
| VisualWebArena | "视觉 WebArena" | 视觉定位的 WebArena；截图是观察 |
| OSWorld | "桌面 Agent 基准" | 369 个任务，在真实 Ubuntu/Windows/macOS 上 |
| GUI 定位 (GUI grounding) | "像素到元素映射" | 模型在 1920x1080 中定位 UI 元素 |
| 操作知识 (Operational knowledge) | "操作系统知识" | 哪个菜单、哪个快捷键、哪个偏好面板 |
| OSWorld-G | "定位套件" | 564 个仅定位样本 + 训练集 |
| OSWorld-Human | "黄金轨迹" | 手动专家动作序列，用于测量效率 |
| 轨迹效率 (Trajectory efficiency) | "步骤超过黄金" | Agent 步骤数除以人类最小值 |

## 扩展阅读

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — 四应用 Web 基准
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — 跨操作系统桌面基准
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — Claude 的基准形态能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — OSWorld 和 WebArena 数据
