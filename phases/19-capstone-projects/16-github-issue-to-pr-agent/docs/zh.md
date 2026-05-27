# 实战项目 16 — GitHub Issue 到 PR 自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都提供相同的 2026 年产品形态：标记一个 issue，获得一个 PR。在云沙箱中运行智能体，验证测试通过，并发布一个带有理由的可审查 PR。困难的部分是自动复现仓库的构建环境、防止凭据泄露、强制执行每仓库预算，以及确保智能体不能 force-push。本实战项目构建自托管版本，并在成本和通过率上与托管替代方案进行比较。

**类型：** 实战项目（Capstone）
**语言：** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**前置课程：** 第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

异步云编程智能体是一个与交互式编程智能体（实战项目 01）不同的产品类别。其 UX 是一个 GitHub 标签。你标记一个 issue `@agent fix this`，一个工作器在云沙箱中启动，克隆仓库，运行测试，编辑文件，验证，并打开一个 PR，PR 正文中包含智能体的理由。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都汇聚于此。

工程挑战是具体的：环境复现（智能体必须从头构建仓库，没有缓存的开发镜像）、不稳定的测试（必须重新运行或隔离）、凭据范围限定（具有最小细粒度权限的 GitHub App）、每仓库每日预算执行，以及禁止 force-push 策略。本实战项目测量通过率、成本和安全性与托管替代方案的对比。

## 概念

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器将工作入队到 ECS Fargate 或 Lambda。工作器将仓库拉入 Daytona 或 E2B 沙箱，使用从仓库推断的通用 Dockerfile（语言、框架）。智能体运行 mini-swe-agent 或 SWE-agent v2 循环，配合 Claude Opus 4.7 或 GPT-5.4-Codex。它迭代：读取代码、提出修复、应用补丁、运行测试。

验证是门控步骤。在 PR 打开之前，完整的 CI 必须在沙箱中通过。计算覆盖率差异；如果为负且超过阈值，PR 打开但被标记为 `needs-review`。智能体将理由作为 PR 描述发布，加上一个 `@agent` 线程，审查者可以 ping 以进行后续操作。

安全通过两个不同的 GitHub 表面进行范围限定：App 提供具有 `workflows: read` 和窄范围仓库内容/PR 作用域的短期安装 token；分支保护（而非应用权限）强制执行"禁止直接写入 `main`"和"禁止 force-push"——应用永远不会被添加到绕过列表中。对 `.github/workflows` 的路径范围只读访问不是真正的 GitHub App 原语，因此智能体对文件编辑的允许列表必须在工作器层面强制执行。每仓库每日预算上限在调度器层面强制执行（例如，每仓库每天最多 5 个 PR，每个 PR 20 美元）。

## 架构

```
GitHub issue 标记为 `@agent fix` 或 PR 评论
            |
            v
    GitHub App webhook -> AWS Lambda 调度器
            |
            v
    ECS Fargate 任务（或 GitHub Actions 自托管运行器）
       - 拉取仓库
       - 推断 Dockerfile（语言、包管理器）
       - Daytona / E2B 沙箱，带目标运行时
       - 克隆 -> git worktree -> 智能体分支
            |
            v
    mini-swe-agent / SWE-agent v2 循环
       Claude Opus 4.7 或 GPT-5.4-Codex
       工具：ripgrep、tree-sitter、读取/编辑、运行测试、git
            |
            v
    验证 CI 在沙箱中通过 + 覆盖率差异检查
            |
            v（已验证）
    git push + 通过 GitHub App 打开 PR
       PR 正文 = 理由 + 差异摘要 + 追踪 URL
       标签：needs-review
            |
            v
    运维人员审查；可以 @-提及智能体进行后续操作
```

## 技术栈

- 触发器：GitHub App 配合细粒度 token；通过 Lambda 或 Fly.io 的 webhook 接收器
- 工作器：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每任务 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：mini-swe-agent 基线或 SWE-agent v2 配合 Claude Opus 4.7 / GPT-5.4-Codex
- 检索：tree-sitter 仓库映射 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率差异门控
- 可观测性：Langfuse 配合每 PR 追踪归档，从 PR 正文链接
- 预算：每仓库每日美元上限；每仓库每天最大 PR 数

## 构建步骤

1. **GitHub App。** 细粒度安装 token：issues 读写、pull_requests 写入、contents 读写、workflows 读取。分支保护（唯一可以做到这一点的表面）强制执行"禁止直接推送到 `main`"和"禁止 force-push"；应用不在绕过列表中。工作器在提议的差异上强制执行"禁止写入 `.github/workflows` 下"作为允许列表检查，因为 GitHub App 权限不是路径范围的。

2. **Webhook 接收器。** Lambda 函数接受 issue 标签/PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行每仓库每日预算。使用仓库 URL、issue 正文和全新的 Daytona 沙箱启动 ECS Fargate 任务。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在，即时生成 Dockerfile。

5. **智能体循环。** mini-swe-agent 或 SWE-agent v2 配合 Claude Opus 4.7。工具：ripgrep、tree-sitter 仓库映射、read_file、edit_file、run_tests、git。硬性限制：20 美元成本、30 分钟墙钟时间、30 个智能体回合。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco/coverage.py 计算覆盖率差异。如果 CI 红色：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 并标记 `needs-review`。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 打开 PR，包含：标题、理由、差异摘要、追踪 URL、成本、回合数。

8. **凭据卫生。** 工作器使用短期 GitHub App 安装 token 运行。日志在归档前脱敏密钥。

9. **评估。** 30 个不同难度的种子内部 issue。测量通过率、PR 质量（差异大小、风格、覆盖率）、成本、延迟。在相同 issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 进行比较。

## 使用方式

```
# 在 github.com 上
  - 用户将 issue #842 标记为 `@agent fix this`
  - 14 分钟后 PR #1903 出现
  - 正文：
    > 修复了由空比较器条目引起的 widget.dedupe() 中的 NPE。
    > 添加了回归测试 widget_test.go::TestDedupeNullComparator。
    > 覆盖率差异：+0.12%
    > 回合数：7  成本：$1.80  追踪：langfuse:...
    > 标签：needs-review
```

## 交付标准

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云工作器，将标记的 issue 转化为可审查的 PR，具有有界成本和范围限定的凭据。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 绿色 + 覆盖率 OK） |
| 20 | PR 质量 | 差异大小、覆盖率差异、风格一致性 |
| 20 | 每个解决 issue 的成本和延迟 | 每个 PR 的美元和墙钟时间 |
| 20 | 安全 | 范围限定的 token、每仓库预算、禁止 force-push、凭据卫生 |
| 15 | 运维人员 UX | 理由评论、重试能力、@-提及后续操作 |
| **100** | | |

## 练习

1. 添加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，并提出稳定它的最小更改。
2. 在三个共享 issue 上比较成本 vs Cursor Background Agents。报告哪些工具在哪些方面胜出。
3. 实现预算仪表盘：每仓库每日成本、每用户成本。对异常发出告警。
4. 构建"预演"模式，打开草稿 PR 而不运行 CI，以便审查者可以低成本检查计划。
5. 添加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | "范围限定的机器人身份" | 具有细粒度权限 + 短期安装 token 的应用 |
| 异步云智能体 | "后台智能体" | 在云沙箱中运行的非交互式工作器，而非终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，如果缺失则生成 Dockerfile |
| 验证 | "沙箱内 CI" | 在打开 PR 之前在工作器内部运行完整测试套件 |
| 覆盖率差异 | "覆盖率保持" | 从基础仓库到智能体分支的测试覆盖率百分比变化 |
| 每仓库预算 | "每日上限" | 在调度器层面强制执行的美元和 PR 数量上限 |
| 理由 | "PR 正文解释" | 智能体对更改内容和原因的摘要；在 PR 正文中必需 |

## 扩展阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 规范异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex（云）](https://openai.com/codex) — 托管竞争者
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 备选商业参考
- [GitHub App 文档](https://docs.github.com/en/apps) — 范围限定的机器人身份
- [Daytona 云沙箱](https://daytona.io) — 参考沙箱
