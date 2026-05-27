# 实战项目 09 — 代码迁移智能体（仓库级语言/运行时升级）

> Amazon 的 MigrationBench（Java 8 到 17）和 Google 的 App Engine Py2-to-Py3 迁移器设定了 2026 年的标准。Moderne 的 OpenRewrite 以规模化方式进行确定性 AST 重写。Grit 使用 codemod 风格的 DSL 解决同样的问题。生产模式结合了两者：一个用于安全重写的确定性基底层，加上一个用于模糊情况的智能体层，一个用于每分支构建的沙箱，以及一个在 PR 打开之前变绿的测试套件。本实战项目要求迁移 50 个真实仓库，并发布带有失败分类的通过率。

**类型：** 实战项目（Capstone）
**语言：** Python（智能体）、Java / Python（目标）、TypeScript（仪表盘）
**前置课程：** 第 5 阶段（NLP）、第 7 阶段（Transformers）、第 11 阶段（LLM 工程）、第 13 阶段（工具）、第 14 阶段（智能体）、第 15 阶段（自主系统）、第 17 阶段（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P14 · P15 · P17
**时间：** 30 小时

## 问题

大规模代码迁移是 2026 年编程智能体最清晰的生产应用之一。真实标准是显而易见的（迁移后测试套件是否通过？），收益是真实的（Java 8 机群迁移是一个团队规模的项目），基准是公开的（MigrationBench 50 仓库子集）。Moderne 的 OpenRewrite 处理确定性方面。智能体层处理 OpenRewrite 配方无法处理的一切：模糊重写、构建系统漂移、长尾语法、传递依赖破坏。

你将构建一个智能体，接收一个 Java 8 仓库（或 Python 2 仓库）并生成一个 CI 绿色的迁移分支。你将测量通过率、测试覆盖率保持、每个仓库的成本，并构建失败分类。与仅确定性基线的并排对比将告诉你智能体的价值实际在哪里。

## 概念

该管道有两层。**确定性基底层**（Java 用 OpenRewrite，Python 用 libcst）安全地运行大部分机械重写：导入、方法签名、空安全编辑、try-with-resources、已弃用 API 替换。它速度快且生成可审计的差异。**智能体层**（OpenAI Agents SDK 或 LangGraph 配合 Claude Opus 4.7 和 GPT-5.4-Codex）处理配方无法处理的情况：构建文件升级（Maven/Gradle/pyproject）、传递依赖冲突、测试不稳定、自定义注解。

每个仓库获得一个预装目标运行时的 Daytona 沙箱。智能体迭代：运行构建、分类失败、应用修复、重新运行。硬性限制：每个仓库 30 分钟、8 美元、20 个智能体回合。如果所有测试通过且覆盖率差异不为负，分支打开一个 PR。否则，仓库被归档到带有证据的失败类别下。

失败分类是交付物。在 50 个仓库中，什么出了问题？传递依赖？自定义注解？构建工具版本？与迁移无关的测试不稳定？每个类别获得计数和示例差异。未来的配方作者可以针对前三名。

## 架构

```
目标仓库
      |
      v
OpenRewrite / libcst 确定性配方
   （安全、快速、可审计，约 70-80% 的修复）
      |
      v
每分支 Daytona 沙箱
      |
      v
智能体循环（Claude Opus 4.7 / GPT-5.4-Codex）：
   - 运行构建 -> 捕获失败
   - 分类失败（构建、测试、lint）
   - 应用修复（补丁或重试配方）
   - 重新运行
   - 预算：30 分钟、8 美元、20 个回合
      |
      v
测试 + 覆盖率差异门控
      |
      v（通过）
打开 PR
      |
      v（失败）
归档到失败类别 + 附加复现
```

## 技术栈

- 确定性基底层：OpenRewrite（Java）或 libcst（Python）
- 智能体：OpenAI Agents SDK 或 LangGraph 配合 Claude Opus 4.7 + GPT-5.4-Codex
- 沙箱：每分支 Daytona devcontainer，预装目标运行时（Java 17 / Python 3.12）
- 构建系统：Maven、Gradle、uv（Python）
- 基准：Amazon MigrationBench 50 仓库子集（Java 8 到 17）、Google App Engine Py2-to-Py3 仓库
- 测试套件：并行运行器，通过 Jacoco（Java）或 coverage.py（Python）测量覆盖率
- 可观测性：Langfuse + 每个仓库的追踪包，包含每个差异块
- 仪表盘：失败分类仪表盘，包含每类计数和示例差异

## 构建步骤

1. **配方阶段。** 首先运行 OpenRewrite（Java）或 libcst（Python）配方。捕获 70-80% 的机械迁移。作为"配方"提交进行提交。

2. **构建试验。** Daytona 沙箱：安装目标运行时，运行构建。如果绿色，跳到测试。如果红色，交给智能体。

3. **智能体循环。** LangGraph 配合工具：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。智能体分类失败（依赖、语法、测试、构建工具）并应用针对性修复。重新运行。

4. **预算上限。** 每个仓库 30 分钟墙钟时间、8 美元成本、20 个智能体回合。任何超限都会停止并归档到"预算耗尽"下，附带当前差异。

5. **测试 + 覆盖率门控。** 构建变绿后，运行测试套件。将覆盖率与基础仓库进行比较。如果覆盖率下降超过 2%，归档到"覆盖率回退"下。

6. **PR 打开。** 成功后，推送分支，打开 PR，附带差异以及应用了哪些配方和智能体编写了哪些提交的摘要。

7. **失败分类。** 对于每个失败的仓库，标记类别：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。构建仪表盘。

8. **50 仓库运行。** 在 MigrationBench 子集上执行。报告每类通过率、每仓库成本、覆盖率保持，以及与仅确定性基线的对比。

## 使用方式

```
$ migrate legacy-java-service --target java17
[配方]    已应用 27 个重写（JUnit 4->5、HashMap 初始化器、try-with-resources）
[构建]    失败：找不到符号 sun.misc.BASE64Encoder
[智能体]  回合 1 分类：removed_jdk_api
[智能体]  回合 2 应用：sun.misc.BASE64Encoder -> java.util.Base64
[构建]    通过
[测试]    412/412 通过；覆盖率 84.1% -> 84.3%
[PR]      已打开 #1841  成本=$3.20  回合=4
```

## 交付标准

`outputs/skill-migration-agent.md` 是交付物。给定一个仓库，它执行确定性配方，然后运行智能体循环以生成绿色迁移分支，或将仓库归档到分类类别下。

| 权重 | 标准 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通过率 | 50 仓库子集 pass@1 |
| 20 | 测试覆盖率保持 | 相对于基础仓库的平均覆盖率差异 |
| 20 | 每个迁移仓库的成本 | 通过运行的 $/仓库 |
| 20 | 智能体/确定性工具集成 | OpenRewrite 处理 vs 智能体编写的修复比例 |
| 15 | 失败分析报告 | 分类完整性及示例 |
| **100** | | |

## 练习

1. 仅使用 OpenRewrite（无智能体）运行迁移管道。将通过率与完整管道进行比较。识别智能体单独产生差异的案例。
2. 实现"lint 清洁"检查：迁移后，运行风格 linter（Java 用 spotless，Python 用 ruff）。如果出现新的 lint 错误，则 PR 失败。测量覆盖率保持但风格回退的比率。
3. 添加"最小差异"优化器：在智能体的分支通过测试后，通过第二次修剪不必要的更改。报告差异大小的减少。
4. 扩展到第三种迁移：Node 18 到 Node 22。重用沙箱包装；将配方层替换为自定义 codemod。
5. 将首次绿色构建时间（TTFGB）作为 UX 指标进行测量。目标：p50 低于 10 分钟。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------------|------------------------|
| 确定性基底层 | "配方引擎" | OpenRewrite / libcst：具有安全保证的声明式 AST 重写 |
| Codemod | "代码修改程序" | 机械地更改源代码的重写规则 |
| 构建漂移 | "工具版本偏差" | 主要版本之间微妙的 Maven / Gradle / uv 行为变化 |
| 失败类别 | "分类桶" | 仓库未迁移的标记原因：依赖、语法、测试、构建工具、预算 |
| 覆盖率差异 | "覆盖率保持" | 从基础仓库到迁移分支的测试覆盖率百分比变化 |
| 智能体回合 | "工具调用轮次" | 智能体循环中的一个规划 -> 行动 -> 观察周期 |
| 预算耗尽 | "触及上限" | 仓库消耗了其 30 分钟/8 美元/20 回合限制而未通过 |

## 扩展阅读

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年规范基准
- [Moderne.io OpenRewrite 平台](https://www.moderne.io) — 确定性基底层参考
- [OpenRewrite 文档](https://docs.openrewrite.org) — 配方编写
- [Grit.io](https://www.grit.io) — 备选 codemod DSL
- [OpenAI 沙箱化迁移 cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 参考
- [Google App Engine Py2 到 Py3 迁移器](https://cloud.google.com/appengine) — 备选迁移基准
- [libcst](https://github.com/Instagram/LibCST) — Python 确定性基底层
- [Daytona 沙箱](https://daytona.io) — 参考每分支沙箱
