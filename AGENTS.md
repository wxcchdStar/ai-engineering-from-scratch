# AGENTS.md

> **Project:** AI Engineering from Scratch — 435-lesson, 20-phase open-source curriculum
> **Core constraints:** 课程内容英文，代码可运行，每课产出可复用 artifact

## Mission

- 体系化 AI 工程教育课程（数学基础 → 自主系统），覆盖 Python/TypeScript/Rust/Julia 四种语言
- 每课遵循固定结构：`phases/XX-phase-name/NN-lesson-name/{code/, docs/en.md, quiz.json}`
- 产出物（prompts/skills/agents/MCP servers）归档至 `outputs/` 目录

## Toolchain

| Action | Command | Authority |
| --- | --- | --- |
| Lesson audit | `python3 scripts/audit_lessons.py` | CI 必过 |
| Catalog rebuild | `python3 scripts/build_catalog.py` | `catalog.json` 必须与文件系统同步 |
| README counts | `python3 scripts/check_readme_counts.py` | README 中的课程计数必须准确 |
| Site build | `node site/build.js` | 解析 README.md + ROADMAP.md 生成 site/data.js |
| Link check | `python3 scripts/link_check.py` | — |
| Python deps | `pip install -r requirements.txt` | requirements.txt |

## Judgment Boundaries

**NEVER**

- 修改 README.md / ROADMAP.md 中的 phase header 格式或 status glyph（✅ 🚧 ⬚），解析器依赖这些精确字符
- 在 lesson 目录外创建新的课程内容
- 删除或重命名已有的 phase/lesson 目录（会破坏 catalog.json 映射）
- 硬编码本地绝对路径到任何可提交文件中

**ASK**

- 新增 phase 或调整 phase 之间的依赖关系
- 修改 `scripts/` 下的审计逻辑
- 变更 `site/build.js` 的解析规则

**ALWAYS**

- 新增/修改 lesson 后运行 `python3 scripts/audit_lessons.py` 验证结构完整性
- 修改 README/ROADMAP 后运行 `node site/build.js` 并确认 `site/data.js` 仅时间戳变化
- lesson 文档使用英文（`docs/en.md`）
- 保持 quiz.json 与 docs/en.md 内容一致

## Karpathy 行为准则

**Think Before Coding — 先澄清再动手**

- 动手前先显式陈述假设。不确定时停下来问，不要猜。
- 存在多种理解时，列出选项让用户选择。

**Simplicity First — 最小代码解决问题**

- 不写超出需求的功能、不为单次使用的代码构建抽象层。
- 自检标准：一个资深工程师看了会不会说"太复杂了"？会就砍。

**Surgical Changes — 只改必须改的**

- 不"顺手改进"相邻代码、注释或格式。
- 匹配项目已有代码风格。
- diff 里每一行改动都应能直接追溯到用户的需求。

## Sub-agent 编排策略

- 将研究、探索、并行分析类任务优先委托给 sub-agent 执行。
- 对可拆分且互不依赖的任务，默认拆分后并行分派给多个 sub-agent。
- 每个 sub-agent 只专注一个任务，避免跨目标执行。

## Language & Output

- 与用户交流使用中文
- 代码注释、lesson 文档、commit message 使用英文
- Git 提交信息简洁，聚焦 "why" 而非 "what"

## Context Map

```yaml
project_structure:
  phases: "20 个阶段的课程内容，每课含 code/ docs/ quiz.json"
  outputs: "课程产出物归档：prompts/ skills/ agents/ mcp-servers/"
  scripts: "CI 与维护脚本：审计、构建目录、链接检查"
  site: "静态站点生成，解析 README/ROADMAP 生成 data.js"
  glossary: "术语表与常见误区"
  .claude/skills: "AI 技能定义（通过 .trae/skills 软链接共享）"
```
