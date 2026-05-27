# Git 与协作 (Git & Collaboration)

> 版本控制不是可选的。你在这里构建的每个实验、每个模型、每节课都会被跟踪。

**类型：** 学习 (Learn)
**语言：** --
**前置要求：** 阶段 0，第 1 课
**时间：** 约 30 分钟

## 学习目标

- 配置 git 身份信息并使用 add、commit、push 的日常工作流
- 创建和合并分支 (branches) 以进行隔离实验，不影响主分支
- 编写 `.gitignore` 文件，排除模型检查点和大文件
- 使用 `git log` 浏览提交历史，理解项目演进过程

## 问题

你即将在 20 个阶段中编写数百个代码文件。没有版本控制，你会丢失工作、破坏无法撤销的内容，并且无法与他人协作。

Git 是工具。GitHub 是代码存放的地方。本课涵盖你在这个课程中需要的内容，不多不少。

## 概念

```mermaid
sequenceDiagram
    participant WD as 工作目录 (Working Directory)
    participant SA as 暂存区 (Staging Area)
    participant LR as 本地仓库 (Local Repo)
    participant R as 远程仓库 (Remote, GitHub)
    WD->>SA: git add
    SA->>LR: git commit
    LR->>R: git push
    R->>LR: git fetch
    LR->>WD: git pull
```

记住三件事：
1. 经常保存（`git commit`）
2. 推送到远程（`git push`）
3. 为实验创建分支（`git checkout -b experiment`）

## 构建

### 步骤 1：配置 git

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

### 步骤 2：日常工作流

```bash
git status
git add file.py
git commit -m "Add perceptron implementation"
git push origin main
```

### 步骤 3：为实验创建分支

```bash
git checkout -b experiment/new-optimizer

# ... 进行修改、提交 ...

git checkout main
git merge experiment/new-optimizer
```

### 步骤 4：使用本课程仓库

```bash
git clone https://github.com/rohitg00/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

git checkout -b my-progress
# 逐步完成课程，提交你的代码
git push origin my-progress
```

## 使用

对于本课程，你只需要以下这些命令：

| 命令 | 使用场景 |
|------|---------|
| `git clone` | 获取课程仓库 |
| `git add` + `git commit` | 保存你的工作 |
| `git push` | 备份到 GitHub |
| `git checkout -b` | 尝试新东西而不破坏主分支 |
| `git log --oneline` | 查看你做了什么 |

就这些。你不需要 rebase、cherry-pick 或 submodules 来完成本课程。

## 练习

1. 克隆此仓库，创建一个名为 `my-progress` 的分支，创建一个文件，提交它，推送它
2. 创建一个 `.gitignore` 文件，排除模型检查点文件（`.pt`、`.pth`、`.safetensors`）
3. 使用 `git log --oneline` 查看此仓库的提交历史，了解课程是如何逐步添加的

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 提交 (Commit) | "保存" | 项目在某个时间点的完整快照 |
| 分支 (Branch) | "一个副本" | 指向某个提交的指针，随着你的工作向前移动 |
| 合并 (Merge) | "合并代码" | 将一个分支的更改应用到另一个分支 |
| 远程 (Remote) | "云端" | 托管在其他地方（GitHub、GitLab）的仓库副本 |
