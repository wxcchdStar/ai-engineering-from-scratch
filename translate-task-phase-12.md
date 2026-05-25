# Phase 12 翻译任务

## 目标

将 `phases/12-multimodal-ai/` 下所有 `docs/en.md` 翻译为简体中文，输出为同目录下的 `docs/zh.md`。

## 已完成（01-16）

| # | Lesson | 状态 |
|---|--------|------|
| 01 | Vision Transformers — Patch Tokens | ✅ |
| 02 | CLIP & Contrastive Pretraining | ✅ |
| 03 | BLIP-2 — Q-Former Bridge | ✅ |
| 04 | Flamingo — Gated Cross-Attention | ✅ |
| 05 | LLaVA — Visual Instruction Tuning | ✅ |
| 06 | Any-Resolution — Patch-n'-Pack | ✅ |
| 07 | Open-Weight VLM Recipes | ✅ |
| 08 | LLaVA-OneVision — Single/Multi/Video | ✅ |
| 09 | Qwen-VL Family — Dynamic FPS | ✅ |
| 10 | InternVL3 — Native Multimodal | ✅ |
| 11 | Chameleon — Early Fusion Tokens | ✅ |
| 12 | Emu3 — Next Token for Generation | ✅ |
| 13 | Transfusion — Autoregressive + Diffusion | ✅ |
| 14 | Show-o — Discrete Diffusion Unified | ✅ |
| 15 | Janus-Pro — Decoupled Encoders | ✅ |
| 16 | MIO — Any-to-Any Streaming | ✅ |
| 17 | Video-Language Temporal Grounding | ✅ |
| 18 | Long Video Million Token | ✅ |
| 19 | Audio-Language Whisper to AF3 | ✅ |
| 20 | Omni Models Thinker-Talker | ✅ |
| 21 | Embodied VLAs OpenVLA Pi0 Groot | ✅ |
| 22 | Document Diagram Understanding | ✅ |
| 23 | ColPali Vision Native RAG | ✅ |
| 24 | Multimodal RAG Cross Modal | ✅ |
| 25 | Multimodal Agents Computer-Use (Capstone) | ✅ |

## 全部完成 ✅

Phase 12 共25课翻译已全部完成。

## 工作流（每篇）

使用 `translate-doc` skill 翻译即可。该 skill 会自动完成 split → 并行翻译 → merge → cleanup 全流程。

## 翻译规范

- **保持不变**：Markdown 格式、Mermaid 图表、Python/JSON 代码块、链接、论文标题/作者名/URL
- **仅翻译**：代码块外的英文叙述性文本
- **术语处理**：首次出现用"中文 (English)"，后续直接用中文；API 名称、框架名、模型名保持英文
- **"What people say / What it actually means" 表格**：左列口语化，右列准确释义

## 项目约束

- 基础路径：`phases/12-multimodal-ai/`
- 输入：`{NN-lesson-name}/docs/en.md`
- 输出：`{NN-lesson-name}/docs/zh.md`
- 按序号逐一翻译
