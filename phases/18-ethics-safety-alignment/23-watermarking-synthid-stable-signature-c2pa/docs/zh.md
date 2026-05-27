# 水印 —— SynthID、Stable Signature、C2PA

> 三种技术构成了 2026 年 AI 生成内容溯源。SynthID（Google DeepMind）——图像水印于 2023 年 8 月推出，文本+视频于 2024 年 5 月推出（Gemini + Veo），文本于 2024 年 10 月通过 Responsible GenAI Toolkit 开源，统一多模态检测器于 2025 年 11 月随 Gemini 3 Pro 推出。文本水印以不可察觉的方式调整下一个 token 的采样概率；图像/视频水印在压缩、裁剪、滤镜、帧率变化后仍然存在。Stable Signature（Fernandez 等人，ICCV 2023，arXiv:2303.15435）——微调潜在扩散解码器，使每个输出包含固定消息；裁剪（10% 内容）的生成图像在 FPR<1e-6 下检测率 >90%。后续"Stable Signature is Unstable"（arXiv:2405.07145，2024 年 5 月）——微调在保持质量的同时移除水印。C2PA —— 加密签名、防篡改的元数据标准（C2PA 2.2 Explainer 2025）。水印和 C2PA 是互补的：元数据可以被剥离但携带更丰富的溯源信息；水印在转码后仍然存在但携带的信息较少。

**类型：** 构建
**语言：** Python（标准库，token 水印嵌入 + 检测）
**前置知识：** Phase 10 · 04（采样），Phase 01 · 09（信息论）
**时间：** ~75 分钟

## 学习目标

- 描述 token 级水印（SynthID-text 风格）及其可检测的机制。
- 描述 Stable Signature 以及攻破它的 2024 年移除攻击。
- 陈述 C2PA 的角色以及为什么它与水印互补。
- 描述关键局限性：模型特定信号、改写下的鲁棒性以及保持语义的攻击（arXiv:2508.20228）。

## 问题

2023-2024 年，深度伪造和 AI 生成内容大规模进入政治和消费者语境。水印是提议的技术溯源信号：在创建时标记生成内容，稍后检测它们。2025 年证据：没有水印是无条件鲁棒的，但与 C2PA 元数据分层结合提供了可用的溯源故事。

## 概念

### 文本水印（SynthID-text 风格）

Kirchenbauer 等人 2023 的机制，由 Google 产品化：

1. 在每个解码步骤，对前 K 个 token 进行哈希，产生词汇表的伪随机分区，分为"绿色"和"红色"集合。
2. 通过向绿色 logits 添加 δ 来偏向绿色集合的采样。
3. 生成内容包含比随机更多的绿色 token。

检测：重新哈希每个前缀，计算生成内容中的绿色 token 数量，计算 z-score。z-score 对带水印文本 >0，对人类文本约等于 0。

属性：
- 对读者不可察觉（δ 足够小，质量损失很小）。
- 在有权访问词汇表分区函数的情况下可检测。
- 对改写不鲁棒——重写文本会破坏信号。

SynthID-text 于 2024 年 10 月通过 Google 的 Responsible GenAI Toolkit 开源。

### Stable Signature（图像）

Fernandez 等人 ICCV 2023。微调潜在扩散解码器，使每个生成图像包含嵌入在潜在表征中的固定二进制消息。检测通过神经解码器从潜在空间中解码。裁剪（至 10% 内容）的图像在 FPR<1e-6 下检测率 >90%。

2024 年 5 月"Stable Signature is Unstable"（arXiv:2405.07145）：微调解码器在保持图像质量的同时移除水印。对抗性生成后微调成本低廉；水印的对抗鲁棒性有限。

### SynthID 统一检测器（2025 年 11 月）

随 Gemini 3 Pro 推出：一个多模态检测器，在一个 API 中读取来自文本、图像、音频和视频的 SynthID 信号。统一了 Google 的溯源技术栈。

### C2PA

内容溯源与真实性联盟（Coalition for Content Provenance and Authenticity）。加密签名、防篡改的元数据标准。C2PA 2.2 Explainer（2025）。C2PA 清单记录溯源声明（谁创建、何时、什么转换），由创建者的密钥签名。

与水印互补：
- 元数据可以被剥离；水印不能（轻易）。
- 元数据丰富（完整溯源链）；水印携带比特。
- C2PA 依赖平台采用；水印自动嵌入。

Google 在搜索、广告和"关于此图像"中整合了两者。

### 局限性

- **模型特定。** SynthID 为来自启用 SynthID 的模型的生成内容加水印。来自没有 SynthID 的模型的生成内容没有水印，因此"无 SynthID 信号"不是真实性的证明。
- **改写。** 文本水印在保持语义的改写下无法存活。
- **转换攻击。** arXiv:2508.20228（2025）展示了破坏文本水印和许多图像水印的保持语义攻击。
- **微调移除。** 根据"Stable Signature is Unstable"，生成后微调移除嵌入的水印。

### EU AI Act 第 50 条

AI 生成内容标注的透明度准则（第一稿 2025 年 12 月，第二稿 2026 年 3 月，预计最终版 2026 年 6 月，参见[European Commission 状态页面](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)）。该准则截至 2026 年 4 月仍为草案，时间表可能变化。要求技术层的监管层。深度伪造必须被标注。

### 在 Phase 18 中的位置

第 22-23 课关于模型发出什么（私有数据、溯源信号）。第 27 课涵盖训练数据治理。第 24 课是要求这些技术措施的监管框架。

## 使用它

`code/main.py` 构建一个玩具文本水印。Token 是整数 0..N-1；带水印的采样偏向哈希定义的绿色集合。检测器计算绿色 token 的 z-score。你可以观察 1000 token 生成内容的检测，观察改写破坏信号，并测量人类文本的误报率。

## 交付它

本课产出 `outputs/skill-provenance-audit.md`。给定一个带有溯源声明的内容部署，它审计：水印机制（如有）、C2PA 签名链（如有）、每个的对抗鲁棒性以及每种模态的覆盖。

## 练习

1. 运行 `code/main.py`。报告带水印的 1000 token 生成内容 vs 人类撰写文本的 z-score。找出 95% 置信阈值下的误报率。

2. 实现一个用同义词替换 30% token 的改写攻击。重新测量 z-score。

3. 阅读 Kirchenbauer 等人 2023 第 6 节关于鲁棒性的内容。为什么文本水印在改写下失败而图像水印在裁剪后仍然存在？

4. 设计一个使用 SynthID-text + C2PA 元数据的部署。描述消费者看到的溯源链。找出每个组件的一个失败模式。

5. 2024 年"Stable Signature is Unstable"结果表明微调移除了图像水印。设计一个限制此攻击的部署控制——例如，要求签名发布微调检查点。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SynthID | "Google 的水印" | 跨模态溯源信号；文本、图像、音频、视频 |
| Token 水印 | "Kirchenbauer 风格" | 通过绿色 token z-score 可检测的偏向采样文本水印 |
| Stable Signature | "图像水印" | 微调解码器水印；ICCV 2023 |
| C2PA | "那个元数据标准" | 加密签名、防篡改的溯源元数据 |
| 改写鲁棒性（Paraphrase robustness） | "改写是否破坏它" | 文本水印属性；目前有限 |
| 微调移除（Fine-tune removal） | "对抗性去水印" | 通过解码器微调移除图像水印的攻击 |
| 跨模态检测器（Cross-modal detector） | "统一 SynthID" | 2025 年 11 月跨模态统一 API |

## 扩展阅读

- [Kirchenbauer 等人 — A Watermark for Large Language Models（ICML 2023，arXiv:2301.10226）](https://arxiv.org/abs/2301.10226) — token 水印机制
- [Fernandez 等人 — Stable Signature（ICCV 2023，arXiv:2303.15435）](https://arxiv.org/abs/2303.15435) — 图像水印论文
- ["Stable Signature is Unstable"（arXiv:2405.07145）](https://arxiv.org/abs/2405.07145) — 移除攻击
- [Google DeepMind — SynthID](https://deepmind.google/models/synthid/) — 跨模态水印
- [C2PA 2.2 Explainer（2025）](https://c2pa.org/specifications/specifications/2.2/explainer/Explainer.html) — 元数据标准
