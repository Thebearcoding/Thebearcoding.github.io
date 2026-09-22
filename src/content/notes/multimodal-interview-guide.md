---
title: 多模态算法学习手册：从原理到面试追问
date: '2026-09-22'
tags:
- 多模态算法
- 大模型面经
- ZealD
- 图解大模型算法
summary: 按知识依赖串起数学、模型、视觉视频、训练和面试的学习入口。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-6e4004f23905" style="display:block;scroll-margin-top:6rem"></span>

这套笔记围绕一个问题组织：**视频中的证据，怎样经过模型变成答案，又怎样通过训练和评估变得可靠？** 学习对象是视觉/视频多模态算法，不是 Agent 框架开发。以余昌叶的 [LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized#header-1) 为图解入口，用论文和官方文档补足机制、推导与边界，再与 ZealD 和已归档面经中的问题连接。

它不要求先会开发大模型系统，但不能跳过矩阵、概率和梯度。本系列提供数学预备课，正文中的小数字能在纸上计算。每个核心模块都尽量回答：要解决什么问题、输入输出是什么、公式为何成立、一个完整算例怎样算、在什么条件下会失败，以及如何检查自己是否学懂。

<span id="mm-38c8bb4d8450" style="display:block;scroll-margin-top:6rem"></span>

## 1. 从哪里开始，不要按文件编号硬读

文件编号保留了原有 Obsidian 链接；实际学习顺序由依赖决定。主线是数学→Transformer→视觉与视频→具体模型→监督训练→实验与面试；PPO/GRPO 是后训练补充线。

| 阅读入口 | 读完应该能够做到 | 对应练习 |
|---|---|---|
| [09-数学与张量预备课](/notes/multimodal-math-prerequisites/) | 区分矩阵乘法与逐元素乘法，算 softmax、对数概率，解释梯度与 KL | 第 1 天 |
| [01-模型骨干与多模态入口](/notes/transformer-attention-rope-gqa/) | 手算注意力，推 RoPE 角差，说明 GQA、RMSNorm、SwiGLU 改了什么 | 第 2–3 天 |
| [04-推理效率与手撕考点](/notes/prefill-decode-video-tokens/) | 画缓存 mask，算 KV 显存，解释首 token 与后续生成瓶颈 | 第 4 天 |
| [06-视觉视频算法面试专项](/notes/vision-video-algorithms/) | 从 patch 推到 CLIP/连接器，再解释采帧、时间、数据和评测 | 第 5–6 天 |
| [02-SFT与DPO的训练信号](/notes/multimodal-sft-lora-dpo/) | 对齐 labels，推 CE/LoRA 梯度和 DPO，识别监督与事实的错位 | 第 7–8 天 |
| [03-从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/) | 从回报推到 PG、GAE、PPO/GRPO，区分估计器、目标与实现 | 第 9–11 天 |
| [07-训练显存与实验排障](/notes/training-memory-debugging/) | 列显存账单，解释 AdamW、混精和累积，设计逐项排障 | 第 12 天 |
| [08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/) | 用固定版本比较 Qwen、DeepSeek、Llama，讲清 MLA/多模态位置与消融 | 第 13 天 |
| [05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/) | 从可追溯问题讲到机制、边界和项目证据，不混淆题源 | 第 14 天 |
| [10-十四天练习与参考解答](/notes/multimodal-fourteen-day-workbook/) | 完成 24 道编号练习及项目、模拟面试任务，按错因返工 | 全程使用 |
| [11-来源、图像许可与知识点覆盖](/notes/multimodal-sources-coverage/) | 查询知识来源、原图、模型版本和本系列未覆盖范围 | 按需查询 |

第一遍不用把所有公式都背下来。每到一个公式，先说出每个变量的含义，再跟算正文数字；第二遍遮住答案独立复现；第三遍换一个数值或前提，看原结论是否仍成立。练习册给出参考解答，目的是让错误暴露出来，而非只让你勾选“读过”。

<span id="mm-b7aa904ef127" style="display:block;scroll-margin-top:6rem"></span>

## 2. 用一张图建立全局，再逐条拆开

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-interview-guide/01-%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png" target="_blank" rel="noopener" aria-label="查看原图：多模态模型结构"><img src="/notes-assets/multimodal-interview-guide/01-%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84.png" alt="多模态模型结构" width="1316" height="990" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">多模态模型结构（点击查看原图）</figcaption>
</figure>

**读图路径**：图像、视频等原始信号经各自编码器转成表示，再进入语言模型参与生成。先指出视觉入口，接着找连接语言空间的部分，最后沿箭头追到输出。图中一根箭头不代表内部只有一步：视频还需要解码、采帧、真实时间信息、空间/时间压缩以及与问题共同条件化。

这幅图不能证明模型一定使用了视觉证据。它只说明可能的信息路径。若关键事件从未被采到，后面的 LLM 再大也没有那部分直接视觉输入；若答案标签经常靠字幕猜出，模型即使收到画面也可能走捷径；若评测只看语言流畅，错误还可能被奖励。第 06 章沿这条链展开，第 02/03 章解释训练信号，第 07 章讨论如何诊断。

图像为原仓库教学图，已保留余昌叶与仓库署名；来源及非商业使用条件见[11-来源、图像许可与知识点覆盖](/notes/multimodal-sources-coverage/)。高清图保存在本地配图目录，博客版本可点击原图放大。

<span id="mm-403617515e50" style="display:block;scroll-margin-top:6rem"></span>

## 3. 面经怎样决定重点，而不是代替教材

| 问题主题 | 当前证据 | 在哪里展开 |
|---|---|---|
| QKV 实现、位置编码、模型家族差异 | ZealD 归档正文自述，Z 类 | 01、08 章及 05 章 Q1–Q3 |
| 项目做什么、数据如何做、失败如何解决 | ZealD 归档正文自述，Z 类 | 05 章项目深挖，02/06/07 章 |
| Prefill/Decode 瓶颈 | 只有既有采集转述，评论原文待核，R 类 | 04 章；不称为本轮已核实原题 |
| ViT、CLIP、Q-Former/MLP、MRoPE、蒸馏、视频时间轴 | 其他作者已归档面经，M 类 | 05 章 Q6–Q14，06/08 章 |
| SFT 后 GRPO 不涨、优势估计疑问 | ZealD 自学/讨论，S 类 | 02、03、07 章 |
| 手算、反例、假设视频实验 | 本教程自行设计，E 类 | 各章和 14 天练习册 |

这里的“真实面经”指作者公开自述有归档，**不是招聘方核证，也不是高频统计**。自动提取的题库含 OCR 碎片等噪声，不能拿条目数量当真实问题数量。问题原帖链接、证据规则与完整 QA 见[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)。

比如“GRPO 不用 Critic”只说了一个差异。深入回答需要继续解释：同题采样的奖励如何归一化；为什么同组全错可能没有信号；为什么平均 surrogate 为零而梯度可以不为零；为什么旧策略和参考策略不能混淆；这些结论在实际 token loss、长度和 KL 约定下如何变化。第 03 章按这个标准展开。

<span id="mm-ef93b0840cda" style="display:block;scroll-margin-top:6rem"></span>

## 4. 两周计划的合理目标

建议每天分成三段：先读相应原理，再亲手做题，最后用自己的话复述并记录卡点。若数学基础弱，可以将一天拆成两天；14 天是学习安排，不是承诺两周掌握整个多模态领域。

第一周建立模型和视频证据链；第二周补齐训练、后训练、评价与项目追问。对视频算法岗位，01/04/06/08 章不能被 RL 长章挤掉；补充线推不完时可以延长，不要只背 PPO/GRPO 名词就继续增加新算法。

掌握一个模块至少要满足四项：

1. 能从输入画到输出，标出关键张量形状或概率条件。
2. 能独立完成一个数值例子，并说清公式用了哪些假设。
3. 能给出一个反例，说明该机制解决不了什么。
4. 能提出可观察的检查或控制变量实验，而不是只说“调参看看”。

“完全弄懂”应以这些能力判断，而不以读过多少页判断。本系列给出可核验的解释、练习与来源，但不把纸面推导等同于真实大规模训练经验。

<span id="mm-c675287ce112" style="display:block;scroll-margin-top:6rem"></span>

## 5. 本版范围与维护边界

本版深入展开多模态算法面试的共同骨架和已归档题目所指向的主题，**不宣称逐页改写整个图解仓库**。Agent 工程、RAG 服务编排、Function Calling、MARL/DDPG/MCTS 等暂不纳入；VideoMAE 已展开 tubelet、tube mask 和重建损失的核心机制，但不等于完整复现；DINOv2 仍是对比与延伸入口。范围逐项列在[11-来源、图像许可与知识点覆盖](/notes/multimodal-sources-coverage/)。

模型比较固定到论文/模型卡版本；不使用未经核实的“最新型号”传闻。Qwen2.5-VL 和 Qwen3-VL 是教学实例，不能据此推断不存在更新产品。新模型值得加入的条件是：有可靠原始资料，而且确实引入需要理解的新机制或面试追问。

返回时从本页选择卡住的模块。若要问进一步问题，带上章节、原式、你算到哪一步和具体疑问，能更快定位是数学、模型机制还是实验假设没有接上。
