---
title: 来源、图像许可与知识点覆盖
date: '2026-09-22'
tags:
- 多模态算法
- 来源
- 图像许可
- 知识覆盖
summary: 记录原论文、图解仓库、面经证据、17幅配图及教程范围。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-7ff656c25143" style="display:block;scroll-margin-top:6rem"></span>

本页记录本系列的知识来源、章节对应和公开配图使用条件。技术核对日期为 2026-09-22；面经采用既有归档，不宣称本轮重新抓取全部小红书帖子。学习入口见[00-图解大模型算法与ZealD面经总览](/notes/multimodal-interview-guide/)。

<span id="mm-37fcc83bd40c" style="display:block;scroll-margin-top:6rem"></span>

## 1. 三类材料各承担什么证据

| 材料 | 用途 | 不支持的结论 |
|---|---|---|
| [LLM-RL-Visualized 图解仓库](https://github.com/changyeyu/LLM-RL-Visualized) | 知识路线与原图辅助；文字中的省略以论文补齐 | 一张示意图等于完整实现；本系列覆盖仓库所有分支 |
| 原论文、固定版本报告、官方 API/模型卡 | 公式、模块、版本配置及实现语义的依据 | 论文在特定实验上的收益会无条件迁移到任何项目 |
| 小红书公开面经/学习笔记的既有归档 | 决定复习方向、提取问题与项目线索 | 招聘方认证、高频率统计、作者诊断已被对照实验证实 |

图解仓库本轮基准快照为 [33cf3e98c04a60dd2ce8e8b03884e6a7b990694e](https://github.com/changyeyu/LLM-RL-Visualized/tree/33cf3e98c04a60dd2ce8e8b03884e6a7b990694e)。正文的便捷章节链接仍可能指向移动分支；需要追溯时以该快照核对。数学例题、解释和自绘图由本教程重新组织，不把原作者配图等同于其认可本教程全部观点。

模型报告固定为 Qwen2.5-VL v1、Qwen3-VL v1、DeepSeek-V2 v5；Llama3.1使用官方模型卡，并注明访问日期。不能根据仓库路径名称猜版本：旧 Qwen2.5-VL 仓库入口在本轮访问时会转向 Qwen3-VL，故采用明确 arXiv 版本。

<span id="mm-902218f66657" style="display:block;scroll-margin-top:6rem"></span>

## 2. 仓库内容到知识点的覆盖表

“展开”表示正文有机制解释及相应推导/算例/边界；“部分”表示只解释与本系列相关的一面；“未纳入”不表示该主题不重要。

| 图解仓库主题 | 本系列展开到哪里 | 对应章节与可检验产物 | 覆盖程度 |
|---|---|---|---|
| 输入/输出、LLM结构、多模态入口（3–8、13） | token/embedding/logits、shift、残差、视觉接入 | 01、02、06；shape链与错位标签表 | 展开 |
| LoRA（10–11）及量化（105） | rank约束、缩放、首步梯度、冻结反传、QLoRA存储机制 | 02、07；参数量及梯度例 | 展开；非完整NF4内核实现 |
| SFT loss、指令与packing（14–16） | 链式概率、CE求导、mask、token权重、跨样本泄漏 | 02、07；逐位置监督与累积算例 | 展开 |
| DPO/β/隐式奖励（17–22） | KL约束最优分布→BT偏好→四log-prob→梯度 | 02；完整推导、β反例 | 展开 |
| Greedy/Beam/采样（26–30、117） | 概率选择、温度、top-k/p、长度与实验口径 | 04；归一化例 | 展开；无搜索服务实现 |
| RL基础、回报、Q/V、MC/TD（35–50） | 状态/动作/转移、Bellman、估计与期望、bootstrap | 03；同轨迹MC/TD/GAE手算 | 展开；不构建通用RL环境 |
| 策略梯度（55及策略梯度PDF） | score-function、reward-to-go、折扣、baseline | 03；逐步推导与两动作例 | 展开 |
| Actor-Critic/GAE/PPO/IS（64–72） | old概率、局部surrogate、双mask、clip、critic target | 03；正负优势四情形与更新伪代码 | 展开；TRPO仅背景，不推完整二阶优化 |
| RLHF/四模型/KL（76–88） | 模型职责、reward/reference/old区别、KL采样边界 | 03、07；信息流与显存角色账单 | 展开；非完整reward模型训练课程 |
| PPO与GRPO、结果/过程反馈（72、95） | 组标准化、token目标、长度、KL、零值非零梯度 | 03；完整算例与失败诊断 | 展开结果监督；过程反馈仅基础对照 |
| 蒸馏（93–94、104） | logits温度、KL与硬标签、序列蒸馏区别 | 05 Q12；温度数例 | 部分；未训练教师/学生 |
| 累积/重计算（106–109） | token加权、激活生命周期、checkpoint边界 | 07；等价条件与损失加权例 | 展开 |
| MHA/GQA/MQA/MLA（111） | head共享、KV账单、低秩吸收与解耦RoPE | 01、04、08；数值缓存比 | 展开 |
| Norm/SwiGLU/RoPE（113–115、118、120–121） | 归一化轴、激活门控、相对旋转与位置偏移 | 01、08；非零均值例与角差推导 | 展开；非所有位置外推变体 |
| Benchmark/分类指标（110、119） | 视频证据、tIoU、Recall@K、P/R/F1与不确定性 | 06、07、10；分桶和配对评估 | 部分；不遍历所有榜单/AUC理论 |

仓库编号用其 README 的 header 编号标识，便于查找，不是本文段落编号。以上条目是原主题到本教程的映射，不代表逐图复制全部原文。

<span id="mm-a261140cf60c" style="display:block;scroll-margin-top:6rem"></span>

## 3. 为视频算法补充的原始资料

图解仓库的多模态总图不能代替视频专项。以下原始资料承担具体机制依据：

| 来源 | 本系列采用的知识点 | 学习位置 |
|---|---|---|
| [Transformer](https://arxiv.org/abs/1706.03762) | 缩放点积、多头与自回归mask | 01 |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[GQA](https://arxiv.org/html/2305.13245v3) | 旋转相对位置、分组KV | 01、04 |
| [RMSNorm](https://arxiv.org/abs/1910.07467)、[GLU Variants](https://arxiv.org/abs/2002.05202) | RMS尺度与SwiGLU门控 | 01 |
| [ViT](https://arxiv.org/abs/2010.11929) | patch投影、空间位置与分类表示 | 06 |
| [CLIP](https://arxiv.org/abs/2103.00020)、[SigLIP](https://arxiv.org/html/2303.15343v2) | 双向CE与配对sigmoid损失 | 06 |
| [BLIP-2](https://arxiv.org/html/2301.12597v3) | query瓶颈与ITC/ITM/ITG监督 | 06 |
| [LLaVA](https://arxiv.org/abs/2304.08485)、[LLaVA-1.5](https://arxiv.org/html/2310.03744v2) | 分阶段视觉接入、线性到MLP连接器 | 06 |
| [TimeSformer](https://arxiv.org/abs/2102.05095)、[VideoMAE](https://arxiv.org/abs/2203.12602) | 时空分解；tubelet、tube mask、遮挡重建损失与算例 | 06；核心机制，不含完整训练复现 |
| [Qwen2.5-VL v1](https://arxiv.org/html/2502.13923v1)、[Qwen3-VL v1](https://arxiv.org/html/2511.21631v1) | 动态分辨率、时间对齐、MRoPE、DeepStack | 08 |
| [DeepSeek-V2 v5](https://arxiv.org/html/2405.04434v5)、[Llama3.1模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) | 固定文本骨干对照、MLA与GQA | 08 |
| [TempCompass](https://arxiv.org/abs/2403.00476)、[Video-MME](https://arxiv.org/abs/2405.21075) | 时间属性与不同视频长度/模态条件评测 | 06、08 |
| [LoRA](https://arxiv.org/abs/2106.09685)、[QLoRA](https://arxiv.org/abs/2305.14314)、[DPO](https://arxiv.org/abs/2305.18290) | 参数高效适配与偏好学习 | 02 |
| [GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 估计器、裁剪更新、组相对学习 | 03 |
| [FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention](https://arxiv.org/abs/2309.06180) | IO感知计算与缓存管理 | 04、07 |
| [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 布尔mask、非方形因果可见关系的API核对 | 04 |

正文推导和算例是本教程组织的教学说明，不是对每篇论文的逐页译文。读论文的追问表在[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)；技术理解仍需区分定义、数学恒等式、优化近似与实验证据。

<span id="mm-4287145288c6" style="display:block;scroll-margin-top:6rem"></span>

## 4. 面经证据索引与日期规则

ZealD公开账号：[个人主页](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5)。本轮复习采用2026-09-20归档中的正文与既有后续记录。公开QA仅作技术主题概括，没有把他的完整帖子重发到本博客。

| 证据标记 | 本系列使用规则 |
|---|---|
| Z | ZealD正文有归档，表述为作者自述 |
| M | 其他作者面经正文有归档，表述为帖子自述 |
| R | 只有既有转述，缺原始评论等信息，不提升为确认原题 |
| S | 自学、求助、项目复盘，不写成被问过 |
| E | 本教程扩展问题、数字例、假设实验，不归给原作者 |

原帖入口及逐题对应见[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)。平台链接可能要求登录，已剥离会话参数。部分归档日期曾由note ID推定，不能作为精确发帖或面试时间，也不据此做频率排名。没有完整日志时，作者或评论者的GRPO归因仅作为假设。

<span id="mm-9c69fd41be9c" style="display:block;scroll-margin-top:6rem"></span>

## 5. 17 张配图的来源与使用位置

以下文件已保存到本套笔记的本地配图目录。01–14 是仓库 PNG，15–17 是本教程自绘 SVG；原图不作去水印、裁署名或重标作者处理。

| 本地图名 | 原仓库章节 / 来源 | 使用位置 |
|---|---|---|
| 01-多模态模型结构.png | [多模态结构，header-7](https://github.com/changyeyu/LLM-RL-Visualized#header-7) | 总览 |
| 02-LLM结构全视图.png | [LLM结构，header-3](https://github.com/changyeyu/LLM-RL-Visualized#header-3) | 01 |
| 03-MHA-GQA-MQA-MLA.png | [注意力类型，header-111](https://github.com/changyeyu/LLM-RL-Visualized#header-111) | 01、08 |
| 04-RoPE位置编码.png | [RoPE，header-120](https://github.com/changyeyu/LLM-RL-Visualized#header-120) | 01 |
| 05-SwiGLU.png | [SwiGLU，header-118](https://github.com/changyeyu/LLM-RL-Visualized#header-118) | 01 |
| 06-SFT交叉熵.png | [SFT Loss，header-14](https://github.com/changyeyu/LLM-RL-Visualized#header-14) | 02 |
| 07-DPO训练全景.png | [DPO全景，header-20](https://github.com/changyeyu/LLM-RL-Visualized#header-20) | 02 |
| 08-奖励回报价值.png | [奖励回报价值，header-44](https://github.com/changyeyu/LLM-RL-Visualized#header-44) | 03 |
| 09-价值函数Q和V.png | [Q与V，header-45](https://github.com/changyeyu/LLM-RL-Visualized#header-45) | 03 |
| 10-策略梯度.png | [策略梯度，header-55](https://github.com/changyeyu/LLM-RL-Visualized#header-55) | 03 |
| 11-GAE.png | [GAE，header-66](https://github.com/changyeyu/LLM-RL-Visualized#header-66) | 03 |
| 12-PPO-Clip.png | [PPO-Clip，header-70](https://github.com/changyeyu/LLM-RL-Visualized#header-70) | 03 |
| 13-GRPO-PPO.png | [PPO与GRPO，header-72](https://github.com/changyeyu/LLM-RL-Visualized#header-72) | 03 |
| 14-PPO四模型.png | [四模型合作，header-82](https://github.com/changyeyu/LLM-RL-Visualized#header-82) | 03（07讨论对应显存角色） |
| 15-视频采帧与证据.svg | 本教程自绘：事件覆盖与信息入口 | 04、06 |
| 16-视觉接入架构.svg | 本教程自绘：视觉塔、连接器与LLM | 06 |
| 17-注意力手算与缓存掩码.svg | 本教程自绘：两token数值与缓存可见表 | 01、04 |

每次引用图后都有读图说明：告诉读者沿哪条箭头、哪些颜色和哪些维度理解，同时注明原图省略的部分。图中示例超参数不是通用规格。

原仓库[图片 LICENSE](https://github.com/changyeyu/LLM-RL-Visualized/blob/33cf3e98c04a60dd2ce8e8b03884e6a7b990694e/LICENSE)允许在所列条件下分享、修改和衍生创作；网络使用须保留图内原创作者与仓库信息，禁止直接商业用途。这里按个人非商业学习笔记引用，不将其称为MIT/Apache或无条件可商用。若未来用于付费课程、商业推广或正式出版，应重新核对适用条件并取得所需授权。

<span id="mm-88b5ba0299e6" style="display:block;scroll-margin-top:6rem"></span>

## 6. 明确没有冒充完成的部分

本系列没有提供所有模型的训练源码、完整PPO/GRPO训练器或NF4/CUDA内核；伪代码用于解释固定目标和梯度职责。手算与数值验证证明的是列出的数学算例，不证明已训练真实大模型，不证明视频方案已取得涨点。

Agent框架、RAG服务、Function Calling、多智能体开发不纳入当前方向。DQN/DDPG、完整TRPO二阶求解、MARL、MCTS、扩散生成模型也没有系统讲授。VideoMAE 已在06展开核心机制，但没有完整训练复现；DINOv2、过程奖励、AUC等只在相邻知识需要时作为对照或延伸，未来若要系统学习，应另立章节。不能只凭本页出现了名字就视为已掌握。

新内容的纳入标准是：有原始来源、与目标题目相关、能解释机制并有自测。文章数量、关键词覆盖和模型发布时间都不能替代这个标准。
