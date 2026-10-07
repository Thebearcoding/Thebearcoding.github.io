---
title: 视频 LLM 的设计取舍：帧数、token 压缩与时间定位
date: '2026-10-06'
tags: [多模态算法, 视频理解, Video-LLM, M-RoPE, Token压缩, 时间定位]
summary: 以 Video-LLaVA、LLaVA-Video、Qwen2-VL、TimeChat、LLaMA-VID、LongVA 为例，讲视频 LLM 怎样在帧数、每帧 token、时间位置和长上下文之间做取舍，以及 Video-MME、MVBench 各自测什么。
draft: false
---

一个视频 LLM 的视觉 token 预算有限，要在“看多少帧”“每帧留多少 token”“模型怎样知道第几秒”三件事之间分配。这一篇不再推导采帧覆盖概率、帧号与真实时间的区别（见 [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) 第 6–7 节），也不再算 KV Cache 和 Prefill 的代价（见 [Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/) 第 4 节）。这里讲的是：几个有代表性的视频 LLM 各自在哪个环节做了什么选择，代价是什么，怎样用实验检验。Qwen2.5-VL 的绝对时间对齐 MRoPE 与 Qwen3-VL 的文字时间戳已在 [模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/) 讲过，本篇只讲 Qwen2-VL 这一代。

阅读入口：[多模态专题总览](/notes/multimodal-interview-guide/)。数值算例为教学构造；论文数字均注明表号。

## 1. 一张预算表：视频 LLM 的设计空间

输入是一段时长 $D$ 秒的视频和一个问题，输出是文本（答案、描述，或带时间戳的定位）。从像素到 LLM 输入，总视觉 token 数可以写成

$$
N_{\text{vis}}=\sum_{f=1}^{F}\frac{P_f}{c_f},\qquad F=\min(\lfloor D\cdot\text{fps}\rfloor,\ F_{\max}),
$$

$P_f$ 是第 $f$ 帧经过视觉塔后的 patch 数，$c_f$ 是这一帧的压缩倍率（池化、合并、query 读取）。每帧压缩倍率可以不同，这正是 SlowFast 式排布的出发点。各模型的选择汇总如下（只列下文核实过的）：

| 模型 | 采帧 | 每帧/每组 token 怎么来 | 时间信息怎么进模型 |
|---|---|---|---|
| Video-LLaVA | 均匀 8 帧，224×224 | LanguageBind 编码，与图像共享两层 MLP 投影 | 帧的先后顺序（序列位置） |
| LLaVA-Video | 最多 64 帧（数据标注时 1 fps） | SigLIP，27×27 网格做 2D 平均池化；慢帧 $p\times p$、快帧 $2p\times2p$ | 帧在序列中的先后顺序 |
| Qwen2-VL | 2 fps，单视频上限 16384 token | 3D 卷积把相邻 2 帧合成一个 tube，再 2×2 合并 | M-RoPE 的时间分量 |
| TimeChat | 96 帧 | Q-Former 抽帧特征，滑动窗口 video Q-Former 再压缩 | 文本“This frame is sampled at 2s.”作为 Q-Former 条件 |
| LLaMA-VID | 每帧 2 个 token | 文本引导的 context token + 池化得到的 content token | 帧的先后顺序 |
| LongVA | 可推到 2000 帧 | 每帧 144 token | 把语言模型上下文延长到 224K 后迁移 |

四个可调旋钮：帧数 $F$、空间分辨率、每帧压缩 $c_f$、时间表示。前三个决定 $N_{\text{vis}}$；第四个不改 token 数，却决定模型能不能说出“第几秒”。

## 2. 统一表示：Video-LLaVA

**问题。** 早期做法给图像和视频各配一个编码器，特征落在不同空间，再各用投影层接进 LLM。Video-LLaVA（Lin et al., arXiv 2311.10122，EMNLP 2024）把这称为“投影前未对齐”：LLM 要靠几层投影去调和两个空间，难以学到图像和视频之间的共性。

**做法。** 用 LanguageBind 的图像编码器和视频编码器（都从 OpenCLIP-L/14 初始化），它们在预训练时已经分别对齐到语言特征空间，作者称之为“投影前对齐”。之后图像与视频**共享**同一个两层 MLP（GeLU）投影。每个视频均匀采 8 帧，每帧按图像预处理到 224×224。训练分两阶段：理解阶段用 558K LAION-CC-SBU 图文对和 702K WebVid 视频文本对；指令阶段用 LLaVA-1.5 的 665K 图像指令和 Video-ChatGPT 的 100K 视频指令（§4.1）。

**结论与边界。** 摘要报告在 MSRVTT、MSVD、TGIF、ActivityNet 上比 Video-ChatGPT 高 5.8%、9.9%、18.6%、10.1%；§4.3.4 称图像与视频联合训练互相促进，与同配置的 LLaVA-1.5† 相比 9 个图像基准中 8 个提升（Figure 5）。作者在局限里写明：只用均匀 8 帧，长视频细节会丢，ActivityNet-QA 上被 Chat-UniVi 超过（论文局限讨论，对应 Table 2）。这正说明了第 1 节的预算问题：8 帧对短视频够用，对长视频不够。

**怎样验证“共享空间”有用。** 论文的对照是把编码器换成未与语言对齐的编码器、或去掉图像数据只训视频，比较视频基准。面试里能说出“控制变量是编码器是否预对齐，数据和投影层不变”即可。

## 3. 帧多还是每帧 token 多：LLaVA-Video 的 SlowFast 排布

**问题。** LLaVA-Video（Zhang et al., arXiv 2410.02713，v3 2025-08-01）用 SigLIP 做视觉塔、Qwen2 做 LLM。每帧 $M=729=27\times27$ 个 token，100 帧就是 72,900 个（论文附录写作 67,600，与 729 对不上，按 $26^2=676$ 才得此数）；附录还说用 Qwen2-72B 时只放得下 8 帧。

**做法。** 视频表示记为 $\mathcal V=(T,M,s,p)$：最多 $T$ 帧，每帧 $M$ 个 token；每隔 $s$ 帧取一帧组成慢帧组，其余是快帧组；慢帧做 $p\times p$ 平均池化，快帧做 $2p\times2p$ 池化，然后按时间顺序把快慢帧交错排列。论文给的总数公式是

$$
N=\left\lfloor \frac{T}{s}\right\rfloor\left\lfloor\frac{M}{p^2}\right\rfloor+\left(T-\left\lfloor\frac{T}{s}\right\rfloor\right)\left\lfloor\frac{M}{4p^2}\right\rfloor .
$$

$s=1$ 时只剩慢帧组，退化为普通表示。论文实验设置给出 7B 用 $(64,\cdot,1,2)$、72B 用 $(64,\cdot,3,2)$（正文 $M$ 写作 679，同文 Table 9 和附录估算都用 729，679 应为笔误）。附录 A.1 指出 SlowFast-LLaVA 等方法会让同一帧既以快帧又以慢帧出现两次，LLaVA-Video 的分组避免了这一点。

**手算**（教学构造，按 `avg_pool2d` 在 27×27 网格上的实际输出算）。$p=2$ 时慢帧池化成 $13\times13=169$ 个；快帧用 $4\times4$ 池化，$\lfloor27/4\rfloor=6$，得 $6\times6=36$ 个。

- $(64,729,1,2)$：$64\times169=10{,}816$。
- $(64,729,3,2)$：慢帧 $\lfloor64/3\rfloor=21$ 帧，$21\times169=3549$；快帧 43 帧，$43\times36=1548$；合计 5097。

注意公式里的 $\lfloor M/p^2\rfloor=\lfloor729/4\rfloor=182$ 与网格池化的 169 不一致。论文 Table 8 的每帧 token 列写的是 169，Table 9 的 $(32,729,1,2)$ 一行写 5,408 $=32\times169$，说明实际按网格池化计数。Table 9 中 $(64,729,3,2)$ 一行写 5,396，用上式两种算法都得不到（5097 或 5757），差异来源论文未说明，本文未能核实。面试时说清“公式是近似、以网格池化后的边长平方为准”即可。

**实验结论**（Table 8，同一训练与推理帧数，VideoMME 无字幕）：32 帧 ×729 为 59.1；110 帧 ×169 为 60.4；440 帧 ×64 为 60.2。三者总 token 分别约 23,328、18,590、28,160。作者据此得出“每帧 token 少、帧数多”更好。但同表 PerceptionTest 从 69.5 降到 68.3、67.2，说明这个结论依任务而变：需要细节的题会因每帧压缩受损。

**边界。** ① 快帧只剩 36 个 token，小字和小物体基本看不见，OCR 类问题要靠慢帧。② 慢帧位置固定（每隔 $s$ 帧），关键事件恰好落在快帧上时只能低分辨率看到。③ Table 8 还显示推理帧数比训练多太多（训 32 测 110）时，PerceptionTest 和 VideoMME 都下降：训练与推理的帧数要匹配。

**数据。** LLaVA-Video-178K：178,510 个视频（0–3 分钟），178K 条描述、960K 条开放式问答、196K 条多选问答，用 GPT-4o 在 1 fps 采帧上标注。

## 4. Qwen2-VL：3D 卷积合并相邻帧与 M-RoPE

**视频输入**（Qwen2-VL，Wang et al., arXiv 2409.12191，v2 2024-10-03，§2.1）。每秒采 2 帧；视觉塔开头用深度为 2 的 3D 卷积，把相邻两帧的同一位置合成一个 3D tube，图像则复制成两帧以保持一致；ViT 之后用一个 MLP 把相邻 2×2 token 合成一个，前后加 `<|vision_start|>`、`<|vision_end|>`。训练时动态调整每帧分辨率，使单个视频不超过 16384 个 token。论文的图像例子：224×224、patch 14 时得到 66 个 token，即 $16\times16/4+2$。

**手算视觉 token**（教学构造，忽略处理器把边长取整到 28 倍数的细节）。

- 60 秒视频，2 fps → 120 帧 → 3D 卷积后 60 组。每帧 448×448 → $32\times32=1024$ 个 patch → 2×2 合并 → 256 个。总数 $60\times256=15{,}360$，加两个边界 token 为 15,362，在 16384 以内。
- 120 秒视频 → 120 组。仍用 448 则 30,720，超出预算。降到 336×336：$24^2/4=144$，$120\times144=17{,}280$，仍超；降到 308×308：$22^2/4=121$，$120\times121=14{,}520$，可行。时长翻倍时，Qwen2-VL 牺牲的是空间分辨率而不是帧率。

**M-RoPE**（§2.1）。把 RoPE 的旋转维拆成时间、高、宽三段。文本三段用同一个位置 ID，等价于 1D RoPE；图像的时间 ID 不变，高宽 ID 随 token 位置变；视频每帧时间 ID 加一。多模态混合时，下一个模态的起始 ID = 上一个模态的最大 ID + 1。Qwen2-VL-7B 的配置 `mrope_section=[16,24,24]`，头维 128 对应 64 个频率对，时间 16 对、高宽各 24 对（官方 `config.json`）。

**手算位置 ID**（教学构造）。3 个文本 token，接一段 4 帧视频，合并后网格为 $t=2$（4 帧经 3D 卷积变 2 组）、$h=2$、$w=3$，共 12 个视觉 token，再接文本。按 transformers v4.46.0 的 `get_rope_index`，时间 ID 以 3D 卷积后的“组”为单位：

| 段 | token | (t, h, w) |
|---|---|---|
| 文本 | 第 0–2 个 | (0,0,0) (1,1,1) (2,2,2) |
| 视频组 0 | 6 个 | t=3；h∈{3,4}；w∈{3,4,5} |
| 视频组 1 | 6 个 | t=4；h∈{3,4}；w∈{3,4,5} |
| 文本 | 下一个 | (6,6,6)，因为前面最大 ID 是 5 |

用 1D RoPE 时，这 12 个视觉 token 会占 3–14，下一个文本从 15 开始；M-RoPE 下从 6 开始。论文说的“降低图像和视频的位置 ID 值，有利于推理时外推到更长序列”就是这个意思。

**边界。** ① 时间 ID 每组加一，与真实秒数无关：2 fps 下一组代表 1 秒，若推理时换成别的 fps，同样的 ID 差对应不同时长。Qwen2.5-VL 改为与绝对时间对齐，见[模型差异笔记](/notes/multimodal-models-paper-reading/)。② 3D 卷积把两帧压进一个 token，组内先后顺序只能靠卷积权重区分，持续不到一组时长的短事件可能被平均掉。③ 宽高大的视频，高宽 ID 可能超过帧数，下一段文本的起点由三者最大值决定。

**验证。** Table 8 用 Qwen2-1.5B + ViT-L 的预训练模型比较 1D-RoPE 与 M-RoPE：视频上 PerceptionTest 46.6→47.4、NextQA 43.9→46.0、STAR 55.5→57.9；图像上有升有降（RealWorldQA 54.5→53.7）。Figure 5 显示 72B 在 Video-MME 中等长度视频上，推理长度超过训练上限 16384 时性能仍稳定。Table 4 中 Qwen2-VL-72B 的 MVBench 为 73.6，Video-MME（无/有字幕）为 71.2/77.8。

## 5. Token 压缩与合并：ToMe 与 LLaMA-VID

**ToMe**（Bolya et al., arXiv 2210.09461，ICLR 2023 oral）。在 ViT 每个 block 的注意力分支与 MLP 分支之间合并相似 token，不需要重新训练。相似度用各 token 的 key 的余弦相似度。二分软匹配（论文 Token Merging 一节）：① 把 token 交替分成 A、B 两组；② A 中每个 token 连一条边到 B 中最相似的 token；③ 只保留最相似的 $r$ 条边；④ 相连的 token 取平均合并；⑤ 拼回两组。每层少 $r$ 个，$L$ 层共少 $rL$ 个。合并后的 token 代表多个 patch，注意力改用比例注意力

$$
A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d}+\log s\right),
$$

$s$ 是每个 token 代表的 patch 数，等价于把这个 key 复制 $s$ 份。

**手算**（教学构造）。6 个 token，A={a1,a2,a3}，B={b1,b2,b3}，key 余弦相似度：a1 对 B 为 (0.9, 0.1, 0.2)，a2 为 (0.3, 0.95, 0.1)，a3 为 (0.2, 0.4, 0.5)。每个 a 选最相似的：a1→b1(0.9)，a2→b2(0.95)，a3→b3(0.5)。$r=2$ 保留前两条，合并后剩 4 个：a3、b1⊕a1、b2⊕a2、b3，两个合并 token 的 $s=2$，其 logit 加 $\log2\approx0.693$。若 a1、a3 都选中 b1 且都被保留，三者会合成一个 $s=3$ 的 token，匹配并不禁止这种情况。规模：576 个 token、$r=16$、24 层，最后剩 $576-16\times24=192$ 个。

**LLaMA-VID**（Li et al., arXiv 2311.17043，ECCV 2024）。每帧只给 LLM 两个 token：**context token** 用用户问题的文本 query $Q_t\in\mathbb R^{M\times C}$ 去读帧特征 $X_t\in\mathbb R^{N\times C}$，

$$
E_t=\mathrm{Mean}\big(\mathrm{Softmax}(Q_tX_t^\top)X_t\big),
$$

softmax 沿 $N$、均值沿 $M$，得到 $1\times C$；**content token** 对帧特征做平均池化（视频时全局池化成 1 个）。手算（教学构造）：1 小时视频按 1 fps 取 3600 帧，每帧 2 个 token 共 7200 个；同样帧数用 LLaVA-Video 的 169/帧则需 608,400 个。

**边界。** ① ToMe 本是 ViT 分类上的加速，接到视频 LLM 时，合并会打乱 token 的空间网格，位置编码（尤其 M-RoPE 的高宽 ID）要另行定义，论文未覆盖这种用法。② LLaMA-VID 的 context token 依赖问题：换个问题就要重新编码全视频，多轮对话无法复用视觉缓存；content token 只有一个时细节基本丢失。③ 所有压缩都会让 OCR、计数、小物体题先掉分。

**怎样验证。** 画“准确率–视觉 token 数–延迟”曲线，并按题型分桶（细节、计数、时序、全局）。只报总分会掩盖“全局题不变、细节题大幅下降”的情况。

## 6. 时间戳感知与时间定位：TimeChat

**任务。** 时间定位（temporal grounding）输入视频和一句描述，输出事件起止 $[t_s,t_e]$（秒）。视频 LLM 直接把时间写成文本，例如“12.0 - 20.0 seconds”。

**TimeChat**（Ren et al., arXiv 2312.02051，CVPR 2024）。两个模块（§3.1）：**时间戳感知帧编码器**把“This frame is sampled at 2s.”这样的文本作为 Q-Former 的条件，在帧级把视觉 token 和时间绑定；**滑动 video Q-Former** 在长度 $L_W$ 的窗口内用 $N_V$ 个 query 抽取 token，窗口按步长 $S$ 滑动，输出 $(T/S)\times N_V$ 个 token，长度随视频变长而增加，而不是固定 32 个。实验设置 $L_W=S=N_V=32$，输入 96 帧，所以视频 token 为 $(96/32)\times32=96$ 个（§4.1；Table 9 的第一行也是 96）。指令数据 TimeIT 含 6 类任务、12 个数据集、125K 条，视频平均 190.8 秒。

**指标手算**（教学构造）。$\mathrm{IoU}=\lvert P\cap G\rvert/\lvert P\cup G\rvert$。四个查询：

| 真值 | 预测 | 交集 | 并集 | IoU |
|---|---|---:|---:|---:|
| [12, 20] | [14, 24] | 6 | 12 | 0.50 |
| [12, 20] | [10, 18] | 6 | 10 | 0.60 |
| [12, 20] | [21, 30] | 0 | 18 | 0 |
| [0, 8] | [0, 6] | 6 | 8 | 0.75 |

R@1, IoU=0.5（每个查询只看第一个预测，IoU ≥ 0.5 记命中）为 3/4=75%；若实现写成严格大于，第一行不算，变成 50%，所以复现时要看评测脚本。R@1, IoU=0.7 为 1/4。mIoU $=(0.5+0.6+0+0.75)/4=0.4625$。第三行在 IoU 上记 0，但它离真值只差 1 秒，IoU 对“相邻但不重叠”的预测没有梯度信号。

**边界。** ① 96 帧铺满 190 秒，相邻帧约 2 秒；事件边界的分辨率受采帧间隔限制，模型写出“12.3 秒”也没有对应帧作证据。② 时间写成文本后要解析，格式错误的输出要单独统计，不能悄悄记 0 或丢弃。③ 片段裁剪后时间戳要加上片段在原视频中的偏移。

**结论**（Table 2，零样本）。Charades-STA 上 TimeChat-7B 的 R@1 为 32.2（IoU=0.5）和 13.4（IoU=0.7），比此前视频 LLM 高 27.5 和 11.8 个点。Table 9 在 YouCook2 上扫描窗口与步长：$(L_W,S)=(16,16)$ 共 192 个 token 时 CIDEr 11.7，高于 $(32,32)$ 的 9.6。

## 7. 长视频：LongVA

**做法**（Zhang et al., arXiv 2406.16852，§3）。先把语言模型（Qwen2-7B-Instruct）在长文本上继续预训练到 224K 上下文，再只用短的图文数据做对齐和指令微调，不用长视频文本对。作者把“长上下文能力从语言迁移到视觉”称为 long context transfer。视频用 UniRes 编码成“拉长的图像”，每帧 144 个 token。为了测长度，构造了合成基准 V-NIAH（视觉大海捞针）。摘要称可处理 2000 帧、超过 200K 视觉 token（$2000\times144=288{,}000$）。

**边界。** V-NIAH 测的是“能不能找到插入的一帧”，不等于能做跨长时间的推理；长上下文还带来 KV 与 Prefill 成本（见 [Prefill/Decode 笔记](/notes/prefill-decode-video-tokens/)）。另一条路线是把长视频切段、先检索相关片段再回答，代价是检索可能漏掉证据。

## 8. 评测：Video-MME 与 MVBench

| | Video-MME（arXiv 2405.21075，CVPR 2025） | MVBench（arXiv 2311.17005，CVPR 2024） |
|---|---|---|
| 规模 | 900 个视频、254 小时、2,700 道题 | 20 个任务，每个任务 200 道，共 4,000 道 |
| 时长 | 短（< 2 分钟）、中（4–15 分钟）、长（30–60 分钟） | 以短片段为主，来自公开数据集 |
| 构造 | 标注员看完整视频，每视频 3 题、每题 4 个选项，再交叉复核 | 静态到动态：把公开标注自动转成多选题，3–5 个选项 |
| 额外条件 | 有/无字幕（744 个视频有字幕）、音频 | 无 |
| 测什么 | 不同时长与模态条件下的综合理解 | 单帧无法解决的时序能力 |

两者都按多选准确率计分。报告 Video-MME 要写明“无字幕/有字幕”两种，例如 Qwen2-VL-72B 为 71.2/77.8（Qwen2-VL Table 4），LLaVA-Video-72B 为 70.5/76.9（LLaVA-Video Table 2）。Video-MME 论文还用纯文本设置检查题目是否必须看视频：Gemini 1.5 Pro 仅凭文本不到 15%（数据构建的质检部分）。

**坑。** ① 帧数、分辨率、字幕是否给出都会改分，不同论文的数字往往不在同一设置下。② 多选题的选项解析规则（只取首字母还是匹配全文）会改分。③ LLaVA-Video Table 2 用星号标出训练集已出现在数据混合里的基准，域内与域外要分开看。④ 4 选 1 的随机水平是 25%，3–5 选项的 MVBench 随机水平随任务不同。

## 9. 面试常问

**同样的 token 预算，多帧低分辨率还是少帧高分辨率？** 看任务。LLaVA-Video Table 8 显示 VideoMME 上 110 帧×169 优于 32 帧×729，但 PerceptionTest 反而下降。正确的回答是按题型分桶画准确率–token 曲线，并保证训练和推理用同样的帧数设置。

**M-RoPE 的三个位置 ID 怎样分配？为什么能外推？** 文本三者相同；图像时间固定、高宽随位置；视频时间每组加一；下一模态从前一模态最大 ID 加一开始。一段视频占用的 ID 范围是 $\max(t,h,w)$ 而不是 token 总数，位置值更小，长度外推更容易。

**模型怎样输出时间？** 三类：位置编码携带时间（Qwen2-VL 的帧序号、Qwen2.5-VL 的绝对时间）；在输入里写时间文本（TimeChat 把时间戳文本送进 Q-Former）；输出端把秒数写成文本再解析。评测用 R@1 at IoU 阈值和 mIoU，要核对阈值是否含等号、格式错误怎样计分。

**ToMe 和平均池化有什么区别？** 池化按固定网格合并，不看内容；ToMe 按 key 相似度合并，合并的未必相邻，需要比例注意力补偿被合并的 token 的权重，而且会打乱空间网格。

## 闭卷验收

不看资料写出视觉 token 总数公式，并说出四个可调旋钮；讲清 Video-LLaVA 的“投影前对齐”与共享投影，以及它为何只适合短视频；写出 LLaVA-Video 的 $\mathcal V=(T,M,s,p)$ 和总数公式，按 27×27 网格算出 $(64,729,3,2)$ 为 5097，并说明为什么公式里的 182 与实际的 169 不一致；算出 Qwen2-VL 一段 60 秒、2 fps、448×448 视频的 token 数，并给出 120 秒时为满足 16384 应怎样降分辨率；对“3 个文本 + 2×2×3 视频网格 + 文本”写出全部 M-RoPE 位置 ID；手算 ToMe 一层的二分匹配和比例注意力的 $\log s$；说出 TimeChat 两个模块和 96 帧对应 96 个视频 token 的来历；对四个查询算 R@1 at IoU=0.5/0.7 与 mIoU，并指出“≥ 还是 >”的影响；最后比较 Video-MME 与 MVBench 的规模、构造方式和各自测什么。

**参考。** [Video-LLaVA](https://arxiv.org/abs/2311.10122)（[官方仓库](https://github.com/PKU-YuanGroup/Video-LLaVA)）；[LLaVA-Video / LLaVA-Video-178K](https://arxiv.org/abs/2410.02713)（[HTML v3](https://arxiv.org/html/2410.02713v3)）；[Qwen2-VL](https://arxiv.org/abs/2409.12191)（[HTML v2](https://arxiv.org/html/2409.12191v2)，[Qwen2-VL-7B-Instruct config](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json)，[transformers v4.46.0 `modeling_qwen2_vl.py`](https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py)）；[TimeChat](https://arxiv.org/abs/2312.02051)（[官方仓库](https://github.com/RenShuhuai-Andy/TimeChat)）；[ToMe](https://arxiv.org/abs/2210.09461)（[官方仓库](https://github.com/facebookresearch/ToMe)）；[LLaMA-VID](https://arxiv.org/abs/2311.17043)（[官方仓库](https://github.com/dvlab-research/LLaMA-VID)）；[LongVA](https://arxiv.org/abs/2406.16852)；[Video-MME](https://arxiv.org/abs/2405.21075)；[MVBench](https://arxiv.org/abs/2311.17005)；[Qwen2.5-VL](https://arxiv.org/abs/2502.13923)。
