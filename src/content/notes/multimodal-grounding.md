---
title: 视觉定位：从 REC 到让 MLLM 写出坐标
date: '2026-10-06'
tags: [多模态算法, 视觉定位, REC, Grounding DINO, Qwen-VL, Kosmos-2, RefCOCO]
summary: 讲清 REC、RES 与开放词表检测的任务定义，GLIP 怎样把检测改写成短语定位，Grounding DINO 的三处跨模态融合，以及 Qwen-VL、Kosmos-2、Qwen2.5-VL 三种坐标表示的换算和 RefCOCO Acc@0.5 的判定。
draft: false
---

检索模型回答“哪张图和这句话相关”，定位模型要回答“这句话说的是图里哪一块”。你做过跨模态检索，熟悉的是全局向量的相似度；视觉定位（grounding）则要求输出一个框或一张掩码，答案能用 IoU 逐像素检查。面试里这一块常问三件事：任务和指标怎么定义，专用检测器（Grounding DINO）怎样把文本融进检测，MLLM 怎样把坐标“写”成 token。本篇按这三条展开，并把同一个框在三种坐标表示之间换算一遍。阅读入口：[多模态专题总览](/notes/multimodal-interview-guide/)。数值算例为教学构造，论文数字注明表号。

## 1. 三个任务：REC、RES、开放词表检测

| 任务 | 输入 | 输出 | 常用指标 |
|---|---|---|---|
| REC（指代表达理解） | 图像 + 一句指代表达，如 “left man in red” | 一个框 $(x_1,y_1,x_2,y_2)$ | Acc@0.5：预测框与真值 IoU 超过 0.5 的比例 |
| RES（指代表达分割） | 图像 + 指代表达 | 一张二值掩码 $H\times W$ | mIoU、oIoU、Prec@X |
| 开放词表检测（OVD） | 图像 + 任意类别名列表 | 每类若干框和分数 | COCO/LVIS 的 AP，常分基类/新类或零样本 |

REC 和 RES 每句话只对应**一个**目标，是 top-1 问题；开放词表检测对每个类别要找出**所有**实例，是检测问题，用 AP 评价。OVR-CNN（CVPR 2021）给出的开放词表设定是：只有一部分类别有框标注，再借助覆盖更多概念的图文对，测试时检测训练中没有框标注的新类别。后来“零样本迁移到 COCO/LVIS”的说法更宽松，训练数据可能已覆盖测试类别的名字，读论文时要看清它属于哪种设定。

**RES 的两个 IoU。** LAVT 的定义：mIoU 是逐样本 IoU 的平均，大小目标权重相同；oIoU 是所有样本交集面积之和除以并集面积之和，偏向大目标。教学构造：小目标交 10、并 20（IoU 0.5），大目标交 900、并 1000（IoU 0.9）。mIoU $=0.7$，oIoU $=910/1020\approx0.892$。同一个模型，小目标做得差时 oIoU 会掩盖问题，所以两个都要报。

## 2. RefCOCO 三兄弟与 Acc@0.5

三个数据集都建在 MSCOCO 图像上，区别在**采集方式**和**语言约束**（Yu et al., ECCV 2016，第 3 节）：

| 数据集 | 采集 | 规模（表达式 / 目标 / 图像） | 平均长度 | 特点 |
|---|---|---|---|---|
| RefCOCO | 两人对战的 ReferItGame | 142,209 / 50,000 / 19,994 | 3.61 词 | 不限制用词，大量 “left”“second from right” |
| RefCOCO+ | 同上 | 141,564 / 49,856 / 19,992 | 3.53 词 | 禁用位置词，只能描述外观 |
| RefCOCOg | 非交互的 Mechanical Turk | 85,474 / 54,822 / 26,711 | 8.43 词 | 句子更长更复杂，每图同类目标较少 |

RefCOCO 和 RefCOCO+ 的测试集分 testA（图中有多个人）和 testB（多个其他物体）；同一论文统计约一半被指代的目标是人。RefCOCOg 有两套划分：Google 划分按目标切分，同一张图可能同时出现在训练和验证里，只有 val；UMD 划分（Nagaraja et al. 2016）按图像切分，有 val 和 test。现在 MLLM 论文报的 RefCOCOg val/test 一般是 UMD 划分。

**判定规则。** 对每条表达式取模型的 top-1 框，与真值算

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A|+|B|-|A\cap B|},\qquad \mathrm{Acc@0.5}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\left[\mathrm{IoU}(\hat b_i,b_i)>0.5\right].
$$

Yu et al. 原文写的是 IoU “larger than 0.5” 算正确。

**手算**（教学构造）。真值框 $b=(160,120,400,360)$，宽高 240，面积 57,600。四个预测：

| 预测框 | 交集 | 并集 | IoU | $>0.5$ | $\ge0.5$ |
|---|---|---|---|---|---|
| A $(170,127.5,390,352.5)$ | 49,500 | 57,600 | 0.859 | 对 | 对 |
| B $(220,120,460,360)$ | 43,200 | 72,000 | 0.600 | 对 | 对 |
| C $(240,120,480,360)$ | 38,400 | 76,800 | 0.500 | 错 | 对 |
| D $(280,120,520,360)$ | 28,800 | 86,400 | 0.333 | 错 | 错 |

B 只右移了 60 像素（框宽的四分之一）就掉到 0.6，右移三分之一框宽、左右各差 80 像素时恰好 0.5。四个样本按 $>0.5$ 得 50%，按 $\ge0.5$ 得 75%。真实数据很少恰好落在 0.5，但像素坐标是否 +1、整数取整等实现差异会移动边界，所以复现别人的数字要用同一份评测脚本。

**边界。** Acc@0.5 不管框有多紧，0.51 和 0.99 一样算对；同类目标少的图上，随便框中一个人也可能命中。MLLM 输出无法解析成框时应算错，并单独报告解析失败率。零样本和微调后的数字不能放在一张表里比较（第 6 节）。

## 3. 把检测改写成短语定位：GLIP

传统检测器的分类头是 $S_{\mathrm{cls}}=OW^\top$，$O\in\mathbb{R}^{N\times d}$ 是 $N$ 个区域特征，$W\in\mathbb{R}^{c\times d}$ 是 $c$ 个类别的权重，类别集合在训练时就固定了。GLIP（CVPR 2022，第 3.1 节）把类别名拼成一句提示词，如 “person. bicycle. car. … toothbrush”，用语言编码器得到 token 特征 $P\in\mathbb{R}^{M\times d}$，换成

$$
O=\mathrm{Enc}_I(\mathrm{Img}),\quad P=\mathrm{Enc}_L(\mathrm{Prompt}),\quad S_{\mathrm{ground}}=OP^\top\in\mathbb{R}^{N\times M}.
$$

损失仍是 $L=L_{\mathrm{cls}}+L_{\mathrm{loc}}$，只是把分类 logits 换成区域与词的对齐分数。$M$ 通常大于 $c$，因为一个类名可能有多个词或被切成多个子词（论文举例 toothbrush 被切成 “tooth#” 和 “#brush”）。所以目标矩阵从 $T\in\{0,1\}^{N\times c}$ 扩展为 $T'\in\{0,1\}^{N\times M}$：短语为正时其所有子词都为正，标点等附加 token 为负；推理时对短语内各 token 的概率取平均。这样一份检测数据和一份短语定位数据可以用同一个损失训练。GLIP 还在两个编码器之间做深度融合（跨模态多头注意力 X-MHA），论文报告用 27M 定位数据预训练后，COCO 零样本 49.8 AP、LVIS 26.9 AP（摘要）。

**手算**（教学构造）。$d=2$，两个区域 $o_1=(1,0)$、$o_2=(0,1)$；提示词 “person. bicycle.” 的三个内容 token：person $(2,0)$、bi# $(0,1)$、#cycle $(0,1)$。则

$$
S_{\mathrm{ground}}=\begin{pmatrix}2&0&0\\0&1&1\end{pmatrix}.
$$

区域 2 对 bicycle 的概率是两个子词 sigmoid 的平均：$\sigma(1)\approx0.731$。如果只监督 bi# 而忽略 #cycle，后一个子词得不到正信号，这就是 $T'$ 扩展要解决的问题。

MDETR（2104.12763）更早一步，用 1.3M 带短语-物体对齐的图文对预训练端到端的检测器，让检测以文本为条件。GLIP 的贡献是把“检测数据也能当定位数据用”写成上面这个等价形式。

## 4. Grounding DINO：三处融合

Grounding DINO（ECCV 2024）在 DETR 类检测器 DINO 上加语言，论文把融合分成三个阶段（第 3.1–3.3 节）。输入是图像和文本（类别名用 “.” 分隔，或一句指代表达），文本编码器是 BERT-base，最长 256 个 token；图像特征来自 Swin 多尺度特征。

1. **特征增强器**（6 层）。图像侧用可变形自注意力，文本侧用普通自注意力，然后做图到文、文到图两个方向的交叉注意力，再接 FFN。目的是在选查询之前就让两种特征互相看到。
2. **语言引导的查询选择**。$X_I\in\mathbb{R}^{N_I\times d}$（$N_I$ 超过一万）、$X_T\in\mathbb{R}^{N_T\times d}$（$N_T<256$），$d=256$：
$$
I_{N_q}=\mathrm{Top}_{N_q}\left(\mathrm{Max}^{(-1)}\left(X_IX_T^\top\right)\right),\qquad N_q=900.
$$
每个图像位置取它与所有文本 token 相似度的最大值，选前 900 个位置。被选中位置给出解码器查询的位置部分（动态锚框），内容部分是可学习向量（“mixed query selection”）。
3. **跨模态解码器**（6 层）。每层依次是自注意力、图像交叉注意力、文本交叉注意力、FFN，比 DINO 的解码层多了文本交叉注意力。

输出时每个查询与文本特征做点积，得到对每个文本 token 的 logits，用 focal loss 做对比式分类；框回归用 L1 和 GIoU。损失权重 L1、GIoU、分类在匈牙利匹配时为 2.0、5.0、2.0，最终损失为 1.0、5.0、2.0（第 3.5 节）。另有子句级文本表示：多个类名拼在一起时，用注意力掩码阻止不相关类名之间互相注意（第 3.4 节）。论文报告 Swin-L 版本 COCO 零样本 52.5 AP（表 2）。

**手算查询选择**（教学构造）。$d=2$，4 个图像位置 $(1,0),(0,1),(1,1),(-1,0)$；2 个文本 token “cat” $(1,0)$、“dog” $(0,2)$。$X_IX_T^\top$ 的四行为 $(1,0),(0,2),(1,2),(-1,0)$，按行取最大得 $(1,2,2,0)$，取 Top-2 选中第 2、3 个位置。第 4 个位置与 cat 方向相反，不会被选成查询。

**边界。** 推理时官方示例用 box_threshold 0.35 筛框、text_threshold 0.25 决定框对应哪些词。阈值决定召回与误检的取舍；对指代表达，模型可能把句中每个名词都框出来（“man holding umbrella” 同时框人和伞），做 REC 要取分数最高的一个框。文本超过 256 token 时类别列表要分批。

**验证。** 消融去掉特征增强器、语言引导查询选择或文本交叉注意力，看 COCO 零样本 AP 与 RefCOCO 的变化（本篇未转录论文消融表的数值）。

## 5. MLLM 怎样输出坐标

专用检测器有回归头，MLLM 只有一个词表，坐标必须变成 token。三种主流做法：

| 模型 | 表示 | 例子（教学构造的同一个框） |
|---|---|---|
| Qwen-VL | 归一化到 $[0,1000)$ 的整数，写成字符串，外包特殊 token | `<ref>man</ref><box>(250,250),(625,750)</box>` |
| Kosmos-2 | 把图像切成 $P\times P$ 网格，左上、右下角各用一个位置 token | `<p>man</p><box><patch_index_0264><patch_index_0755></box>` |
| Qwen2.5-VL | 模型输入尺寸下的绝对像素坐标，常用 JSON | `{"bbox_2d": [161, 119, 403, 357], "label": "man"}` |
| Shikra（补充） | 归一化到 0–1 的小数，保留 3 位，直接写在自然语言里 | `[0.250,0.250,0.625,0.750]` |

**Qwen-VL**（2308.12966）：框归一化到 $[0,1000)$，写成 “(x_topleft,y_topleft),(x_bottomright,y_bottomright)”，前后加 `<box>`、`</box>`；被指代的文字用 `<ref>`、`</ref>` 标出。视觉端经交叉注意力适配器（可学习查询向量作 query、视觉特征作 key）压缩成 256 个视觉 token，查询-键对里加了二维绝对位置编码，以保留位置信息；第一阶段输入 224×224，多任务阶段提到 448×448。RefCOCO val 89.36（表 6）。

**Kosmos-2**（2306.14824，第 3.1、3.3 节）：宽和高各均分 $P$ 段得到 $P\times P$ 个格子，新增 $P\times P$ 个位置 token，框用左上角和右下角所在的格子表示，用格子中心像素还原坐标。论文取 $P=32$，即 1024 个位置 token；输入 224×224 时每格 7×7 像素。训练数据 GrIT 约 9100 万图、1.15 亿文本片段、1.37 亿个框（第 2 节）。

**Qwen2.5-VL**（2502.13923，第 2.1.2、2.2.1 节）：改用“基于输入图像实际尺寸”的坐标，理由是相对坐标无法表示物体的真实大小和位置。视觉编码器 patch 14、相邻 2×2 合并，所以输入宽高会被缩放到 28 的倍数。RefCOCO val：7B 90.0、72B 92.7（表 6）。注意 Qwen3-VL 技术报告（2511.21631，第 3.2.4 节）又改回 $[0,1000]$ 归一化坐标，理由是对分辨率和长宽比变化更稳健，后处理更简单。所以说“Qwen 用绝对坐标”时必须带版本号。

### 5.1 同一个框，三种写法（教学构造）

原图 640×480（宽×高），框 $(160,120,400,360)$ 像素，归一化后为 $(0.25,0.25,0.625,0.75)$。

**Qwen-VL。** $x\times1000/640$、$y\times1000/480$：得 `(250,250),(625,750)`。论文没写取整方式，这里恰好整除。还原时 $x=250/1000\times640=160$，与输入图被缩放到 448 无关，这是归一化坐标的好处。

**Kosmos-2。** 按 HuggingFace `processing_kosmos2.py` 的实现（$P=32$）：左上角 $u_x=\lfloor0.25\times32\rfloor=8$、$u_y=8$；右下角 $l_x=\lceil0.625\times32-1\rceil=19$、$l_y=\lceil0.75\times32-1\rceil=23$。索引 $=\text{row}\times32+\text{col}$：左上 $8\times32+8=264$，右下 $23\times32+19=755$。还原取格子中心：$x_1=8/32+1/64=0.265625$，$x_2=19/32+1/64=0.609375$，$y_1=0.265625$，$y_2=23/32+1/64=0.734375$，乘回图像尺寸得 $(170,127.5,390,352.5)$，正是第 2 节的预测 A。**光是量化就让 IoU 只剩 0.859**，对 Acc@0.5 无害，但对 Acc@0.9 这类更严格的阈值或小目标就是硬上限：一个 10 像素宽的物体在 640 宽的图上不到一格（一格 20 像素）。

**Qwen2.5-VL。** 按 `qwen_vl_utils.smart_resize`（factor 28）：高 $480/28\approx17.14\to17\times28=476$，宽 $640/28\approx22.86\to23\times28=644$，面积在默认上下限内，输入为 644×476。框换到输入尺寸：$x$ 乘 $644/640$，$y$ 乘 $476/480$，得 $(161,119,402.5,357)$，模型输出整数，如 $(161,119,403,357)$。还原：$161\times640/644=160$，$403\times640/644\approx400.5$，$119\times480/476=120$，$357\times480/476=360$。

### 5.2 缩放后的绝对坐标：最常见的坑（教学构造）

原图 2560×1920，设 max_pixels $=1280\times28\times28=1{,}003{,}520$。先四舍五入到 28 的倍数得 2548×1932，面积约 492 万超过上限，于是 $\beta=\sqrt{2560\times1920/1003520}\approx2.213$，宽 $\lfloor2560/\beta/28\rfloor\times28=1148$，高 $\lfloor1920/\beta/28\rfloor\times28=840$。原图框 $(640,480,1600,1440)$ 在输入尺寸下是 $(287,210,717.5,630)$。

- 正确还原：$x$ 乘 $2560/1148$、$y$ 乘 $1920/840$，得约 $(640,480,1601,1440)$，IoU 约 0.999。
- 忘了还原、直接当原图坐标：与真值交集 $78\times150=11{,}700$，并集 1,090,920，IoU $\approx0.011$，全错。
- 两个方向的缩放比不同（$1148/2560\approx0.448$，$840/1920=0.4375$），只用一个比例还原也会系统性偏移。

评测 Qwen2.5-VL 时，要从处理器拿到真实的输入宽高（例如 `image_grid_thw` 乘 14），而不是假设模型看到的是原图。反过来，造 SFT 数据时，框也必须按同一个 smart_resize 换到输入尺寸，否则训练标签与模型看到的像素对不上。

### 5.3 三种表示的取舍

- **归一化整数**（Qwen-VL、Qwen3-VL）：与分辨率无关，换算简单；但同一个数值在宽图和窄图上代表不同的像素距离，模型要从图像推断尺度。
- **网格 token**（Kosmos-2）：每个角只要 1 个 token，解码快；精度受 $P$ 限制，新增 $P^2$ 个嵌入要从头学。
- **绝对像素**（Qwen2.5-VL）：坐标数值与物体真实尺度一致；但依赖预处理，换分辨率或换处理器就可能错位，数值位数也随图像变大而变多。
- 文本坐标的共同弱点：交叉熵把 “625” 写成 “626” 和写成 “125” 的惩罚可能差不多，损失不反映几何距离。这也是 [多模态 RL](/notes/multimodal-rl/) 里用 IoU 当可验证奖励的动机之一。

## 6. 怎样验证与比较

- **指标对齐**：同一份解析和 IoU 脚本，写明阈值是 $>$ 还是 $\ge$，坐标是否 +1，解析失败算错。
- **设定对齐**：Kosmos-2 在 RefCOCO val 上 52.32（表 3）是零样本，Qwen-VL 的 89.36 是训练过 RefCOCO 类数据后的结果，两者不能直接比出“谁更强”。
- **分桶**：testA/testB、RefCOCO+ 无位置词、RefCOCOg 长句分开看；按目标面积分桶检查小目标。
- **消融**：同一模型换坐标格式（归一化 vs 绝对）、换输入分辨率，看 Acc@0.5 与 Acc@0.75；对 Grounding DINO 类模型做模块消融。
- **人工抽检**：看错例是“框对了物体但不紧”（IoU 0.3–0.5）还是“框错了物体”（IoU 接近 0），两者对应的改进方向不同。后者常与指代消歧和幻觉有关，见 [幻觉与评测](/notes/multimodal-hallucination-eval/)。

检索与定位的关系：检索给出“哪张图”，定位给出“图里哪里”。文档图像检索后再做区域定位，就是 [多模态检索](/notes/multimodal-embedding-retrieval/) 的自然延伸；视频里的时间定位见 [视频理解](/notes/multimodal-video-understanding/)。

## 7. 面试常问

**Q1：RefCOCO 和 RefCOCO+ 差在哪？为什么后者更难？** RefCOCO+ 采集时禁用了位置词，只能靠外观区分同类目标，模型不能靠 “left/right” 这类空间词取巧。两者都有 testA（人）和 testB（物体）。

**Q2：Grounding DINO 和 GLIP 的核心区别？** GLIP 把检测分类头换成区域-词对齐分数，在编码器阶段做深度融合；Grounding DINO 在 DETR 类框架上把融合扩到三处：特征增强器、语言引导的查询选择、解码器里的文本交叉注意力。

**Q3：Qwen-VL 和 Qwen2.5-VL 的坐标有什么不同，评测要注意什么？** 前者归一化到 $[0,1000)$，后者用输入图像（smart_resize 之后）的绝对像素坐标，必须按输入尺寸和原图尺寸分别在 $x$、$y$ 上还原。Qwen3-VL 又改回 $[0,1000]$ 归一化。

**Q4：Kosmos-2 的位置 token 有多少个，精度上限在哪？** $P=32$，共 1024 个，一个框用两个 token。精度受格子大小限制，教学例子中仅量化就把 IoU 降到 0.859。

**Q5：为什么 Acc@0.5 高不代表定位好？** 它只看是否过线，不看框有多紧；严格阈值、按面积分桶和 RES 的 mIoU/oIoU 能补上这一点。

## 闭卷验收

不看本文，写出 REC、RES、开放词表检测各自的输入输出和指标，说出 RefCOCO、RefCOCO+、RefCOCOg 的三点区别；写出 GLIP 的 $S_{\mathrm{ground}}=OP^\top$ 并解释为什么目标矩阵要扩到子词；按顺序讲出 Grounding DINO 的三个融合模块并手算一次查询选择；把一个 640×480 图上的框分别写成 Qwen-VL 归一化坐标、Kosmos-2 的两个位置 token 和 Qwen2.5-VL 的绝对坐标，再从 token 还原回来；最后对 2560×1920 的图走一遍 smart_resize，说清忘记还原坐标会怎样。

**参考。** [Modeling Context in Referring Expressions（RefCOCO/RefCOCO+）](https://arxiv.org/abs/1608.00272)；[Generation and Comprehension of Unambiguous Object Descriptions（RefCOCOg）](https://arxiv.org/abs/1511.02283)；[Modeling Context Between Objects for Referring Expression Understanding（UMD 划分）](https://arxiv.org/abs/1608.00525)；[TFDS RefCOCO 数据卡（划分说明）](https://www.tensorflow.org/datasets/catalog/ref_coco)；[OVR-CNN](https://arxiv.org/abs/2011.10678)；[LAVT](https://arxiv.org/abs/2112.02244)；[MDETR](https://arxiv.org/abs/2104.12763)；[GLIP](https://arxiv.org/abs/2112.03857)；[Grounding DINO](https://arxiv.org/abs/2303.05499) 与 [官方仓库](https://github.com/IDEA-Research/GroundingDINO)；[Kosmos-2](https://arxiv.org/abs/2306.14824) 与 [HuggingFace processing_kosmos2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/kosmos2/processing_kosmos2.py)；[Shikra](https://arxiv.org/abs/2306.15195)；[Qwen-VL](https://arxiv.org/abs/2308.12966)；[Qwen2.5-VL 技术报告](https://arxiv.org/abs/2502.13923) 与 [官方博客](https://qwenlm.github.io/blog/qwen2.5-vl/)；[qwen-vl-utils vision_process.py](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py)；[Qwen3-VL 技术报告](https://arxiv.org/abs/2511.21631)。
