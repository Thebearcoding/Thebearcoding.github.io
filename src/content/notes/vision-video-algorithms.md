---
title: 为什么图像和视频能够进入语言模型
date: '2026-09-22'
tags:
- 多模态算法
- 视频理解
- ViT
- CLIP
- SigLIP
- BLIP2
- 视觉证据
summary: 从ViT、CLIP、SigLIP和连接器到时间建模、采帧证据与可靠评测。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-111ffa2025cb" style="display:block;scroll-margin-top:6rem"></span>

本章沿着一个问题展开：**模型回答“他先拿杯子还是先开柜门”时，答案究竟怎样受到画面的约束？** 输入中必须有证据，视觉网络必须保留证据，时间表示必须区别事件先后，训练目标必须奖励依赖证据的回答，评测也必须检验这件事。任何一环缺失，流畅的输出都可能只是猜测。

先理解[01-模型骨干与多模态入口](/notes/transformer-attention-rope-gqa/)中的矩阵乘法、Attention、位置编码和残差。本章解释 patch、对齐、连接器、视频建模和评估；具体模型版本见[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)，可回溯的面试题见[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)。全部算例和拟开展实验均为教学构造。

<span id="mm-f9764a456398" style="display:block;scroll-margin-top:6rem"></span>

## 1. 输出任务决定需要保留什么信息

分类把输入映射到预定义类别；检索对候选对象排序；视觉问答在图像与问题条件下生成答案；grounding（定位）要求把语言对应到像素区域或时间区间。它们可以共享视觉编码器，但监督和评价不同。

对于“男子把杯子放回架子”的视频，分类可能只需“放置物品”，检索需从一万个视频中找到它，问答可能区分“左侧还是右侧”，时间定位还需输出动作起止。一个全局向量保留了“男子、杯子、架子”也许足以检索，却可能丢掉左右关系和事件边界。因此先定义任务和证据，再讨论模型大小。

<span id="mm-266abc32ae91" style="display:block;scroll-margin-top:6rem"></span>

## 2. ViT 如何把像素变成 token

RGB 图像是 $X\in\mathbb R^{H\times W\times3}$。设高宽均能被 patch 边长 $p$ 整除，切成非重叠小块，块数为

$$
N=\frac H p\frac W p.
$$

每块展平为 $3p^2$ 维，乘共享可训练矩阵 $E\in\mathbb R^{3p^2\times d_v}$，成为 $d_v$ 维 patch embedding。**各空间位置共享映射**，不是给每个位置训练一个独立词典。kernel size 和 stride 都等于 $p$ 的卷积可实现同一操作。

$224\times224$、$p=16$ 时，$N=196$；每块输入长度 $16\cdot16\cdot3=768$。若 $d_v=1024$，序列为 $[B,196,1024]$。标准分类 ViT 可以加可学习的 CLS token，长度变197；VLM可能去掉CLS，保留196个空间特征。取哪一层、是否池化、是否加CLS都要讲具体配置。[ViT 原论文](https://arxiv.org/abs/2010.11929)

<span id="mm-c396be40e3d1" style="display:block;scroll-margin-top:6rem"></span>

### 为什么需要空间位置

内容相同的patch放在不同位置，会改变“人在车左边还是右边”。标准ViT给patch加位置向量；其他视觉塔可用二维RoPE或相对位置。固定位置表用于新分辨率时常需空间插值；动态分辨率仍需要位置规则。插值位置表不能补出低清图里已经丢失的文字笔画。

随后，self-attention让patch彼此交换信息，逐token的MLP做非线性变换，残差连接保留原信息。图像编码通常双向可见；自回归文本Decoder通常限制未来可见性。两者共享Transformer运算，但输入、mask、位置、训练目标不同。分类可对CLS或池化特征接分类头；VLM保留多个patch，更有机会读取局部证据。

<span id="mm-35562fb939bd" style="display:block;scroll-margin-top:6rem"></span>

### 分辨率与计算

高宽从224都变448，patch数从196变784，是四倍；全局Attention的分数矩阵从 $196^2$ 变 $784^2$，是十六倍。投影和MLP成本则主要随token数线性增加，因此不能说整个模型计算必然十六倍。高分辨率有利于小字和小物体，但仍取决于原图是否清晰、后面是否再次压缩。

<span id="mm-fd8b5110b35d" style="display:block;scroll-margin-top:6rem"></span>

## 3. CLIP：训练怎样让图像与文字可以比较

图像编码器 $f$ 与文本编码器 $g$ 分别输出向量，经投影进入同维空间。维度相同只表示能做点积，**语义相近来自训练**。记

$$
u_i=\frac{f(I_i)}{\|f(I_i)\|_2},\qquad
v_j=\frac{g(T_j)}{\|g(T_j)\|_2},\qquad S_{ij}=\frac{u_i^\top v_j}{\tau}.
$$

归一化后点积是余弦相似度，避免主要靠向量长度抬高分数。batch有 $B$ 对图文，对角线为配对。图到文的一行回答“该图最对应哪条文本”；文到图的一列回答反向问题。[CLIP 原论文](https://arxiv.org/abs/2103.00020)

$$
\mathcal L_{I\to T}=-\frac1B\sum_i\log\frac{e^{S_{ii}}}{\sum_j e^{S_{ij}}},
\qquad
\mathcal L_{T\to I}=-\frac1B\sum_j\log\frac{e^{S_{jj}}}{\sum_i e^{S_{ij}}}.
$$

对称CLIP损失取两个方向的平均。行softmax与列softmax的分母不同，一个方向不会自动满足另一个方向的全部排序约束。

<span id="mm-3180f40cecc2" style="display:block;scroll-margin-top:6rem"></span>

### 手算到梯度，理解模型被推向哪里

令 $B=2$，$S=\begin{bmatrix}2&0\\0&2\end{bmatrix}$。第一张图选第一条文本的概率为 $e^2/(e^2+1)=0.8808$，交叉熵为0.1269；另一行与列方向相同，因此双向平均仍为0.1269。如果两个候选分数都为2，正确概率变为0.5，损失为 $\log2=0.6931$。

一行交叉熵对分数的导数是

$$
\frac{\partial\ell_i}{\partial S_{ij}}=p_{ij}-\mathbf1[j=i].
$$

正配对梯度为 $-0.1192$，负配对为 $+0.1192$；梯度下降抬高正例分数、压低负例分数。若对未除温度的相似度求导，还多一个 $1/\tau$。更小温度既使分布尖锐，也改变梯度尺度，不能直接称为更好。常见实现训练的是对数逆温度 $\alpha=\log(1/\tau)$，再以 $e^\alpha$ 放大点积；$\alpha$ 本身不等于温度。

<span id="mm-2fd4bd143554" style="display:block;scroll-margin-top:6rem"></span>

### 零样本分类如何出现

把“猫、狗、车”各写成描述句，经文本编码器形成候选向量；待测图与它们比较，最大分数给出类别。“零样本”是无需用该分类任务的标注集拟合新分类头，不代表预训练未见过猫或对应词语。候选类别和措辞改变会影响结果，softmax概率也未经天然校准。

<span id="mm-3bad15891296" style="display:block;scroll-margin-top:6rem"></span>

### 难负例、假负例、大batch

两张猫图可能都有正确描述“一只猫”；另一条caption虽然不是原配对，却不是语义错误。硬当负例会产生冲突。难负例应当“相似但确实不匹配”，例如相同对象、不同动作顺序；疑似重复与多正确描述应清理、降权或多正例建模。

普通梯度累积先后计算微批次梯度，**不会自动产生跨微批次负例**。若需要跨设备/跨批候选，要说明特征收集、梯度传播或缓存。大batch提供更多比较，但收益受假负例、数据和训练预算限制。

千万级大库可用双塔向量索引先召回，再用更贵的图文交互模型精排。单向量检索把对象压成一个向量；多向量方法保留多个局部向量，可聚合局部匹配，但索引、存储和计算更贵。精排能改变候选顺序，却无法找回召回阶段根本没选中的对象。

<span id="mm-181579337b2c" style="display:block;scroll-margin-top:6rem"></span>

## 4. SigLIP 改变的是配对学习目标

CLIP行softmax在候选间分配总和为1的概率；SigLIP独立判断每对图文是否匹配。设 $y_{ij}=+1$ 表示匹配、$-1$ 表示不匹配，$z_{ij}=a\,u_i^\top v_j+b$，按原论文伪代码的归一化习惯：

$$
\mathcal L_{\rm sigmoid}=\frac1B\sum_{i,j}\log(1+e^{-y_{ij}z_{ij}}).
$$

每对损失可独立计算，不需要全局softmax分母；实现常用稳定的logsigmoid/softplus。$a>0$为尺度，$b$为偏置。batch有 $B$ 正例、$B(B-1)$ 负例，偏置与负例配比重要。SigLIP仍需要负例，不能只把softmax换成sigmoid就认为实现完整。[SigLIP 算法1](https://arxiv.org/html/2303.15343v2)

手算：2×2分数矩阵对角线2、非对角线0。正例每项0.1269，负例每项0.6931；四项加和除 $B=2$，得到0.8201。若负配对降到 $-2$，四项都为0.1269，损失降至0.2539。这个数不能与CLIP的0.1269比较模型强弱，二者目标与归一化不同。

CLIP/SigLIP视觉塔受到语言监督，[DINOv2](https://arxiv.org/abs/2304.07193)等方法采用图像自监督；这改变表征学习来源，但不能只凭名字断言“前者只有全局信息，后者一定更会局部”。目标任务的冻结特征分类、检索和细粒度评测才能检验。某VLM采用SigLIP也不等于其整个生成训练继续用SigLIP损失。

<span id="mm-2a6cd8569c6a" style="display:block;scroll-margin-top:6rem"></span>

## 5. 连接器：从可匹配向量到可条件生成

<figure style="margin:1.5rem 0">
<a href="/notes-assets/vision-video-algorithms/16-%E8%A7%86%E8%A7%89%E6%8E%A5%E5%85%A5%E6%9E%B6%E6%9E%84.svg" target="_blank" rel="noopener" aria-label="查看原图：视觉接入架构"><img src="/notes-assets/vision-video-algorithms/16-%E8%A7%86%E8%A7%89%E6%8E%A5%E5%85%A5%E6%9E%B6%E6%9E%84.svg" alt="视觉接入架构" width="1100" height="330" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">视觉接入架构（点击查看原图）</figcaption>
</figure>

沿图的视觉编码器→连接器→LLM走一遍：第一段产生视觉特征，第二段调整维度或数量，第三段结合问题生成文字。图中视觉token是连续向量，不是词表里的一串“图像单词”。再反向追问：答案loss会经过哪些箭头回传？这决定视觉特征如何学会影响生成。此图为本学习文档自绘的简化结构，未表示某个版本的全部模块。

<span id="mm-a45ec59d7225" style="display:block;scroll-margin-top:6rem"></span>

### 线性投影、MLP、空间合并

视觉特征 $Z:[N,d_v]$ 经 $ZW+b$ 变成 $[N,d_l]$，长度不变。两层MLP计算 $\phi(ZW_1+b_1)W_2+b_2$，增加非线性表达；若去掉中间非线性，两层仍可合并为一层。

**普通逐位置MLP不必压缩token数量。** 若先把空间相邻2×2特征拼在通道上，再经MLP，长度才减少四倍。例如 $[196,1024]$ 可以直接变 $[196,4096]$；也可先拼成 $[49,4096]$ 后映射。前者保留更多空间位置，后者减少LLM长度，两者成本和细节保留不同。

<span id="mm-c78b3e202acf" style="display:block;scroll-margin-top:6rem"></span>

### Query resampler 为什么可以输出固定长度

让 $K$ 个可学习query去读取 $N$ 个视觉位置，一次cross-attention可简写为

$$
Z'=\operatorname{softmax}(QK_v^\top/\sqrt d)V_v\in\mathbb R^{K\times d}.
$$

其中 $Q:[K,d]$，$K_v,V_v:[N,d]$。输出长度由query数决定，$K<N$时形成信息瓶颈。query是训练参数，其语义分工由学习形成，不能预设每个槽天然对应某个物体。Q-Former是包含自注意力、交叉注意力和特定训练mask的完整模块；resampler是更宽泛的压缩器称呼。

<span id="mm-67be5a139bb0" style="display:block;scroll-margin-top:6rem"></span>

### BLIP-2 的三个目标为什么不能合成一句“图文对齐”

BLIP-2先训练Q-Former抽取与语言相关的信息，再把其输出投影到冻结LLM的输入维度。第一阶段的ITC、ITM、ITG共用模块，却改变query与文字的可见关系；第二阶段才主要让LLM学会消费这些视觉提示。原论文用32个query，并不意味着所有模型都固定32个。[BLIP-2 第3节及图2–3](https://arxiv.org/html/2301.12597v3)

| 目标 | 要回答的问题 | query与文字能否互看 | 产生什么监督 |
|---|---|---|---|
| ITC，对比 | 图像和候选文本能否独立编码后检索 | 两路隔离 | 配对相似度相对负例更高 |
| ITM，匹配 | 充分交互后，这一图文对是否匹配 | 双向交互 | 匹配/不匹配分类 |
| ITG，生成 | 只凭图像和此前文字，下一词是什么 | query不看文字；文字看query及过去文字 | 逐token生成CE |

ITC中的一个图像有多个query输出，原方法用其中与文本CLS最相似的一项作为图文匹配分数；ITM则对交互后的query做分类并聚合。两者不是把同一个余弦分数换个名字。ITG要求视觉信息先经过query瓶颈再到文字，无法从冻结视觉塔直接绕过去。

用一个**自拟泄漏反例**理解mask：如果训练检索时让图像query直接读取正确caption“红杯子”，即使遮掉图像，query也可能靠复制caption完成匹配。训练loss很好，却没有学到可独立检索的图像表示。相反，ITM的任务本来就允许拿到待判断的图和文字，交互不是泄漏；它需要判断关系是否真实。判断泄漏的依据是任务在推理时允许什么信息，而不是“一律不许看文本”。

ITG则与SFT的shift相同：输入“红、杯”预测下一词“子”时，当前预测不能看答案“子”。不能只给文字内部加因果mask，却让query先读取整句，再把未来信息传回来。由此可见，同一张多模态结构图必须连同训练mask才能定义一个真正的算法。

最后用一个两query的自拟数例理解压缩：若三个视觉位置的值为标量 $[1,3,5]$，两个query分别给出权重 $[0.5,0.5,0]$ 和 $[0,0.25,0.75]$，其输出为2和4.5。query数决定输出槽数，权重决定每槽汇总什么；仅靠这两数通常不能唯一恢复三个原值。这是“省token但有信息瓶颈”的具体含义。

<span id="mm-970ed41c8d88" style="display:block;scroll-margin-top:6rem"></span>

### LLaVA 的训练为何分阶段

原始LLaVA先冻结视觉塔和LLM，用图文配对训练线性连接器；随后用视觉指令数据训练连接器和语言模型，使模型按问题回答。LLaVA-1.5将连接器改为两层MLP，并调整数据与分辨率。跨版本结果混合多个因素，不能把全部收益归因于MLP。[原始LLaVA](https://arxiv.org/abs/2304.08485)、[LLaVA-1.5](https://arxiv.org/html/2310.03744v2)

生成目标是

$$
\mathcal L=-\sum_{t\in\text{答案位置}}\log p_\theta(y_t\mid I,q,y_{<t}).
$$

图像与问题作为条件，答案位置算loss。视觉token未直接被要求预测词表标签，但答案梯度仍能经注意力回传到可训练视觉模块。冻结LLM参数并不意味着可以切断经过LLM的全部梯度：要训练前方projector，仍需损失对LLM输入的梯度。label mask详见[02-SFT与DPO的训练信号](/notes/multimodal-sft-lora-dpo/)。

冻结视觉塔保留已有表征、减少训练负担；面对领域变化或细节任务，解冻可能有益，也可能过拟合。应比较冻结、局部解冻和全参在相同数据上的质量与成本，不把某一论文方案当通用规定。

<span id="mm-7139c73f5e1e" style="display:block;scroll-margin-top:6rem"></span>

## 6. 时间建模：为什么“逐帧看过”不等于知道先后

逐帧编码给出 $Z:[F,P,d]$。如果平均所有帧，$\frac1F\sum_f Z_f$ 在交换帧顺序后不变，无法从这个结果恢复动作先后。时间位置告诉网络顺序和间隔，跨帧交互允许比较状态，时序监督告诉网络哪些变化决定答案，三者作用不同。

<span id="mm-e4c40ecb3709" style="display:block;scroll-margin-top:6rem"></span>

### 时空Attention怎样交换信息

联合时空Attention对 $FP$ 个token交互，分数规模约 $F^2P^2$。TimeSformer的divided版本先让同一空间位置跨时间交互，再在每帧内做空间交互，分别使用不同投影。分数规模约 $PF^2+FP^2$。$F=8,P=196$ 时分别为2,458,624和319,872。这里只比较简化分数数量，未计投影、MLP、CLS和内核成本；分解改变信息路径，不是数学等价的免费优化。[TimeSformer第3.2节](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf)

为什么需要两步？令 $z_{f,p}$ 表示第f帧、第p个patch。时间步先在 $z_{1,p},\ldots,z_{F,p}$ 之间交换信息；空间步再在同一帧的各p之间交换。远处patch的信息通过中间路径传播，而联合Attention可直接对任意时空对评分。

一个自拟反例：杯子从左边移到右边，同一空间格在两帧里未必对应同一物体。时间Attention比较的是固定空间位置，不是已经做好的对象跟踪；空间交互和多层组合才有机会关联移动轨迹。只平均各帧分类分数，也不能等价地恢复这种联系。实际效果需要顺序依赖题和同预算消融验证。

<span id="mm-2478d43b0565" style="display:block;scroll-margin-top:6rem"></span>

### VideoMAE：遮挡的是预训练输入，不是让推理时漏帧

VideoMAE用视频立方块作为token，通过高比例tube mask，让同一空间位置在时间上共享遮挡；重编码器只处理可见块，轻解码器结合mask位置恢复被遮块像素。原论文采用 $2\times16\times16$ 的时间—空间块，并对遮挡位置计算重建MSE；这和文字CE是不同的监督对象。[VideoMAE第3.3节](https://arxiv.org/html/2203.12602v3)

设遮挡块集合为 $\mathcal M$，每块像素目标向量宽度为 $D_{\rm pix}$，预测为 $\hat x_i$，可用下面的教学形式理解按元素平均的重建损失：

$$
\mathcal L_{\rm recon}=\frac1{|\mathcal M|D_{\rm pix}}\sum_{i\in\mathcal M}\|\hat x_i-x_i\|_2^2.
$$

目标可以按块归一化，需与具体训练协议一致。自拟小例：一块目标 $[0,1]$、预测 $[0.2,0.7]$，MSE为 $(0.04+0.09)/2=0.065$。对第二个预测值的梯度是 $0.7-1=-0.3$，下降更新将其往1推；它没有直接监督“拿起杯子”这个动作类别，更没有给自然语言回答打分。

tube mask的动机是避免过于容易的复制：只遮第2帧的一小格而第1、3帧同位置都可见时，静态背景几乎可原样抄回；跨时间一起遮住该位置，会让重建更多依赖其他空间与时空线索。这不保证学到因果动作理解，也不是建议VLM推理时故意删掉90%的证据。预训练时创造学习任务，与推理时保留回答所需证据，处在不同阶段。

以16帧、224方图为例，上述cube配置产生 $8\cdot14\cdot14=1568$ 个token。若恰好只保留10%，重编码器只看到约157个（具体整数由mask实现决定）；解码器仍需处理恢复位置，不能把整个训练耗时简单说成原来的10%。接到语言问答还需连接器与跨模态监督，重建目标不会自动教会指令跟随。

<span id="mm-c70f0a9b09d0" style="display:block;scroll-margin-top:6rem"></span>

## 7. 抽帧、空间细节和真实时间共用预算

<figure style="margin:1.5rem 0">
<a href="/notes-assets/vision-video-algorithms/15-%E8%A7%86%E9%A2%91%E9%87%87%E5%B8%A7%E4%B8%8E%E8%AF%81%E6%8D%AE.svg" target="_blank" rel="noopener" aria-label="查看原图：视频采帧与证据"><img src="/notes-assets/vision-video-algorithms/15-%E8%A7%86%E9%A2%91%E9%87%87%E5%B8%A7%E4%B8%8E%E8%AF%81%E6%8D%AE.svg" alt="视频采帧与证据" width="1100" height="390" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">视频采帧与证据（点击查看原图）</figcaption>
</figure>

图中横轴是原视频时间，采样点是模型真正收到的帧，彩色区间是回答所需事件。先看采样点是否覆盖证据，再看模型的理解能力。图为教学自绘，不能当成采样方法已经在真实数据上胜出的结果。

10秒视频在 $\{0,2,4,6,8\}$ 秒采帧，红灯只在 $[4.9,5.1]$ 秒出现，五帧全部漏掉。如果模型猜对，也不证明它看到了红灯。改为5次独立均匀瞬时采样，至少一次覆盖概率为

$$
1-(1-0.2/10)^5\approx9.61\%.
$$

这个模型忽略帧曝光、运动模糊和采样相关性，只解释短事件稀疏。增帧、分段、运动/镜头线索、先粗后细都可能改善覆盖，却增加成本或偏置；运动驱动也可能追着镜头晃动跑。

视觉token可粗写 $N_v=FP/c$，$c$为压缩倍率；具体模型可能将相邻帧组成tubelet，添加分隔符或动态裁剪。8帧×196为1568个；每帧压到32得到256个，时间位置仍有8个而局部细节可能损失。如果同预算只保留一张高分辨率帧，空间细节好却容易漏事件。应比较按短事件、OCR、顺序分桶的准确率—token—延迟曲线。

帧编号也不等于时间：$[0,0.1,8]$ 秒与 $[0,4,8]$ 秒编号都为 $[0,1,2]$，持续时间却不同。精确定位必须保留真实采样时间与片段相对原视频的偏移。解码帧率、模型采样FPS和最终帧数是三个概念；可变帧率文件尤其不能随意用帧号除名义FPS代表真实时间。具体Qwen时间编码见[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)。

<span id="mm-1bf951a075ce" style="display:block;scroll-margin-top:6rem"></span>

## 8. 数据决定模型被教会什么

视频问答样本应记录源视频标识、片段起止、问题、答案、证据区间、任务类型、字幕/音频条件和标注来源。只存“视频路径+答案”，难以复查漏采、歧义或推理错误。

“开门后把杯子放在哪里”需要看见开门和放置。剪掉开门却保留问题，会把缺失证据样本当普通正例。合成caption也可能从静态截图脑补“随后走开”，学生便学到错误时间关系。质检要覆盖可解码、视听同步、近重复、答案证据、字幕泄漏和题型，而非只筛语言通顺。

先按源视频及近重复簇划分train/validation/test，再切片采帧。同源相邻片段跨集合会泄漏背景、人物或字幕。验证集用于选择模型和超参；测试集应在方案固定后评估，反复用测试集选择采样方法也是适配。

90%数据若是静态识别，总loss很低也不意味着学会先后关系。增加顺序、计数、定位和“无法判断”样本，同时保留原能力回归集。重采样、加权loss和难例挖掘会改变训练分布，必须在目标测试分布检查收益与校准。

字幕“下一站北京”不证明画面出现北京地标，破裂声也不一定来自可见的玻璃杯。时间对齐解决各信号何时对应；模态消融检验依赖；人工核查检验依赖是否支持答案。三者不能互相替代。

<span id="mm-2af183ed36e9" style="display:block;scroll-margin-top:6rem"></span>

## 9. 评价必须检验证据、能力和不确定性

<span id="mm-4a29116e9bcb" style="display:block;scroll-margin-top:6rem"></span>

### 指标跟随任务

分类/选择题报告准确率并按题型、时长分桶。单正例检索的Recall@K是正确对象进入前K的查询比例；多相关对象时应按官方定义区分命中率与召回率。时间定位例子：真值 $[4,8]$、预测 $[6,10]$，交集2秒、并集6秒，IoU=$1/3$，在IoU≥0.5阈值下失败。

自由文本答案要区分事实正确、证据支持、时空关系、格式遵循。BLEU/ROUGE不充分反映时序和幻觉。LLM-as-Judge要固定rubric及输入条件，以人工样本校准，检查长度偏好、位置偏差以及judge是否见到完整视频证据。训练reward与最终评价应有独立核查，避免奖励漏洞同时成为评价漏洞。

<span id="mm-08f458d07f86" style="display:block;scroll-margin-top:6rem"></span>

### 用可控干预检查捷径

对答案应该随动作反转而改变的题，保留背景与物体，反转时间顺序，看输出是否相应改变；去字幕测字幕依赖；遮关键区域测局部证据。但干预可能产生不自然输入，因此还需不影响答案的控制干预，不能把所有性能下降都当因果证明。

[TempCompass](https://arxiv.org/abs/2403.00476)强调时间属性辨别；[Video-MME](https://arxiv.org/abs/2405.21075)覆盖不同长度和模态条件。它们不等价。比较要固定官方划分、采样、字幕/音频、解析和评分协议，模型名相同也不保证实验可比。

假设100道独立同分布题答对70道，近似标准误为 $\sqrt{0.7\cdot0.3/100}=0.0458$，粗略95%区间约 $[0.61,0.79]$。涨2个百分点可能是噪声。同题比较要考虑配对差异，同源问题有相关性时bootstrap应按源视频抽样。这个算例不是任何模型的实测显著性结论。

<span id="mm-b13554ffbe32" style="display:block;scroll-margin-top:6rem"></span>

## 10. 将失败诊断变成可证伪项目

假设固定视觉塔、LLM、训练数据和视觉token上限，比较均匀8帧与分段8帧。在100道短事件题中人工标证据，分别记录“覆盖至少一帧”“覆盖全部必要事件”“答题正确”。看到放置但没看到开门，可能仍缺完整时序证据。

若覆盖改善，先对同一道题做配对比较，并在两方案共同覆盖的题目交集上比较答题能力；不能直接比较两组不同“已覆盖子集”的准确率，因为样本难度可能不同。若新增覆盖题的改进与总体变化吻合，而共同覆盖交集无明显差异，才更支持证据覆盖这一解释。覆盖相同但时序题改善，可进一步检验时间表示；加帧后OCR下降，检查是否迫使每帧分辨率下降。特征probe能预测顺序仅是相关证据，不能独自证明LLM忽视了信息。

诊断顺序是原视频/标注→实际采样与时间戳→视觉编码和压缩→跨帧信息→生成监督→评价协议。然后才讨论[03-从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)是否提供有效学习信号。没有视觉证据或可检验reward时，增加RL不会可靠补出缺失事件。

<span id="mm-e671af5c5023" style="display:block;scroll-margin-top:6rem"></span>

## 练习与解释

1. **336×336图像，patch14，去CLS后多少patch？2×2合并后多少token？** $24^2=576$，合并后144。时间分组与特殊token尚未计入。
2. **一图两句正确描述，单正例CLIP有什么问题？** 另一句被当负例，梯度互相排斥；可多正例建模或清理重复。
3. **为什么MLP不必压缩长度，resampler可以？** 普通MLP逐位置变通道；固定query通过cross-attention聚合输入，输出槽数由query数决定。
4. **冻结LLM能否训练前面的projector？** 可以，LLM参数不更新但仍需对输入求导，不能随意截断整条计算图。
5. **帧打乱但答案不变，能断言没有时间编码吗？** 不能；问题可能无关时间、证据漏采或字幕捷径。要用答案应随顺序改变的题和控制组。
6. **70%变72%是否宣布有效？** 还需样本量、配对错误变化、相关性与重复实验；总分不证明稳定收益或机制。

回到[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)练口述，再把通用机制放进[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)的具体版本。掌握标准是能解释信息如何流到输出，并指出它什么时候会失败。
