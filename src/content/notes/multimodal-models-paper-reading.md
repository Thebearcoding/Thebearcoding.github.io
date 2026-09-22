---
title: 模型差异为什么必须落到具体版本
date: '2026-09-22'
tags:
- Qwen2.5-VL
- Qwen3-VL
- DeepSeek-V2
- Llama3.1
- MLA
- 论文阅读
summary: 固定Qwen2.5-VL、Qwen3-VL、DeepSeek-V2和Llama3.1，解释机制和论文消融。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-1dd4111c63af" style="display:block;scroll-margin-top:6rem"></span>

“DeepSeek、Qwen、LLaMA有什么区别”是[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)中的正文自述题。回答它不能靠家族标签：一个家族可以同时有dense与MoE、文本与视觉、base与instruct。本章用**固定版本**解释不同模块解决的问题，再把论文阅读变成可检验的理解。

本轮核对时间为2026-09-22。材料固定为Qwen2.5-VL报告v1、Qwen3-VL报告v1、DeepSeek-V2报告v5及Llama3.1官方模型卡；这些是便于学习的实例，未宣称覆盖截至当日所有新品。旧笔记出现的“Qwen3.5标题配Qwen3.8链接”不作为依据。另一个实际陷阱是：Qwen2.5-VL旧GitHub地址当前会返回Qwen3-VL正文，所以**链接名字本身不是版本证据**。

<span id="mm-1bc5221ce50b" style="display:block;scroll-margin-top:6rem"></span>

## 1. 一张比较表怎样帮助理解，而非背型号

| 固定版本 | 输入与主干 | 注意力/FFN关键机制 | 视觉与时间侧 | 比较时真正要问 |
|---|---|---|---|---|
| Llama3.1 8B/70B/405B | 文本，自回归Decoder | GQA；dense模型 | 这组型号没有视觉塔 | KV共享节省什么，仍缓存多少历史 |
| DeepSeek-V2（236B/激活21B） | 文本，自回归Decoder | MLA + DeepSeekMoE | 此模型不等于DeepSeek-VL | 压缩缓存与专家稀疏分别改变什么 |
| Qwen2.5-VL 7B | 图像/视频/文本，Qwen2.5骨干 | 该配置LLM 28层、4个KV头、head维128 | window ViT、2×2 merger、绝对时间对齐MRoPE | 动态分辨率与时间间隔如何进入模型 |
| Qwen3-VL 8B | 图像/视频/文本，Qwen3 dense骨干 | 不把dense配置推广到30B-A3B等MoE型号 | SigLIP2视觉塔、Interleaved-MRoPE、DeepStack、文字时间戳 | 多层特征与位置频谱分别解决什么 |

来源：[Llama3.1官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)、[DeepSeek-V2官方仓库](https://github.com/deepseek-ai/DeepSeek-V2)、[Qwen2.5-VL表1](https://arxiv.org/html/2502.13923v1)、[Qwen3-VL第2节](https://arxiv.org/html/2511.21631v1)。

这张表不是同预算性能排名：文本模型与VLM任务不同，总参数量也差距很大。它用来定位模块。若评估视觉任务，必须再选真正支持视觉的对照模型，并固定输入分辨率、帧数、prompt、生成长度和评分协议。MoE的激活参数描述每token参与的部分，不意味着部署时只需存储这些参数。

<span id="mm-cd714342b610" style="display:block;scroll-margin-top:6rem"></span>

## 2. Qwen2.5-VL：把空间分辨率和真实时间纳入输入

<span id="mm-cdb7767d13b9" style="display:block;scroll-margin-top:6rem"></span>

### 问题：固定方形输入会损失什么

把高宽比很大的文档强行压成固定小方图，既可能扭曲结构，也可能抹掉小字。动态分辨率使图像尺寸对应可变长度的视觉序列，并受最小/最大像素预算控制。它不代表完全不resize：报告明确将高宽调整为28的倍数，以配合patch14与2×2空间合并。

设预处理后图像为 $H'\times W'$，则视觉塔patch数为

$$
N_{\rm patch}=\frac{H'W'}{14^2},\qquad
N_{\rm visual}=\frac{N_{\rm patch}}4=\frac{H'W'}{28^2}.
$$

对于 $448\times448$，得到1024个patch，合并后256个LLM视觉token。高宽都变896时得到4096个patch、1024个视觉token。这里未计特殊token、文字、视频时间分组与padding，实际预算必须看processor输出而非只套公式。[Qwen2.5-VL第2.1节](https://arxiv.org/html/2502.13923v1)

<span id="mm-0f7b935f2c61" style="display:block;scroll-margin-top:6rem"></span>

### Window Attention为什么降低视觉成本

全局Attention让每个patch与所有patch交互，分数数量约 $N^2$。若每窗固定含 $w$ 个patch，共约 $N/w$ 窗，则分数约 $(N/w)w^2=Nw$，随图像token数线性增长。代价是单个窗口层不能直接联系远处内容，因此仍要有全局交互路径。

Qwen2.5-VL报告的ViT为32层，索引7、15、23、31使用全局Attention，其余多为window attention，最大窗口112×112像素即8×8 patch。注意这是**视觉编码器内部**的窗口，不代表LLM的整个输入都改成局部可见。视觉塔节省的计算与merger缩短LLM输入是两种收益，不能混为一个“加速比例”。

<span id="mm-512d574f83b4" style="display:block;scroll-margin-top:6rem"></span>

### 空间merger怎样接入语言模型

报告中视觉隐藏维为1280。空间相邻四个patch拼接时形成5120维输入，再由merger投到对应LLM隐藏维；7B型号的LLM隐藏维为3584。可把单张448图的形状链记成

$$
[1024,1280]\longrightarrow[256,4\cdot1280]\longrightarrow[256,3584].
$$

这个链条说明“减少长度”发生在空间分组，“匹配语言维度”发生在投影。合并并不保证无损：如果四个patch分别含小字笔画，压缩是否保留它们由模型与训练共同决定。详细连接器原理见[06-视觉视频算法面试专项](/notes/vision-video-algorithms/)。

<span id="mm-18c2391dbb3d" style="display:block;scroll-margin-top:6rem"></span>

### MRoPE究竟对什么旋转

一维RoPE按位置 $m$ 给Q/K各二维子空间施加旋转角 $m\omega_j$。MRoPE让不同子空间使用时间 $t$、高度 $h$、宽度 $w$ 的位置值。对于同图相邻两个patch，时间坐标相同而高宽不同；对于同空间位置的不同视频时刻，主要区别在时间坐标。文本的三组位置ID可取相同值，以对应一维序列位置。

这是坐标如何参与Attention的规则，不是“直接把三张位置图加到像素上”。二维视觉塔位置与LLM中的多模态位置也属于不同计算阶段，不能只说“用了RoPE”就省略发生在哪里。

Qwen2-VL的时间位置主要对应帧序列；Qwen2.5-VL把时间维ID与真实时间对齐，配合动态FPS训练。0、0.1、8秒与0、4、8秒不应只看作三个等间距帧。用 $t_{\rm id}\propto t_{\rm seconds}$ 可理解设计意图，但确切缩放、起点和分组应读取processor，不能把教学比例式当官方配置。[报告第2.1.2—2.1.3节](https://arxiv.org/html/2502.13923v1)

<span id="mm-d30e9467d862" style="display:block;scroll-margin-top:6rem"></span>

### 面试追问与可检验假设

如果增加分辨率后OCR提高但TTFT变差，可以检查patch、merger输出与KV长度是否按预算增长；如果不同FPS的同一事件被预测成不同持续时间，先核对真实时间戳和processor。动态FPS描述采样训练分布，不等于模型自动发现所有关键事件。任何采样规则漏掉的证据仍不能靠时间编码恢复。

<span id="mm-be61bfe9873b" style="display:block;scroll-margin-top:6rem"></span>

## 3. Qwen3-VL：它对上述问题作了哪些改变

Qwen3-VL报告固定了四个dense规模2B/4B/8B/32B及两种MoE规模30B-A3B、235B-A22B。它仍由视觉编码器、MLP merger和语言Decoder组成，不应因“统一多模态”宣传词而推断不存在视觉塔。[官方仓库](https://github.com/QwenLM/Qwen3-VL)

<span id="mm-5ba59e9e06d3" style="display:block;scroll-margin-top:6rem"></span>

### 视觉初始化改变，不等于生成目标变成对比损失

报告使用SigLIP2视觉编码器并继续做动态分辨率训练；2B/4B配较小的Large视觉塔，其余所述配置使用SO-400M。它保留2×2空间merger。SigLIP2是一种视觉预训练来源，后续VLM训练还要让视觉特征支持文字生成；不要把[06-视觉视频算法面试专项](/notes/vision-video-algorithms/)中的SigLIP原始配对loss直接说成Qwen3-VL所有训练阶段的loss。[报告第2节](https://arxiv.org/html/2511.21631v1)

<span id="mm-bb895a29990d" style="display:block;scroll-margin-top:6rem"></span>

### Interleaved-MRoPE为什么要交错频率

普通MRoPE将不同维度段分配给时间/高/宽；由于RoPE频率随维度变化，连续分段可能使各轴拥有不均衡的频谱。直观地说，如果时间维大多拿到某一段快或慢频率，它对长短时间变化的表达就不均衡。

交错分配让各轴覆盖多种频率，而不是各占一个连续频带。教学类比是三个轴分别拿到“快、中、慢”的尺子；并不意味着精确代码按t-h-w逐一循环，也不保证任意长度外推。它改变位置频率分配，不改变原始视频是否被采到。[报告第2.1节](https://arxiv.org/html/2511.21631v1)

<span id="mm-b239e7135f8b" style="display:block;scroll-margin-top:6rem"></span>

### DeepStack为何不必增加输入序列长度

只把视觉塔最后一层接到LLM，可能忽略中间层仍保留的细节。Qwen3-VL从视觉塔三个层级取特征，用独立merger映射，再加入前几个LLM层对应的视觉位置隐藏状态。简化理解为

$$
h_{\ell,\mathcal V}\leftarrow h_{\ell,\mathcal V}+P_\ell(z_{k_\ell}),
$$

$\mathcal V$ 表示视觉位置集合，$P_\ell$含相应映射；这只是帮助读图的残差示意，不是完整源码。它在已有位置补充信息，所以不必像把特征拼在序列尾部那样增加上下文长度，但投影、激活和数据移动仍有成本。

论文第5.12.2节给出控制条件下的DeepStack消融；对面试而言，重点是读取其模型、训练token和评估阶段是否固定，而非只背平均分提升。如果拿不同规模或不同训练数据的两模型比较，就不能把差异归因于DeepStack。[报告第2.2节及消融](https://arxiv.org/html/2511.21631v1)

<span id="mm-7af7d8170f3f" style="display:block;scroll-margin-top:6rem"></span>

### 显式文字时间戳为何再次出现

Qwen2.5-VL把时间ID绑定绝对时间；长视频可能产生较大且稀疏的位置ID，学习还依赖FPS覆盖。Qwen3-VL改用时间组前的显式文字时间戳，允许秒或时分秒形式。时间文本让模型直接读时间，代价是增加一些上下文token；MRoPE仍承担空间—时间位置组织，不能说成“完全不需要多模态位置编码”。[报告第2.3节](https://arxiv.org/html/2511.21631v1)

一个值得做的E类实验是：保留画面不变，只改变错误时间戳，看定位结果是否跟着错；再用正确时间戳恢复。它检验时间标记的依赖，也暴露错误预处理的风险。若声称这提高了真实时序理解，还需时间顺序反事实与视觉证据核验。

<span id="mm-11693aec212b" style="display:block;scroll-margin-top:6rem"></span>

## 4. DeepSeek-V2的MLA：缓存“压缩状态”而非全部展开KV

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-models-paper-reading/03-MHA-GQA-MQA-MLA.png" target="_blank" rel="noopener" aria-label="查看原图：MHA-GQA-MQA-MLA"><img src="/notes-assets/multimodal-models-paper-reading/03-MHA-GQA-MQA-MLA.png" alt="MHA-GQA-MQA-MLA" width="1772" height="1113" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">MHA-GQA-MQA-MLA（点击查看原图）</figcaption>
</figure>

读图时只比较每种Attention在历史位置保存什么。MHA保存多个头的K/V；GQA减少KV头并共享；MLA保存供多个头恢复/计算的压缩潜变量。别把图中不同颜色当成计算免费：压缩改变参数化，仍需投影和高效实现。原图来自[LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized)，署名保留；公式以[DeepSeek-V2第2.1节](https://arxiv.org/html/2405.04434v5)为准。

<span id="mm-ddb7e37f2573" style="display:block;scroll-margin-top:6rem"></span>

### 从普通KV到低秩联合表示

设某token隐藏状态为列向量 $h_t$。MLA先算

$$
c_t^{KV}=W^{DKV}h_t,\qquad
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}.
$$

$c_t^{KV}$的维度 $d_c$ 比全部头展开的KV小。关键不是“推理时把已经缓存的KV随意做PCA”，而是模型训练时就采用这个低秩联合参数化。推理可以保存 $c_t^{KV}$，不必永久保存每头完整内容K/V。

为什么有机会直接在压缩空间算分数？忽略RoPE时，

$$
q^\top k=q^\top W^{UK}c=((W^{UK})^\top q)^\top c.
$$

矩阵乘法结合律把上投影移到查询一侧，省掉对每个历史token重新展开K。V的线性上投影也可与输出投影结合；这种重排是否真正提速仍依赖内核与实现，并不从公式自动等于某固定吞吐倍数。

<span id="mm-ea98d8c5e6a4" style="display:block;scroll-margin-top:6rem"></span>

### RoPE为什么不能随意吸收进去

若K额外按各历史位置旋转，点积中出现与位置相关的旋转矩阵，不能一般地与 $W^{UK}$交换顺序。于是本来希望合并的矩阵之间插入了位置依赖，破坏简单吸收。

DeepSeek-V2采用解耦RoPE：内容部分保留压缩表示，另外生成携带位置的query分量及共享key分量 $k_t^R$。因此缓存不是只有 $c_t^{KV}$，还要保存位置key；每token每层的理想缓存元素数为 $d_c+d_R$。Q压缩虽然可减少部分训练激活，却不是减少历史KV缓存的原因。[MLA公式9—19](https://arxiv.org/html/2405.04434v5)

<span id="mm-af5657e391eb" style="display:block;scroll-margin-top:6rem"></span>

### 手算一个教学缓存比

假设普通MHA有32头，每头128维，则每token每层保存 $2\cdot32\cdot128=8192$ 个元素。假设某MLA配置 $d_c=512,d_R=64$，则为576个，理想缓存比为 $8192/576\approx14.22$。**这组数字是教学配置，不是声称DeepSeek-V2全部头数与模型配置如此。** dtype、层数、batch、上下文会继续放大绝对字节数；权重、工作区、解压中间量不在这条缓存公式里。

<span id="mm-1bed69a1a924" style="display:block;scroll-margin-top:6rem"></span>

### MLA与MoE为什么是两条线

MLA解决Attention的历史表示和缓存。MoE在FFN位置用router为token选择部分专家，改变每token激活计算和参数容量；专家权重驻留、负载不均衡和跨卡通信依然可能成为瓶颈。DeepSeek-V2把二者结合，因此不能把全部推理收益归因于MLA，也不能说“MoE是Decoder-only的替代架构”。官方236B总参数/21B激活是该固定型号描述，不是DeepSeek家族永恒属性。

<span id="mm-63226720a431" style="display:block;scroll-margin-top:6rem"></span>

## 5. Llama3.1提供怎样的对照

Llama3.1官方文本系列为8B、70B、405B，均使用GQA；模型卡声明128k上下文。这里的“支持长度”不等于所有位置检索、长视频或多跳推理在该长度都可靠；文本型号本身也没有视觉输入。[官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md)

把它与MLA比较时，GQA节省缓存靠减少KV头，仍直接保存各KV头的历史键和值；MLA采用潜变量及位置key。把它与Qwen-VL比较时，应分开文本骨干差异与视觉接入差异。若面试官实际指的是某视觉Llama或另一个DeepSeek版本，应重新查该版本资料，而非沿用本表。

另一个容易混淆的轴是base与instruct：base主要延续训练分布的文本，instruct经过后训练更能遵循指令；两者可能共享主干结构但行为不同。不能因为结构相同就用不同chat template或解码条件比较，更不能把后训练表现全部归于Attention结构。

<span id="mm-e081e01362b0" style="display:block;scroll-margin-top:6rem"></span>

## 6. 阅读论文时，怎样从“看过”变成“能解释”

论文不是型号介绍页。对于一个机制，应形成四个可检查的句子：旧方法在哪里失败；作者改了哪个运算或监督；成立依赖哪些条件；什么实验能区分这个解释与替代解释。下面是围绕本教程的阅读路线，读者可据此检查自己是否真正理解。

| 论文 | 应读懂的问题与机制 | 需要守住的假设 | 对应面试追问 |
|---|---|---|---|
| [ViT](https://arxiv.org/abs/2010.11929) | patch投影、位置、全局交互如何做分类 | 输入分辨率与位置适配，数据规模 | 与文本Transformer哪里相同、哪里不同 |
| [CLIP](https://arxiv.org/abs/2103.00020) | 双塔为何可检索，双向CE如何给梯度 | 原配对正确，负例可能有噪声 | 归一化/温度/假负例，手写InfoNCE |
| [SigLIP](https://arxiv.org/html/2303.15343v2) | 配对二分类怎样去掉全局softmax归一化 | 正负比例与偏置，归一化惯例 | 是否还需要负例，能否直接比loss值 |
| [BLIP-2](https://arxiv.org/html/2301.12597v3) | query瓶颈、ITC/ITM/ITG与冻结LLM适配 | mask防泄漏，压缩可能丢细节 | query为何可学，三种目标各补什么 |
| [LLaVA-1.5](https://arxiv.org/html/2310.03744v2) | 简单MLP与指令数据为何形成强基线 | 数据、分辨率和模型共同变化 | MLP与Q-Former如何公平比较 |
| [Qwen2.5-VL](https://arxiv.org/html/2502.13923v1) | 动态分辨率、window ViT、时间对齐 | 处理器和时间戳正确，预算有限 | 视觉token怎么算，FPS变化怎样处理 |
| [Qwen3-VL](https://arxiv.org/html/2511.21631v1) | 频率交错、多层视觉注入、时间文本 | 消融条件可比，时间文本有成本 | 为什么DeepStack不增加视觉序列长度 |
| [DeepSeek-V2](https://arxiv.org/html/2405.04434v5) | 低秩KV与矩阵吸收，解耦RoPE | 结合律不等于交换律，缓存包含位置key | MLA与GQA和MoE分别区别在哪 |
| [TempCompass](https://arxiv.org/abs/2403.00476) | 怎么检验真正时序能力 | 静态/语言捷径与评分协议 | 反转帧序列后如何设计控制实验 |
| [Video-MME](https://arxiv.org/abs/2405.21075) | 长短视频、多模态条件下怎样评估 | 输入和计分协议一致 | 总分提高是否代表时序提高 |

<span id="mm-ecd5f1aebbfa" style="display:block;scroll-margin-top:6rem"></span>

### 用一篇论文做完整的思考示范

以“加入多层视觉特征可保留细节”为假设。机制层面必须解释这些特征来自视觉塔哪些层、怎样映射、加入LLM哪里，输出shape是否一致；不能只写“增强融合”。证据层面需要同骨干、同数据、同训练量的消融。若OCR和图表提升而动态动作判断没有提升，结论应限定为相应能力，而非自动扩张成“视频理解全面提升”。

替代解释包括参数增加、数据更多、训练更久和输入分辨率改变。若控制它们后仍有收益，才能更强地支持多层特征的作用。原论文实验是作者在其条件下的证据；迁移到自己的低资源数据，仍是待验证假设。

读懂公式还意味着知道等号在什么条件成立。例如MLA矩阵吸收使用结合律；加入位置旋转后不能直接交换。正是这个细节解释为何需要解耦RoPE，比背“压缩KV”多了一层能够迁移的理解。

<span id="mm-6cbc2f8c6129" style="display:block;scroll-margin-top:6rem"></span>

## 练习与答案

1. **448×448在patch14、2×2空间合并时产生多少视觉token？896×896呢？** 分别256和1024；不含特殊token及视频时间分组。
2. **Qwen2.5-VL-7B的28层、4KV头、维128，8192长度、BF16、batch1，KV多少？** $2\cdot28\cdot8192\cdot4\cdot128\cdot2=469762048$ 字节，即448MiB。仅是此配置的KV，不含权重/视觉塔/激活。
3. **DeepStack额外注入三层特征就使上下文长度三倍吗？** 不必。若加到已有视觉位置，长度保持；计算、激活成本仍会增加。
4. **MLA只缓存512维潜变量就够吗？** 要看具体实现；DeepSeek-V2还需解耦RoPE的key。这里512也只是教学数值，不能推广。
5. **动态FPS能保证采到0.2秒事件吗？** 不能。训练适配不同采样频率与单段输入证据覆盖是不同问题。
6. **Qwen3-VL优于某文本模型就证明其Attention更好吗？** 不成立：输入模态、骨干、数据、训练预算和评价不同，需要受控消融。

返回[00-图解大模型算法与ZealD面经总览](/notes/multimodal-interview-guide/)，或用[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)检查能否不用型号标签把原因说清楚。
