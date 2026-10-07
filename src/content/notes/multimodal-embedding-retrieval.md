---
title: 多模态 Embedding 与检索：从 CLIP 双塔到 VLM2Vec
date: '2026-10-06'
tags: [多模态算法, 多模态检索, Embedding, VLM2Vec, 对比学习, GradCache]
summary: 讲清 VLM 怎样变成 embedding 模型：E5-V 的提示式表征、VLM2Vec 的 last-token 与 MMEB、GME，再讲池化、难负例与假负例、GradCache、Matryoshka 与二值化，最后给出一个 VLM embedding 项目的基线、消融与评测方案。
draft: false
---

CLIP 式双塔把图像和文本分别编码成向量，检索快，但一个向量只能表示“一张图”或“一句话”，处理“这张图里的裙子换成红色”这类图文混合查询很吃力，也不会按指令切换任务。另一条路线是把生成式 VLM 改造成 embedding 模型：图文混合输入进同一个解码器，取某个位置的隐藏状态作为向量。这一篇先讲这条路线的三个代表工作（E5-V、VLM2Vec 与 MMEB、GME），再展开训练里真正决定效果的工程问题：池化取哪个 token，负样本从哪里来、哪些其实是“假负例”，怎样用 GradCache 在有限显存下把 batch 做大，怎样用 Matryoshka 和二值化压缩索引，以及怎样评测。InfoNCE 与 CLIP 损失的推导见 [对比学习](/notes/contrastive-learning/) 和 [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) 第 3 节，本篇不重复。

阅读入口：[多模态专题总览](/notes/multimodal-interview-guide/)。数值算例为教学构造；论文数字注明表号，代码细节注明仓库提交。

## 1. 两条路线：双塔 CLIP 与基于 VLM 的 embedding

**输入输出。** 双塔：图像 $[B,3,H,W]$ 经视觉塔得到 $[B,d]$，文本 $[B,L]$ 经文本塔得到 $[B,d]$，两塔参数不共享，L2 归一化后做点积。VLM embedding：查询和目标都写成一段可能含图像占位符的 token 序列，经同一个 VLM（视觉塔 + 连接器 + LLM 解码器）得到最后一层隐藏状态 $H\in\mathbb R^{B\times L\times d}$，再池化成 $[B,d]$。以 Qwen2-VL-2B 为例，$d$ 就是 LLM 的 hidden size 1536（官方 `config.json`）。

| | CLIP 式双塔 | VLM embedding |
|---|---|---|
| 输入 | 单图或单句 | 任意图文交错序列，可带任务指令 |
| 编码器 | 两个独立编码器 | 一个解码器（因果注意力） |
| 向量来源 | 各塔的 CLS 或池化后再投影 | 通常是最后一个 token 的末层隐藏状态 |
| 代价 | 小、快，可预先离线编码 | 参数量大，一张图可能产生上千个视觉 token |
| 典型弱点 | 组合查询、指令跟随 | 编码慢、显存大，batch 难做大 |

**边界。** VLM embedding 仍然是“单向量”方法：查询和候选各自独立编码，最后只比较一个点积，不是精排里的交叉注意力。它的优势来自编码器更强、输入更灵活，不改变“召回只能比较向量”这一结构。所以召回之后照样可以接交互式精排，道理见 [召回与索引的成套更新](/notes/wangshusen-recommender-retrieval/)。

## 2. E5-V：用提示把输入“压进一个词”

E5-V（Jiang et al., arXiv 2407.12580）要解决的问题是：VLM 原本为生成训练，直接取最后一个 token 的隐藏状态，图像向量和文本向量各自聚成一团（论文 Figure 3 的“模态鸿沟”），没法跨模态检索。它借用文本 embedding 里 PromptEOL 的做法，给输入套一个让模型“用一个词总结”的提示（§3.1）：

```
<text>\n Summary above sentence in one word:
<image>\n Summary above image in one word:
```

然后取**最后一个 token** 的隐藏状态作为向量。直觉是：模型要预测的下一个词必须概括整段输入，而图和文都被要求压到“一个词”的同一个空间里。论文报告，这种提示本身就能大幅缩小模态鸿沟（Figure 3b）。

**只用文本训练。** 有了统一空间，E5-V 只在 NLI 句对上做对比学习（约 27.3 万对，每条带一个正例和一个难负例），训练时干脆去掉视觉编码器和投影层，只训练 LLM（§3.2）。基座是 LLaVA-NeXT-8B，QLoRA，1000 步，batch 768（§4）。损失是标准的带难负例 InfoNCE：

$$
\mathcal L=-\log\frac{e^{\cos(h_i,h_i^+)/\tau}}{\sum_{j=1}^{N}\big(e^{\cos(h_i,h_j^+)/\tau}+e^{\cos(h_i,h_j^-)/\tau}\big)}.
$$

**证据。** Table 6 对比三种表征：直接取最后 token（Last）、套提示但去掉 “in one word:”（Prompt）、完整提示（Our）。不微调时七项任务平均分别是 9.2、10.3、61.9；用同样的文本数据微调后是 68.2、75.6、80.7。Table 7 显示，换成 55.8 万 CC3M 图文对训练，效果反而不如纯文本训练。

**边界。** E5-V 的提示是人工写的，换任务要换提示（论文为 FashionIQ、CIRR 各写了一套）；纯文本训练依赖“提示已经消除模态鸿沟”这个前提，换一个基座要重新验证。在 MMEB 上它的总体分数很低（见第 3 节 Table 2 的 13.3），说明“一个词”的提示不足以表达分类、VQA、定位这类需要按指令取信息的任务。

## 3. VLM2Vec 与 MMEB

VLM2Vec（Jiang et al., arXiv 2410.05160，ICLR 2025，会议信息取自官方仓库 README）同时给出一个基准 MMEB 和一个训练框架。

**模型。** 任何 VLM 都可以做骨干，论文用 Phi-3.5-V 和 LLaVA-1.6。查询前面加任务指令，目标端不加（§3.1，式 (1)）：

```
q_inst = [IMAGE_TOKEN] Instruct: {task_definition} \n Query: {q}
```

查询和目标分别过同一个模型，取**最后一层、最后一个 token** 的向量 $h_q,h_t$。损失是对 in-batch 负例（加上有的话的难负例）做 InfoNCE，打分函数 $\phi(h_q,h_t)=\exp(\cos(h_q,h_t)/\tau)$，$\tau=0.02$（§3.1、§4）。

**MMEB 的组成**（§2、Table 1）。36 个数据集分属 4 类元任务，所有任务都改写成“给定指令和查询，从候选里挑出目标”的排序问题：

| 元任务 | 数据集数 | 训练用（分布内） | 只评测（分布外） | 例子 |
|---|---:|---:|---:|---|
| 分类 | 10 | 5 | 5 | ImageNet-1K、N24News；分布外如 ImageNet-A、ObjectNet |
| VQA | 10 | 6 | 4 | OK-VQA、DocVQA、ChartQA；分布外如 ScienceQA、TextVQA |
| 检索 | 12 | 8 | 4 | MSCOCO t2i/i2t、CIRR、WebQA；分布外如 OVEN、FashionIQ |
| 视觉定位 | 4 | 1 | 3 | MSCOCO；分布外如 RefCOCO、Visual7W-Pointing |
| 合计 | 36 | 20 | 16 | |

每个查询有 1000 个候选，指标是 Precision@1（排第一的候选是否正确）。作者在附录 A.2 / Table 5 比较了候选数，候选太少会很快饱和，1000 是成本与难度的折中。训练集取 20 个分布内数据集，超过 5 万条的随机采 5 万，共 66.2 万条（§4）。

**训练设置**（§4）。batch 1024，最大文本长度 256，2K 步，LoRA rank 8；用 GradCache 时子批大小 4，累积到总 batch 1024；8 张 H100。Phi-3.5-V 用 4 个子图裁块；LLaVA-1.6 统一缩放到 1344×1344 或 336×336。

**主要结果**（Table 2，Precision@1 平均）。最好的变体是 LLaVA-1.6 + LoRA + 1344×1344：总体 62.9，分布内 67.5，分布外 57.1。不在 MMEB 上微调的基线中，最好的是 UniIR（CLIP_SF）44.7；CLIP 37.8，E5-V 13.3。同骨干 Phi-3.5-V，全参微调 55.9，LoRA 60.1；LLaVA-1.6 降到 336×336 时为 55.0。

**消融。** ① Table 3（Phi-3.5-V，bs=256）：全参 52.0，LoRA rank 4/8/16/32 分别为 58.4、58.2、50.8、53.4，rank 不是越大越好。② Figure 4：batch、训练步数、子图数增大时效果逐渐上升，作者特别强调 batch，理由是多模态数据缺难负例，只能靠大 batch 提供更多随机负例（§3.2、§4.3.2）。③ Table 4：去掉指令后 VLM2Vec 从 52.0 降到 34.8；CLIP 加上指令反而从 37.8 降到 26.7。④ Figure 5：只在检索任务上训练的模型，迁移到其他元任务的效果最好，作者归因于检索任务的模态组合最丰富。

**官方仓库里的 2B 版本**（`TIGER-AI-Lab/VLM2Vec`，本文核对的提交为 `d00b5dc`，2026-09-21）。README 记录：2025-02 发布了基于 Qwen2-VL 2B 和 7B 的 VLM2Vec，MMEB 总分 60.1 和 65.8；2025-06 起 main 分支换成 V2 代码，V1 代码归档到 `v1` 分支。`TIGER-Lab/VLM2Vec-Qwen2VL-2B` 的模型卡正文写的是基于 `Qwen/Qwen2-VL-2B-Instruct`、LoRA、batch 2048、每个子数据集最多 10 万条、只用 in-batch 负例，分数为分类 59.0、VQA 49.4、检索 65.4、定位 73.4、总体 60.1。注意卡片元数据的 `base_model` 一栏写成了 7B，与正文不一致，以正文和加载代码为准。V2 的主力模型 VLM2Vec-V2.0 也基于 Qwen2-VL-2B（README；论文 arXiv 2507.04590 §4：batch 1024、LoRA rank 16、$\alpha=32$、$\tau=0.02$、GradCache、视频取 8 帧，MMEB-V2 共 78 个任务，Table 2 总分 58.0）。

仓库里 V2 的 2B 训练脚本 `experiments/public/train/train_v2-qwen2vl-2B.sh` 的关键参数：

| 参数 | 值 | 含义 |
|---|---|---|
| `--model_name` | `Qwen/Qwen2-VL-2B-Instruct` | 骨干 |
| `--lora --lora_r` | 16 | LoRA |
| `--pooling` / `--normalize` / `--temperature` | `eos` / True / 0.02 | 取最后一个有效 token，L2 归一化 |
| `--grad_cache` | True | 打开 GradCache |
| `--per_device_train_batch_size` | 128 | 8 卡共 1024 |
| `--gc_q_chunk_size` / `--gc_p_chunk_size` | 8 / 8 | 查询、目标两侧的子批大小 |
| `--interleave_batch_size` | 64 | 连续 64 条来自同一数据集（代码已标记为弃用，新名字是 `homogeneous_batch_size_per_device`） |
| `--learning_rate` / `--max_steps` / `--warmup_steps` | 5e-5 / 5000 / 100 | 线性调度 |

**边界。** MMEB 的分数是“1000 选 1”的 Precision@1，候选来自同一数据集，不等于百万级库里的召回率；分布外的 16 个数据集与训练集同属四类元任务，“分布外”指数据集未参与训练，不是任务类型全新。

## 4. GME：训练数据的配比与合成

GME（Zhang et al., arXiv 2412.16855，CVPR 2025）同样用 Qwen2-VL 做骨干、取最后一个 token 的末层隐藏状态（§4.1），重点在数据。它把检索分为单模态、跨模态、融合模态（查询或候选是图文混合）三类，在每类各 10 万条的对照实验里发现混合三类数据的模型最均衡（§4.1、Figure 3）。融合模态数据少，于是用 LLM 从维基段落生成查询、抽取实体、配图（检索或文生图），合成 110 万条（§4.2）。最终训练 800 万条，2B 与 7B 两个尺寸，LoRA rank 8，学习率 1e-4，温度 0.03，每张图最多 1024 个视觉 token，每条样本带 8 个负例（§5.1）。评测用自建的 UMRB（47 个子任务，含 BEIR、M-BEIR、ViDoRe 等），指标按子任务用 NDCG@10、NDCG@5、Recall@5/10（Table 3 标题）。附录还提到一个实际差距：同尺寸纯文本 embedding 模型在纯文本检索上仍然更强（gte-Qwen2-7B-instruct 60.25 对 GME-7B 55.63）。

## 5. 池化：last token 还是 mean

**为什么解码器常取最后一个 token。** 因果注意力下，位置 $t$ 只看得到 $1..t$。只有最后一个有效 token 看过整段输入；前面的 token 在“还不知道后文”时就定型了。mean pooling 把所有位置平均，等于把大量“没看完输入”的向量也算进来。注意力 mask 的细节见 [Transformer 与多模态入口](/notes/transformer-attention-rope-gqa/)。

**手算**（教学构造）。一个查询 3 个 token，末层隐藏状态（$d=3$，各行已单位化）：

$$
H=\begin{bmatrix}1&0&0\\0.6&0.8&0\\0&0.6&0.8\end{bmatrix},\qquad c_1=(0,0.6,0.8),\quad c_2=(1,0,0).
$$

把第 1 行想成指令前缀 “Instruct”，第 3 行是读完内容之后的位置，$c_1$ 是正确目标。

- last token：$h_3=(0,0.6,0.8)$，$\cos(h_3,c_1)=1$，$\cos(h_3,c_2)=0$，选 $c_1$。
- mean：$\bar h=(1.6,1.4,0.8)/3$，$\lVert(1.6,1.4,0.8)\rVert=\sqrt{5.16}\approx2.272$，$\cos(\bar h,c_1)=1.48/2.272\approx0.652$，$\cos(\bar h,c_2)=1.6/2.272\approx0.704$，选错成 $c_2$。

指令前缀越长，mean pooling 越被前缀主导。反过来，last token 把全部信息压在一个位置上，这个位置的表示好坏完全取决于训练。

**padding 的坑。** batch 内长度不同要补 pad。若右补齐一个 pad，隐藏状态为 $(0.5,0.5,0.5)$：直接取 `hidden[:, -1]` 拿到的是 pad，与 $c_1$、$c_2$ 的余弦为 0.808、0.577；不带 mask 的平均除以 4，得到 $(0.525,0.475,0.325)$，余弦 0.700、0.674。两种错法都会悄悄改变分数。VLM2Vec 仓库的 `_pooling` 用 `attention_mask` 找每行最后一个值为 1 的位置，注释写明左右补齐都适用；`eos`、`last_token` 都映射到 `last`。

**边界。** ① 生成模型的 chat 模板末尾可能有固定的结束符或换行，“最后一个 token”到底是哪个，训练和推理必须一致。② 双向编码器（BERT 类）没有这个不对称，mean 常常更稳；结论不能直接搬。③ 也有工作在 LLM 上改双向注意力或加潜在注意力池化，这里不展开。

**验证。** 同数据、同步数，只换 `--pooling last/mean`，比较 MMEB 四类元任务与分布外平均；另写单测：同一条样本单独编码与混在不同长度的 batch 里编码，向量的余弦应接近 1。

## 6. 负样本：in-batch、难负例与假负例

**in-batch 负例。** batch 有 $B$ 对 $(q_i,t_i)$，打分矩阵 $S=QT^\top\in\mathbb R^{B\times B}$，第 $i$ 行的标签是 $i$，其余 $B-1$ 个目标都当负例。多卡时先 all-gather 所有卡的向量再算，每个查询的负例数是“全局 batch − 1”。VLM2Vec 的 `DistributedContrastiveLoss` 把本卡那一份替换回带梯度的原张量，loss 乘以卡数，抵消 DDP 的梯度平均。它只算查询到目标一个方向，不是 CLIP 那样的对称损失。

**难负例**是与查询相似、但确实不匹配的候选，例如同类别的另一张图、只差一个属性的商品。常见来源：① 用当前模型或一个现成检索器（如 BM25、CLIP）召回 top-$K$，去掉正例后从中取；② 利用数据结构，例如同一张图里的其他区域、同一视频的相邻片段；③ 规则改写，例如把颜色、数量、左右换掉。常见做法是跳过排名最靠前的几个再采样，因为排得最靠前的“负例”最可能其实是正确答案。

**假负例**是被当作负例、实际上与查询匹配的候选。多模态数据里非常普遍：两张图都配得上“一只猫”；MMEB 这种格式下，分类任务的目标是类别名，同一 batch 里两张狗的图片对应同一个目标文本 “dog”，彼此就成了假负例；VQA 里大量答案是 “yes”、“2” 这类短文本，同理。若再按数据集组 batch（同一数据集的连续样本），重复目标的概率会更高。这是本文对数据格式的分析，不是某篇论文的结论。

**手算：一个假负例能有多大影响**（教学构造）。$\tau=0.02$，一个查询对正例的余弦是 0.80，对三个“负例”分别是 0.82（其实是重复的正确答案）、0.50、0.30。logits 为 $(40,41,25,15)$：

$$
\mathcal L=-\log\frac{e^{40}}{e^{40}+e^{41}+e^{25}+e^{15}}\approx1.313,
$$

正例概率 $0.269$，那个假负例独占 $0.731$。梯度把正例往上推 $0.731$，同时把一个正确答案往下压同样的量。把它屏蔽掉，损失降到约 $3\times10^{-7}$；不屏蔽而是在分母里给它乘权重 0.5 或 0.1，损失为 0.858 或 0.240。温度越低，这种“一个错标签主导整行”的现象越严重。

**常见处理**（通用做法）：① 去重：对同一 batch 内文本完全相同或图像近重复的目标，在打分矩阵里把这些位置 mask 掉，或当作多正例；② 阈值过滤：用当前模型或教师模型打分，负例分数超过“正例分数 − margin”或超过某个绝对阈值就丢弃；③ 降权：在分母里给可疑负例乘一个小于 1 的权重；④ 采样上避开：难负例从 top-$K$ 中跳过最靠前的若干名。阈值和 margin 没有通用值，要在验证集上扫。

**边界。** 过滤太狠会把真正的难负例也去掉，模型学不会细粒度区分；用当前模型打分来过滤还会自我强化。**验证**：人工抽查被过滤掉的负例里有多少确实是正确答案（过滤的精确率），再比较打开、关闭过滤时分布外的平均分和各元任务分数。

## 7. GradCache：在固定显存下把 batch 做大

**问题。** 对比损失的每一行都依赖整个 batch 的目标向量，所以不能像交叉熵那样简单地切成微批次累积梯度：普通梯度累积每次只有微批次内的负例。直接用大 batch，又要同时存下所有样本的激活值，显存随 batch 线性增长。

**原理**（Gao et al., arXiv 2101.06983，RepL4NLP 2021，§3.2–3.3）。损失对参数的梯度可以分两段写：

$$
\frac{\partial\mathcal L}{\partial\Theta}=\sum_{j}\sum_{i\in\hat S_j}\underbrace{\frac{\partial\mathcal L}{\partial f(s_i)}}_{u_i}\ \frac{\partial f(s_i)}{\partial\Theta}.
$$

$u_i$ 只依赖所有样本的向量数值，不依赖编码器内部；给定 $u_i$ 后，各样本对参数的梯度彼此独立。于是分四步：① 不建计算图，逐个子批前向，得到全部向量；② 只在向量上建图算损失，反传得到每个向量的梯度 $u_i$ 并缓存；③ 逐个子批重新带图前向，把缓存的 $u_i$ 当作上游梯度反传，累积参数梯度；④ 全部子批处理完后更新一次参数。论文称这与直接用大 batch 得到的梯度完全相同；多卡时在第 ① 步之后做一次 all-gather。

**代码怎样写。** VLM2Vec 仓库 `src/grad_cache/grad_cache.py`：`forward_no_grad` 在 `torch.no_grad()` 下前向，同时用 `RandContext` 记下每个子批的随机数状态；第 ③ 步在同一随机状态下重算，保证 dropout 一致；反传用一个代理标量 `torch.dot(reps.flatten(), gradient.flatten()).backward()`，它对 `reps` 的梯度正好是缓存的 $u$；`no_sync_except_last=True` 让 DDP 只在最后一个子批同步梯度。`GradCacheLateProcessTrainer.training_step` 里有一个容易忽略的分支：只有在分布式且卡数大于 1 时才走 GradCache，单卡时直接 `model(queries, targets)` 整批前向，此时 `gc_*_chunk_size` 不起作用（以 `d00b5dc` 为准）。

**手算：显存与计算**（教学构造，激活量为假设值）。8 卡，每卡 128 个查询 + 128 个目标，向量维度 1536。假设带计算图时每个样本的激活约 0.6 GB。

- 直接整批：每卡 $256\times0.6=153.6$ GB，超过 80 GB。
- GradCache 子批 8：激活峰值 $8\times0.6=4.8$ GB，另加 bf16 权重约 4.4 GB（按 2.2B 参数估）与 LoRA 的优化器状态。缓存本身很小：本卡 256 个 fp32 向量 $256\times1536\times4\text{ B}\approx1.57$ MB，梯度同样大；all-gather 后 2048 个向量约 12.6 MB；$1024\times1024$ 的 fp32 打分矩阵约 4.2 MB。
- 计算：记一次前向为 $F$，反向约 $2F$。普通训练 $3F$；GradCache 多一次无图前向，为 $4F$，多约 33%。若本来就开了梯度检查点（$4F$ 变 $5F$），多约 25%。LoRA 下参数梯度少，反向可能小于 $2F$，额外开销的占比反而更大。论文在 DPR 上实测约多 20% 时间（§4，Figure 1）。每卡每步要跑 $128/8=16$ 个子批，两侧共 32 次无图前向和 32 次带图前向反传。
- 负例数：GradCache 每个查询有 1023 个 in-batch 负例。若改用梯度累积，每次微步 8 卡各 8 条，gather 后只有 63 个负例。

**边界。** ① “梯度完全相同”的前提是两次前向数值一致：随机状态要复现；BatchNorm 这类依赖 batch 统计量的层会让子批与整批不同。② 第 ① 步在 `no_grad` 下用的是更新前的参数，第 ③ 步也是，这一点没有问题；但如果中途改了模型状态（例如切换 train/eval），两次前向就不一致。③ GradCache 只解决显存，不解决假负例：batch 越大，撞到重复目标的概率越高。**验证**：小 batch 下关掉 GradCache 与打开 GradCache，各算一步参数梯度，逐元素比较应在数值误差内一致；训练时记录每步耗时与峰值显存。

## 8. Matryoshka 与二值化：压缩索引

**Matryoshka 表征**（Kusupati et al., arXiv 2205.13147，NeurIPS 2022）。在一组嵌套维度 $\mathcal M$（论文 ResNet50 例子是 $\{8,16,\dots,1024,2048\}$）上同时优化前 $m$ 维的损失（式 (1)，$c_m$ 默认全为 1）：

$$
\min\ \frac1N\sum_{i}\sum_{m\in\mathcal M}c_m\,\mathcal L\big(z_{i,1:m}\big).
$$

用于对比学习时，查询和目标两侧都截断到前 $m$ 维，并且每个维度各自做归一化（§3、附录 C）。训好后，前 $m$ 维本身就是可用的向量，部署时按预算截断。论文还给出自适应检索：先用低维向量召回一个短名单（例如 $D_s=16$、$K=200$），再用高维向量重排（§4.3.1）。VLM2Vec 的论文与仓库训练参数里没有 Matryoshka 损失，对它的向量直接截断属于“没训练过的压缩”，必须单独评测。

**手算截断**（教学构造）。8 维向量 $q=(0.3,-0.2,0.5,0.1,-0.4,0.2,-0.1,0.6)$，$d_1=(0.2,-0.1,0.4,-0.05,-0.3,0.3,0.1,0.5)$，$d_2=(-0.3,0.2,0.1,0.4,0.3,-0.2,-0.1,0.2)$。全维余弦 $\cos(q,d_1)\approx0.941$，$\cos(q,d_2)\approx-0.103$。只取前 4 维并重新归一化：$\cos(q_{1:4},d_{1,1:4})=0.275/(0.6245\times0.4610)\approx0.955$，$\cos(q_{1:4},d_{2,1:4})\approx-0.117$。这里排序没变，但这是构造出来的；没有 Matryoshka 训练时，前 $m$ 维不保证承载主要信息。

**二值化**（通用工程做法）。把归一化后的每一维按符号变成 1 bit（大于 0 记 1，否则记 0），8 bit 打包成一个字节；距离用 Hamming 距离 $\mathrm{popcount}(a\oplus b)$，在 CPU 上就是异或加位计数。上例中：

| 向量 | 符号码 | 十六进制 | 与 $q$ 的 Hamming 距离 | 浮点余弦 |
|---|---|---|---:|---:|
| $q$ | 10110101 | B5 | 0 | 1 |
| $d_1$ | 10100111 | A7 | 2 | 0.941 |
| $d_2$ | 01111001 | 79 | 4 | −0.103 |

$\mathrm{B5}\oplus\mathrm{A7}=00010010$，两个 1；$\mathrm{B5}\oplus\mathrm{79}=11001100$，四个 1。注意 $d_1$ 的第 4 维是 $-0.05$、$q$ 的第 4 维是 $0.1$，都接近 0，却贡献了一位差异：靠近 0 的坐标最容易翻转。

**存储账**（教学构造）。64 维 fp32 每条 $64\times4=256$ B，fp16 128 B，64 bit 二值码 8 B，压缩 32 倍；100 万条分别约 256 MB、128 MB、8 MB。Qwen2-VL-2B 的 1536 维 fp32 每条 6144 B，100 万条约 6.1 GB；二值化后 192 B，约 192 MB；若先用 Matryoshka 截到 256 维再二值化，只要 32 B。

**为什么符号码能近似角度。** 若先对向量做一次随机旋转，每一位就相当于一个随机超平面。两个向量被一个随机过原点的超平面分开的概率，等于它们夹角 $\theta$ 与 $\pi$ 之比（在两向量张成的平面里，分界线方向均匀分布，落在夹角内的比例就是 $\theta/\pi$）。所以 $b$ 位码的期望 Hamming 距离约为 $b\,\theta/\pi$。不旋转、直接对原坐标取符号时，各位不独立，这个关系只是粗略近似。

**常见做法：二值召回 + 浮点重排。** 先用 Hamming 距离取 $r\times K$ 个候选（$r$ 是放大倍数），再用浮点查询向量对它们重新打分取 top-$K$；Sentence Transformers 文档的 embedding quantization 一节就是这样描述的。int8 标量量化是另一种折中，需要用一批校准数据估计每维的取值范围。更一般的乘积量化见 [语义 ID](/notes/recsys-semantic-id/)。

**验证。** 固定评测集，画“每条存储字节 vs Recall@K”曲线：浮点全维、截断到若干维、二值化、二值化加重排。

## 9. 评测：Recall@K、Precision@1 与协议

单正例时，$\mathrm{Recall@}K$ 就是正确目标进入前 $K$ 的查询比例：

$$
\mathrm{Recall@}K=\frac{1}{\lvert Q\rvert}\sum_{q\in Q}\mathbf 1\big[\mathrm{rank}(t_q^+)\le K\big].
$$

MMEB 的 Precision@1 在单正例下等于 Recall@1。多正例时要区分“至少命中一个”（Hit@K）和“命中的正例占全部正例的比例”，论文、榜单各有口径，引用时写清楚。GME 的 UMRB 按子任务用 NDCG@10、NDCG@5 或 Recall@5/10，与 MMEB 的分数不能直接比较。

**协议上必须固定的东西**：候选集（MMEB 每题 1000 个，换成全库检索会难得多）；查询端指令的写法（Table 4 说明指令影响很大）；图像分辨率与视觉 token 上限；是否归一化；平局怎样处理。复现时先用官方权重加官方脚本跑出与论文、模型卡一致的数字，再改任何东西。

抖音、快手、小红书等已经上线的多模态表征系统怎样构造训练数据、怎样接入下游、线上收益多大，见 [工业界多模态表征](/notes/multimodal-industrial-representation/)；按面试深挖顺序整理的问题见 [表征项目面试题库](/notes/multimodal-representation-interview/)。

## 10. 做一个 VLM embedding 项目

目标：以 Qwen2-VL-2B 为骨干，在 MMEB 训练集上复现 VLM2Vec 式训练，并做一组能解释因果的消融。**不承诺具体分数。**

**基线。** ① 零样本：CLIP / SigLIP 与未训练的 Qwen2-VL-2B（直接取最后 token，以及套 E5-V 式提示）；② 官方 `VLM2Vec-Qwen2VL-2B` 权重在本地复测，与模型卡的 60.1 对齐，确认评测脚本无误；③ 自己按官方配置训练的版本。

**消融**（每次只改一项，其余照基线）：

| 消融 | 设置 | 看什么 |
|---|---|---|
| 池化 | last vs mean（mask 正确） | 四类元任务与分布外平均；长指令任务是否差异更大 |
| batch / GradCache | 全局 batch 256、1024、2048 | 分数随 batch 的变化，耗时与峰值显存 |
| 难负例 | 无；用第一轮模型挖 top-$K$（跳过最前几名） | 检索、定位类是否提升，分类是否下降 |
| 假负例处理 | 无；同 batch 重复目标 mask；阈值过滤；降权 | 被过滤负例的人工抽查精确率；分类、VQA 是否受益 |
| 同源 batch | 随机混合 vs 每 64 条同一数据集 | 与假负例处理交叉做，看是否互相抵消 |
| 维度与比特 | 全维 fp32、截断到 256/512 维、二值化、二值化加重排 | 每条字节数对 Recall@K 的曲线 |

**评测。** 主指标沿用 MMEB 的 Precision@1，按元任务、分布内外分开报。另建一个“大库”设定：把多个数据集的候选合并成十万级库，报 Recall@1/5/10，模拟真实召回。至少两个随机种子；每个数据集 1000 条左右的查询，差异在一个点以内时不下结论，可以按查询做 bootstrap 估计区间。

**常见坑。** ① `--normalize` 默认是 False，忘了打开时 logits 是未归一化点积除以 0.02，数值会爆，损失不收敛。② 单卡跑仓库代码时 GradCache 分支不生效，显存不够会以为是 chunk 设置问题。③ 训练与评测的指令模板、图像分辨率、视觉 token 上限不一致。④ padding 方向与池化位置不匹配（第 5 节）。⑤ 评测集与训练集的图像重叠，例如 MSCOCO 同时出现在多个子集里，要按图像去重。⑥ 分类类任务同 batch 重复目标导致的假负例（第 6 节）。⑦ 用 MMEB 的 1000 候选分数去推断百万库上的召回。⑧ 二值化前没有归一化或没有中心化，导致某些维几乎全是同一个符号，信息量很低；可以统计每一位取 1 的比例，接近 0 或 1 的位基本没用。

## 11. 面试常问

**VLM embedding 为什么取最后一个 token 而不是平均？** 因果注意力下只有最后一个有效 token 看过整段输入，前面的位置没看到后文；平均会被长指令前缀主导（第 5 节的手算）。前提是 mask 正确找到最后一个有效位置，且训练和推理模板一致。双向编码器没有这个不对称。

**GradCache 和梯度累积有什么区别？** 梯度累积每个微批独立算损失，负例只有微批内的样本；GradCache 先无图前向拿到全 batch 的向量，算出每个向量的梯度并缓存，再逐子批重算反传，负例是整个 batch，梯度与大 batch 一致。代价是多一次前向，论文实测约多 20% 时间。

**假负例怎么处理？** 先识别来源：重复文本目标、近重复图像、一图多描述。处理手段有 mask 或改成多正例、按分数阈值过滤、降权、难负例采样时跳过最靠前的几名。要抽查被过滤的负例，确认过滤的精确率，并在验证集上调阈值。

**E5-V 和 VLM2Vec 的区别？** E5-V 靠“用一个词总结”的提示统一图文空间，只用文本句对训练，零样本可用；VLM2Vec 在 36 个数据集组成的 MMEB 上做多任务对比训练，查询端加任务指令，MMEB 上远高于 E5-V（Table 2：62.9 对 13.3）。

**二值化后召回掉点怎么办？** 二值召回放大候选数，再用浮点向量重排；或用 Matryoshka 先截断再量化，做一条存储与召回的权衡曲线，按预算选点。

## 闭卷验收

不看资料能画出 CLIP 双塔与 VLM embedding 的输入输出与张量形状；写出 E5-V 的两条提示，说出它为什么只用文本训练也能做图文检索，以及 Table 6 里 Last、Prompt、Our 三者的差距；说出 VLM2Vec 的查询模板、取向量的位置、$\tau$，以及 MMEB 的 36 个数据集怎样分成 4 类、20 个训练与 16 个只评测，候选数和指标是什么；说出官方 2B 版本的骨干与训练脚本中 GradCache 相关的三个参数；用第 5 节的矩阵手算 last 与 mean 的选择结果，并指出 padding 怎样让两种池化都出错；用 logits $(40,41,25,15)$ 算出假负例下的损失并解释为什么低温度放大问题；写出 GradCache 的梯度分解和四个步骤，估算子批 8 时的激活峰值、缓存大小与额外计算；写出 Matryoshka 的损失，算出 64 维 fp32 与 64 bit 码的存储比，对两个 8 位码算 Hamming 距离；最后讲出一个 VLM embedding 项目的基线、六组消融和八个常见坑。

**参考。** [E5-V](https://arxiv.org/abs/2407.12580)（[代码](https://github.com/kongds/E5-V)）；[VLM2Vec](https://arxiv.org/abs/2410.05160)（[HTML v3](https://arxiv.org/html/2410.05160v3)）；[VLM2Vec-V2](https://arxiv.org/abs/2507.04590)；[VLM2Vec 官方仓库](https://github.com/TIGER-AI-Lab/VLM2Vec)（提交 `d00b5dc`：`src/model/model.py`、`src/loss.py`、`src/trainer.py`、`src/grad_cache/grad_cache.py`、`src/arguments.py`、`experiments/public/train/train_v2-qwen2vl-2B.sh`）；[VLM2Vec-Qwen2VL-2B 模型卡](https://huggingface.co/TIGER-Lab/VLM2Vec-Qwen2VL-2B)；[Qwen2-VL-2B-Instruct config](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/config.json)；[GME](https://arxiv.org/abs/2412.16855)；[GradCache](https://arxiv.org/abs/2101.06983)（[代码](https://github.com/luyug/GradCache)）；[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)（[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract.html)）；[Sentence Transformers：Embedding Quantization](https://sbert.net/examples/sentence_transformer/applications/embedding-quantization/README.html)；[CLIP](https://arxiv.org/abs/2103.00020)。
