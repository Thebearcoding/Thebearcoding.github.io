---
title: 十四天练习册：从手算到多模态面试
date: '2026-09-22'
tags:
- 多模态算法
- 两周学习
- 练习
- 面试
- 答案详解
summary: 24道编号练习附完整参考解答，配合论文项目与闭卷模拟验收。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-1c10f85e7c47" style="display:block;scroll-margin-top:6rem"></span>

这份练习册以纸笔和普通计算器为主。每天先读指定正文，做题时遮住参考解答，完成后用不同颜色写出错误原因。基础路线每天约两小时；推导不熟时可以拆成两天。进度由能否独立解释决定，日历只负责安排复习节奏。

每天留下三件小成果：一页手算、一张你自己能解释的结构图、一段约两分钟的口述。所有数字题都是教学构造，不是模型跑分；第 13、14 天的项目方案也不能在面试中说成已经完成的实验。

<span id="mm-0e7fa6327abb" style="display:block;scroll-margin-top:6rem"></span>

## 第 1 天：向量、条件概率与 softmax

阅读[读懂大模型公式之前：数学与张量预备课](/notes/multimodal-math-prerequisites/)前五节。

**练习 1**：$x=(2,1)$，$W=\begin{bmatrix}1&0&-1\\0&2&1\end{bmatrix}$，求 $xW$。再说明结果是几个 token、几个特征。

**参考解答**：依次与三个列向量点积，得到 $(2,2,-1)$。这是一个 token 的三个新特征；不是三个 token。若有 5 个输入位置，$X:[5,2]$，结果为 $[5,3]$。把行数和通道宽度混淆，会在多头 reshape 时继续出错。

**练习 2**：两个 logits 为 $(0,\log3)$，求 softmax；目标是第二类时，交叉熵是多少？若第一、第二个回答 token 的正确条件概率为 0.75、0.4，整段概率与负 log-prob 是多少？

**参考解答**：指数为 $(1,3)$，概率 $(0.25,0.75)$，交叉熵 $-\log0.75\approx0.28768$。整段概率为 $0.75\cdot0.4=0.3$，负 log-prob 为 $-\log0.3\approx1.20397$；也等于两个负 log-prob 相加。加概率不符合链式分解。

**闭卷讲解**：为什么模型喜欢相加 log-prob？如果你只说“数值稳定”，再补上连乘变求和的概率依据。

<span id="mm-6eac9282895c" style="display:block;scroll-margin-top:6rem"></span>

## 第 2 天：Attention 的每一行在做什么

阅读[从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)第 1–3 节，照着原图从右至左解释完整模型，再聚焦单头。

**练习 3**：设 $Q=K=\begin{bmatrix}1&0\\0&1\end{bmatrix}$，$V=\begin{bmatrix}2&0\\0&4\end{bmatrix}$，$d_h=2$，使用因果 mask。求两行输出。

**参考解答**：分数与正文单位向量例相同。第一行只看自己，输出 $(2,0)$；第二行概率约 $(0.33024,0.66976)$，输出约 $(0.66048,2.67905)$。更改 V 不一定更改本题的分数，因为题目把 Q/K 固定；真实模型 Q/K/V 都来自输入投影，改输入时三者一般都变。

**练习 4**：把未来位置的分数改成很大，但因果 mask 正确，过去位置的输出是否改变？若改的是 padding mask 呢？

**参考解答**：不可见分数加 $-\infty$ 后概率仍为 0，过去输出不受影响。padding mask 控制无效 key 是否进入同一行的归一化；若漏掉一个高分 padding key，可见内容权重会被错误稀释。两类 mask 都是可见性，均不等于训练 label mask。

<span id="mm-abd5316d34c9" style="display:block;scroll-margin-top:6rem"></span>

## 第 3 天：RoPE、归一化与门控

阅读[从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)第 5–7 节。把“旋转”解释到二维矩阵，不只背名称。

**练习 5**：$q=k=(1,0)$，每个位置旋转 $30^\circ$；$m=1,n=3$，求旋转后点积。把两位置同时加 5，结果变不变？

**参考解答**：角差为 $(3-1)30^\circ=60^\circ$，点积为 $\cos60^\circ=0.5$。两位置同时平移，角差仍为 $60^\circ$，这一纯 RoPE 点积不变；这不代表有其他位置机制、边界和不同可见范围的完整模型输出一定不变。

**练习 6**：对 $x=(3,4)$，$g=(1,1)$，忽略 $\epsilon$，分别算 RMSNorm 和无仿射 LayerNorm。SwiGLU 某通道的 gate=1、up=2 时门控乘积是多少？

**参考解答**：RMSNorm 为 $(3,4)/\sqrt{12.5}\approx(0.84853,1.13137)$；LayerNorm 去掉均值 3.5、除标准差 0.5，得 $(-1,1)$。门控乘积为 $2\sigma(1)\approx1.46212$，完整 FFN 还要做输出投影。若误把 SiLU 当 sigmoid，会在负输入与幅值上理解错。

<span id="mm-c28d154fc4a9" style="display:block;scroll-margin-top:6rem"></span>

## 第 4 天：缓存、GQA 与推理阶段

阅读[Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/)第 1–4 节。

**练习 7**：历史 3 个 token，一次新增 2 个 token。写出 $[T_q,T_k]$ 和可见矩阵。用一句话解释新 token 的 RoPE 索引。

**参考解答**：形状为 $[2,5]$，可见矩阵为 $\begin{bmatrix}1&1&1&1&0\\1&1&1&1&1\end{bmatrix}$。新位置使用 3、4（从零编号），不重新从 0 开始。第一行未来新 token 仍不可见；第二行可以看到它前面的全部位置。

**练习 8**：32 层、batch 2、4 KV 头、每头 128 维、BF16，增加 512 个视觉 token，缓存增加多少？若 query heads=16，共享关系是什么？

**参考解答**：

$$
2\cdot32\cdot2\cdot512\cdot4\cdot128\cdot2
=67{,}108{,}864\ {\rm bytes}=64\ {\rm MiB}.
$$

每 4 个 Q 头共享一套 K/V。第一项 2 表示 K/V，最后一项 2 表示 BF16 字节数；二者不能漏掉。该结果不包含模型权重或训练激活。

**追问**：为什么更少 KV 头不等于少看历史 token？因为每个 Q 仍然要对可见位置匹配，共享改变的是 K/V 组数。

<span id="mm-673e801731a6" style="display:block;scroll-margin-top:6rem"></span>

## 第 5 天：ViT、CLIP 与图文匹配

阅读[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)的视觉编码、CLIP 与 SigLIP 部分。

**练习 9**：$224\times224$ RGB 图，以 $16\times16$ patch 切分，每 patch 展平宽度和 token 数分别多少？高宽都变为 448 时呢？

**参考解答**：每 patch 展平 $16\cdot16\cdot3=768$ 维，初始 patch 数 $14^2=196$。高宽翻倍后为 $28^2=784$，四倍；不是两倍。这里未计 CLS、动态裁剪或空间合并，因此不可冒充所有处理器的实际输出。

**练习 10**：图文 logits 矩阵 $S=\begin{bmatrix}2&0\\0&2\end{bmatrix}$，对角线为正确配对，求双向 CLIP 损失（两个方向平均）。如果第二条 caption 其实也适合第一张图，发生什么问题？

**参考解答**：每行正确概率 $e^2/(e^2+1)\approx0.88080$，损失 $\log(1+e^{-2})\approx0.12693$。两行两列都相同，平均后仍为 0.12693。第二 caption 被当作负例却可能语义正确，是假负例；更多负例并不保证每条监督都更可靠。

**闭卷讲解**：CLIP 的检索表示为何不能直接替代按视觉问题生成答案？答题中要出现输入、输出和训练目标三项区别。

<span id="mm-116d92223733" style="display:block;scroll-margin-top:6rem"></span>

## 第 6 天：连接器、帧采样与事件证据

阅读[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)的连接器与视频部分，暂不进入 RL。

**练习 11**：8 帧，每帧 196 个视觉 token，压缩后每帧 32 个，比较总量。若问题是辨认手表上的小字，和“先拿杯子还是先开门”，压缩风险分别是什么？

**参考解答**：1568 降至 256。前者依赖空间细节，压缩可能消掉小字；后者依赖事件覆盖和先后关系，可能因漏帧或时间表示不足失败。不能用“省了 84% token”直接证明质量可接受，应按这两类问题分别评估。

**练习 12**：10 秒视频事件持续 0.2 秒，理想化独立均匀随机采 5 个瞬时帧，至少命中一次的概率是多少？均匀固定采样是否服从同一公式？

**参考解答**：一次没命中概率 $0.98$，五次全没命中为 $0.98^5$，命中至少一次约 9.61%。固定时间网格不独立随机，其结果取决于事件窗口与采样相位，不能直接套此公式。这个差别是采样假设，而不是计算器误差。

**纸上实验**：画 0、2、4、6、8 秒五条竖线，再画 4.9–5.1 秒事件窗口。看到“证据完全未输入”时，解释为什么单靠增加后训练无法可靠恢复它。

<span id="mm-6ed5664e3030" style="display:block;scroll-margin-top:6rem"></span>

## 第 7 天：SFT 的逐位置监督

阅读[SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/)的 shift 表、交叉熵和数据部分。复习第 2 天的 attention mask。

**练习 13**：简化序列为 $[\text{BOS},\text{问},\text{答},\text{EOS}]$，仅监督“答”和 EOS，按标签与输入同位置的 convention 写 labels；说明哪两个 logits 参与预测。

**参考解答**：labels 可写为 $[-100,-100,\text{答},\text{EOS}]$。做一次 shift 后，位置 1 的 logits 预测位置 2 的“答”，位置 2 的 logits 预测 EOS。若模型内部已经 shift，就不能自己再次 shift。prompt 对应标签忽略，不代表回答看不到 prompt。

**练习 14**：某 token 的三类概率为 $(0.1,0.7,0.2)$，目标为第 2 类。交叉熵对三个 logits 的梯度是什么？

**参考解答**：$p-\mathrm{onehot}(2)=(0.1,-0.3,0.2)$。梯度下降倾向提高目标 logit、压低其他 logits。这个局部方向没有核查训练答案的事实真伪，错误标签也会被有效学习。

<span id="mm-adf10ab57ca6" style="display:block;scroll-margin-top:6rem"></span>

## 第 8 天：LoRA 与偏好学习

阅读[SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/)的 LoRA、QLoRA、DPO 推导。

**练习 15**：$4096\times4096$ 权重上加 rank=8 LoRA，不计偏置，新增多少参数？A 随机、B=0 时，为何首步仍能学习？两者都为零呢？

**参考解答**：新增 $8(4096+4096)=65{,}536$ 个参数。$BA=0$ 保留初始函数；B 的梯度含非零 A，一般可更新，A 首步梯度含 B 而为零。若 A/B 都零，两者梯度都可能被另一零矩阵挡住。这是常用初始化的代数原因，不是随机种子的习惯。

**练习 16**：chosen/rejected 的当前 log-prob 为 $-2,-3$，reference 都为 $-2.5$，$\beta=0.2$。求 DPO logit 与 loss。若固定 reward 下将 KL 正则系数 $\beta$ 增大，精确最优策略的取舍是什么？

**参考解答**：相对提升为 $0.5,-0.5$，差为 1，logit 为 0.2，loss $\log(1+e^{-0.2})\approx0.59814$。在正文 $\mathbb E r-\beta KL$ 的约定里，增大 $\beta$ 加强靠近 reference 的权衡；这不保证有限步实际训练的波动必然更小。

<span id="mm-5d00211e9442" style="display:block;scroll-margin-top:6rem"></span>

## 第 9 天：从奖励到策略梯度

阅读[从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)第 1–4 节；只要到 baseline，不必一天读完整个 RL 长章。

**练习 17**：奖励 $(0,1,2)$、$\gamma=0.9$，从后向前算 $G_2,G_1,G_0$。若当前 $V(s_0)=0.5,V(s_1)=1$，第一步 MC 与 TD 目标是什么？

**参考解答**：$G_2=2,G_1=1+0.9(2)=2.8,G_0=0.9(2.8)=2.52$。MC 目标为 2.52；TD 为 $0+0.9(1)=0.9$。二者分别依赖实际完整未来与下一状态的估计。

**练习 18**：单步策略好动作概率 $p=\sigma(\theta)=0.4$、奖励为1；坏动作奖励0，求真实期望奖励梯度。减基线 $b=0.4$ 后期望是否改变？

**参考解答**：$J=p$，梯度 $p(1-p)=0.24$。采好动作后的样本梯度为 $(1-0.4)(1-0.4)=0.36$；采坏动作后为 $(0-0.4)(-0.4)=0.16$；加权后 $0.4(0.36)+0.6(0.16)=0.24$。这里 b 不依赖当前动作，才符合基线证明条件。

<span id="mm-a583072e90ed" style="display:block;scroll-margin-top:6rem"></span>

## 第 10 天：GAE 与终止边界

阅读[从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)第 5 节。

**练习 19**：沿用昨天奖励，$V=(0.5,1,1.5)$，最后真正终止，$\lambda=0.8$。先算每步 TD 残差，再反向算 GAE。

**参考解答**：$\delta=(0.4,1.35,0.5)$，$\gamma\lambda=0.72$。依次得 $A_2=0.5$、$A_1=1.35+0.72(0.5)=1.71$、$A_0=0.4+0.72(1.71)=1.6312$。若 $\lambda=1$，$A_0=2.02=G_0-V(s_0)$，这是检查 telescoping 的好方法。

**练习 20**：某步 $r=1,V(s)=0.5,V(s_{\rm final})=2,\gamma=0.9$。若真正 terminal，$\delta$ 是多少？若只是时间限制截断，允许 bootstrap，但不跨 reset 累加下一 episode 的优势，如何处理？

**参考解答**：terminal 不 bootstrap，$\delta=1-0.5=0.5$。可 bootstrap 的截断用 reset 前的 final observation，$\delta=1+0.9(2)-0.5=2.3$；优势递推的 continuation mask 为0，不能接上 reset 后新 episode 的优势。两个 mask 控制不同边界。

<span id="mm-6560cc0b46ee" style="display:block;scroll-margin-top:6rem"></span>

## 第 11 天：PPO 与 GRPO 的训练信号

阅读[从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)第 6–9 节。重看 PPO 四模型原图，逐一说明是否更新、输出什么。

**练习 21**：$\epsilon=0.2$，分别求 $(A,\rho)=(2,1.3),(-2,0.7),(2,0.7),(-2,1.3)$ 的 clipped surrogate。

**参考解答**：分别取 $\min(2.6,2.4)=2.4$、$\min(-1.4,-1.6)=-1.6$、$\min(1.4,1.6)=1.4$、$\min(-2.6,-2.4)=-2.6$。后两种仍会惩罚朝错误方向走的概率变化；不能说“ratio 超出范围，所有梯度都零”。

**练习 22**：同题奖励 $[1,0,0,1]$，按总体标准差归一优势是多少？若新旧策略相同，平均 surrogate 为0，是否没梯度？

**参考解答**：均值与标准差均为0.5，优势 $[1,-1,-1,1]$。新旧 ratio 为1，因此正负项平均值为0；它们对不同回答概率的导数不一样，梯度通常非零。只有所有奖励相同使各优势均为0时，这个优势项才无更新信号（其他正则另论）。正文的两动作例在 $p=0.5$ 处给出 $J=0,\nabla J=0.5$。

<span id="mm-b99bc08c2eda" style="display:block;scroll-margin-top:6rem"></span>

## 第 12 天：评估、显存与排障

阅读[训练显存与实验排障：把机制变成可检查的量](/notes/training-memory-debugging/)和[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)的评估部分。

**练习 23**：事件定位真区间 $[4,8]$，预测 $[6,10]$，求 tIoU。另有分类结果 TP=18、FP=6、FN=12，求 precision、recall、F1。

**参考解答**：区间交集2、并集6，tIoU=$1/3$。precision=$18/24=0.75$，recall=$18/30=0.6$，F1=$2PR/(P+R)=2/3$。tIoU 是时间区间重叠；F1 是分类或检索集合指标，不能互相替代。宏/微平均等汇总口径还需另报。

**练习 24**：两个 micro-batch 有效 token 数分别为2和8，各自平均 loss 为1和3。直接平均两次 loss 与按全 token 平均分别多少？

**参考解答**：直接平均为2；全 token 平均为 $(2\cdot1+8\cdot3)/10=2.6$。若想与完整大 batch 的 token-mean 目标一致，梯度累积必须按有效 token 加权。把 batch 调小后累积有助控制单步激活，但不会自动让 CLIP 不同 micro-batch 共享全部负例。

**排障口述**：SFT loss 下降、视频顺序题不涨，按数据/标签→采帧证据→时间表示→视觉依赖→评估协议提出检查；每一步说出一个你能实际观察的量。

<span id="mm-55191841294e" style="display:block;scroll-margin-top:6rem"></span>

## 第 13 天：读论文与讲项目

阅读[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/)，选一篇固定版本报告和一篇训练方法论文。

**任务**：给每篇写五行：它解决的问题、相对基线改了什么、关键图的一条完整路径、结果成立的条件、你会做的一个消融。不能把论文摘要中的成绩直接说成自己的结果。

**参考完成样式**：对于“视频时间信息怎样进入模型”，固定版本名称和报告链接；画出帧采样→时间戳/位置坐标→视觉编码/合并→LLM 的路径；提出“相同帧集合、相同token预算，打乱时间或移除时间信息”的对照；按先后/持续时间/OCR题型分别看影响。这个设计测试的是时间表示贡献，不能同时更换模型和数据后仍把涨点归因于时间编码。

**项目追问**：数据怎样拆分，如何证明没有同源视频泄漏？参考答案要落到源视频ID、近重复片段/字幕、train/val/test互斥检查，而不是只说“随机划分80/10/10”。

<span id="mm-37a5643355c4" style="display:block;scroll-margin-top:6rem"></span>

## 第 14 天：闭卷模拟面试与定向返工

让自己按顺序回答：QKV 怎样实现；RoPE 的相对位置如何出现；Prefill/Decode 为什么不同；CLIP、BLIP-2、LLaVA 各解决什么；视频抽帧和 token 压缩怎样评估；SFT loss mask 如何核验；GRPO 没涨如何诊断；介绍一个你真实完成的项目。

每题按四项检查：定义是否准确、机制是否能展开、是否能给具体算例/证据、是否知道适用边界。可各记0或1，共4分；这是自拟复习量表，不是面试通过率预测。分数只是定位返工：缺机制回正文，缺算例重做练习，缺证据回原始论文/项目日志，缺边界找反例。

最后重新做第2、4、7、10、11天各一道题。间隔后仍能独立写出来，才说明记住的是关系；看到熟悉词就点头，尚不足以回答继续追问。

本练习册的知识依据集中于[来源、图像许可与知识点覆盖](/notes/multimodal-sources-coverage/)；题目按学习目标组织，不代表任何招聘方的题库或“高频程度”统计。
