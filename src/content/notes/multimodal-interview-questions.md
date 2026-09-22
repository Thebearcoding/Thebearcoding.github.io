---
title: 多模态算法复习问答与项目推演
date: '2026-09-22'
tags:
- 多模态算法
- 复习问答
- 项目推演
- 论文答辩
- 自测
summary: 按14个知识主题组织复习问答，用机制、算例、边界和项目推演检查理解。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-475fb8e237be" style="display:block;scroll-margin-top:6rem"></span>

这份题单按知识点组织复习问答：先说明机制，再用算例、反例和项目追问检查理解。答案是本教程根据论文与机制组织的教学讲解，不代表某公司的题库或面试频率统计。

先学[从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)、[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)，再用本页闭卷复述。每题都可以从四个层次检验：

- **机制：** 输入、输出与关键运算是什么。
- **计算：** 能否手算张量形状、概率或成本。
- **边界：** 哪些假设不成立时，结论会失效。
- **验证：** 能否提出可观察的对照实验或检查。

<span id="mm-117301abe651" style="display:block;scroll-margin-top:6rem"></span>

## 1. 知识点题单与学习入口

| 编号 | 问题主题 | 检查重点 | 先读 |
|---|---|---|---|
| Q1 | DeepSeek、Qwen、LLaMA的区别 | 固定版本，按相同维度比较 | [模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/) |
| Q2 | Q/K/V如何实现 | 投影、拆头、mask与缓存形状 | [从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/) |
| Q3 | 位置编码 | 置换性质、相对位置与真实时间 | [从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)、[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/) |
| Q4 | Prefill/Decode瓶颈 | 计算、访存与端到端时延拆分 | [Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/) |
| Q5 | 项目做什么、数据怎么来、问题如何解决 | 假设、控制变量、证据与限制 | 本页第3节 |
| Q6 | ViT与文本Transformer，如何做分类 | patch表示、交互与输出头 | [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |
| Q7 | 对比学习、InfoNCE/CE、归一化与温度 | 匹配目标、尺度与监督假设 | [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |
| Q8 | Q-Former与MLP取舍，为什么两层MLP | 非线性、token预算与信息瓶颈 | [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |
| Q9 | Qwen2.5-VL视觉塔与图文对齐；MRoPE | 空间压缩、时间对齐与版本差异 | [模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/) |
| Q10 | 冻结视觉塔、训练projector/LLM的原因 | 适配能力、训练成本与梯度路径 | [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |
| Q11 | 千万级库难负例挖掘、视频数据清洗 | 假负例、数据质量与分割泄漏 | 本页、[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |
| Q12 | 蒸馏温度高低的影响 | 软分布、梯度尺度与教师误差 | 本页 |
| Q13 | 手写图像MLP，分类为何用CE而非MSE | 输入组织、目标函数与梯度 | 本页、[SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/) |
| Q14 | 视频中物体出现时间轴与描述 | 检索、时序定位与事实性评价 | 本页、[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/) |

<span id="mm-d09dd09a341e" style="display:block;scroll-margin-top:6rem"></span>

## 2. 从口述答案推到追问

<span id="mm-97b38d3d7f00" style="display:block;scroll-margin-top:6rem"></span>

### Q1：模型家族有什么区别

**完整回答。** 比较应落到具体版本和同一维度。比如Llama3.1是文本自回归模型，使用GQA；DeepSeek-V2是采用MLA与MoE的文本模型；Qwen2.5-VL还包含视觉塔、空间merger与多模态位置表示。前两者可用于解释文本骨干取舍，但不能直接当成相同视觉任务的竞品比较。进一步比较参数总量、激活参数、KV缓存、视觉输入、训练阶段和目标任务成本；型号表与机制见[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/)。

**为什么会继续追问。** 只说“DeepSeek是MoE、Qwen是dense”说明你背的是标签。MoE通常改变FFN，GQA/MLA改变Attention缓存，视觉塔解决视觉输入，三者不在同一层面。追问“MLA为什么仍要缓存RoPE部分”“2×2 merger减少什么成本”是在检查你是否理解模块的因果关系。

**练习。** 把Qwen3-VL的8B dense与30B-A3B MoE分别填表。不能由A3B推断只需加载3B权重，也不能由8B/30B名称直接推断视觉表现；实际推理取决于权重驻留、路由、上下文与实现。

<span id="mm-9c04faa285d4" style="display:block;scroll-margin-top:6rem"></span>

### Q2：Q/K/V如何实现

**完整回答。** 输入 $X:[B,T,d]$ 经不同可训练投影得到Q、K、V，拆头后计算 $QK^\top/\sqrt{d_h}$，加合法可见性mask，沿key维softmax，再乘V。多头输出拼接后经输出投影回到 $d$。Q/K决定读取权重，V提供被汇总的内容；它们都由同一层输入经过训练得到，而不是三个静态词典。

**追问的依据。** 要能说明每步形状，尤其Q长度与K长度在增量解码时不同。GQA中32个Q头、8个KV头时，每4个Q头共用一组KV。历史K/V能复用，是因为因果前向中未来token不会改变旧位置的层内表示；位置偏移错误会破坏这一条件。手算与实现检查路线见[从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)、[Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/)。

**练习。** 两token单位投影时，第二行softmax约为 $(0.330,0.670)$。解释为什么输出是V的加权和，而非选一个最大值。再解释softmax后直接清零未来项为什么导致概率归一化错误。

<span id="mm-c780c7445e98" style="display:block;scroll-margin-top:6rem"></span>

### Q3：位置编码解决了什么

**完整回答。** 无位置、且不额外引入顺序结构的self-attention具有置换等变性：输入重排，输出对应重排，无法凭内容确定相对间隔。因果mask本身带有方向和可见集结构，所以不能据此说所有无显式位置的因果模型完全不知道顺序。显式位置编码进一步提供位置、距离和多尺度变化。

RoPE把Q/K的二维子空间按位置旋转，两位置的点积出现相对角差。旧cache中的K已按原位置旋转，新token需要真实后续位置。视频还要描述高、宽和时间，真实时间间隔也不等于帧序号。MRoPE与Qwen两代时间编码见[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/)。

**练习。** 第1和第3位置相差 $2\omega$；把两者都移动5个位置，相对差不变。再想帧时间0、0.1、8秒如何与0、4、8秒区分。只说“视频也加RoPE”没有回答这个问题。

<span id="mm-f9d4d508656c" style="display:block;scroll-margin-top:6rem"></span>

### Q4：Prefill与Decode瓶颈

**完整回答。** Prefill并行处理整个输入并建立KV缓存；Decode在缓存上每次生成新token，输出步数通常串行。长输入Prefill有较大矩阵运算，Decode常反复读取权重与KV，低batch时容易受带宽和时延制约。实际瓶颈还随batch、上下文、模型与内核变化，不能说“永远算力/永远带宽”。

视频端还需单独测解码采帧、视觉塔、LLM Prefill、首token延迟TTFT、后续token时延TPOT与KV峰值。否则把整个TTFT归因于LLM会误诊。完整成本例子见[Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/)。

<span id="mm-6e60d715a532" style="display:block;scroll-margin-top:6rem"></span>

### Q6—Q8：ViT、CLIP、连接器连起来怎么讲

**完整回答。** ViT先把图像变成patch序列，通过空间位置和双向交互学习视觉特征；文本Transformer的输入来自tokenizer，mask和训练目标由任务决定。分类头可接CLS或池化特征，生成式VLM则保留局部token供语言模型读。CLIP对全局图文表示做匹配训练；InfoNCE把每个候选集合当分类问题，归一化控制向量尺度，温度控制分布与梯度。

连接器再把视觉特征映射到LLM隐藏维。逐位置两层MLP增加非线性，但默认不改变token数；固定query的Q-Former能压缩长度，同时引入信息瓶颈和训练要求。要比较二者，应固定视觉塔、LLM、数据、视觉token预算和评测，而不是用不同论文总分判断模块优劣。完整手算、SigLIP区别和BLIP-2三种目标见[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)。

**追问。** 两层MLP如果去掉中间激活，为什么仍是线性映射？同图有两个正确caption时，单正例标签会怎样？BLIP-2的ITC为什么不让query看正确文字？这三问分别检查表达能力、监督假设和信息泄漏。

<span id="mm-f7266f074437" style="display:block;scroll-margin-top:6rem"></span>

### Q9—Q10：版本机制与冻结策略

**完整回答。** 以Qwen2.5-VL为例，动态分辨率改变空间token数，window attention降低视觉塔局部交互成本，spatial merger合并空间特征；与真实时间对齐的MRoPE让不同采样FPS下仍表达时间间隔。Qwen3-VL还调整视觉塔、频率分配、层间视觉注入和时间戳形式，不能把两个版本混讲。详见[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/)。

冻结视觉塔可减少可训练参数和过拟合风险；适配任务时训练projector/LLM，使语言生成学会消费视觉证据。若域差异大，冻结也可能限制性能。比较冻结、部分解冻、全参时，记录同数据质量、回归能力和成本；冻结LLM时仍可让梯度穿过它传到projector，冻结不等于no-grad整条路径。

<span id="mm-42c891234dcf" style="display:block;scroll-margin-top:6rem"></span>

### Q11：难负例与视频脏数据

**完整回答。** 难负例是语义相近却确实错误的候选，不是分数越高越值得作为负例。先用当前双塔向量批量召回候选，过滤同源、近重复及可能正确的配对，再用教师交互模型或人工复核筛选。大库的索引召回率、更新频率、候选难度分布和假负率都要测；训练时保留一定随机负例，避免只适配狭窄混淆。

视频脏样本要沿输入链处理：解码失败、空帧、音画错位、时间戳偏移、重复片段、caption脑补、答案不被画面支持、字幕泄漏。每类错误都应记录过滤比例和抽检精度，别只回答“清洗了坏数据”。数据分割按源视频/近重复簇，防止同一事件跨训练测试集。

<span id="mm-daabc0e249e4" style="display:block;scroll-margin-top:6rem"></span>

### Q12：蒸馏温度改变什么

**完整回答。** Teacher和student的logits经 $p_i^{(T)}=\operatorname{softmax}(z_i/T)$ 形成软分布。$T$升高会更平滑，让非最大类别的相对关系更明显；太高可能把有用差异抹平，太低则近似硬标签。常见目标是 $T^2D_{\rm KL}(p_{\rm teacher}^{(T)}\|p_{\rm student}^{(T)})$ 与硬标签CE加权；$T^2$用于补偿温度造成的梯度尺度变化，不是所有蒸馏配方都必须照搬。[知识蒸馏原论文](https://arxiv.org/abs/1503.02531)

算例：logits为 $(2,0)$，$T=1$时概率约 $(0.881,0.119)$；$T=2$时约 $(0.731,0.269)$。后者保留更多第二类信息，但教师也可能是错的。不同词表不能直接逐token对齐KL；只能获取教师生成文本时属于序列/数据蒸馏，监督信息与logits蒸馏不同。

**追问。** Teacher在OCR细节上错误、student却被强力拟合，怎样发现？需要独立人工核验与真实标签回归，不以蒸馏loss低作为教师正确性的证明。

<span id="mm-379c884d3205" style="display:block;scroll-margin-top:6rem"></span>

### Q13：分类为什么常用CE，MSE可不可以

**完整回答。** CE来自类别分布的负对数似然，真实类别 $y$ 的loss为 $-\log p_y$；对logits导数为 $p-\operatorname{onehot}(y)$。模型非常自信地答错时，正确类别的logit仍得到较强纠正。概率MSE也能用于分类，但梯度还经过softmax Jacobian，饱和时可能较弱，优化性质不同，因此数学可行不等于同样好训练。

图像MLP要说明输入组织：整图展平会丢掉显式局部结构且参数随分辨率增加；按patch共享投影再聚合保留更多结构；卷积或ViT是不同的归纳偏置。手撕时给出shape和最小输入，不应只写一个Linear层就声称完整图像分类器。CE推导见[SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/)。

<span id="mm-f8ecf43b4630" style="display:block;scroll-margin-top:6rem"></span>

### Q14：怎样输出物体出现的时间轴

**完整回答。** 先把任务定义为“给定目标对象或开放词表描述，输出若干出现区间、证据和描述”。以低成本采帧与视觉检索生成候选时段，再在候选区间密采样或做时序定位，核对对象身份并合并相邻区间，最后基于证据生成描述。目标小、遮挡多或出现短暂时，采样与对象跟踪会直接影响召回。

指标要分开：是否找到目标、时间区间IoU、多个区间的精确率/召回率、描述事实性、总延迟。同一对象反复出现不能强行并成覆盖中间空白的一大段；片段内部秒数还要换回原视频时轴。上述是供学习讨论的方案设计，效果需要实验验证。

<span id="mm-747db0c7a143" style="display:block;scroll-margin-top:6rem"></span>

## 3. Q5：项目与论文深挖，结论必须有证据

一个可追问的项目可以用“问题→假设→干预→证据→限制”讲清。以下是**假设性视频项目设计**，不代表已经取得结果：

“均匀8帧可能漏掉长视频中的短事件。可检验的假设是，同样视觉token预算下，分段候选加局部细采样能改善证据覆盖。对照实验固定视觉塔、LLM、训练集、分辨率预算和解码设置，比较覆盖全部必要事件的比例、时序问答准确率、OCR回归和TTFT。再单独做时间戳消融，检验提升来自覆盖还是时间表达。”

这里每一句都能继续追问：关键事件如何标？采样策略额外花了多少计算？局部补帧是否降低其他帧分辨率？同源视频是否泄漏？重复种子和置信区间如何？如果只能提供总分，机制主张仍不充分。

| 面试官追问 | 实际要证明什么 | 应准备的材料 |
|---|---|---|
| 为什么做这个问题 | 存在具体失败模式 | 原基线错例与任务定义 |
| 为什么不选另一方法 | 比较条件公平 | 同数据/预算替代基线 |
| 数据如何做 | 标签支持目标能力 | schema、证据区间、质检抽样 |
| 损失为什么这样设计 | 优化目标与最终能力匹配 | 梯度方向、反例、目标消融 |
| 提升是否显著 | 不是偶然或挑样本 | 配对结果、重复实验、区间 |
| 为什么OCR下降 | 改动有可解释代价 | 题型/分辨率分桶 |
| 你的贡献是什么 | 能具体承担技术决策 | 自己负责的模块、日志与失败修正 |
| 若不涨怎么办 | 能排查而非堆改动 | 输入→表示→监督→评价的诊断链 |
| 如果重做 | 下一步可检验 | 最高优先级假设及验收条件 |

论文讲解也用同样逻辑。先说明原方法不能解决的具体现象，再写核心运算和假设，再看控制变量消融是否支持作者解释。不要把主结果表里的所有涨点当某一个模块的因果证据。阅读路线和逐篇追问见[模型差异为什么必须落到具体版本](/notes/multimodal-models-paper-reading/)。

<span id="mm-e7472077fac1" style="display:block;scroll-margin-top:6rem"></span>

## 4. 后训练的两道延伸自测

**PPO优势估计。** 从期望值写到采样估计时，引入了哪些误差？回答时应区分真实价值函数、估计器与实现，说明采样噪声、价值估计误差与偏差—方差取舍。推导集中在[从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)。

**SFT后GRPO没有提升。** 可以怎样检查初始化策略、成功轨迹可达性和组内奖励差异？在视频后训练中，先检查模型是否收到视觉证据，再检查多次采样有没有可区分的好坏答案与可靠奖励。一次训练未提升不足以推出“SFT一定损害RL”或“必须换prompt”；应结合固定评测、采样结果和奖励分布提出可检验的解释。

<span id="mm-6760bce3245e" style="display:block;scroll-margin-top:6rem"></span>

## 闭卷验收

任选一题，给出完整机制，再画数据/梯度流，算一个数字，指出一个失败前提，并提出一个可测对照。若讲到“对齐、动态分辨率、时序融合、显存优化”后只能重复术语，就回到对应原理章。

项目题则必须区分已经测量的结果与拟开展实验。完整、有边界的回答，比编造项目增益或背一串所谓标准结论更能承受追问。
