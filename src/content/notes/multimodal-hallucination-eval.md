---
title: 多模态幻觉与评测：POPE、CHAIR、MME、MMBench 与解码侧缓解
date: '2026-10-06'
tags: [多模态, 幻觉, 评测, POPE, CHAIR, MMBench, VCD, OPERA, RLHF-V]
summary: 从幻觉类型和成因讲起，手算 CHAIR、POPE、MME 的 acc+ 和 MMBench 的 CircularEval，再讲 VCD、OPERA 两种解码侧缓解与 RLHF-V 的数据侧做法，以及每个指标会在哪里失真。
draft: false
---

多模态大模型最常被追问的缺陷是**幻觉**：回答流畅，却说出图里没有的物体、属性或关系。这一篇回答三个问题：幻觉怎样量化（CHAIR、POPE、AMBER），通用能力怎样评测、为什么各基准的判分协议设计成那样（MME、MMBench、MMMU、SEED-Bench），以及有哪些缓解方法（解码侧的 VCD、OPERA，数据侧的 RLHF-V）。DPO 的推导见 [SFT 与 DPO](/notes/multimodal-sft-lora-dpo/)，解码与 KV 缓存见 [Prefill 与 Decode](/notes/prefill-decode-video-tokens/)，本篇不重复。

阅读入口：[多模态专题总览](/notes/multimodal-interview-guide/)。数值算例为教学构造；论文数字均注明表号，判分细节注明官方代码。

## 1. 幻觉的类型与成因

按出错的对象分，常见三类（AMBER 就是按这三类出题的）：

| 类型 | 例子 | 典型度量 |
|---|---|---|
| 存在 | 图里只有叉子，模型说“桌上有刀叉” | CHAIR、POPE、AMBER 存在类 |
| 属性 | 红色的车说成蓝色；两只猫说成三只 | AMBER 属性类、MME 的 color/count |
| 关系 | 人站在车旁，说成“人坐在车里” | AMBER 关系类 |

还有一种划分是按任务形式：**生成式**（让模型自由描述，事后抽取物体核对）和**判别式**（直接问“图里有没有 X”，答 yes/no）。前者贴近真实用法，后者便宜、稳定。两者的结论不一定一致。

成因只写论文里有证据的部分：

- **语言先验。** CHAIR 的原论文（Rohrbach et al., EMNLP 2018）分析图像描述模型后指出，幻觉往往来自语言模型式的预测，而不是视觉理解错误；在标准句子指标上分数最高的模型，幻觉不一定最少（摘要）。
- **训练数据统计。** POPE 论文发现，大视觉语言模型（LVLM）容易幻觉出在视觉指令数据里**出现频率高**的物体，以及与图中真实物体**经常共现**的物体（论文分析部分）。这正是 POPE 设计 popular 和 adversarial 两种负采样的原因。
- **注意力模式。** OPERA 观察到，许多幻觉与自注意力里的“知识聚合”模式有关：生成后段时，模型主要看前面少数几个“总结 token”，而不是图像 token（第 5 节展开）。
- **回答倾向。** 在 POPE 的 Table 3 里，LLaVA、mPLUG-Owl、MultiModal-GPT 对存在性问题几乎全答 yes（yes 比例 95%–100%）。这是指令数据里正例太多造成的偏向，与看没看懂图无关。

## 2. CHAIR：在自由描述里数幻觉物体

**输入输出。** 输入：一组图像各自的生成描述，加上每张图的真实物体集合（MSCOCO 的 80 类）；输出：两个比例。**定义**（Rohrbach et al. 2018，§2.1）：

$$
\mathrm{CHAIR}_i=\frac{\lvert\{\text{hallucinated objects}\}\rvert}{\lvert\{\text{all objects mentioned}\}\rvert},\qquad
\mathrm{CHAIR}_s=\frac{\lvert\{\text{sentences with hallucinated object}\}\rvert}{\lvert\{\text{all sentences}\}\rvert}.
$$

$\mathrm{CHAIR}_i$ 是实例级（物体粒度），$\mathrm{CHAIR}_s$ 是句子级（只要含一个幻觉物体整句就算错）。两者都是越低越好。在 LVLM 的长描述上，“句子”通常指一整条回答。

**物体怎样抽取。** 原文的流程：分词并把名词变单数；用 MSCOCO 同义词表把词映射到 80 类，例如 “player” 映射为 person；处理双词复合词，保证 “hot dog” 不会被误判为 dog。真实物体同时取自 COCO 分割标注和人写的参考描述，因为只用其一会人为抬高幻觉率。原文主要在 Karpathy test 划分上报告。

**手算**（教学构造，三张图三条描述）：

| 图 | 真实物体 | 描述 | 抽出的物体 | 幻觉 |
|---|---|---|---|---|
| 1 | person, dog, frisbee | A man throws a frisbee to a dog near a bench. | person（man）, frisbee, dog, bench | bench |
| 2 | cat, couch, remote | A cat sleeps on a couch next to a remote. | cat, couch, remote | 无 |
| 3 | hot dog, cup, dining table | A hot dog and a cup on a dining table, with a knife and a fork. | hot dog, cup, dining table, knife, fork | knife, fork |

提及物体共 $4+3+5=12$ 个，幻觉 $1+0+2=3$ 个，$\mathrm{CHAIR}_i=3/12=25\%$。三条描述里有两条含幻觉，$\mathrm{CHAIR}_s=2/3\approx66.7\%$。注意第 3 条：如果抽取器不处理复合词，会多抽出一个 dog，$\mathrm{CHAIR}_i$ 变成 $4/13$。同一物体被重复提及时是否去重，取决于具体实现（本文未逐行核对）；本例没有重复。

**边界与失败场景。**

1. **描述越长，CHAIR 越高。** OPERA 用同一套 500 张 COCO val2014 图、同一提示 “Please describe this image in detail.” 测 LLaVA-1.5 的 beam search：最大新 token 为 512 时 $\mathrm{CHAIR}_s=48.8$、$\mathrm{CHAIR}_i=13.9$（Table 1）；限制到 64 时只有 18.8 和 5.9（Table 2）。比较两个模型时必须控制生成长度。
2. **惩罚“说得多”，不奖励“说得全”。** 只说 “A dog.” 的模型 CHAIR 为 0。所以要配一个覆盖率指标，例如 AMBER 的 Cover（第 4 节）。
3. **对提示敏感。** POPE 论文 §3.2 用两条意思相近的指令测同一批模型，CHAIR 结果差别很大，有的模型排名还会变。
4. **词表封闭。** 只认 COCO 80 类，80 类以外的幻觉（例如“窗外有一座教堂”）完全漏掉。

**怎样验证。** 抽 50 条描述人工标幻觉物体，和自动抽取结果对比，算抽取的精确率与召回率；比较模型时固定提示、解码方式和最大长度，并同时报告平均描述长度。

## 3. POPE：把幻觉问成 yes/no

POPE（Li et al., EMNLP 2023, arXiv 2305.10355）把生成式评测改成判别式“轮询”，绕开 CHAIR 对抽取和提示的依赖。

**构造**（论文 POPE 方法部分与实验设置）。从 MSCOCO 验证集随机选 500 张真实物体多于 3 个的图，每张图问 6 个问题，模板是 “Is there a/an <object> in the image?”。其中 3 个问真实物体（答案 yes），3 个问不存在的物体（答案 no），比例 1:1，每种设置共 3000 题。不存在的物体有三种采法：

| 设置 | 负样本怎么来 | 例子（图里有 dining table, fork, cup） |
|---|---|---|
| Random | 从图中不存在的类别里随机采 | giraffe |
| Popular | 整个数据集里出现最频繁、且本图不存在的前 $k$ 类（$k=l/2$，$l=6$） | 例如高频类 person、car |
| Adversarial | 按与本图真实物体的共现频率排序，取本图不存在的前 $k$ 类 | knife（常与 fork 同现） |

例子一列为教学构造。对没有标注的图（A-OKVQA、GQA），论文用分割模型 SEEM 自动得到物体。

**指标。** 以 yes 为正类，报告 Accuracy、Precision、Recall、F1 和 yes 比例。官方 `evaluate.py` 的判分值得记住：只取回答的第一句，去掉逗号后按空格切词，含 `No`、`not` 或 `no` 就判为 no，**其余一律判为 yes**。所以答非所问、拒答都会算成 yes，这会推高 yes 比例和 FP。

**手算**（教学构造，100 题，50 正 50 负）。

*模型 A，random 设置*：正例里答 yes 45 个，负例里答 yes 15 个，即 TP=45、FN=5、FP=15、TN=35。Accuracy $=(45+35)/100=0.80$；Precision $=45/60=0.75$；Recall $=45/50=0.90$；F1 $=2\times0.75\times0.90/(0.75+0.90)=1.35/1.65\approx0.818$；yes 比例 $=60/100=0.60$。

*模型 A，adversarial 设置*：负例换成共现物体，FP 涨到 30，TN=20，正例不变。Accuracy $=0.65$，Precision $=45/75=0.60$，Recall 仍为 0.90，F1 $=1.08/1.50=0.72$，yes 比例 0.75。Recall 不变、Precision 掉了，说明下降全部来自“把共现物体当成存在”。

*模型 B，永远答 yes*：TP=50、FP=50。Accuracy 0.5，Precision 0.5，Recall 1.0，F1 $=2\times0.5\times1/1.5\approx0.667$，yes 比例 1.0。这与论文 Table 3 里 MultiModal-GPT 的一行（Precision 50.00、Recall 100.00、F1 66.67、Yes 100.00）完全吻合。**所以只看 F1 会被骗**：一个毫无判别力的模型也有 0.667。正例占一半时，yes 比例应接近 0.5，明显偏离就说明模型有回答倾向。

**论文里的结果**（Table 3，MSCOCO）。InstructBLIP 的 F1 在 random、popular、adversarial 下分别是 89.29、83.45、78.45，yes 比例从 55.20 升到 68.97；LLaVA（初版）三种设置的 F1 是 68.65、67.72、66.98，yes 比例 95%–99%。三种设置依次变难，与第 1 节的频率、共现成因一致。

**边界。** ① 只测存在性，不测属性和关系。② 只测“问了才答”的判别能力，模型在 POPE 上分数高，自由描述时照样可能幻觉。③ 问题模板固定，可能被专门针对它的数据刷分。④ 负类全部来自 COCO 80 类。

**怎样验证。** 三种设置都报，并附 yes 比例；把 FP 按负样本类别统计，看错误是否集中在高频或共现类别；换问法（例如 “Does the image contain …?”）复测，看结论是否稳定。

## 4. AMBER：生成与判别放进同一个基准

AMBER（arXiv 2311.07397）不依赖 LLM 判分，同时覆盖生成式任务和判别式任务，判别题分存在、属性、关系三类。规模：1004 张图，14220 条提示，其中生成式 1004 条、判别式 13216 条（存在 4924、属性 7628、关系 1664），标注涉及 337 种物体（§3.1、附录 A.1.2）。

生成式指标（§3.2.2，式 (2)–(5)），$R'_{obj}$ 是回答中抽出的物体，$A_{obj}$ 是标注的真实物体，$H_{obj}$ 是标注的“容易被想象出来”的幻觉目标物体：

$$
\mathrm{CHAIR}(R)=1-\frac{\mathrm{len}(R'_{obj}\cap A_{obj})}{\mathrm{len}(R'_{obj})},\quad
\mathrm{Cover}(R)=\frac{\mathrm{len}(R'_{obj}\cap A_{obj})}{\mathrm{len}(A_{obj})},\quad
\mathrm{Cog}(R)=\frac{\mathrm{len}(R'_{obj}\cap H_{obj})}{\mathrm{len}(R'_{obj})}.
$$

Hal 是“该回答是否含任何幻觉”的 0/1 指标，相当于逐条的 $\mathrm{CHAIR}_s$。综合分 $\mathrm{AMBER\ Score}=\tfrac12(1-\mathrm{CHAIR}+\mathrm{F1})$（式 (6)）。用第 2 节的图 1 算：CHAIR $=1-3/4=0.25$，Cover $=3/3=1$。

**和 POPE 的一个差别。** AMBER 的 Precision、Recall 是在**答案为 no 的幻觉题**上算的（§3.2.2），也就是以 no 为正类；POPE 以 yes 为正类。两个基准的 F1 不能直接横比。

## 5. 通用评测：判分协议比题目更值得记

| 基准 | 规模与形式 | 判分要点 |
|---|---|---|
| MME（2306.13394） | 感知 10 个子任务、认知 4 个；每张图两道 yes/no 题 | 子任务分 = acc + acc+，满分 200；感知满分 2000，认知 800 |
| MMBench（2307.06281，ECCV 2024） | 3217 道选择题，能力分 3 层：2 个 L-1、6 个 L-2、20 个 L-3；有中文版 MMBench-CN | CircularEval；先规则匹配选项，失败再用 GPT 对齐 |
| MMMU（2311.16502，CVPR 2024） | 11.5K 道大学水平题，6 个学科大类、30 个学科、183 个子领域、30 种图像类型；94% 为选择题 | 零样本；规则抽取答案，选择题解析失败时随机选一个，开放题判错；micro 平均准确率 |
| SEED-Bench（2307.16125） | 19K 道选择题，12 个评测维度，图像与视频 | 选择题有人工标注的标准答案，判分不需要人或 GPT |

**MME：acc 与 acc+。** 每张图配两道题，一道答案是 yes，一道是 no，提示末尾加 “Please answer yes or no.”。acc 按题算，随机猜为 50%；acc+ 要求同一张图两道题都对，随机猜为 25%。官方 `calculation.py` 的解析：回答恰好是 yes/no 就直接用；否则只看**小写后的前 4 个字符**，含 yes 判 yes，含 no 判 no，都不含记为 other，other 一律算错。注意这与 POPE“其余判 yes”的默认方向相反。

*手算*（教学构造，某子任务 4 张图 8 道题）：

| 图 | yes 题 | no 题 | 本图两题全对 |
|---|---|---|---|
| 1 | 对 | 对 | 是 |
| 2 | 对 | 错（答了 yes） | 否 |
| 3 | 对 | 对 | 是 |
| 4 | 错（回答 “There is a …”，记 other） | 对 | 否 |

acc $=6/8=75\%$，acc+ $=2/4=50\%$，子任务分 $=75+50=125$（满分 200）。一个永远答 yes 的模型：acc $=50\%$，但每张图的 no 题都错，acc+ $=0$，得 50 分，低于随机猜的期望 $50+25=75$ 分。acc+ 的作用就是惩罚这种回答倾向。

**MMBench：CircularEval。** $N$ 个选项的题喂给模型 $N$ 次，每次把选项连同答案循环移位；只有 $N$ 次全部答对才算这道题对（§4.3）。模型输出先用启发式规则匹配选项字母，匹配不上再把问题、选项和模型输出交给 ChatGPT/GPT-4，让它对齐到某个选项，仍对不上就返回伪选项 “Z”（§4.1）。

*手算*（教学构造；移位方向以官方实现为准，这里取左移）。题目“图中汽车是什么颜色”，正确答案是 red：

| 轮次 | A | B | C | D | 正确字母 | 模型回答 | 对否 |
|---|---|---|---|---|---|---|---|
| 1 | blue | red | green | yellow | B | B | 对 |
| 2 | red | green | yellow | blue | A | A | 对 |
| 3 | green | yellow | blue | red | D | D | 对 |
| 4 | yellow | blue | red | green | C | B | 错 |

VanillaEval 只看第 1 轮，算对；CircularEval 下第 4 轮错，整题判错。一个永远答 “A” 的模型，VanillaEval 下正确答案恰好在 A 的题全对，CircularEval 下一题都过不了。均匀乱猜的模型，单轮正确率 1/4，四轮全对的概率为 $(1/4)^4=1/256\approx0.39\%$。论文 Table 2（dev 集）报告了改用 CircularEval 后的掉分：GPT-4v 为 74.3（降 10.8），Qwen-VL-Chat 为 59.5（降 17.4），MiniGPT4-7B 为 32.7（降 24.1）。越弱的模型掉得越多，说明 VanillaEval 下有相当一部分“对”来自位置偏好和运气。

**MMMU 的定位。** 它考学科知识加推理，不是专门测幻觉。摘要里 GPT-4V 和 Gemini Ultra 分别只有 56% 和 59%。题目取自学术材料，公开题目存在被训练数据收录的风险，需要警惕污染。

**边界。** ① 用 GPT 抽取答案会引入判分器本身的误差和版本漂移，要固定版本并报告规则匹配的成功率。② 基准分数涨了不等于幻觉少了：MME 的感知分里只有 existence、count、position、color 直接和幻觉相关。③ 视频评测（Video-MME 等）见 [视频理解](/notes/multimodal-video-understanding/)，Agent 基准的判分坑见 [Agent 评测](/notes/agent-eval/)。

## 6. 解码侧缓解之一：VCD

VCD（Leng et al., CVPR 2024, arXiv 2311.16922）不训练，只改解码：同一个问题，分别用原图 $v$ 和加噪图 $v'$ 算一次 next-token logits，两者做对比。加噪图的视觉信息被破坏，模型在它上面给出的高分 token，更多反映语言先验和统计偏置；减掉它，就能压低“不看图也会说”的 token。

**输入输出。** 每一步需要两次前向（原图、加噪图各一次，可以拼成 batch=2），输出都是词表大小 $\lvert V\rvert$ 的 logits；解码开销约为原来的两倍。

**加噪**（式 (2)）：用扩散前向过程给图像加 $T$ 步高斯噪声，$q(v_t\mid v_{t-1})=\mathcal N(v_t;\sqrt{1-\gamma}\,v_{t-1},\gamma I)$。

**对比分布**（式 (3)）：

$$
p_{\mathrm{vcd}}(y\mid v,v',x)=\mathrm{softmax}\big[(1+\alpha)\,\mathrm{logit}_\theta(y\mid v,x)-\alpha\,\mathrm{logit}_\theta(y\mid v',x)\big].
$$

**自适应可信度约束**（式 (4)）：只保留在原图分布下足够可信的 token，

$$
V_{\mathrm{head}}(y_{<t})=\{y_t\in V:\ p_\theta(y_t\mid v,x,y_{<t})\ge\beta\max_w p_\theta(w\mid v,x,y_{<t})\},
$$

不在 $V_{\mathrm{head}}$ 里的 token 概率置 0。论文附录 A：$\gamma=0.1$、$\alpha=1$、$\beta=0.1$；$T$ 在 MME 和 LLaVA-Bench 上取 500，在 POPE 上取 999。摘要报告 POPE 上 F1 最多提升 7.4，MME 上最多提升 18%。

**手算**（教学构造，$\alpha=1$、$\beta=0.1$）。上文是 “On the table there is a”，图里有 fork 和 cup，没有 knife：

| token | 原图 logit | 加噪图 logit | 原图概率 | VCD logit $=2l-l'$ | 无约束 VCD 概率 | 有约束 VCD 概率 |
|---|---:|---:|---:|---:|---:|---:|
| fork | 2.8 | 1.0 | 0.366 | 4.6 | 0.569 | 0.643 |
| knife | 3.0 | 2.9 | 0.447 | 3.1 | 0.127 | 0.143 |
| cup | 2.0 | 0.5 | 0.164 | 3.5 | 0.189 | 0.214 |
| giraffe | 0.0 | −3.0 | 0.022 | 3.0 | 0.115 | 0（被截掉） |

原图下贪心会选 knife（0.447），它在加噪图下几乎不掉分（3.0 到 2.9），说明它主要靠“叉子旁边常有刀”的先验。对比后 fork 变成第一。再看 giraffe：原图概率只有 0.022，但它在加噪图下掉得更多，对比后 logit 反而升到 3.0，无约束时拿到 0.115 的概率。约束阈值 $\beta\cdot\max p=0.1\times0.447\approx0.045$，giraffe 低于阈值被截掉。这就是式 (4) 存在的原因：对比只该在“原图下本来就合理”的候选之间重新排序，不能把原本不可能的 token 抬起来。

**边界。** ① $\alpha$ 太大会压掉本来就该依赖语言知识的 token（虚词、常识）。② 噪声步数决定 $v'$ 里还剩多少图像信息，不同基准的最优 $T$ 不同（论文本身就用了 500 和 999 两档）。③ 两次前向，延迟和显存都要翻倍算。④ 论文实验在修改后的分布上直接采样；和 beam search 组合时效果需要自己测。

**怎样验证。** 固定其他解码参数，扫 $\alpha$、$\beta$、$T$，同时看 POPE（判别）和 CHAIR + 覆盖率（生成）；再测一个通用基准，确认没有伤到一般能力。

## 7. 解码侧缓解之二：OPERA

OPERA（Huang et al., CVPR 2024, arXiv 2311.17911）也不训练，作用在 **beam search** 上。

**观察。** 自注意力图里会出现“柱状”模式：某个 token 之后生成的很多 token 都把注意力集中在它身上。作者称之为总结 token，并发现幻觉常常在这种模式出现之后开始，此时模型在过度信任总结 token，而忽略了图像 token。注意力图怎么读见 [Transformer 笔记](/notes/transformer-attention-rope-gqa/)。

**过度信任惩罚**（式 (3)–(6)）。取已生成 token 之间、大小为 $k\times k$ 的局部注意力窗口（不含图像和提示 token），乘以缩放系数 $\sigma$，上三角置零；对每一列，把该列从对角线往下的值连乘，取连乘最大的那一列，记为惩罚 $\phi(\omega_{<t})$。连乘越大，说明后面的 token 越一致地盯着同一个位置。候选 token 的得分改为

$$
p(x_t\mid x_{<t})=\mathrm{Softmax}\big[\mathcal H(h_t)-\alpha\,\phi(\omega_{\le t})\big]_{x_t},
$$

只对每个 beam 的 top-$N_{can}$ 个候选计算。

**回溯重分配**（式 (7)(8)）。记录最近若干个 token 的“最大列”位置，如果同一位置出现的次数达到阈值 $r$，就认为总结 token 已经主导了生成：回滚到该位置，在候选里排除原先选的 token，重新选择，最多回滚 $\beta$ 次。

**超参与结果。** 默认 $N_{beam}=5$、$N_{can}=5$、$\sigma=50$、$\alpha=1$、$r=15$、$\beta=5$（实现细节）。LLaVA-1.5 上：最大 512 个新 token 时，$\mathrm{CHAIR}_s$/$\mathrm{CHAIR}_i$ 从 beam search 的 48.8/13.9 降到 44.6/12.8（Table 1）；最大 64 个 token 时从 18.8/5.9 降到 14.2/5.2（Table 2）。

**边界。** 只适用于 beam search；需要拿到注意力权重，而常用的融合注意力内核只返回输出、不返回权重（本文的工程判断），要另外计算；回滚会增加解码步数，延迟不稳定。

## 8. 数据侧缓解：RLHF-V

前两种方法只改推理，不改模型。RLHF-V（Yu et al., CVPR 2024, arXiv 2312.00849）改的是偏好数据和损失。

**数据。** 不让标注员在两条回答之间二选一，而是让他们**直接改写**模型回答里的幻觉片段：原回答为 $y_l$，改好的为 $y_w$。共 1.4K 条。好处是偏好信号精确到片段，标注员不用在“都有错”的两条回答里勉强挑一条。

**DDPO**（§3.1，式 (5)）。把 DPO 里的序列对数概率换成加权版本，$y_u$ 是未改动片段的 token，$y_c$ 是被改动片段的 token：

$$
\log\pi(y\mid x)=\frac1N\Big[\sum_{y_i\in y_u}\log p(y_i\mid x,y_{<i})+\gamma\sum_{y_i\in y_c}\log p(y_i\mid x,y_{<i})\Big],\qquad N=\lvert y_u\rvert+\gamma\lvert y_c\rvert.
$$

$\gamma>1$ 让被改的片段占更大权重，论文取 $\gamma=5$；除以 $N$ 是为了不让长回答占便宜。例（教学构造）：一条回答 20 个未改 token，对数概率之和 −10，5 个改动 token 之和 −5；普通平均是 $-15/25=-0.6$，DDPO 是 $(-10+5\times(-5))/(20+25)=-35/45\approx-0.78$，改动片段在这一项里的权重从 $5/25=20\%$ 升到 $25/45\approx56\%$。

**结果。** 摘要：基座模型的幻觉率降低 34.8%，优于用 10K 条标注数据训练的同期工作 LLaVA-RLHF。基座是 Muffin（BEiT-3 视觉编码器 + 13B Vicuna）。

**边界。** 人工改写成本高、难以扩展；改写只覆盖标注员看出来的幻觉。后续工作用 AI 反馈或自动构造偏好对来替代人工，本篇不展开；多模态 RL 的可验证奖励见 [多模态 RL](/notes/multimodal-rl/)。

## 9. 面试常问

**POPE 的 F1 很高，能说明模型幻觉少吗？** 不能。一个永远答 yes 的模型 F1 也有 0.667。要同时看 yes 比例（理想接近 0.5）和三种设置下的掉分；另外 POPE 只测存在性和“被问到时”的判别，生成描述时仍要用 CHAIR 一类指标。

**CHAIR_i 和 CHAIR_s 有什么区别，各有什么坑？** 前者是幻觉物体占提及物体的比例，后者是含幻觉物体的描述占比。描述越长两者越高，只说一句也能拿 0，而且只认 COCO 80 类、对提示敏感。比较时要固定长度并配覆盖率。

**MME 为什么要 acc+？MMBench 为什么要 CircularEval？** 两者都在压制“蒙对”。acc+ 要求同一张图的 yes 题和 no 题都对，永远答 yes 的模型 acc+ 为 0。CircularEval 让答案轮换位置，只有每次都对才算对，位置偏好和运气都会被剔除，乱猜的通过率从 1/4 降到 1/256。

**VCD 的公式和可信度约束为什么必要？** $\mathrm{softmax}[(1+\alpha)l(v)-\alpha l(v')]$，减掉加噪图下的 logits 以压低语言先验。没有约束时，原图下不可能、但加噪后更不可能的 token 会被对比放大，所以只在原图概率不低于 $\beta$ 倍最大概率的候选里重新排序。

**解码侧和数据侧方法怎么选？** 解码侧不用训练、即插即用，但每次推理都付出代价（VCD 两次前向，OPERA 依赖 beam search 和注意力权重）。数据侧要标注和训练，但改的是模型本身，推理没有额外开销。按本文的看法，线上服务优先做数据侧，解码侧适合用来快速验证“问题是否出在语言先验”。

## 闭卷验收

不看资料写出 $\mathrm{CHAIR}_i$、$\mathrm{CHAIR}_s$ 的定义，对三条描述手算两者，并说明 “hot dog” 为什么要特殊处理；说出 POPE 的构造（500 张图、每图 6 题、正负 1:1）和三种负采样各采什么，对一组 TP/FP/TN/FN 算出 accuracy、precision、recall、F1 和 yes 比例，并解释永远答 yes 为什么 F1 是 0.667；说出官方 POPE 与 MME 判分脚本对无法解析的回答分别怎么处理；手算一个子任务的 acc、acc+ 和 MME 分数；画出一道四选一题在 CircularEval 下的四轮选项，判定对错；写出 AMBER 的 Cover 和 AMBER Score，指出它的 precision 以哪一类为正；写出 VCD 的对比式和可信度约束，手算一个 logit 调整例子；讲清 OPERA 的总结 token、柱状注意力、惩罚项和回滚条件；写出 RLHF-V 的 DDPO 加权对数概率，并说出 $N$ 的作用。

**参考。** [CHAIR: Object Hallucination in Image Captioning](https://arxiv.org/abs/1809.02156)；[POPE](https://arxiv.org/abs/2305.10355)（[官方仓库](https://github.com/RUCAIBox/POPE)，`evaluate.py`）；[AMBER](https://arxiv.org/abs/2311.07397)；[MME](https://arxiv.org/abs/2306.13394)（[官方评测工具](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)，`tools/eval_tool.zip` 中的 `calculation.py`）；[MMBench](https://arxiv.org/abs/2307.06281)；[MMMU](https://arxiv.org/abs/2311.16502)；[SEED-Bench](https://arxiv.org/abs/2307.16125)；[VCD](https://arxiv.org/abs/2311.16922)（[官方仓库](https://github.com/DAMO-NLP-SG/VCD)）；[OPERA](https://arxiv.org/abs/2311.17911)；[RLHF-V](https://arxiv.org/abs/2312.00849)。
