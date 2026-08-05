---
title: 把 VLM 拆到原子:做 MiniMind-V 课程的三个收获
date: 2026-08-05
tags: [多模态, VLM, 教学相长]
summary: 为一个 65M 的视觉语言模型做了 12 课互动课程,过程中真正学到的三件事。
---

给 [MiniMind-V 做完 12 课互动课程](https://athebear.me/minimind-v-course/course/)之后,回头看有三个收获值得单独记下来。

## 一、"多模态"在实现上小得惊人

课程做到第 7 课时最震撼:VLM 相对 LLM 的核心改动**不到 50 行**——找到序列里的 64 个 `<|image_pad|>` 占位符,把投影后的视觉特征按位置贴进去,结束。此后 LLM 根本不区分 token 的出身。

所谓"让语言模型看见",实现上只是给句子扩充了词汇。

## 二、训练策略就是三个 freeze 开关

$$
\underbrace{\text{SigLIP2}}_{\texttt{冻结,95M}} \;\to\; \underbrace{\text{MLP 投影}}_{\texttt{从零训练,1.2M}} \;\to\; \underbrace{\text{MiniMind LLM}}_{\texttt{首末层微调,64M}}
$$

预训练阶段(freeze_llm=2)只训 1.2M 参数的投影层,学习率敢开到 4e-4;SFT 阶段(freeze_llm=1)解冻首末层,学习率降 80 倍到 5e-6——**动"白纸"可以大步走,动"有知识的参数"必须小步精修**。这个直觉比任何调参手册都好记。

## 三、给别人讲,是最好的 debug

做第 8 课的反向传播动画时,我原本想画"梯度流到冻结的 LLM 边界就熄灭"——查了真实源码才发现不对:梯度必须**借道**穿过冻结层的激活链才能到达上游的投影层,`requires_grad=False` 只是"参数不记账",真正断路的是视觉编码器的 `no_grad`。

不做课程,这个理解误区我可能带很多年。

---

下一步:在[手敲复现仓库](https://github.com/Thebearcoding/minimind-v-course)里,把这 12 课的内容逐课写成代码。
