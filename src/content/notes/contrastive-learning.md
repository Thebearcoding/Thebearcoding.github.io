---
title: 对比学习
date: '2026-08-11'
tags:
- machine-learning
- self-supervised-learning
- representation-learning
summary: 拉近正样本、推远负样本:自监督表示学习的核心范式,以及 CLIP 的工程实践数字。
draft: false
obsidian: true
---

## 定义

对比学习(Contrastive Learning)是一种自监督表示学习方法，通过比较样本之间的相似性来学习有意义的特征表示。其核心思想是：将相似样本（正样本对）的表示拉近，将不相似样本（负样本对）的表示推远。

用 InfoNCE 写出来就是一行:

$$
\mathcal{L} = -\log \frac{\exp\big(\mathrm{sim}(q, k^{+})/\tau\big)}{\sum_{i=0}^{K} \exp\big(\mathrm{sim}(q, k_i)/\tau\big)}
$$

分子是查询 $q$ 与正样本 $k^{+}$ 的相似度,分母是它与全部候选(1 个正 + $K$ 个负)的相似度之和——本质是一道"$K{+}1$ 选 1"的分类题,温度 $\tau$ 控制这道题的"严格程度"。

## 要点

- **正样本对**：语义相似的样本对，如同一图像的不同增强版本、图像与其描述文本
- **负样本对**：语义不同的样本对，通常从同一批次中随机采样
- **InfoNCE 损失**：最常用的对比学习目标函数，基于噪声对比估计
- **温度参数**：控制相似度分布的平滑程度，通常设为 0.07 左右
- **大批次训练**：需要大量负样本以提供有效的对比信号
- **CLIP 的实践**：batch size = 32768（通过梯度累积），可学习温度参数初始化为 log(1/0.07)
- **效率优势**：对比学习比预测式方法（如图像标题生成）效率高 4 倍

## 示例

典型的对比学习流程：

```python
# InfoNCE 损失计算
def info_nce_loss(query, positive_key, negative_keys, temperature=0.07):
    # 计算正样本相似度
    pos_sim = torch.sum(query * positive_key, dim=-1) / temperature

    # 计算所有负样本相似度
    neg_sim = torch.mm(query, negative_keys.T) / temperature

    # 对比损失
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(len(query), dtype=torch.long)
    loss = F.cross_entropy(logits, labels)
    return loss
```

## 相关概念

- 多模态表示学习 - 对比学习在多模态场景的应用
- CLIP模型 - 使用对比学习的视觉-语言模型
- InfoNCE损失 - 对比学习的核心损失函数
- 大规模预训练 - CLIP 的大 batch 对比学习训练策略
- 双塔架构 - 对比学习常用的架构设计
- 困难负样本挖掘 - 对比学习在 grounding 场景中常借助 hard negatives 强化判别力

## 参考资料

- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)](https://arxiv.org/abs/1911.05722)
