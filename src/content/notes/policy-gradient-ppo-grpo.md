---
title: 从RL基础推到PPO与GRPO
date: '2026-09-22'
tags:
- 强化学习
- 策略梯度
- PPO
- GRPO
- 优势函数
- 多模态后训练
summary: 贯通回报、价值、策略梯度、GAE、PPO和组相对优化，含完整数值例。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-ebfae00f1f0a" style="display:block;scroll-margin-top:6rem"></span>

如果只知道“SFT 拟合标准答案，RL 最大化奖励”，遇到“为什么要减 baseline”“PPO 裁剪为什么取 min”“GRPO loss 为零为什么还有梯度”仍很容易卡住。本篇沿着一个问题推下去：**奖励只告诉我们一次结果好不好，怎样把它变成可训练的 token 概率更新？**

先读 [02-SFT与DPO的训练信号](/notes/multimodal-sft-lora-dpo/) 的 log-prob、梯度和 reference。正文分三段：第 1–4 节解决“回报怎样变成梯度”；第 5–7 节解决“怎样估优势、重复利用采样”；第 8–10 节解决“语言模型如何组采样并约束更新”。不涉及 Agent 框架开发，图中的 agent 只是强化学习里“采取动作的策略”这个通用称呼。

ZealD 在[学习记录](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5/6aa79968000000000b001eeb)中提到推导困难。这是教程的切入点；他的个人解释和项目结果不是数学证明，下面的算例也不是其真实实验数据。

<span id="mm-6f825518a1d6" style="display:block;scroll-margin-top:6rem"></span>

## 1. 把语言生成写成一次有终点的决策过程

<span id="mm-3b143f295139" style="display:block;scroll-margin-top:6rem"></span>

### 1.1 状态、动作、奖励各是什么

在时间 $t$，策略 $\pi_\theta(a_t\mid s_t)$ 根据状态 $s_t$ 给动作概率，采出 $a_t$ 后得到奖励 $r_t$ 和下一状态 $s_{t+1}$。本文约定 $r_t$ 是**动作 $a_t$ 之后**收到的奖励；有些教材把同一个量写为 $R_{t+1}$，只是下标不同，不能混写进同一递推式。

语言生成里，状态是“视频/图像、问题和当前已生成前缀”，动作是下一个 token，下一状态通常是把它追加到前缀。最后完成回答时可由规则或奖励模型给一个分数。若是正确性二值奖励，中间 token 的任务奖励通常为 0，终点才给 0 或 1。这种反馈稀疏，但一个后续答案的成败仍与此前各 token 的选择有关。

一次 rollout 就是从开始到终止或采样边界的一条生成轨迹。EOS 可能是真正终止；达到最大生成长度是什么含义，需要按目标规定，不能把“采集停止”与“任务自然完成”自动等同。

<span id="mm-0e74523cee99" style="display:block;scroll-margin-top:6rem"></span>

### 1.2 奖励、回报、价值不能互换

令一条轨迹有 $T$ 次动作，折扣因子 $0\le\gamma\le1$：

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kr_{t+k},
\qquad
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],
\qquad
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

$r_t$ 是一步反馈；$G_t$ 是这一次实际轨迹的未来奖励和；$V^\pi$ 是从某状态出发、继续按策略行动时回报的期望；$Q^\pi$ 多固定了第一步动作。若是有限时域任务，时间/剩余步数应包含在状态中，或把价值写成 $V_t^\pi$，否则同样画面在不同剩余时间下可能并无同一个价值。

优势衡量“固定这个动作，比当前状态下平均选动作好多少”：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

它不是任务的绝对分数。某答案得到 0.6 分，在组内均值 0.2 时相对较好，在均值 0.9 时相对较差。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/08-%E5%A5%96%E5%8A%B1%E5%9B%9E%E6%8A%A5%E4%BB%B7%E5%80%BC.png" target="_blank" rel="noopener" aria-label="查看原图：奖励回报价值"><img src="/notes-assets/policy-gradient-ppo-grpo/08-%E5%A5%96%E5%8A%B1%E5%9B%9E%E6%8A%A5%E4%BB%B7%E5%80%BC.png" alt="奖励回报价值" width="1513" height="1013" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">奖励回报价值（点击查看原图）</figcaption>
</figure>

**读图先分颜色。**橙色点是路径上的即时奖励，绿色是每条路径求和后的回报，红色是从同一起点继续走的平均表现。图按不折扣的例子把四条回报 $1.6,2.6,2.1,2.0$ 平均为 $2.075$。若它们是四次独立采样，这个数是价值的 Monte Carlo 估计；只有这些可能结果恰好等概率且穷尽分布时，才等于真实期望。价值定义不是“一定采四条再平均”。

<span id="mm-b07de21c6ac7" style="display:block;scroll-margin-top:6rem"></span>

## 2. Bellman、Monte Carlo 与 TD：从哪获得价值

<span id="mm-765ed43323d8" style="display:block;scroll-margin-top:6rem"></span>

### 2.1 Bellman 关系是在拆分期望

由 $G_t=r_t+\gamma G_{t+1}$，先固定 $(s,a)$、再对下一状态和未来动作取期望：

$$
Q^\pi(s,a)=
\mathbb E\left[r_t+\gamma V^\pi(s_{t+1})\mid s_t=s,a_t=a\right].
$$

当前状态还没选动作，因此：

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

第一式平均环境转移和反馈的不确定性，第二式平均策略选动作的不确定性。真正终止后的价值为零。不能把一次观察的 $r+\gamma V(s')$ 直接当成一般随机环境里真实的 $Q(s,a)$。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/09-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0Q%E5%92%8CV.png" target="_blank" rel="noopener" aria-label="查看原图：价值函数Q和V"><img src="/notes-assets/policy-gradient-ppo-grpo/09-%E4%BB%B7%E5%80%BC%E5%87%BD%E6%95%B0Q%E5%92%8CV.png" alt="价值函数Q和V" width="1703" height="1555" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">价值函数Q和V（点击查看原图）</figcaption>
</figure>

**图下方两棵树是两种不同的平均。**左树从状态出发，先按 $\pi(a\mid s)$ 选动作，所以由 Q 加权得到 V；右树先固定动作，再按转移概率走到不同状态，所以由“奖励加下一状态 V”得到 Q。上方最大的 $Q$ 表示在给定后续策略下该动作的期望回报更高，不等于这条动作每次采样都最好。

例：某动作即时奖励为 0，之后以各 50% 概率进入价值 0 或 2 的状态，$\gamma=0.9$。真实 Q 为 $0.9(0.5\times0+0.5\times2)=0.9$；单次目标只会是 0 或 1.8。随机目标可以高于或低于期望。

<span id="mm-3575b135e6bf" style="display:block;scroll-margin-top:6rem"></span>

### 2.2 Model-based 与 model-free 在这里怎么区分

若已知奖励与转移分布，可以直接算 Bellman 期望，反复评估策略，这属于利用环境模型的动态规划思路。模型未知时，可以采样来估价值，Monte Carlo 和 TD 是两种基本方法。

语言 token 追加的状态变化虽然简单，外部任务反馈、用户交互或视觉事实评分未必都已知。不能仅因“下一状态等于前缀加 token”，就说整个 LLM 强化学习问题天然有完整环境模型。

<span id="mm-4c19b8624b0b" style="display:block;scroll-margin-top:6rem"></span>

### 2.3 同一条轨迹算两种目标

设三步奖励为 $(0,1,2)$，$\gamma=0.9$，第三步后真正终止。Monte Carlo 从后往前计算：

$$
G_2=2,\qquad
G_1=1+0.9(2)=2.8,\qquad
G_0=0+0.9(2.8)=2.52.
$$

假设价值网络目前给出 $V(s_0)=0.5,V(s_1)=1,V(s_2)=1.5$。MC 对 $V(s_0)$ 的目标是完整回报 2.52；TD(0) 只看一步反馈和下一状态预测，目标是 $0+0.9V(s_1)=0.9$。

MC 不用下一状态预测作 bootstrap，但要等完整未来反馈，方差通常较大。TD 用自己当前估值构造新目标，能更早更新，但会受到估值误差影响。以学习率 0.1 作一次表格型价值更新，MC 把 0.5 改成 $0.5+0.1(2.52-0.5)=0.702$；TD 改成 $0.5+0.1(0.9-0.5)=0.54$。这里不是在比较谁一步就更“正确”，而是在比较目标的信息来源。[仓库 MC/TD 图](https://github.com/changyeyu/LLM-RL-Visualized#header-48)。

<span id="mm-ef79f9834791" style="display:block;scroll-margin-top:6rem"></span>

## 3. 策略梯度：离散 token 为什么也能训练

<span id="mm-be3d6771d271" style="display:block;scroll-margin-top:6rem"></span>

### 3.1 求导的是概率，不是对采样出的文字求导

先令有限轨迹且 $\gamma=1$，整局回报 $R(\tau)=\sum_t r_t$。优化目标：

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[R(\tau)].
$$

虽然采到哪个离散 token 不可直接连续求导，但“采到这条轨迹的概率”可对模型参数求导。假设初始状态、环境转移和奖励规则没有直接依赖 $\theta$，轨迹概率为：

$$
p_\theta(\tau)=p(s_0)\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid s_t)P(s_{t+1},r_t\mid s_t,a_t).
$$

把期望写成求和，求导，再乘除概率：

$$
\begin{aligned}
\nabla_\theta J
&=\sum_\tau R(\tau)\nabla_\theta p_\theta(\tau)\\
&=\sum_\tau p_\theta(\tau)R(\tau)\nabla_\theta\log p_\theta(\tau)\\
&=\mathbb E_\tau\left[R(\tau)\nabla_\theta\log p_\theta(\tau)\right].
\end{aligned}
$$

这里要求概率及导数满足可交换求导与求和/积分的常规条件。对轨迹乘积取 log，只有策略项含可训练参数：

$$
\nabla_\theta J=
\mathbb E_\tau\left[
R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$

因此可以对采样动作的 log-prob 反向传播，再乘一次反馈形成权重。若奖励本身也对 $\theta$ 求导，那是另一个目标，不能仍省略它的导数；常规策略优化将 rollout 反馈视为固定观测。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/10-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6.png" target="_blank" rel="noopener" aria-label="查看原图：策略梯度"><img src="/notes-assets/policy-gradient-ppo-grpo/10-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6.png" alt="策略梯度" width="1593" height="1234" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">策略梯度（点击查看原图）</figcaption>
</figure>

**图上方从期望变为样本均值。**第一行是分布上的理论梯度，第二行用 N 条轨迹平均估计。左下记录轨迹，右下策略与环境交互，再用回报加权的 log-prob 梯度更新。红色更新式用加号，因为它在最大化 J；若优化器执行梯度下降，代码里的 loss 需要取负号。图用整局回报，下一节会解释为什么可换成 reward-to-go。

<span id="mm-3802e105653d" style="display:block;scroll-margin-top:6rem"></span>

### 3.2 为什么过去奖励可以丢掉

给定动作前的历史，过去奖励已经确定；当前动作概率的 score function 平均为零：

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
[\nabla_\theta\log\pi_\theta(a\mid s)]
=\sum_a\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}
=\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

于是过去奖励乘当前 score 的期望为零，去掉它们不改变期望，只保留从当前动作开始的未来回报。对本节 $\gamma=1$：

$$
\nabla_\theta J=\mathbb E_\tau\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t.
$$

恢复折扣时必须同步修改目标。若目标明确为从初始状态计算的
$J_\gamma=\mathbb E\sum_t\gamma^tr_t$，且 $G_t$ 从当前时刻重新计折扣，则：

$$
\nabla_\theta J_\gamma
=\mathbb E_\tau\sum_t\gamma^t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t.
$$

外部 $\gamma^t$ 不能无声消失。采用折扣状态访问分布等约定时可把它吸收进取期望方式；常见 LLM 完整回答优化也常取 $\gamma=1$。本篇后面的语言生成 PPO/GRPO 以这一无折扣回答目标讨论，GAE 数例另用 $\gamma=0.9$ 展示递推。

<span id="mm-65ffbae06f30" style="display:block;scroll-margin-top:6rem"></span>

### 3.3 两动作手算检验梯度

单状态只有好/坏动作，好动作概率 $p=\sigma(\theta)=0.4$，奖励分别 1 和 0。期望回报 $J=p$，直接求导得到 $p(1-p)=0.24$。

好动作的 $\nabla_\theta\log p=1-p=0.6$，坏动作的 $\nabla_\theta\log(1-p)=-p=-0.4$。用奖励乘它们，单次样本梯度分别为 0.6 和 0；采样期望为 $0.4(0.6)+0.6(0)=0.24$。这证明在本例中，不需要对离散采样操作本身求导，也可无偏地估计期望回报梯度。

<span id="mm-4fa6b8f46246" style="display:block;scroll-margin-top:6rem"></span>

## 4. Baseline 为什么不改变方向，却可能减小噪声

设 $b(s)$ 只依赖动作前状态。由上一节 score 均值为零：

$$
\mathbb E_a\left[b(s)\nabla\log\pi_\theta(a\mid s)\right]=0.
$$

所以可把 $G_t$ 换成 $G_t-b(s_t)$。常选状态价值 $V^\pi(s_t)$，得到相对于状态平均表现的优势估计。价值预测越合理，很多“这个状态本来就容易/困难”的波动就被减掉；基线选择不佳则不保证每种情况下都降方差。

继续两动作例，取 $b=0.4$。好、坏动作的加权梯度分别：

$$
(1-0.4)(0.6)=0.36,\qquad
(0-0.4)(-0.4)=0.16.
$$

期望仍为 $0.4(0.36)+0.6(0.16)=0.24$。这里失败动作也有了有效梯度：它比平均表现差，应降低其概率。

实现 actor loss 时，baseline/advantage 应作为固定权重，即 stop-gradient 或 detach。即使价值网络与 actor 共享某些参数，也不能顺着 actor loss 中的优势再反传一条额外路径；价值网络用独立的拟合目标学习。这个操作与“价值永远不训练”完全不同。

<span id="mm-5d7fab07f3d1" style="display:block;scroll-margin-top:6rem"></span>

## 5. GAE：怎样把多个 TD 残差组合成优势

<span id="mm-43900f216233" style="display:block;scroll-margin-top:6rem"></span>

### 5.1 从 TD 误差到多步累计

用当前价值预测构造单步残差：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t).
$$

它表示“收到的奖励加下一状态估值”比原先预测好多少。若价值预测准确，$\delta_t$ 对未来随机性取条件期望后提供当前动作优势；真实网络有误差，因此用多个未来残差折中：

$$
\hat A_t^{\rm GAE}
=\sum_{k=0}^{T-t-1}(\gamma\lambda)^k\delta_{t+k}
=\delta_t+\gamma\lambda\hat A_{t+1}.
$$

$\lambda\in[0,1]$ 控制未来残差的权重。$\lambda=0$ 只用一步 TD；$\lambda=1$、末端价值处理正确时，残差中的中间价值相消，得到 $G_t-V(s_t)$。较小 $\lambda$ 更依赖 bootstrap 的准确性，较大 $\lambda$ 更多使用实际未来反馈；这就是常说的偏差/方差权衡，不是对任意任务的性能排序。[GAE 原论文](https://arxiv.org/abs/1506.02438)。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/11-GAE.png" target="_blank" rel="noopener" aria-label="查看原图：GAE"><img src="/notes-assets/policy-gradient-ppo-grpo/11-GAE.png" alt="GAE" width="1503" height="811" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">GAE（点击查看原图）</figcaption>
</figure>

**图左边先算每个 delta，右边再从末端倒推 A。**最末一步没有下一状态价值，隐含它是真终止。彩色下标帮助对应同一时间步。图上两条等式适用于同一连续轨迹；它没有显式展示多 episode、padding 或采样截断，所以不能当成带所有边界条件的代码。

<span id="mm-47655a02658c" style="display:block;scroll-margin-top:6rem"></span>

### 5.2 把前面的三步例子算到底

沿用 $r=(0,1,2)$、$\gamma=0.9$、$V=(0.5,1,1.5)$，末端真终止：

$$
\delta_0=0+0.9(1)-0.5=0.4,
\quad\delta_1=1+0.9(1.5)-1=1.35,
\quad\delta_2=2-1.5=0.5.
$$

取 $\lambda=0.8$，$\gamma\lambda=0.72$：

$$
\hat A_2=0.5,\quad
\hat A_1=1.35+0.72(0.5)=1.71,\quad
\hat A_0=0.4+0.72(1.71)=1.6312.
$$

如果 $\lambda=1$：

$$
\hat A_0=0.4+0.9(1.35)+0.9^2(0.5)=2.02
=G_0-V(s_0)=2.52-0.5.
$$

这最后一步是检查递推与 MC 是否一致的办法；它也说明 reward、return、advantage 的数值不能混用。

<span id="mm-b27174c89651" style="display:block;scroll-margin-top:6rem"></span>

### 5.3 两个 mask：能否 bootstrap，能否接下一步优势

为了不把新 episode 接到旧 episode 上，定义两个含义不同的开关：

$$
\delta_t=r_t+\gamma b_tV(s_{t+1}^{final})-V(s_t),
\qquad
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

$b_t$ 表示任务在这个边界后是否仍有应估计的未来回报；$c_t$ 表示当前 buffer 的下一条记录是否是同一条连续轨迹、可以继续累计优势。不能用同一个笼统 done 替代所有情况。

| 边界 | $b_t$ | $c_t$ | 原因 |
|---|---:|---:|---|
| 正常连续一步 | 1 | 1 | 既 bootstrap，也接后续残差 |
| 真正任务终止 | 0 | 0 | 未来价值为零，不接下一局 |
| 外部时间限制后 reset，但任务价值仍定义为继续 | 1 | 0 | 用 reset 前 final state 估值，不接新局 |
| 本次 rollout buffer 结束、环境未终止 | 1 | 0 | bootstrap 端点；本片段没有下一项优势 |

例如边界奖励 1、当前 V 为 2、final state V 为 3、$\gamma=0.9$。真终止 $\delta=1-2=-1$；外部截断应 bootstrap 时 $\delta=1+0.9(3)-2=1.7$。若误把 reset 后新初始状态的 V 填进来，目标也错。

有的实现先把 bootstrap 值并入末步奖励，再用统一的结束 mask；这是等价组织方式之一，不能在奖励和 $\delta$ 中重复加两次。[Gymnasium 的 termination/truncation 说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。LLM 达到最大生成长度可被定义成失败终止、截断样本或其他处理，需明确任务目标，不能仅靠 API 名字判断。

<span id="mm-0a6744186274" style="display:block;scroll-margin-top:6rem"></span>

## 6. PPO：为什么要记住采样时的概率

<span id="mm-79fbddb30611" style="display:block;scroll-margin-top:6rem"></span>

### 6.1 Old policy 不是 reference policy

一个 rollout 是由某个确定版本的策略 $\pi_{\rm old}$ 采出来的。训练过程中当前策略 $\pi_\theta$ 在改变；如果把同一批样本重复训练数次，当前动作概率与采样时就不同了。于是记录旧概率，计算：

$$
\rho_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\rm old}(a_t\mid s_t)}
=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\rm old}(a_t\mid s_t)\right).
$$

$\rho=1$ 表示当前概率未变；1.3 表示变为旧值的 1.3 倍，不是增加 1.3 个百分点。旧 log-prob 必须固定，不能随每个梯度 step 一起更新，否则分母失去“采样快照”的意义。

若生成时用了 temperature 或 top-p，实际采样分布可能与未处理 logits 的 softmax 不同。训练和生成引擎还可能有数值差异。需要明确 old log-prob 按哪种分布记录、当前概率是否采用一致约定；不能把任何来自旧模型的数字都称为精确行为概率。标准公式先在采样/概率约定一致的前提下理解。

对固定状态下的动作分布，概率比可用于重要性加权。但单步 $\rho_t$ 没有同时校正整条状态访问分布；PPO 的表达是局部替代目标，不是任意大步更新下真实期望回报的精确恒等式。

<span id="mm-123159d2980d" style="display:block;scroll-margin-top:6rem"></span>

### 6.2 min 与 clip 一起才形成 PPO-Clip

优势 $\hat A_t$ 已在 rollout 后计算并固定。PPO-Clip 最大化：

$$
J_{\rm clip}=
\mathbb E_t\left[
\min\left(\rho_t\hat A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

$\epsilon>0$ 给出相对概率比的裁剪范围。实际最小化的 actor loss 是 $\mathcal L_{\rm actor}=-J_{\rm clip}$。这里 clip 并不是“让概率永远不超范围”；它让某些方向继续变化不再改善这条样本的 surrogate。

取 $\epsilon=0.2$：

| 优势 | $\rho$ | 原项 $\rho A$ | 裁剪项 | 取 min | 此方向含义 |
|---:|---:|---:|---:|---:|---|
| 2 | 1.3 | 2.6 | 2.4 | 2.4 | 好动作已提高过多，局部项饱和 |
| $-2$ | 0.7 | $-1.4$ | $-1.6$ | $-1.6$ | 坏动作已压低过多，局部项饱和 |
| 2 | 0.7 | 1.4 | 1.6 | 1.4 | 好动作被压低，仍需纠正 |
| $-2$ | 1.3 | $-2.6$ | $-2.4$ | $-2.6$ | 坏动作被提高，仍需纠正 |

当 A 为负，乘法改变大小关系，所以不能把 min 直接移到“两个 ratio”上而不分情况。也不能说“只要 ratio 出界，该样本梯度都为零”。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/12-PPO-Clip.png" target="_blank" rel="noopener" aria-label="查看原图：PPO-Clip"><img src="/notes-assets/policy-gradient-ppo-grpo/12-PPO-Clip.png" alt="PPO-Clip" width="1557" height="1567" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">PPO-Clip（点击查看原图）</figcaption>
</figure>

**先看上方红蓝两项，再看下方两图。**蓝线是未裁剪 ratio，红线是裁剪后的系数；紫线是根据 A 正负选出来的有效系数。左下 A 为正，只在提高过头的一侧变平；右下 A 为负，则在降低过头的一侧变平。真正目标还要乘 A，因此紫线并不是两种情况下同方向的收益曲线。

即使该样本局部项饱和，其他样本的梯度仍会改变共享网络，并继续影响这个动作概率。因此 PPO 没有给每个更新施加严格 trust region；需要监测新旧策略差异和训练行为。原论文还有 KL-penalty 版本，这里明确讨论 PPO-Clip。[PPO 原论文](https://arxiv.org/abs/1707.06347)。

<span id="mm-3a958565ea44" style="display:block;scroll-margin-top:6rem"></span>

### 6.3 Critic 怎样训练，哪些量不能跟着求导

actor 需要优势，critic 需要拟合未来回报。常见 GAE 管线用：

$$
\hat R_t=\operatorname{stopgrad}\big(\hat A_t+V_{\rm old}(s_t)\big),
\qquad
\mathcal L_V=\frac12\mathbb E_t\big(V_\psi(s_t)-\hat R_t\big)^2.
$$

本例的 $\hat R_0=1.6312+0.5=2.1312$，是 $\lambda$-return 目标，不是 MC 的 2.52。$\psi$ 是 critic 参数，$V_{\rm old}$ 是收集这批轨迹时固定的估值。若不断用正在变化的目标反向穿透 actor/critic，就不再是上述更新。

教学伪代码展示梯度路径，省略所有张量维度和 mask：

    old_logp、优势、value_target = 固定的 rollout 统计
    new_logp、new_value = 当前模型前向
    ratio = exp(new_logp - stop_gradient(old_logp))
    actor_loss = -mean(min(ratio * stop_gradient(优势),
                           clip(ratio) * stop_gradient(优势)))
    critic_loss = 0.5 * mean((new_value - stop_gradient(value_target)) ** 2)

有些实现另加 entropy 奖励：$H(\pi)=-\sum_a\pi(a|s)\log\pi(a|s)$，鼓励分布暂时保留探索空间；也有 value clipping。它们是可选设计，应与核心 clip 目标分开报告，不能把所有框架配置当成 PPO 定义。

<span id="mm-4034e0a25d1d" style="display:block;scroll-margin-top:6rem"></span>

## 7. 语言模型 RLHF 中，四种模型各干什么

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/14-PPO%E5%9B%9B%E6%A8%A1%E5%9E%8B.png" target="_blank" rel="noopener" aria-label="查看原图：PPO四模型"><img src="/notes-assets/policy-gradient-ppo-grpo/14-PPO%E5%9B%9B%E6%A8%A1%E5%9E%8B.png" alt="PPO四模型" width="1472" height="958" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">PPO四模型（点击查看原图）</figcaption>
</figure>

**图中锁的意义是更新职责。**actor 生成回答、按策略目标更新；critic 估每个前缀的未来回报、按价值损失更新；reward model 给任务/偏好分数，reference 给参考概率。图里的“回答”不是本篇记号 $\hat A_t$ 的优势；同一个字母在不同配图里可能含义不同，要按框和箭头判读。

| 角色 | 产生的量 | 常见更新方式 |
|---|---|---|
| 当前策略/actor $\pi_\theta$ | 下一 token 概率 | 每个策略训练 step 更新 |
| 旧策略 $\pi_{\rm old}$ | rollout 时的动作概率 | 每批新采样前同步；可只保存 log-prob |
| reference $\pi_{\rm ref}$ | 限制偏离的基准概率 | 一段训练中通常冻结；分阶段方案可重置 |
| reward model/规则 | 回答质量分数 | 常在策略内循环固定 |
| critic $V_\psi$ | 前缀未来回报预测 | 拟合价值目标 |

“四模型”通常指 actor、reference、reward、critic，旧策略的角色常由 actor 快照或缓存 log-prob 实现，不一定额外常驻一个同尺寸模型。reward 可能是可验证规则，actor/critic 也可能共享骨干，因此不能把角色数直接乘成显存数。

RLHF-PPO 常在任务 reward 中加入相对 reference 的逐 token 惩罚，例如在采样策略冻结时，用 $\log\pi_{\rm old}(a_t|s_t)-\log\pi_{\rm ref}(a_t|s_t)$ 构造固定 KL reward，再算 GAE。最后一个 token 另加任务结果分数。注意这个 reference 与 ratio 分母里的 old 并不相同：前者控制长期偏离，后者记录这批数据从何而来。

一个 rollout 内循环可理解为：当前 actor 同步为 old → 生成回答并保存旧概率/价值 → 打分与构造优势/价值目标 → 当前 actor 与 critic 在这些固定目标上更新若干次 → 再采新数据。这样才能解释“为什么不应无限重复用同一批 rollout”。

<span id="mm-8dec335589cb" style="display:block;scroll-margin-top:6rem"></span>

## 8. GRPO：用同题多答案形成相对学习信号

<span id="mm-52300c272e0a" style="display:block;scroll-margin-top:6rem"></span>

### 8.1 一组回答怎样替代显式价值网络

对同一条件 $q$ 采样 $G$ 条回答 $o_1,\ldots,o_G$。这里 q 包含视频和实际采样帧，不只是问题字符串。每条输出得到任务分数 $R_i$，结果监督版本可定义：

$$
\bar R=\frac1G\sum_iR_i,\qquad
s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\qquad
\hat A_i=\frac{R_i-\bar R}{s_R+\varepsilon_{\rm num}}.
$$

这是用总体标准差作教学约定；一些代码用样本标准差、换归一化方式或不除 std，数值会不同，必须检查实现。$\varepsilon_{\rm num}$ 是防除零的小常数，与 PPO 裁剪宽度 $\epsilon$ 不同。

例如 $G=4$，奖励 $[1,0,0,1]$，均值 0.5、标准差 0.5，忽略极小稳定项后优势为 $[1,-1,-1,1]$。若全错或全对，所有分子均为 0；加小常数能避免除零，无法创造优劣方向。结果监督把同一条回答的 $\hat A_i$ 赋给它的所有生成 token，学习的是这条回答相对同题其他回答的表现。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/policy-gradient-ppo-grpo/13-GRPO-PPO.png" target="_blank" rel="noopener" aria-label="查看原图：GRPO-PPO"><img src="/notes-assets/policy-gradient-ppo-grpo/13-GRPO-PPO.png" alt="GRPO-PPO" width="1542" height="1311" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">GRPO-PPO（点击查看原图）</figcaption>
</figure>

**上半图沿价值网络到 GAE，看优势怎样产生；下半图沿多条输出到组内奖励，看优势来源怎样改变。**下半图没有 critic，但仍有采样、评分和 reference。KL 箭头直接作用于优化目标，不必塞入组内 reward。图省略了每个 token 的 ratio、mask 和长度平均，完整表达在下一节。

<span id="mm-588aa7f2f0fe" style="display:block;scroll-margin-top:6rem"></span>

### 8.2 组均值不直接继承“动作无关基线”的证明

第 4 节要求 baseline 不依赖当前所选动作。组均值却包含当前回答自己的奖励。令同题回答独立同分布，$g_i=\nabla\log\pi(o_i|q)$；仅中心化、不除 std 时：

$$
\begin{aligned}
\mathbb E[(R_i-\bar R)g_i]
&=\mathbb E[R_i g_i]-\frac1G\mathbb E[R_i g_i]
-\frac1G\sum_{j\ne i}\mathbb E[R_j]\mathbb E[g_i]\\
&=(1-1/G)\mathbb E[R_i g_i].
\end{aligned}
$$

最后用到 $\mathbb E[g_i]=0$。这显示包含自身的组均值带来缩放；再除随机 std、按长度平均和做裁剪后，不能称为“精确无偏地替代 $V(s)$”。GRPO 的价值在于实用的组相对 surrogate 和省去额外 critic，不在于与原始 policy gradient 每一项严格等价。

<span id="mm-bdeb29ea7cc0" style="display:block;scroll-margin-top:6rem"></span>

## 9. 把 GRPO 写成完整的 token 目标

<span id="mm-6981770e77dc" style="display:block;scroll-margin-top:6rem"></span>

### 9.1 每个 token 的 ratio，为什么不等于整段 ratio

令 $m_{i,t}=1$ 表示第 i 条回答第 t 个有效生成 token，$\ell_i=\sum_t m_{i,t}>0$。状态 $s_{i,t}=(q,o_{i,<t})$，定义：

$$
\rho_{i,t}=
\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}
{\pi_{\rm old}(o_{i,t}\mid q,o_{i,<t})}.
$$

经典结果监督 GRPO 的一种写法是最大化：

$$
J_{\rm GRPO}=
\mathbb E_{q,\{o_i\}\sim\pi_{\rm old}}
\frac1G\sum_{i=1}^{G}\frac1{\ell_i}
\sum_t m_{i,t}
\left[
\min\big(\rho_{i,t}\hat A_i,\,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)
-\beta k_{i,t}
\right].
$$

其中 $k_{i,t}$ 是相对 reference 的 KL 项或其估计，$\beta$ 是该项权重。实现最小化 $-J_{\rm GRPO}$；优势、旧概率、mask 和采样回答固定，梯度流向当前策略。

整段回答概率是逐 token 概率乘积，所以整段 ratio 是 $\prod_t\rho_{i,t}$。例如两个 token 的 ratio 为 1.1、0.9，序列 ratio 为 0.99，逐 token 平均为 1；对两者做 clip 更不会等价。[DeepSeekMath §4.1](https://arxiv.org/abs/2402.03300)使用逐 token 目标；仓库 PDF 的回答级简写只能辅助理解方向。

每条回答先除自身长度，再在组内平均，让每条回答总权重接近一致；如果改成全 batch token 平均，长回答会占更多权重。两种聚合会改变梯度，不能在面试中混用却声称是同一个精确 loss。

<span id="mm-d5ce7b7ab20e" style="display:block;scroll-margin-top:6rem"></span>

### 9.2 KL 项究竟是什么

固定前缀时，精确 KL 为：

$$
D_{\rm KL}(\pi_\theta\Vert\pi_{\rm ref})
=\sum_a\pi_\theta(a|s)\log\frac{\pi_\theta(a|s)}{\pi_{\rm ref}(a|s)}.
$$

可以全词表求和，也可用采样估计。DeepSeekMath 给出的一种非负逐样本表达，令 $u=\pi_{\rm ref}(a|s)/\pi_\theta(a|s)$：

$$
k=u-\log u-1.
$$

由 $\log u\le u-1$，每个 $k\ge0$，相同策略时为 0。当动作按当前 $\pi_\theta$ 采样且支持集适当时，$\mathbb E_{\pi_\theta}[u-1]=0$，因此这个表达的期望等于上述 KL。若训练时沿用 old policy 样本，当前参数已改变，就需要关注采样分布与校正，不能说“任意 off-policy batch 下都无偏”。不同库的估计和梯度处理可能不同。

例：当前概率 0.4、reference 为 0.2，则 $u=0.5$，$k=0.5-\log0.5-1\approx0.19315$。单独的 log-ratio 与这个非负样本表达数值不同，但不要据此判为互相矛盾；它们是不同估计形式。KL 约束也不能保证不发生 reward hacking。

<span id="mm-e4d08898f9a0" style="display:block;scroll-margin-top:6rem"></span>

### 9.3 为什么 loss=0 仍可能在学习

取一个单 token 二动作问题，old policy 对好坏动作各为 0.5；当前策略初始也为 0.5。恰好组采样到一好一坏，奖励 $[1,0]$，优势 $[1,-1]$，不加 KL。此时两 ratio 都是 1：

$$
J=\frac12(1\times1+1\times(-1))=0.
$$

但设好动作概率 $p=\sigma(\theta)$，old 概率作为常数固定，$p=0.5$ 处：

$$
\nabla_\theta\rho_{\rm good}
=\frac{p(1-p)}{0.5}=0.5,\qquad
\nabla_\theta\rho_{\rm bad}=-0.5.
$$

因此：

$$
\nabla_\theta J=\tfrac12(0.5\times1+(-0.5)\times(-1))=0.5.
$$

目标数值为零，梯度却会提高好动作概率。这里优势被 detach；若误将 old 概率也设成随当前参数变化的同一张量，会得到错误的常数 ratio 梯度。

真正“组内奖励全部相同”时，每一个 $\hat A_i=0$，这一组的优势驱动梯度才为零；KL、其他正则或其他组仍可能有梯度。这是分析 GRPO 日志时非常重要的区别。

<span id="mm-f162a9be47f6" style="display:block;scroll-margin-top:6rem"></span>

### 9.4 结果监督与过程监督的差异

结果监督只给完整回答打分，再把组相对结果赋给各 token。过程监督可给中间推理步骤评分，使前缀能区分“前半段正确、后半段出错”。例如两个步骤的归一化奖励为 $0.2,-0.1$，采用后续步骤奖励和作为 token 优势时，第一步之前 token 的优势为 $0.1$，第一步之后、第二步结束前为 $-0.1$。

这是另一种反馈粒度，不代表每个写得详细的推理链都能被可靠评分；视频场景还需要步骤与画面证据、时间戳对应。过程奖励模型的标注成本与误判也会成为新问题。

<span id="mm-21bba906bc9e" style="display:block;scroll-margin-top:6rem"></span>

## 10. 从 ZealD 的项目问题形成可证伪的解释

[ZealD 项目复盘](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5/6aaea91d0000000026017d0b)可引出“SFT 后 GRPO 为什么没提高”。没有完整实验数据时，答案应是待验证的机制，而不是替项目宣布唯一原因。

首先在 RL 前测 base/SFT 的单次成功率 pass@1 与多次采样至少一次成功的比例。若单次成功概率是 p，且独立同分布采 G 次，理论至少成功一次为 $1-(1-p)^G$；真实输出可能受采样设置和重复模式影响，不能把这个独立假设当事实。pass@G 高只说明候选集中有潜力，不代表自动能学会选出它。

随后看同题组内是否全对、全错、reward std 近零。若所有格式都合格却事实都错，只奖励格式不会创造视觉事实的方向。若训练奖励提高而独立集不提高，则可能是奖励漏洞、标签问题、分布过窄或泄漏，需要具体 bad case 验证。

再看更新强度：新旧 log-prob 差、ratio、clip fraction、梯度范数。clip fraction 常统计 ratio 越界比例，但越界不总等于 surrogate 梯度饱和，优势正负也要看。当前与 old 的 KL 衡量一次更新偏移，当前与 reference 的 KL 衡量相对基准偏移，两者不能混报。

视频组采样还须固定同一题的实际视觉证据。若每条回答随机看到不同帧，组内分数同时混入“谁看到了关键动作”的变化。这可被设计成特殊训练目标，但已经不是固定同一条件下比较回答质量的简单解释。

**面试回答示范。**“我先看起始策略能否采出正确答案，以及奖励在同题组内是否有区分。再核对答案解析、证据覆盖、截断、advantage 和 mask。若有有效信号但更新仍小，检查 ratio、clip 和 KL；若训练 reward 上升而验证不升，检查奖励投机和数据泄漏。只有固定模型、数据拆分、预算和评测做对照，才归因到 SFT 分布或 GRPO 参数。”

上述是教程扩展题；真实面试题与作者学习分享的来源等级见 [05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)。这套后训练知识服务于多模态算法分析，具体视频证据评测见 [06-视觉视频算法面试专项](/notes/vision-video-algorithms/)。

<span id="mm-60b13accbf90" style="display:block;scroll-margin-top:6rem"></span>

## 11. 自测与答案

**题 1：**$A=-3,\rho=0.6,\epsilon=0.2$，PPO 原项、裁剪项和结果是什么？

**答案：**原项 $-1.8$、裁剪项 $-2.4$，取 $-2.4$。负优势已向降低概率方向走过界，这一局部项不再奖励继续下降；不代表整个模型所有梯度为零。

**题 2：**真终止与外部截断都让采样停止，为什么不能都令 $V_{next}=0$？

**答案：**前者任务没有后续回报，后者可能只是收集窗口结束。若目标仍计入之后回报，就需 final state bootstrap；但两者都不能把 reset 后新 episode 的优势串进来，所以 bootstrap 与递推连续性是两个开关。

**题 3：**$G=4$，奖励 $[1,1,1,1]$，加数值 epsilon 能产生非零优势吗？

**答案：**不能。分子 $R_i-\bar R$ 都是 0。epsilon 只解决分母数值问题；KL 或别的组仍可能驱动参数变化。

**题 4：**为何不能把 $\pi_{\rm old}$ 换成 $\pi_{\rm ref}$ 来算 PPO ratio？

**答案：**ratio 分母应代表数据采样概率。reference 可能来自更早阶段，并不是这批动作的行为策略；替换它改变了目标和权重。两模型起始时可以相同，但角色不同。

**题 5：**已算出平均 policy loss 为零，能否断言 GRPO 停训？

**答案：**不能。先看每个样本优势是否全零与实际梯度；正负优势抵消可使标量为零，而对参数的导数仍非零，第 9.3 节给出了数值反例。

<span id="mm-33defd2084a5" style="display:block;scroll-margin-top:6rem"></span>

## 来源与配图边界

- 余昌叶图解仓库：[奖励与价值](https://github.com/changyeyu/LLM-RL-Visualized#header-42)、[MC/TD](https://github.com/changyeyu/LLM-RL-Visualized#header-48)、[策略梯度](https://github.com/changyeyu/LLM-RL-Visualized#header-55)、[GAE](https://github.com/changyeyu/LLM-RL-Visualized#header-66)、[PPO/GRPO](https://github.com/changyeyu/LLM-RL-Visualized#header-72)、[PPO 四模型](https://github.com/changyeyu/LLM-RL-Visualized#header-82)。
- [仓库 12 页策略梯度 PDF](https://github.com/changyeyu/LLM-RL-Visualized/blob/master/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%28Policy%20Gradient%29-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%28PPO%26GRPO%E7%AD%89%29%E4%B9%8B%E6%A0%B9%E5%9F%BA.pdf)：第 3–5 页用于概率推导直觉，第 9–11 页算法比较高度简化。尤其回答级 GRPO ratio 不能逐项替代原论文 token 目标。
- 原始依据：[GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[DeepSeekMath HTML §4.1](https://arxiv.org/html/2402.03300)。
- ZealD：[推导学习记录](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5/6aa79968000000000b001eeb)、[传统 RL 学习](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5/6aaaa176000000000b036985)、[项目复盘](https://www.xiaohongshu.com/user/profile/68ff42af000000003702b1e5/6aaea91d0000000026017d0b)。
- 原图署名和使用条件见 [00-图解大模型算法与ZealD面经总览](/notes/multimodal-interview-guide/)。公式边界由本文另行说明；图解不能替代具体算法和实现定义。
