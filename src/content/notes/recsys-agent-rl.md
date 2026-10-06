---
title: Agent RL 预备：Search-R1、verl 与检索器接入
date: '2026-10-06'
tags: [Agent, 强化学习, Search-R1, verl, GRPO, 检索增强]
summary: 从“为什么一次 RAG 不够”讲到 Search-R1 的多轮 rollout、检索 token 掩码与 EM 奖励，再看 verl 怎样组织 actor、rollout 与工具调用，最后把本地检索服务接进 RL 训练。
draft: false
---

RAG 的做法是用问题检索一次，把文档拼进上下文再生成。多跳问题往往要先查到中间实体，才知道下一步该查什么。这一篇讲怎样用强化学习训练 LLM **在推理过程中自己决定查什么、查几次、何时作答**：先讲 Search-R1 的 rollout 模板、检索 token 掩码和结果奖励，再讲 verl 框架怎样承载这件事，最后讲检索器作为环境接入时的输入输出、成本与评估。PPO、GRPO 的推导见 [从 RL 基础推到 PPO 与 GRPO](/notes/policy-gradient-ppo-grpo/)，本篇不重复。

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。数值算例为教学构造；论文数字均注明表号，代码细节注明仓库版本。

## 1. 问题设定：为什么不只做一次 RAG

把“LLM + 搜索引擎”写成一个序列决策过程。状态是问题加上已生成的文本与已返回的文档；动作是下一个 token，其中有几个 token 序列有特殊含义：写出 `<search> q </search>` 就会触发一次检索，写出 `<answer> a </answer>` 则本局结束。环境就是检索器：输入查询 $q$，返回 top-$k$ 段落。

一次检索的 RAG 有三个硬伤。第一，**查询固定**：只能用原问题检索，问题里没有出现的桥接实体（例如“这款香水的代言歌手是谁”）查不到。第二，**次数固定**：简单题浪费上下文，难题又不够。第三，**检索结果无法被纠正**：第一次查偏了，模型没有机会换个说法再查。Search-R1 论文把已有方案分成两类：RAG，以及把搜索当工具、靠提示（IRCoT、ReAct）或监督微调（Toolformer）来调用。它指出提示类方法泛化差，监督类方法需要大量高质量轨迹标注，而且检索操作不可导，没法端到端反传（[§1–2](https://arxiv.org/abs/2503.09516)）。

RL 的思路是只给最终答案打分，让模型在自己采样的轨迹上试错。检索器不可导没有关系：策略梯度只需要对**模型自己生成的 token** 求 log 概率的梯度，检索器被当作环境的一部分。

## 2. Search-R1 的 rollout：模板、标签与环境循环

Search-R1（Jin et al., arXiv 2503.09516，v5 于 2025-08-05 更新）用四对标签组织轨迹（论文 §3.2–3.3、Table 1）：

| 标签 | 谁写 | 作用 |
|---|---|---|
| `<think>…</think>` | 模型 | 推理 |
| `<search>…</search>` | 模型 | 查询；检测到 `</search>` 就暂停生成去检索 |
| `<information>…</information>` | 环境 | 包住检索返回的 top-$k$ 段落，拼回上下文 |
| `<answer>…</answer>` | 模型 | 最终答案，本局结束 |

Table 1 的模板大意是：每次拿到新信息先在 `<think>` 里推理；缺知识就用 `<search>` 调搜索引擎，结果放在 `<information>` 里返回；可以搜任意多次；不再需要外部知识时直接在 `<answer>` 里给答案，不要解释。作者刻意只约束结构，不规定“必须反思”或“必须搜索”，以便观察 RL 自然学出的行为。

**环境循环**（论文 Algorithm 1）：设最大动作预算 $B$。每一轮让模型生成，遇到 `</search>`、`</answer>` 或 EOS 停下。如果是搜索，就解析查询、检索、把 `<information>d</information>` 追加到序列后继续；如果是答案就返回；两者都不是，就追加一句“我的动作不对，重新想”。官方代码 `search_r1/llm_agent/generation.py` 里的提示文字与论文略有不同，是 “My previous action is invalid. If I want to search, I should put the query between <search> and </search>…”。代码先跑 `max_turns` 轮允许检索的生成，再追加一轮不允许检索的生成，所以一条轨迹最多检索 `max_turns` 次。

**张量形状**（以官方仓库为准）。一个训练 step 有 $N=$ `train_batch_size` 个问题，每个采 $n$ 条轨迹（GRPO 的组大小；PPO 时 $n=1$）。拼好后：

- `input_ids`：$(Nn,\ L_p+L_r)$，左边是提示（截到 `max_start_length`），右边是所有轮次的生成与检索结果；
- `responses`、`info_mask`、`loss_mask`、`advantages`：都是 $(Nn,\ L_r)$；
- `token_level_rewards`：$(Nn,\ L_r)$，只有最后一个有效 token 位置非零，其余为 0（`main_ppo.py` 的 `RewardManager`）。

每次返回的观测在拼接前截断到 `max_obs_length` 个 token（论文附录 B.2 写的是 500）。

## 3. 检索 token 掩码：只对自己写的 token 求梯度

轨迹里混着两种 token：模型生成的，和环境塞进来的文档。PPO 和 GRPO 的损失默认对整段 response 求平均。Search-R1 定义指示函数 $I(y_t)$：模型生成的 token 取 1，检索得到的 token 取 0，只在 $I=1$ 的位置上计算目标（论文式 (2)(3)）：

$$
\mathcal J_{\mathrm{GRPO}}(\theta)=\mathbb E\Bigg[\frac1G\sum_{i=1}^{G}\frac{1}{\sum_t I(y_{i,t})}\sum_{t:\,I(y_{i,t})=1}\min\big(\rho_{i,t}\hat A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_{i,t}\big)-\beta\,\mathbb D_{\mathrm{KL}}[\pi_\theta\Vert\pi_{\mathrm{ref}}]\Bigg],
$$

其中 $\rho_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t};\mathcal R)/\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t};\mathcal R)$。论文特别说明，KL 项同样只在生成 token 上计算。

**为什么要掩码。** 论文的原话是：对检索 token 做同样的优化“可能导致非预期的学习动态”。展开来说（这是本文的解释）：文档 token 不是策略选的动作，$\nabla\log\pi_\theta(\text{doc token})$ 和这一步决策好不好无关。不掩码时，优势为正的轨迹会推着模型去“背诵”维基段落，优势为负的轨迹又会压低模型对一段正确文本的概率，这些梯度全是噪声，而且文档 token 往往占了大头。注意掩码只作用在损失上：文档仍在注意力范围内，模型照样以它为条件生成后文。

**代码怎样实现。** 拼接时，`generation.py` 同时维护两份 response：一份是真实 token，另一份把观测段替换成 pad，由后者生成 `info_mask`。`ray_trainer.py` 的 `_create_loss_mask` 取 `info_mask` 的 response 部分作为 `loss_mask`；`dp_actor.py` 在 `state_masking=true` 时用它替换 `response_mask` 去算策略损失。训练日志里的 `state_tokens/coverage` 就是 `loss_mask.sum()/response_mask.sum()`。verl 自己的 agent loop 也是同一个约定：`response_mask` 对 LLM 生成的 token 记 1，对工具返回的 token 记 0（verl 文档 *Agent Loop*）。

**手算：一条两轮检索轨迹**（教学构造，token 数虚构）。

| 段 | 内容 | 谁写 | token 数 | `attention_mask` | `loss_mask` |
|---|---|---|---:|---:|---:|
| P | 模板 + 问题 | 数据 | 40 | 1 | 不在 response 内 |
| s1 | `<think>…</think><search>Curious fragrance</search>` | 模型 | 18 | 1 | 1 |
| o1 | `<information>Doc 1(Title: …) …</information>` | 环境 | 60 | 1 | 0 |
| s2 | `<think>…</think><search>Britney Spears birthplace</search>` | 模型 | 15 | 1 | 1 |
| o2 | `<information>…</information>` | 环境 | 55 | 1 | 0 |
| s3 | `<think>…</think><answer>McComb, Mississippi</answer>` | 模型 | 12 | 1 | 1 |

response 长度 $18+60+15+55+12=160$，计入损失的 token 有 $18+15+12=45$ 个，coverage $=45/160=0.28125$。按式 (3) 的逐条平均，每个生成 token 权重 $1/45$。如果不掩码，每个 token 权重 $1/160$，生成 token 合计只占 $45/160\approx28\%$，剩下约 72% 的梯度权重落在维基原文上。问题内容取自论文附录 Table 9 的案例。

**边界。** ① `<information>` 标签是环境插入的，也要掩掉；`</search>` 是模型写的，要计入。② 无效动作时插入的纠错提示同样属于观测，官方代码也把它放进被掩的段。③ 多轮 chat 模板下，逐轮分词拼接的结果可能与整段分词不一致。verl 采用“增量分词”，并默认在 rollout 结束后比对两种结果（文档 *Multi-turn Rollout Support*，`tokenization_sanity_check_mode`）。

**验证。** 写单测：把 `loss_mask==1` 的 token 解码出来，断言里面不含任何 `<information>` 段内容；训练时监控 coverage，突然变成 1 说明掩码失效。论文的消融（Qwen2.5-7B-base、PPO）：有掩码时七个数据集平均 EM 为 0.431，无掩码时为 0.343（Table 4）；3B-base 上分别为 0.303 和 0.262（附录 Table 6）。

## 4. 奖励与优势：EM 结果奖励，PPO 与 GRPO

**奖励。** 只有结果奖励（论文式 (4)）：

$$
r_\phi(x,y)=\mathrm{EM}(a_{\mathrm{pred}},a_{\mathrm{gold}}).
$$

官方 `qa_em.py` 先做 SQuAD 式归一化（小写、去标点、去冠词 a/an/the、合并空白），与任一参考答案完全相等记 1，否则 0。有个容易踩的坑：打分时解码的是**提示加回答**，而模板里本身就有示例 `<answer> xxx </answer>`，所以代码要求至少匹配到两个 `<answer>` 块并取最后一个，只有一个就判 0。论文明确说没有加格式奖励，因为模型已经能守住格式；也不训练神经奖励模型（§3.4）。仓库里另有带格式分的 `qa_em_format.py`，对应作者第二篇实证研究（arXiv 2505.15117），其摘要结论是格式奖励能提升最终表现，中间检索奖励作用有限。

**PPO**：优势用 GAE 算，附录 B.2 里 $\lambda=1$、$\gamma=1$，critic 学习率 1e-5，policy 学习率 1e-6。**GRPO**：同一问题采 $G$ 条，组内标准化：

$$
\hat A_i=\frac{r_i-\mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})+\varepsilon},
$$

再把 $\hat A_i$ 广播到第 $i$ 条轨迹所有 `loss_mask=1` 的 token 上。

**手算：组大小 4、奖励 0/1**（教学构造）。同一问题采 4 条轨迹，奖励 $r=(1,0,0,0)$，均值 0.25。

- 总体标准差（除以 $G$）：$\sqrt{(0.75^2+3\times0.25^2)/4}=\sqrt{0.1875}\approx0.433$，优势为 $(1.732,-0.577,-0.577,-0.577)$。
- 样本标准差（除以 $G-1$）：$\sqrt{0.75/3}=0.5$，优势为 $(1.5,-0.5,-0.5,-0.5)$。

Search-R1 内置的 verl 和 verl v0.7.0 的 `compute_grpo_outcome_advantage` 都用 `torch.std`，默认是样本标准差，即第二种。两种的优势之和都为 0。再看两种特殊组：$r=(1,1,0,0)$ 时，样本标准差为 $\sqrt{1/3}\approx0.577$，优势为 $(0.866,0.866,-0.866,-0.866)$；$r=(1,1,1,1)$ 或全 0 时，优势全为 0，这一组不贡献策略梯度。组大小为 1 时，代码把均值设为 0、标准差设为 1，优势就等于奖励本身。这和论文附录 H 说的“退化为 REINFORCE”一致。上面那条两轮轨迹如果是第 1 条，它的 45 个生成 token 每个都拿到优势 1.5。

**怎样选。** 论文 §5.1 / Table 3 / 附录 F 的结论：GRPO 收敛更快，因为 PPO 的 critic 要先热身；PPO 更稳，GRPO 训练久了会出现奖励坍塌；两者最终的训练奖励相近。因此默认用 PPO。Table 3 中 7B-base 的平均 EM，PPO 为 0.431，GRPO 为 0.350；3B-base 则是 GRPO 0.312 略高于 PPO 0.303。可见没有哪个算法全面占优。

**验证。** 监控每个 step 中“组内奖励全相同”的比例。这个比例接近 1 时，GRPO 几乎没有学习信号，排查思路见 [PPO/GRPO 笔记第 10 节](/notes/policy-gradient-ppo-grpo/)。

## 5. 论文实验设置与主要结论

只列能在论文里核到的（§4.3、附录 B.2）：

| 项 | 设置 |
|---|---|
| 基座 | Qwen2.5-3B / 7B（Base 与 Instruct）；附录 C 补充 14B |
| 检索器 / 语料 | E5；2018 年维基百科 dump（Karpukhin et al., 2020） |
| top-$k$ | 3（所有检索类方法统一） |
| 训练数据 | NQ 与 HotpotQA 训练集合并 |
| 评估 | 7 个数据集的 EM：NQ、HotpotQA 为域内；TriviaQA、PopQA、2Wiki、Musique、Bamboogle 为域外 |
| 训练 | 单机 8×H100，500 步，总 batch 512、mini-batch 256，最大序列 4096，回答与检索内容各最多 500 token，FSDP + CPU offload，vLLM rollout |
| RL 超参 | KL 系数 $\beta=0.001$，clip $\epsilon=0.2$，温度 1.0，GRPO 每题采 5 条，最大动作预算 $B=4$ |

**主要结论。** ① Table 2：7B 上 Search-R1-base 平均 EM 0.431，RAG 0.304，拒绝采样 0.348；3B 上 Search-R1-instruct 0.325，RAG 0.270。摘要里说的 24%/20% 相对提升，与“相对最强基线”的口径吻合（$0.431/0.348\approx1.24$，$0.325/0.270\approx1.20$）；贡献列表里写的是相对 RAG 的 41%（$0.431/0.304\approx1.42$）。同一版论文里两种口径并存，引用时要说清基线。② §5.2 / 附录 E：Instruct 模型收敛更快，但最终训练奖励与 Base 接近。③ §5.3：前 100 步回答长度先降（模型去掉废话），之后随搜索次数增加而上升；有效搜索次数随训练增加。④ 附录 G / Table 7：top-$k$ 取 1、3、5 时平均 EM 分别为 0.375、0.431、0.400。作者推测 $k=1$ 召回不足，$k=5$ 噪声多。⑤ 附录 H / Table 8：组大小 1、3、5 的平均 EM 为 0.410、0.363、0.350，组越大收敛越快，但更容易坍塌。

这些数字依赖它的检索器、语料和训练步数，不能当作你复现的预期值。

## 6. 相关工作一句话对照

- **R1-Searcher**（arXiv 2503.05592）：两阶段、只用结果奖励的 RL，不用过程奖励，也不用蒸馏冷启动。
- **ReSearch**（arXiv 2503.19470，NeurIPS 2025）：同样不依赖推理步骤的监督数据，把搜索当推理链的一部分，在 Qwen2.5-7B/32B 上训练。
- **DeepResearcher**（arXiv 2504.03160）：在真实网页搜索环境里端到端 RL，不局限于固定语料，用多智能体结构抽取网页信息。
- **ZeroSearch**（arXiv 2505.04588）：训练时用一个经过 SFT 的 LLM 模拟搜索引擎生成文档，并按课程逐步降低文档质量，省掉真实搜索 API 的费用与噪声。

## 7. verl：谁在算什么

verl 是 HybridFlow（Sheng et al., arXiv 2409.19256，EuroSys 2025）的开源实现。仓库现已迁到 `verl-project/verl`，旧地址会重定向。Search-R1 的仓库就是在早期 verl 上改出来的。

**角色。** 官方 *HybridFlow Programming Guide* 以 PPO 为例，定义了三个 worker group：`ActorRolloutRef`（actor 训练、rollout 生成、reference 算 log 概率，可以合在同一个进程里）、`Critic` 和 `Reward`（奖励模型；规则奖励由 `RewardManager` 直接用函数计算）。GRPO 不需要 critic。各角色的职责与 [PPO/GRPO 笔记第 7 节](/notes/policy-gradient-ppo-grpo/) 的四模型一一对应。

**single-controller 与 hybrid。** RL 可以看成两层数据流：控制流（先 rollout，再算优势，再训练）和计算流（每个模型的分布式前向、反向）。verl 让**控制流跑在单个进程里**（driver 中的 `RayPPOTrainer.fit`），计算流由多进程 worker 执行，driver 通过 WorkerGroup 远程调用、分发数据。这样写新算法只需要改单进程里的训练循环，代价是 driver 与 worker 之间有数据往返。论文摘要概括为把 single-controller 和 multi-controller 两种范式混合使用。

**Hybrid engine。** actor 的训练（FSDP/Megatron 切分）和生成（vLLM/SGLang 切分）放在同一组 GPU 上轮流进行，切换阶段时对权重重新切分。论文称之为 3D-HybridEngine，摘要说它做到零内存冗余，并显著降低通信开销。配置里的 `actor_rollout_ref.hybrid_engine: True` 对应这一点。README 列出的 rollout 后端有 vLLM、SGLang 和 HF Transformers。

**多轮工具调用在文档里的位置**（verl v0.7.0，tag 提交日期 2026-01-05）：*Multi-turn Rollout Support*（`docs/sglang_multiturn/multiturn.rst`）讲 `multi_turn` 开关、工具 YAML（继承 `BaseTool`）、MCP 工具和增量分词；*Search Tool Integration*（`search_tool_example.rst`）就是 Search-R1 式的配方；*Agent Loop*（`docs/advance/agent_loop.rst`）定义 `response_mask`。版本提醒：v0.8.0 起 `examples/sglang_multiturn` 已从仓库移除，当前 main 分支的文档（标注 2026-05-09 更新）注明内置的 `SearchTool` 参考实现已删除，需要用户继承 `BaseTool` 自己实现。所以照着做时要么固定到 v0.7.0，要么自己写工具类。

## 8. 一份 GRPO 配置逐字段读

来源：verl v0.7.0 的 `examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh`，配合 `search_multiturn_grpo.yaml`；字段含义以 `docs/algo/grpo.md` 和 `rollout.yaml` 的注释为准。

| 字段 | 示例值 | 含义 |
|---|---|---|
| `algorithm.adv_estimator` | `grpo` | 默认是 `gae`（PPO）；改成 grpo 后不用 critic |
| `data.train_batch_size` | 512 | 每 step 的**问题数**；轨迹数 = 512 × `rollout.n` |
| `actor_rollout_ref.rollout.n` | 5 | 组大小，必须大于 1 |
| `actor.ppo_mini_batch_size` | 256 | 全局 mini-batch，代码里会再乘以 n；`ppo_epochs` 默认为 1 时，每 step 更新参数 2 次 |
| `actor.ppo_micro_batch_size_per_gpu` | 8 | 只影响显存，不改变算法 |
| `actor.use_kl_loss` / `kl_loss_coef` / `kl_loss_type` | True / 0.001 / `low_var_kl` | KL 直接加在损失上（k3 估计），不进奖励 |
| `algorithm.use_kl_in_reward` | False | 与上一行互斥 |
| `actor.clip_ratio`、`loss_agg_mode` | 默认 0.2、`token-mean` | 文档提醒：原 GRPO 是逐条平均，verl 示例默认按 token 平均 |
| `actor.optim.lr` / `lr_warmup_steps_ratio` | 1e-6 / 0.285 | 与 Search-R1 附录 B.2 一致 |
| `rollout.name` | `sglang` | 生成后端 |
| `rollout.multi_turn.enable` | True（写在 yaml 里） | 打开多轮 |
| `rollout.multi_turn.max_assistant_turns` | 2 | 助手最多说几轮，相当于检索次数上限 |
| `rollout.multi_turn.tool_config_path` | `search_tool_config.yaml` | 工具定义：`retrieval_service_url`、`num_workers: 120`、`rate_limit: 120`、`timeout: 30`，以及 OpenAI function schema（参数是 `query_list`） |
| `rollout.multi_turn.max_tool_response_length` | 默认 256 | 工具返回截断长度，默认从中间截 |
| `data.max_response_length` / `rollout.max_model_len` | 3000 / 15000 | 多轮累计的回答长度与引擎上下文上限 |
| `data.return_raw_chat` | True | 多轮需要原始消息列表 |

注意它与 Search-R1 原仓库的差别：verl 版用的是 OpenAI 风格的函数调用和 chat 模板，不是 `<search>` 纯文本标签；工具返回段的拼接格式是 `Doc 1 (Title: …)\n…`，与原仓库的 `Doc 1(Title: …) …` 略有不同。两边的模型不能混用评测脚本。

## 9. 检索器怎样接入 RL

**作为 HTTP 服务。** Search-R1 把检索器和训练分开部署：`retrieval_launch.sh` 用 FastAPI 在 8000 端口启动 `retrieval_server.py`，默认参数是 e5（`intfloat/e5-base-v2`，BERT-base，768 维）、Flat 索引、`topk 3`、`--faiss_gpu`。编码时查询加前缀 `query: `，段落加前缀 `passage: `，mean pooling 后做 L2 归一化，FAISS 按内积检索。接口：

```
POST /retrieve
{"queries": ["Curious fragrance", "Britney Spears birthplace"], "topk": 3, "return_scores": true}
→ {"result": [[{"document": {"id": "...", "contents": "\"Title\"\ntext"}, "score": 0.83}, ...], [...]]}
```

（score 数值为示意。）输入是长度为 $Q$ 的查询列表，输出是 $Q\times k$ 个文档；rollout 侧把同一轮所有轨迹的查询批成一次请求。语料是 jsonl，每行包含 `id` 和 `contents`，后者首行为标题。仓库还支持 BM25、HNSW 近似索引和在线搜索 API（`docs/retriever.md`）。据 verl 文档，wiki-18 的索引加语料下载约 60–70 GB，解压后约 132 GB；GPU 版检索服务每卡占 5–7 GB 显存，CPU 版精度更低、会拖累训练。

**冻结还是联合训练。** Search-R1 和 verl 示例都冻结检索器：它只是环境，策略学的是“怎样写查询、怎样用结果”。联合训练在论文里没有做，下面是本文的工程分析。你熟悉 [对比学习](/notes/contrastive-learning/)，可以用策略产生的查询加上“段落里是否含有答案”的弱标签去微调 E5。但有三个代价：① 每更新一次检索器，就要重新编码整个语料、重建索引，否则查询向量和文档向量不在同一空间，道理同 [召回第 9 节](/notes/wangshusen-recommender-retrieval/) 的“模型与索引成套更新”；② 环境变了，同一策略在前后两个检索器下的奖励分布不同，RL 更难稳定；③ 检索器可能迁就策略写出的怪查询。比较稳妥的顺序是：先冻结检索器训好策略；再离线用策略的查询日志训练新检索器；最后固定新检索器重新跑 RL，并在旧检索器上做交叉评测。

**奖励设计与奖励黑客。** EM 严格但稀疏。F1 或子串 EM（仓库里的 `subem_check`）更宽松，模型却可能学会把答案写长、列出多个候选去“蹭”匹配。格式奖励可能让模型只学会守格式而不答对。“检索命中答案”的中间奖励可能鼓励无意义的反复搜索（2505.15117 的摘要说这类奖励作用有限）。抽查是必须的：每隔若干 step 打印高奖励轨迹，看答案是不是从 `<information>` 原样抄来的、是否正确，以及有没有出现“不搜就答”或“搜满上限”的退化模式。

**调用次数与成本。** 每个 step 的检索请求上界是 问题数 × $n$ × 最大轮数。Search-R1 的 GRPO 脚本是 512 × 5 × `max_turns=2`，最多 5120 次（论文实验用的是 $B=4$，与脚本不同）。控制手段有：`max_turns` 或 `max_assistant_turns`、观测截断（`max_obs_length` 或 `max_tool_response_length`）、top-$k$、工具的并发与限流（`num_workers`、`rate_limit`、`timeout`）。如果用在线搜索 API，费用会随 rollout 数线性增长，这正是 ZeroSearch 要解决的问题。

**评估。** 沿用论文口径：7 个 QA 数据集，用 EM，区分域内（NQ、HotpotQA）和域外。同时报告平均检索次数、平均回答长度和无效动作率。verl 文档提醒，完整验证集有 51k 条，验证一次约 6000 秒，所以示例里默认 `val_before_train=False`。开发阶段先固定一个子集。

## 10. 最小可行项目：2–3 周、单机 4 卡

目标是复现“检索 + RL”的完整闭环，并跑出一组有因果解释的消融。**不承诺具体分数。**

| 周 | 做什么 | 验收 |
|---|---|---|
| 第 1 周前半 | 部署检索服务（e5 + Flat-GPU，或 BM25 / HNSW 以省显存）；用 `retrieval_request.py` 压测 | 批量 QPS 与延迟有记录；手查 20 个问题，top-3 是否合理 |
| 第 1 周后半 | 用 Search-R1 仓库或 verl v0.7.0 示例跑通 Qwen2.5-3B（建议 Instruct，收敛更快）的 GRPO；`train_batch_size`、`n_gpus_per_node` 按 4 卡调小，打开 offload | 跑完前几十 step 不报错；mask 单测与奖励单测通过 |
| 第 2 周 | 正式训练；同一检索器下跑基线：直接回答、一次 RAG、不检索的 R1 | 训练奖励、有效搜索次数、回答长度三条曲线；在 NQ / HotpotQA 子集上比较 EM |
| 第 3 周 | 消融：有无检索 token 掩码；top-$k$=1/3；`max_turns`=1/2 | 每个结论写成“因为……所以……”，并附 3 条 bad case |

**风险与对应的验证。** ① 显存：检索服务和训练抢卡。把检索放 CPU（HNSW）或单独占一张卡，并记录对 EM 的影响。② 掩码或分词错位：用解码单测，加上 coverage 监控。③ 奖励抽取错误：构造“只有模板里的 `<answer>`”“答案大小写不同”“多个参考答案”等用例做单测。④ GRPO 奖励坍塌（论文 Fig. 2a）：每 50 步存一次 checkpoint，按验证集挑选，必要时换 PPO 或缩小学习率。⑤ 全对或全错的组太多：先测起始模型在训练集上的 pass@5，过滤掉过难或过易的题。⑥ 数据泄漏：评测集与训练集去重；语料是 2018 年的，答案已经变化的题要剔除。

## 11. 面试常问

**ReAct 与 RL 训练出的 agent 有什么区别？** ReAct 靠提示让模型交替写推理和动作，模型权重不变，行为取决于示例和模型本身的能力。RL agent 的接口可以完全相同（思考、搜索、观测、作答），但策略在自己的轨迹上按结果奖励更新过，学到的是“这种题该怎么查、查到什么程度停”。Search-R1 的 Table 2 中，提示类的 IRCoT 和 Search-o1 在 7B 上的平均 EM 为 0.239 和 0.206，低于 Search-R1。代价是要有可验证的奖励和大量 rollout。

**为什么要 mask 检索 token？** 它们不是策略的动作，对它们的 log 概率求梯度不对应任何决策。不掩的话，梯度大半落在文档文本上，等于在用奖励信号做“背维基”或“反背维基”的语言模型训练。KL 也要一起掩。实验证据见 Table 4。

**怎样判断信息够了，怎样防止死循环？** 学习层面：“何时写 `<answer>`”本身就是策略的动作，结果奖励会惩罚过早作答（答错）；不写答案也拿 0 分，所以一直搜索不会有收益。论文附录 I 的案例显示，模型在信息已经足够后，还会多做一次验证性检索。工程层面要有硬上限：最大动作预算 $B$ 或 `max_turns`、总长度上限、观测截断、无效动作提示。还可以加上重复查询检测，或把调用次数计入代价，Search-R1 README 列出的 OTC 就是专门研究“最优工具调用次数”的工作。评估时报告平均检索次数，以及触顶比例（搜满上限仍未作答的轨迹占比）。

## 闭卷验收

不看资料写出 Search-R1 的四对标签以及各自由谁生成；画出 Algorithm 1 的环境循环，说出三种停止条件；写出带掩码 $I(y_t)$ 的 GRPO 目标，解释为什么 KL 也要掩；对一条两轮检索轨迹手算 `loss_mask` 和 coverage；对奖励 $(1,0,0,0)$ 分别用总体标准差和样本标准差算组内优势；说出 EM 奖励的归一化步骤和“至少两个 `<answer>`”的原因；讲清 verl 中 single-controller、hybrid engine、`ActorRolloutRef` 的含义；逐字段解释 `rollout.n`、`train_batch_size`、`ppo_mini_batch_size`、`use_kl_loss`、`multi_turn.max_assistant_turns`；写出检索服务的请求与返回格式；最后讲出冻结检索器与联合训练检索器各自的代价，以及一个 2–3 周、4 卡的复现与消融计划。

**参考。** [Search-R1](https://arxiv.org/abs/2503.09516)（[HTML v5](https://arxiv.org/html/2503.09516)）；[Search-R1 官方仓库](https://github.com/PeterGriffinJin/Search-R1)（`search_r1/llm_agent/generation.py`、`search_r1/search/retrieval_server.py`、`verl/utils/reward_score/qa_em.py`、`train_grpo.sh`）；[An Empirical Study on RL for Reasoning-Search Interleaved LLM Agents](https://arxiv.org/abs/2505.15117)；[HybridFlow](https://arxiv.org/abs/2409.19256)；[verl 仓库](https://github.com/volcengine/verl)（[v0.7.0 tag](https://github.com/volcengine/verl/tree/v0.7.0)）；[verl 文档：HybridFlow Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)、[GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)、[Multi-turn Rollout](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)、[Search Tool Integration](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)、[Agent Loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)；[DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)；[PPO](https://arxiv.org/abs/1707.06347)；[E5](https://arxiv.org/abs/2212.03533)；[DPR（wiki-18 语料）](https://arxiv.org/abs/2004.04906)；[ReAct](https://arxiv.org/abs/2210.03629)；[R1-Searcher](https://arxiv.org/abs/2503.05592)；[ReSearch](https://arxiv.org/abs/2503.19470)；[DeepResearcher](https://arxiv.org/abs/2504.03160)；[ZeroSearch](https://arxiv.org/abs/2505.04588)。
