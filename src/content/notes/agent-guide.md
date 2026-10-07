---
title: Agent 算法学习手册：知识地图与阅读顺序
date: '2026-10-06'
tags: [Agent, 学习路线, 工具调用, RAG, 强化学习, 评测]
summary: Agent 算法岗要会什么、各篇按什么顺序读、每篇读完该答出什么问题，以及怎样从多模态与跨模态检索背景接到 Agent 训练与评测。
draft: false
---

这一篇是 Agent 专题的入口，回答三个问题：**Agent 算法岗**（偏模型、训练与评测，不是搭应用）到底考什么；专题里的各篇按什么顺序读，每篇读完应当能答出什么；做过多模态和跨模态检索的人，哪些已有知识可以直接迁移，哪些要补。总览本身少用公式，但第 4 节有一个 ReAct 轨迹的上下文长度与步数估算，要求能在纸上算出来。多模态部分的入口见 [多模态专题总览](/notes/multimodal-interview-guide/)。

本篇数值算例为教学构造；涉及的论文数字都注明出处，核对记录见同名 `.verify.md`。

## 1. Agent 算法岗要会什么

先给一个可以落到代码里的定义：**Agent** 是一个在循环里运行的 LLM 策略。每一步它读入当前上下文（任务、历史动作、环境返回的观测），输出一段文本；文本中如果有结构化的动作（工具调用、检索查询、代码、点击坐标），外部程序就执行它，把结果作为观测追加回上下文，直到模型给出最终答案或撞到步数上限。ReAct（Yao et al., arXiv 2210.03629，ICLR 2023）把这种“推理与动作交替”的写法定型，摘要里的实验覆盖 HotpotQA、FEVER、ALFWorld 和 WebShop。

围绕这个循环，算法岗的问题可以分成六类：

| 问题 | 要会的东西 | 典型追问 |
|---|---|---|
| 动作怎么表示 | tool schema、消息格式、结构化输出、约束解码 | 模型输出的 JSON 参数非法怎么办 |
| 多步怎么控制 | CoT、ReAct、Plan-and-Execute、反思、记忆 | 上下文塞满了怎么办；什么时候停 |
| 知识从哪来 | RAG 全链路：切块、检索、重排、生成 | 检索召回高但答案不忠实，问题出在哪 |
| 怎样接外部系统 | MCP 的 host/client/server，多智能体编排 | 什么时候多智能体反而更差 |
| 怎样训练 | 工具调用 SFT 数据构造，多轮 Agent RL，奖励设计 | 环境返回的 token 为什么要掩掉；奖励被黑怎么发现 |
| 怎样评测与设计 | 基准、pass@k 与 pass^k、轨迹失败分析、系统设计题 | 一次跑分 60% 能不能上线 |

前三类偏推理时的机制，后三类是算法岗和应用岗拉开差距的地方：你要能说清**训练信号从哪来**、**评测数字能说明什么**。

## 2. 知识地图：分层看依赖

下面按依赖从下往上排。上层默认你会下层；面试追问通常是从上层往下层挖。

| 层 | 内容 | 依赖下层的什么 | 对应篇 |
|---|---|---|---|
| L0 模型基础 | 注意力、位置编码、KV 缓存、prefill 与 decode | — | [Transformer](/notes/transformer-attention-rope-gqa/)、[Prefill 与 Decode](/notes/prefill-decode-video-tokens/) |
| L1 单步动作 | 消息格式、tool schema、Function Calling、约束解码、并行调用 | 模型怎样按模板生成 | [agent-tool-calling](/notes/agent-tool-calling/) |
| L2 多步控制 | CoT、Self-Consistency、ToT、ReAct、Plan-and-Execute、Reflexion、记忆与上下文管理 | 单步调用可靠；上下文长度的代价 | [agent-planning-memory](/notes/agent-planning-memory/) |
| L3 外部知识 | RAG 全链路、多模态 RAG、自适应检索 | 检索与 embedding；多步控制 | [agent-rag](/notes/agent-rag/) |
| L4 协议与编排 | MCP、多智能体框架 | 工具调用；多步控制 | [agent-mcp-multiagent](/notes/agent-mcp-multiagent/) |
| L5 训练 | 工具调用 SFT → Search-R1 式 RL → 信用分配、过程奖励、rollout 工程 | SFT/DPO、PPO/GRPO；L1–L3 的轨迹格式 | [recsys-agent-rl](/notes/recsys-agent-rl/)、[agent-rl-advanced](/notes/agent-rl-advanced/) |
| L6 评测 | 基准、可靠性指标、失败分析、污染与成本 | 能复现一条轨迹并判断对错 | [agent-eval](/notes/agent-eval/) |
| L7 系统设计 | 需求 → 架构 → 容量/延迟/成本 → 失败处理 → 评测迭代 | 以上全部 | [agent-system-design](/notes/agent-system-design/) |

读图方式：一个面试问题落在哪一层，就先在那一层给机制，再往下补一层的原因。比如“多轮工具调用的 RL 训练为什么不稳”属于 L5，回答要落到 L1 的消息拼接（环境 token 混进 response）和 L0 的长序列代价。

## 3. 阅读顺序与每篇的验收问题

**主线**（按顺序读）：

| 序 | 篇目 | 读完应当能回答 |
|---|---|---|
| 1 | agent-guide（本篇） | Agent 算法岗的六类问题各对应哪篇；一条 ReAct 轨迹的上下文长度怎样随步数增长？ |
| 2 | [agent-tool-calling](/notes/agent-tool-calling/) | 一次 Function Calling 从 tool schema 到模型输出、再到工具结果回填，消息序列长什么样；工具调用 SFT 数据怎样造、怎样过滤？ |
| 3 | [agent-planning-memory](/notes/agent-planning-memory/) | ReAct 与 Plan-and-Execute 各在什么任务上更合适；上下文快满时，截断、摘要和向量记忆各丢掉了什么？ |
| 4 | [agent-rag](/notes/agent-rag/) | 召回、重排、生成三段各用什么指标定位问题；文档图像检索（如 ColPali）和先 OCR 再检索的区别在哪？ |
| 5 | [agent-mcp-multiagent](/notes/agent-mcp-multiagent/) | MCP 里 host、client、server 各管什么，tools、resources、prompts 有何区别；什么条件下多智能体不如单智能体？ |
| 6 | [recsys-agent-rl](/notes/recsys-agent-rl/)（已有） | Search-R1 的 rollout 怎样拼接，为什么要掩掉检索 token；verl 中谁负责 rollout、谁负责训练，检索器怎样作为环境接入？ |
| 7 | [agent-rl-advanced](/notes/agent-rl-advanced/) | 多轮轨迹只有终局奖励时，信用怎样分到每一步；结果奖励、格式奖励、过程奖励各会被怎样“黑”？ |
| 8 | [agent-eval](/notes/agent-eval/) | SWE-bench、GAIA、τ-bench、BFCL 各测什么、用什么指标；pass@k 与 pass^k 为什么会给出相反的印象？ |
| 9 | [agent-system-design](/notes/agent-system-design/) | 给你“设计一个 deep research Agent”，怎样在 45 分钟内讲完需求、架构、成本估算、失败处理和评测迭代？ |

**前置**（已有多模态笔记，按需回补）：

| 篇目 | 在 Agent 专题里用在哪 |
|---|---|
| [数学与张量预备课](/notes/multimodal-math-prerequisites/) | softmax、对数概率、KL，读 RL 和约束解码的前提 |
| [Transformer、RoPE 与 GQA](/notes/transformer-attention-rope-gqa/) | 理解 chat 模板为什么是“一条长序列”，以及注意力 mask |
| [Prefill、Decode 与视频 token](/notes/prefill-decode-video-tokens/) | 多轮 Agent 的延迟与 KV 显存估算，前缀缓存为什么省钱 |
| [对比学习](/notes/contrastive-learning/) | RAG 的稠密检索器、难负例 |
| [SFT 与 DPO](/notes/multimodal-sft-lora-dpo/) | 工具调用 SFT 的 labels 对齐与掩码，偏好数据 |
| [从 RL 基础推到 PPO 与 GRPO](/notes/policy-gradient-ppo-grpo/) | Agent RL 的全部目标函数，本专题不重推 |
| [训练显存与实验排障](/notes/training-memory-debugging/) | 长轨迹 RL 的显存账与排障 |

**同批多模态补充**（做多模态 Agent 时读）：[多模态 embedding 与检索](/notes/multimodal-embedding-retrieval/)（RAG 检索器的多模态版本）、[多模态 RL](/notes/multimodal-rl/)（可验证奖励在 VLM 上的用法）、[视觉定位](/notes/multimodal-grounding/)（GUI Agent 输出点击坐标的基础）、[多模态幻觉与评测](/notes/multimodal-hallucination-eval/)（与 RAG 的 faithfulness 是同一类问题）、[视频理解](/notes/multimodal-video-understanding/)（长观测的压缩思路）。

建议节奏：主线 1–5 是推理时机制，可以较快过完；6–7 是 Agent 算法岗的核心，要配合代码读；8 和 9 放在最后，因为它们要求把前面的东西串起来。

## 4. 手算：一条 ReAct 轨迹的上下文与步数预算

**设定**（教学构造，所有 token 数虚构）。系统提示加 tool schema 共 800 token，用户问题 50 token。每一步模型生成“思考 + 工具调用”共 60 token，工具返回的观测截断到 400 token。做 5 次工具调用后，模型生成 40 token 的最终答案。模型上下文上限 8192。

**每次调用模型时的输入长度。** 每完成一步，上下文增加 $60+400=460$ 个 token。第 $t$ 次调用（$t=1,\dots,6$，第 6 次是写答案）的输入长度为

$$
L_t = 850 + 460\,(t-1),
$$

依次为 850、1310、1770、2230、2690、3150，最终序列长度 $3150+40=3190$。

**没有前缀缓存时的 prefill 总量**：每次调用都把整段上下文重新算一遍，

$$
\sum_{t=1}^{6} L_t = 6\times850 + 460\times(0+1+2+3+4+5) = 5100 + 6900 = 12000.
$$

**有前缀缓存时**：之前的 token 已有 KV，只需 prefill 新增部分，即首次的 850 加上 5 段观测 $5\times400=2000$，共 2850（模型自己生成的 token 在 decode 时已写入缓存）。两者之比 $12000/2850\approx4.2$。步数越多差距越大：无缓存的总量随步数平方增长，有缓存的随步数线性增长。机制见 [Prefill 与 Decode](/notes/prefill-decode-video-tokens/)。

**步数上限。** 要求 $850+460n+40\le8192$，得 $n\le7302/460\approx15.9$，最多 15 步工具调用。验算：$n=15$ 时总长 7790，$n=16$ 时 8250，超限。

**放到 RL 训练里看。** response 部分长 $3190-850=2340$，其中模型生成的只有 $5\times60+40=340$，占 $340/2340\approx14.5\%$。如果不掩掉观测 token，约 85% 的梯度权重落在工具返回的文本上，这正是 [recsys-agent-rl 第 3 节](/notes/recsys-agent-rl/) 讲的检索 token 掩码要解决的问题。

**多步可靠性。** 假设每一步工具调用独立地以 0.95 的概率正确，5 步全对的概率是 $0.95^5\approx0.774$。单步 95% 看着很高，串成 5 步就只剩约 77%。真实步骤之间不独立，这个数只用来说明误差会累积，评测时要报告轨迹级成功率，而不是单步准确率。

**边界。** ① 观测长度在真实任务里方差很大（网页、代码文件），用均值估算会低估尾部，要看 P95。② 摘要式记忆会让 $L_t$ 不再线性增长，但每次摘要本身也要一次模型调用。③ 前缀缓存要求前缀逐 token 不变；如果每步都改写系统提示（比如插入当前时间），缓存就失效。

**怎样验证。** 在自己的 Agent 日志里统计每步观测长度的分布、每条轨迹的步数分布和触顶比例（撞到步数或长度上限仍未作答的轨迹占比），用实测分布替换上面的常数重算。

## 5. 覆盖清单：必会、常问、加分

**必会**（答不出来基本过不了算法面）：

- ReAct 循环与消息格式，Function Calling 的 tool schema 和回填方式 → [agent-tool-calling](/notes/agent-tool-calling/)
- 上下文长度怎样随步数增长，截断与摘要的取舍 → 本篇第 4 节、[agent-planning-memory](/notes/agent-planning-memory/)
- RAG 全链路与各段指标，稠密/稀疏/混合检索 → [agent-rag](/notes/agent-rag/)
- 多轮 RL 的 rollout 拼接、环境 token 掩码、GRPO 组内优势 → [recsys-agent-rl](/notes/recsys-agent-rl/)、[PPO 与 GRPO](/notes/policy-gradient-ppo-grpo/)
- 至少三个基准的任务和指标，pass@k 与 pass^k 的区别 → [agent-eval](/notes/agent-eval/)

**常问**：

- 工具调用 SFT 数据的构造与过滤 → [agent-tool-calling](/notes/agent-tool-calling/)
- 约束解码与结构化输出，并行工具调用 → [agent-tool-calling](/notes/agent-tool-calling/)
- Reflexion 一类“用文字反思而不更新权重”的方法与 RL 的区别 → [agent-planning-memory](/notes/agent-planning-memory/)
- MCP 的角色划分与三类 server 功能 → [agent-mcp-multiagent](/notes/agent-mcp-multiagent/)
- 奖励设计：结果、格式、过程奖励，奖励黑客的发现方法 → [agent-rl-advanced](/notes/agent-rl-advanced/)
- 轨迹级失败分析：把失败归到规划、调用、检索还是作答 → [agent-eval](/notes/agent-eval/)
- 设计题：deep research Agent 的架构与成本估算 → [agent-system-design](/notes/agent-system-design/)

**加分**：

- PRM 与 ORM 的取舍，长轨迹的信用分配与训练稳定性 → [agent-rl-advanced](/notes/agent-rl-advanced/)
- 多模态 RAG（文档图像检索）与 GUI 定位 → [agent-rag](/notes/agent-rag/)、[视觉定位](/notes/multimodal-grounding/)
- 什么时候多智能体不如单智能体 → [agent-mcp-multiagent](/notes/agent-mcp-multiagent/)
- 评测污染、结果方差与成本归一 → [agent-eval](/notes/agent-eval/)
- 多模态客服 Agent 的设计 → [agent-system-design](/notes/agent-system-design/)

## 6. 从多模态背景接上 Agent

你做过跨模态检索、了解 VLM 训练，这些不是另起炉灶，而是能直接对上 Agent 链路里的几个环节。

| 你已有的 | 在 Agent 里对应 | 要补的差异 |
|---|---|---|
| 双塔检索、对比学习、Recall@K | RAG 的稠密检索器与评测 | 检索的终点是“生成的答案对不对”，召回高不等于答案忠实；还要会重排和混合检索 |
| VLM 把图像 token 接进 LLM | 多模态 Agent 的观测：截图、文档页、视频帧 | 观测是每步新增的，长度预算按步累加（第 4 节） |
| 视频 token 压缩、采帧 | 长观测截断、摘要记忆 | 压缩丢掉的信息会在后续步骤里才暴露，要用轨迹级指标评 |
| 视觉定位、输出框坐标 | GUI Agent 输出点击位置 | 坐标错一点，环境里的动作就全错，奖励是二值的 |
| SFT 的 labels 掩码 | 工具调用 SFT 只在 assistant 段算损失 | 多轮里工具返回也在序列中，必须掩掉 |
| PPO/GRPO 单轮 | 多轮 Agent RL | rollout 要和环境交互、异步、有失败；奖励更稀疏 |
| 多模态幻觉评测 | RAG 的 faithfulness、Agent 编造工具结果 | 要能区分“检索没给到”和“给到了但没用” |

面试时的讲法：先说明自己熟悉的那一环（例如检索器的难负例和 Recall@K），再说明它放进 Agent 之后哪个指标变了，最后给出一个实验来证明这种迁移有效。例如：“检索器 Recall@5 提升后，下游 EM 是否同步提升；如果没有，就去查重排或生成段。”不要把多模态项目硬说成 Agent 项目，讲清楚两者之间的接口就够了。

## 7. 面试常问

**Agent 和普通的多轮对话有什么区别？** 区别在于有没有环境。Agent 的输出里有可执行的动作，执行结果作为观测回到上下文，决定下一步；对话只有用户输入。因此 Agent 要处理动作格式错误、工具失败、步数上限和观测过长等问题，这些在普通对话里都不存在。

**为什么多步 Agent 的评测要看 pass^k？** τ-bench（arXiv 2406.12045）把 pass^k 定义为 k 次独立试验全部成功的概率，用 $\binom{c}{k}/\binom{n}{k}$ 对任务取期望（§3）；pass@k 是至少一次成功。线上用户每次只跑一次、而且要每次都对，所以可靠性要看 pass^k。该文摘要报告 gpt-4o 一类函数调用 Agent 在任务上的成功率低于 50%，retail 域的 pass^8 低于 25%。

**多轮工具调用 RL 和单轮 GRPO 最大的工程差异是什么？** rollout 中途要等环境返回，序列里混有环境 token 需要掩码，轨迹长度方差大导致 batch 内负载不均，工具服务的并发与超时会直接影响训练吞吐。细节见 [recsys-agent-rl](/notes/recsys-agent-rl/) 和 [agent-rl-advanced](/notes/agent-rl-advanced/)。

**上下文窗口很长了，还需要记忆模块吗？** 需要。第 4 节说明了无缓存时 prefill 总量随步数平方增长；窗口长也不代表模型能用好远处的信息。是否需要摘要或外部记忆，要看任务的步数分布和长上下文下的实测准确率，而不是只看窗口上限。

## 闭卷验收

不看资料写出 Agent 循环的定义和停止条件，说出算法岗的六类问题各对应专题里哪一篇；按第 3 节的表，逐篇说出读完应能回答的问题；画出 L0 到 L7 的分层，说明“多轮 RL 不稳”要往下挖到哪两层；对系统提示 800、问题 50、每步生成 60、观测 400、5 步、答案 40 的轨迹，手算各次调用的输入长度、无缓存与有缓存的 prefill 总量（12000 与 2850）、8192 窗口下的最大步数（15）、RL 中生成 token 占比（约 14.5%）；解释单步 95% 串成 5 步为何只剩约 77%；写出 pass^k 的定义并说明它与 pass@k 的区别；最后用自己的项目举一个例子，讲清多模态检索的哪一环怎样接进 Agent，以及用什么实验证明。

**参考。** [ReAct](https://arxiv.org/abs/2210.03629)；[Toolformer](https://arxiv.org/abs/2302.04761)；[Reflexion](https://arxiv.org/abs/2303.11366)；[Search-R1](https://arxiv.org/abs/2503.09516)；[τ-bench](https://arxiv.org/abs/2406.12045)（[HTML](https://arxiv.org/html/2406.12045)）；[SWE-bench](https://arxiv.org/abs/2310.06770)；[GAIA](https://arxiv.org/abs/2311.12983)；[Model Context Protocol 规范](https://modelcontextprotocol.io/specification/latest)；[verl 仓库](https://github.com/volcengine/verl)。
