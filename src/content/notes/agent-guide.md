---
title: Agent 算法学习手册：知识地图、阅读顺序与覆盖清单
date: '2026-10-06'
tags: [Agent, 学习路线, 工具调用, 上下文工程, 复习问答]
summary: 把 Agent 拆成接口、控制循环、上下文与记忆、检索、协议与协作、评测、训练入门、训练进阶、系统设计九层，每层对应一篇教程；用四个手算把成本和可靠性落到数字上，回答八道高频八股题，最后给一份可以逐项勾掉的覆盖清单。
draft: false
---

这套笔记围绕一个问题组织：**一个 LLM 怎样在多步交互中可靠地完成任务，又怎样用数据和训练让它更可靠？** 面向的是算法岗：重点是机制、可计算的代价、可验证的评测和训练方法，不是某个 Agent 框架的 API 用法。每篇教程都有手算例子，论文结论注明出处，可以在原文核对。

先分清两个词。Anthropic 的 *Building effective agents* 把基于 LLM 的系统分成两类：**workflow**（LLM 和工具按预先写好的代码路径编排）和 **agent**（LLM 自己动态决定流程和工具使用）。文章的建议是先找最简单的方案，只有在必要时才增加复杂度，因为 agent 系统通常是用延迟和成本换效果。这一句话本身就是很多面试题的答案起点。

## 1. 知识地图：九层，每层一篇

| 层 | 要回答的问题 | 教程 |
|---|---|---|
| 接口 | 模型怎样“调用”一个函数？参数怎样保证能执行？ | [工具调用](/notes/agent-tool-calling/) |
| 控制循环 | 下一步做什么？什么时候停？失败了怎么办？ | [规划与记忆](/notes/agent-planning-memory/) 第 1–4 节 |
| 上下文与记忆 | 窗口里放什么？放不下怎么办？跨会话记什么？ | [规划与记忆](/notes/agent-planning-memory/) 第 5–8 节 |
| 检索 | 外部知识怎样进来？召回和生成怎样分开测？ | [Agent RAG](/notes/agent-rag/) |
| 协议与协作 | 工具怎样标准化接入？什么时候拆成多个 Agent？ | [MCP 与多智能体](/notes/agent-mcp-multiagent/) |
| 评测 | 涨了 6 个点是真的吗？裁判可信吗？ | [Agent 评测](/notes/agent-eval/) |
| 训练入门 | 怎样用 RL 训练模型自己决定查什么？ | [Agent RL 预备](/notes/recsys-agent-rl/) |
| 训练进阶 | 长度偏差、熵坍塌、多轮信用分配、奖励黑客 | [Agent RL 进阶](/notes/agent-rl-advanced/) |
| 系统设计 | 一道开放的设计题怎样在 45 分钟内讲清楚？ | [Agent 系统设计](/notes/agent-system-design/) |

这九层不是互相独立的。一个典型的追问链是：“你的 Agent 为什么会死循环？”（控制循环）→“上下文里有没有重复的工具结果？”（上下文）→“怎样评估修复是否有效？”（评测）→“能不能通过训练让它学会停？”（RL）。读的时候随时想“这一层的失败会在哪一层表现出来”。

多模态方向的对应入口是 [多模态学习手册](/notes/multimodal-interview-guide/)。两个专题的交汇点有三处：[多模态 embedding](/notes/multimodal-embedding-retrieval/) 可以当 Agent 的检索器；[视觉定位](/notes/multimodal-grounding/) 是 GUI Agent 点击屏幕元素的基础；[多模态 RL](/notes/multimodal-rl/) 和 Agent RL 共用同一套 GRPO 机制和同样的奖励黑客问题。

## 2. 阅读顺序

| 顺序 | 教程 | 读完应该能够做到 |
|---|---|---|
| 1 | [工具调用](/notes/agent-tool-calling/) | 画出五步循环，写出 schema 与 `tool_calls`，手算前缀缓存的节省 |
| 2 | [规划与记忆](/notes/agent-planning-memory/) | 对比 ReAct 与先规划再执行，手算记忆检索打分与压缩次数 |
| 3 | [Agent RAG](/notes/agent-rag/) | 手算 RRF，用“是否召回 × 是否答对”四格表定位问题 |
| 4 | [MCP 与多智能体](/notes/agent-mcp-multiagent/) | 写出 `tools/call` 往返，手算单 Agent 与多 Agent 的 token 账 |
| 5 | [Agent 评测](/notes/agent-eval/) | 区分 pass@k 与 pass^k，手算 kappa 与 McNemar |
| 6 | [Agent RL 预备](/notes/recsys-agent-rl/) | 讲清 Search-R1 的检索 token 掩码与 GRPO 组内优势 |
| 7 | [Agent RL 进阶](/notes/agent-rl-advanced/) | 解释长度偏差与 Clip-Higher，说出 DAPO 四个改动的分量 |
| 8 | [Agent 系统设计](/notes/agent-system-design/) | 用固定框架答一道设计题，手算成本、延迟和并发 |

第一遍跟着正文算一遍数字；第二遍遮住答案自己算；第三遍改一个前提（窗口变小、工具变多、成功率变低），看结论怎样变。

## 3. Prompt、框架、后训练的边界

面试中常被问：“一个 Agent 能力不够，你会改 prompt、改框架，还是做后训练？” 三者解决的问题不同：

| 手段 | 改变的是什么 | 适合 | 代价与边界 |
|---|---|---|---|
| Prompt / 上下文工程 | 模型这一次看到的信息与指令 | 模型能力足够，只是不知道规则、格式或背景 | 不改变能力上限；上下文越长越贵，也越容易被稀释 |
| 框架（编排代码） | 调用顺序、工具集、重试、校验、停止条件 | 流程可预知，或需要硬约束（权限、预算、格式校验） | 框架写死的路径越多，灵活性越低；不能修正模型本身的判断错误 |
| 后训练（SFT / RL） | 模型参数，即“在这种情况下倾向于怎么做” | 同类任务大量重复、有可验证的结果、提示已经调不动 | 需要数据和训练成本；可能损害其他能力；要有可靠的评测防止奖励黑客 |

**判断顺序**：先用评测集定位失败类型。格式错、不知道规则，先改 prompt；违反硬约束（越权、超预算、死循环），改框架加检查；同一类判断反复出错，而且能写出验证器，才考虑后训练。后训练的收益要用“同一评测集、同一脚手架”的配对比较来证明，方法见 [Agent 评测](/notes/agent-eval/) 第 6 节。

## 4. 四个手算：把成本和可靠性落到数字上

**① 前缀缓存的节省**（教学构造）。系统提示加工具定义共 1500 token，每轮新增 350 token，共 6 轮。不用前缀缓存时每轮重新 prefill 全部历史：$\sum_{t=1}^{6}(1500+350t)=16\,350$ token；用缓存时只算新增部分：$1500+350\times6=3600$ token，约为前者的 $1/4.5$。前提是前缀逐字节不变，工具列表里插一个时间戳缓存就从那里失效。原理见 [Prefill 与 Decode](/notes/prefill-decode-video-tokens/) 第 2 节。

**② 窗口能撑几步**（教学构造）。窗口 8192 token，系统提示加工具 1500，预留输出 1024，可用于历史的是 $8192-1500-1024=5668$。每步（调用加结果）450 token，最多 $\lfloor5668/450\rfloor=12$ 步（12 步 5400 token，13 步 5850 token 就超了）。所以长任务必须压缩上下文或把信息外部化，见 [规划与记忆](/notes/agent-planning-memory/) 第 7 节。

**③ RL 时模型自己生成的 token 占多少**（教学构造）。一条轨迹：模型 3 轮各写 120 token，最后作答 40 token，共 400 个生成 token；3 次工具返回各 600 token，共 1800 个观测 token。response 部分里模型生成的只占 $400/(400+1800)=18.2\%$。如果不对观测 token 做损失掩码，八成以上的梯度权重会落在工具返回的文本上，这就是 [Agent RL 预备](/notes/recsys-agent-rl/) 第 3 节检索 token 掩码要解决的问题。

**④ 多步可靠性**（教学构造）。每步成功率 0.95、步骤独立时，5 步全对的概率 $0.95^5=0.774$，10 步只剩 $0.95^{10}=0.599$。提升可靠性只有两条路：提高单步成功率，或者加上能发现错误的检查与重试，后者的收益取决于检查器的召回率，见 [规划与记忆](/notes/agent-planning-memory/) 第 4 节。

## 5. 八道高频八股

每道先口述 2 分钟，再对照下面的要点和对应教程。

**1. Prompt、框架与后训练的边界。** 见第 3 节的表：prompt 改信息，框架改流程和硬约束，后训练改模型的倾向；按“格式与规则 → 硬约束 → 反复出现的判断错误”的顺序选择，后训练要有验证器和配对评测。

**2. 记忆方案与多级记忆。** 三层：短期（上下文窗口）、工作（任务内的待办列表、已确认事实、草稿文件）、长期（向量库、数据库，跨会话）。核心是“写什么”和“读什么”：写入时去重、打重要性分；读取时按相关性、时近性、重要性打分取 top-$k$（Generative Agents 的做法，权重要在标注数据上调）。还要处理过时信息的更新与删除。详见 [规划与记忆](/notes/agent-planning-memory/) 第 5–6 节。

**3. Skill 与渐进式披露。** Anthropic 在 2025 年 10 月发布 Agent Skills（同年 12 月作为开放标准发布）：一个 skill 是一个目录，核心是带 YAML frontmatter（必填 `name` 和 `description`）的 `SKILL.md`，可以附带脚本和参考文件。**渐进式披露**分三级：启动时只把每个 skill 的名字和描述放进系统提示；模型判断某个 skill 相关时，才读入完整的 `SKILL.md`；`SKILL.md` 里引用的其他文件，只在需要时再打开。手算（教学构造）：装了 30 个 skill，每个元数据约 100 token、正文约 2000 token。全部预载要 $30\times2100=63\,000$ token；渐进式披露下常驻只有 $30\times100=3000$ token，一个任务用到 2 个 skill 时也只需 $3000+2\times2000=7000$ token。这是“上下文是有限资源”在工具层面的应用。

**4. 任务规划与停止条件。** 规划有 ReAct（每步边想边做）、先规划再执行、两者混合（粗计划加逐步执行，假设被推翻时重新规划）。停止条件要分两层：**学习层**，模型自己判断信息已足够并作答（结果奖励会惩罚过早作答）；**工程层**，硬上限：最大步数、最大 token、最大费用、超时，以及“计划里的待办全部勾掉”这类显式完成条件。评估时报告平均步数和触顶比例。

**5. 从 Function Calling 到 MCP 再到 Skill。** 三者解决的问题不同：Function Calling 是模型侧的能力，按 schema 生成调用；MCP 是应用与工具之间的协议，把 $M$ 个应用接 $N$ 个工具的适配从 $M\times N$ 降到 $M+N$；Skill 是打包“程序性知识”（怎样做一类任务的说明、脚本、模板）的方式，配合渐进式披露控制上下文开销。一个 skill 里可以写“用哪个 MCP 工具、按什么步骤”，三者是叠加关系，不是替代关系。详见 [工具调用](/notes/agent-tool-calling/) 与 [MCP 与多智能体](/notes/agent-mcp-multiagent/)。

**6. 死循环怎么处理。** 先分原因：工具一直报同样的错（错误信息没帮模型做决策）；结果被截断或压缩掉，模型不记得已经调过（上下文问题）；目标本身不可达（任务问题）。对策分三层：检测（同一工具同一参数重复调用、连续 $n$ 步无新信息）、干预（回填明确的错误说明与替代方案、强制换策略或请求用户澄清）、兜底（步数和费用上限、超限后返回部分结果并说明原因）。训练层面可以在 RL 奖励里加轮数代价，但系数要小，否则模型宁可不查直接猜，见 [Agent RL 进阶](/notes/agent-rl-advanced/) 第 6 节。

**7. 上下文过长与上下文衰减。** 两个问题要分开。**过长**是装不下：用压缩、外部化、截断工具返回、子 Agent 隔离来控制。**衰减**是装得下但用不好：Anthropic 的 *Effective context engineering* 称之为 context rot，上下文 token 越多，模型准确回忆其中信息的能力越差；Chroma 的技术报告 *Context Rot* 测了 18 个模型，发现即使在简单任务上，性能也会随输入变长而越来越不稳定；Lost in the Middle 则发现放在中间的信息最容易被忽略。所以“窗口够大”不等于“全塞进去”：只放当前步骤需要的信息，关键信息放在开头或结尾。详见 [规划与记忆](/notes/agent-planning-memory/) 第 7 节和 [Agent RAG](/notes/agent-rag/) 第 3、8 节。

**8. 多智能体什么时候更好。** 三个理由：子任务可以并行、需要上下文隔离、子任务需要差别很大的工具集。Anthropic 的多智能体调研系统在内部评测上比单 Agent 高 90.2%，但 token 用量约是普通对话的 15 倍（单 Agent 约 4 倍），只适合价值足够高的任务。三个理由都不成立时，单 Agent 加好的规划和上下文管理通常更省、更可靠。详见 [MCP 与多智能体](/notes/agent-mcp-multiagent/) 第 6–9 节。

## 6. 覆盖清单

每读完一篇、能闭卷讲清楚，就勾掉一项。

- [ ] 工具调用的五步循环；`tool_calls` 与 `tool` 消息；为什么参数是不可信输入
- [ ] chat 模板怎样渲染工具与调用；解析器何时工作；前缀缓存手算
- [ ] 约束解码能保证什么、不能保证什么；strict 模式的要求
- [ ] SFT 时为什么掩掉工具返回；BFCL 的题型与 AST 判定
- [ ] ReAct 与 CoT 的实测差异；先规划再执行；Reflexion 什么时候有用
- [ ] 三层记忆；Generative Agents 打分手算；压缩、外部化与 MemGPT
- [ ] RAG 每个环节的失败；RRF 手算；四格诊断表；长上下文能否替代 RAG
- [ ] MCP 的三个角色与三类能力；`initialize`、`tools/list`、`tools/call`；安全风险
- [ ] 多智能体的三个理由、四种结构、失败模式；单 Agent 与多 Agent 的 token 账
- [ ] pass@k 与 pass^k 及其无偏估计；LLM 裁判的偏差与 kappa；McNemar 检验
- [ ] Search-R1 的四对标签、检索 token 掩码、EM 奖励、GRPO 组内优势
- [ ] 长度偏差、Clip-Higher、动态采样、超长惩罚；多轮信用分配；奖励黑客
- [ ] Skill 与渐进式披露；Function Calling、MCP、Skill 的关系
- [ ] 死循环的检测、干预与兜底；上下文过长与衰减的区别
- [ ] 系统设计题框架；deep research 主例的成本、延迟与并发手算；多模态客服副例

## 闭卷验收

不看资料说出 workflow 与 agent 的区别和“先找最简单方案”的原则；画出九层知识地图并说出每层对应的教程；用一张表讲清 prompt、框架、后训练各改变什么、适合什么、代价是什么；手算前缀缓存的节省、8192 窗口能撑的步数、RL 轨迹中生成 token 的占比、5 步和 10 步的全对概率；对八道八股每道口述 2 分钟；最后对照覆盖清单，把没勾掉的项按阅读顺序补上。

**参考。** [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)；[Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)；[Anthropic: Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)；[Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)；[Chroma: Context Rot](https://research.trychroma.com/context-rot)；[Model Context Protocol 规范](https://modelcontextprotocol.io/specification)；[ReAct](https://arxiv.org/abs/2210.03629)；[Generative Agents](https://arxiv.org/abs/2304.03442)；[Lost in the Middle](https://arxiv.org/abs/2307.03172)；[Search-R1](https://arxiv.org/abs/2503.09516)；[DAPO](https://arxiv.org/abs/2503.14476)。
