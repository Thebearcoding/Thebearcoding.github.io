---
title: 工具调用：ReAct、Toolformer 与 Function Calling
date: '2026-10-06'
tags: [Agent, 工具调用, ReAct, Toolformer, ToolLLM, Function Calling, 约束解码]
summary: 从 ReAct 的思考-动作-观测轨迹和 Toolformer 的损失过滤讲起，逐字拆开 OpenAI 与 Qwen 的工具调用消息格式，手算一次并行调用的 token 数，再讲 ToolBench/DFSDT、APIGen 怎样造 SFT 数据，以及 JSON schema 与语法约束解码。
draft: false
---

Agent 的“动作”大多是工具调用：搜索、算数、查库、调 API。这一篇回答四个问题：模型怎样在文本里表达“我要调用工具”（ReAct 的轨迹格式、Function Calling 的消息格式）；怎样不靠人工标注教会模型调用（Toolformer 的自监督过滤）；工具调用的 SFT 数据怎样大规模地造出来（ToolBench 的 DFSDT、APIGen 的三级校验）；怎样保证输出一定能被解析（JSON schema 与语法约束解码）。用 RL 训练工具调用见 [Agent RL 预备](/notes/recsys-agent-rl/) 和 [Agent RL 进阶](/notes/agent-rl-advanced/)，规划与记忆见 [规划与记忆](/notes/agent-planning-memory/)，BFCL 等评测见 [Agent 评测](/notes/agent-eval/)，本篇不重复。

阅读入口：[Agent 专题总览](/notes/agent-guide/)。数值算例为教学构造；论文数字注明表号，消息格式注明文档或仓库版本与访问日期（2026-10-06）。

## 1. 问题设定：工具调用的三方接口

一次工具调用涉及三方：**模型**写出“调用哪个函数、参数是什么”；**执行器**（你的代码）解析这段输出、真正执行、把结果写回上下文；**工具**本身是黑盒，输入输出都是文本或 JSON。模型从不直接执行任何东西，它只生成 token。所以“工具调用能力”拆成三件可以分别评测的事：该不该调（相关性判断）、调哪个（选择）、参数对不对（填参与格式）。

在 RL 的语言里，这是一个序列决策过程：状态 $c_t=(o_1,a_1,\dots,o_{t-1},a_{t-1},o_t)$ 是到目前为止的上下文，动作 $a_t$ 是一次工具调用或最终回答，观测 $o_{t+1}$ 是工具返回。ReAct 论文 §2 正是这样写的。后面所有格式，无论纯文本标签还是 JSON，都只是这个循环的不同序列化方式。

## 2. ReAct：思考、动作、观测交替

ReAct（Yao et al., arXiv 2210.03629，ICLR 2023）的核心只有一句话：把动作空间扩充为 $\hat{\mathcal A}=\mathcal A\cup\mathcal L$，$\mathcal L$ 是语言空间。语言动作叫 thought，它不改变环境、不产生观测，只是更新上下文 $c_{t+1}=(c_t,\hat a_t)$，用来分解目标、从观测里摘信息、调整计划（§2）。

**轨迹格式。** 知识问答任务（HotpotQA、FEVER）上，论文设计了一个很弱的维基百科 API，只有三种动作（§3.1）：`search[entity]` 返回词条页前 5 句，词条不存在时返回最相近的 5 个实体名；`lookup[string]` 返回页面里下一句含该字符串的句子，相当于浏览器的 Ctrl+F；`finish[answer]` 结束任务。一条轨迹是若干个 Thought / Action / Observation 三元组：

```
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
```

（取自论文附录 C.1 的 HotpotQA ReAct 提示，原文排版为两栏表格。）知识类任务每步都写 thought（dense），ALFWorld、WebShop 这类动作很长的任务只在关键处写（sparse），由模型自己决定。执行器的工作是：模型写完 `Action k: ...` 就停下，解析动作、执行，把 `Observation k: ...` 拼回去再继续生成。

**设置与结论**（PaLM-540B，少样本提示）。HotpotQA 用 6 条、FEVER 用 3 条人工写的轨迹作示例，作者注明再多不提升（§3.2 脚注 2）。

| 方法 | HotpotQA EM | FEVER Acc |
|---|---:|---:|
| Standard | 28.7 | 57.1 |
| CoT | 29.4 | 56.3 |
| CoT-SC（21 条采样） | 33.4 | 60.4 |
| Act（去掉 thought） | 25.7 | 58.9 |
| ReAct | 27.4 | 60.9 |
| CoT-SC → ReAct | 34.2 | **64.6** |
| ReAct → CoT-SC | **35.1** | 62.0 |

（Table 1。）三点结论：① ReAct 两项都优于 Act，说明 thought 对“怎样行动”有用；② HotpotQA 上 ReAct 略低于 CoT（27.4 对 29.4），FEVER 上更高；③ 两者结合最好：ReAct 7 步（FEVER 5 步）内没答出就退回 CoT-SC，或 CoT-SC 多数票不过半就转 ReAct。决策任务上提升更明显：ALFWorld 最好一组提示的成功率 71%，Act 为 45%，模仿学习基线 BUTLER 为 37%（Table 3）；WebShop 成功率 40.0%，Act 30.1%，IL 29.1%，IL+RL 28.7%（Table 4）。

**失败模式**（Table 2，人工标注 200 条 HotpotQA 轨迹）。CoT 失败的 56% 是幻觉；ReAct 失败里 47% 是推理错误，其中有一种 ReAct 特有的模式：反复生成之前的 thought 和 action，跳不出循环；23% 是搜索返回空或无用信息，之后模型很难恢复。这两条正是工程上要加“重复动作检测”和“最大步数”的原因。

**微调。** 作者用 ReAct 生成的 3,000 条答对的轨迹微调 PaLM-8B/62B：提示时 ReAct 在小模型上是四种方法里最差的，微调后变成最好的，8B 微调版超过 62B 的所有提示方法（§3.3、Fig. 3）。这是“工具调用要靠训练而非提示”的早期证据，也是后面 ToolBench 这类数据集的出发点。

**边界。** ReAct 的格式是纯文本约定，`Search[...]` 里出现方括号、换行或模型漏写 `Action` 前缀都会解析失败；观测直接拼进上下文，长网页会挤爆窗口（截断策略见 [规划与记忆](/notes/agent-planning-memory/)）。**验证**：除了任务指标，统计每条轨迹的步数、无效动作率和“重复动作”比例；做 thought 消融（即 ReAct 对 Act）。

## 3. Toolformer：让模型自己标注哪里该调用

ReAct 靠提示，Toolformer（Schick et al., arXiv 2302.04761，NeurIPS 2023）则把调用写进训练数据，而且不要人工标注：让模型自己在预训练文本里插入候选调用，只保留**确实让后文更好预测**的那些，再在这份数据上微调。

**表示。** 调用 $c=(a_c,i_c)$ 线性化为 $e(c)=$ `<API>` $a_c(i_c)$ `</API>`，带结果为 $e(c,r)=$ `<API>` $a_c(i_c)\to r$ `</API>`。实际实现里这三个特殊符号用已有的 `[`、`]`、`->` 表示，不改词表（§2 脚注 1）。例：`Out of 1400 participants, 400 (or [Calculator(400 / 1400) → 0.29] 29%) passed the test.`

**三步构造数据**（§2、Fig. 2）：

1. **采样位置与调用。** 用带少量示例的提示 $P(x)$ 让模型读文本 $x$，计算每个位置开始调用的概率 $p_i=p_M(\texttt{<API>}\mid P(x),x_{1:i-1})$，保留 $p_i>\tau_s$ 的位置，最多 top-$k$ 个；每个位置采样最多 $m$ 个候选调用。
2. **执行**，得到结果 $r_i$。
3. **按损失过滤。** 定义从位置 $i$ 起的加权交叉熵

$$
L_i(z)=-\sum_{j=i}^{n}w_{j-i}\,\log p_M(x_j\mid z,x_{1:j-1}),
$$

$z$ 是放在前面的前缀。比较两种情况：

$$
L_i^{+}=L_i\big(e(c_i,r_i)\big),\qquad L_i^{-}=\min\big(L_i(\varepsilon),\ L_i(e(c_i,\varepsilon))\big),
$$

只保留 $L_i^{-}-L_i^{+}\ge\tau_f$ 的调用。$L_i^-$ 取“不调用”和“调用但不给结果”中较小的一个，这样能排除“只是参数文本本身帮了忙”的调用。权重让调用只对附近的 token 负责（§4.1）：

$$
w_t=\frac{\tilde w_t}{\sum_s\tilde w_s},\qquad \tilde w_t=\max(0,\ 1-0.2\,t).
$$

一个容易漏的细节：计算损失时 $e(c_i,r_i)$ 是**作为前缀**放在整段文本前面，而不是插在位置 $i$，因为此时模型还没见过插在文中间的调用，硬插会拉高困惑度（§2 脚注 3）。过滤通过后，再把调用插回原位置得到 $x^*$，用普通语言模型目标微调。

**超参**（附录 A）。默认 $\tau_s=0.05$、$\tau_f=1.0$、$k=5$、$m=5$；计算器和翻译只在启发式筛过的小子集上生成，所以改为 $\tau_s=0$、$k=20$、$m=10$，并放宽到 $\tau_f=0.5$。基座 GPT-J（6.7B），语料是 CCNet 子集；五个工具是问答（Atlas）、维基搜索（BM25）、计算器、日历、翻译（NLLB 600M）。

**手算：一次过滤判断**（教学构造，仿 Fig. 2 的 Pittsburgh 例子）。文本 “Pittsburgh is also known as **the Steel City.**”，在 “as” 之后的位置 $i$ 有两个候选：$c^1=$ QA(“What other name is Pittsburgh known by?”) → “Steel City”；$c^2=$ QA(“Which country is Pittsburgh in?”) → “United States”。

先算权重：$\tilde w=(1,0.8,0.6,0.4,0.2,0,\dots)$，和为 3.0，归一化后 $w_0..w_4=(0.3333,0.2667,0.2,0.1333,0.0667)$，第 6 个及以后的 token 权重为 0。设后 5 个 token 的负对数似然如下：

| 前缀 | the | Steel | City | . | 下一词 | 加权损失 |
|---|---:|---:|---:|---:|---:|---:|
| 无调用 $\varepsilon$ | 0.5 | 4.0 | 1.0 | 0.3 | 0.8 | 1.5267 |
| $e(c^1,\varepsilon)$ | 0.5 | 3.8 | 1.0 | 0.3 | 0.8 | 1.4733 |
| $e(c^1,\text{Steel City})$ | 0.4 | 0.3 | 0.2 | 0.3 | 0.8 | 0.3467 |
| $e(c^2,\text{United States})$ | 0.5 | 3.9 | 1.0 | 0.3 | 0.8 | 1.5000 |

例如第一行：$0.5\times0.3333+4.0\times0.2667+1.0\times0.2+0.3\times0.1333+0.8\times0.0667=1.5267$。对 $c^1$：$L^-=\min(1.5267,1.4733)=1.4733$，$L^+=0.3467$，差 1.1267，大于等于 $\tau_f=1.0$，保留；若 $\tau_f=2.0$ 则丢弃。对 $c^2$（为简化，设它的“不给结果”损失也是 1.4733），$L^-=1.4733$，$L^+=1.5000$，差为 $-0.0267$，丢弃。这正是 Fig. 2 的判断：答案碰巧出现在后文的调用才被留下。

**边界。** ① $\tau_f$ 决定数据量：问答工具在 $\tau_f=0.5/1.0/2.0$ 下分别保留 51,987 / 18,526 / 5,135 条，计算器只有 3,680 / 994 / 138 条（Table 2）。② 过滤准则衡量的是“对预测后文有用”，不是“事实正确”：论文 Table 10 里有一个与上下文无关、却因降低困惑度而被保留的搜索调用（得分 0.92）。③ 每条调用独立采样，所以模型学不会“先查日期再拿日期去问答”这种链式调用（§4.2.5）。④ 能力随规模出现：GPT-2 系列约 775M 参数起才明显受益（§4.4、Fig. 4）。

**验证。** 推理时只要 `<API>` 进入 top-$k$ 就开始调用，论文用 $k=10$，且每个输入最多调用一次（§4.2）。结果：LAMA 的 T-REx 子集 53.5（GPT-J 31.9，GPT-3 175B 39.8，Table 3）；数学题 ASDiv / SVAMP / MAWPS 为 40.4 / 29.4 / 44.0，超过 GPT-3（Table 4）；QA 仍落后 GPT-3（Table 5）；关掉 API 时 WikiText 与 CCNet 困惑度与只在 CCNet 上微调的模型相同（Table 8），说明没损伤语言建模。Table 9 的 $k$ 消融显示 $k=1$ 时 T-REx 上只有 40.3% 的样本调用工具，$k=10$ 时为 98.1%。

## 4. Function Calling 的消息格式

现在的主流做法不再让模型写 `Search[...]` 这种自由文本，而是在 API 层面把工具声明为 JSON Schema，模型输出结构化的调用，执行结果用专门的角色写回。下面以官方文档为准。

**工具声明**（两家相同的骨架）：

```json
{"type": "function", "function": {"name": "get_weather", "description": "Get current weather of a city.",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
```

**OpenAI 兼容的 Chat Completions 消息**（以 Qwen 文档里用 OpenAI 客户端的示例为准）：助手消息带 `tool_calls` 数组，每项有 `id`、`type: "function"`、`function.name`、`function.arguments`；执行结果用 `{"role": "tool", "tool_call_id": ..., "content": ...}` 回传。**OpenAI Responses API**（*Function calling* 指南，developers.openai.com，访问于 2026-10-06；该指南现在只给 Responses 的示例）里调用是输出数组中的一个条目 `{"type": "function_call", "call_id": "call_12345xyz", "name": "get_weather", "arguments": "{\"location\":\"Paris, France\"}"}`，结果用 `{"type": "function_call_output", "call_id": ..., "output": ...}` 回传。注意 `arguments` 是 **JSON 编码后的字符串**，不是对象。其他要点：`tool_choice` 可取 `auto`（默认，调零个或多个）、`required`（至少调一个）、`none`、指定函数或 `allowed_tools`；`parallel_tool_calls` 设为 false 时一轮最多调一个；`strict: true` 要求每个 object 都写 `additionalProperties: false`、`properties` 里的字段全部列进 `required`，可选字段写成 `["string","null"]`。指南写明 Responses 会尽量把 schema 规范化为 strict，Chat Completions 默认仍是非 strict；并提醒微调模型在一轮里调用多个函数时，这些调用的 strict 会被关闭。

**Qwen**（Hermes 风格）。Qwen 文档推荐对 Qwen3 使用 Hermes 风格的工具调用，并给出 vLLM 部署命令 `vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes`；vLLM 文档也写明 Qwen2.5 的 chat template 已内置 Hermes 风格，用 `hermes` 解析器。消息列表与 OpenAI 一致（`assistant` 带 `tool_calls`，`tool` 带结果），差别在**模板渲染成什么 token**。下面是用 `Qwen/Qwen2.5-7B-Instruct` 的 `tokenizer_config.json`（HF 仓库提交 a09a354，2025-01-12）里的 chat template 渲染的实际文本：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_weather", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
北京和上海今天哪个更热？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"city": "北京"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "上海"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temp_c": 31}
</tool_response>
<tool_response>
{"temp_c": 28}
</tool_response><|im_end|>
<|im_start|>assistant
北京更热：31°C 对 28°C。<|im_end|>
```

四个要点。① 工具声明被写进 system 提示，模型“看到”的工具就是这段 JSON 文本，所以 `description` 写得好坏直接影响选择。② 调用写在 `<tool_call>` 标签里，一轮可以连写多个，这就是并行调用。③ 工具结果没有独立角色：连续的 `tool` 消息被合并进**一个 `user` 轮**，各自包在 `<tool_response>` 里。④ 结果和调用之间没有 id 对应，只靠顺序；OpenAI 格式则靠 `tool_call_id` 配对。

**一个格式坑。** Qwen2.5 模板对参数做的是 `tool_call.arguments | tojson`。如果你照 OpenAI 的习惯把 `arguments` 存成字符串，渲染出来是 `"arguments": "{\"city\": \"北京\"}"`，被二次转义，训练和推理的格式就不一致了。Qwen3-8B 的模板（提交 b968826，2025-07-26）加了判断：字符串原样输出，否则 `tojson`。所以自己拼 SFT 数据时，要先确认用的是哪一版模板，再决定把参数存成对象还是字符串，并用渲染结果做单测。

## 5. 并行调用：消息序列与 token 计数

**手算**（教学构造；token 数用 Qwen2.5-7B-Instruct 的 tokenizer 实测，上面那段渲染文本）。问题“北京和上海今天哪个更热”需要两次独立的查询。

| 段 | 谁写 | token 数 |
|---|---|---:|
| system（含 1 个工具声明） | 模板 | 155 |
| user 问题 | 数据 | 13 |
| 助手轮头（im_start、assistant、换行） | 模板 | 3 |
| 两个 `<tool_call>` 块 + 中间换行 + 轮尾 im_end | **模型** | 19 + 1 + 19 + 1 = 40 |
| 换行、tool 轮头、两段 `<tool_response>`、轮尾、下一个助手头 | 模板与环境 | 1 + 39 = 40 |
| 最终回答 + 轮尾 im_end | **模型** | 14 |
| 末尾换行 | 模板 | 1 |

单个调用 `<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>` 是 19 个 token，其中 `<tool_call>`、`</tool_call>` 各是一个特殊 token，“北京”一个 token。整段 267 个 token；模型自己写的是 $40+14=54$ 个，SFT 时只有这 54 个（加上你选择计入的轮尾）参与损失，其余是模板、用户和工具返回，要掩掉，道理同 [Search-R1 的检索 token 掩码](/notes/recsys-agent-rl/)。

**并行对串行。** 同一件事改成串行（先查北京、拿到结果、再查上海）：

| | 生成调用次数 | 每次的提示长度 | 提示总长（无前缀缓存） | 模型生成 token | 序列总长 |
|---|---:|---|---:|---:|---:|
| 并行 | 2 | 171、252 | 423 | 40 + 14 = 54 | 267 |
| 串行 | 3 | 171、216、261 | 648 | 20 + 20 + 14 = 54 | 276 |

生成的 token 一样多，差别在**往返次数**：并行少一次 LLM 调用，两个工具可以同时执行，端到端延迟大约少一个“prefill + 工具执行”的周期；有前缀缓存（prefix caching）时重复 prefill 的代价会小很多，但往返的串行等待省不掉。prefill 与 decode 的成本模型见 [prefill 与 decode](/notes/prefill-decode-video-tokens/)。

**边界。** ① 只有彼此**无依赖**的调用能并行；“先查航班号再查航班状态”必须串行，模型把有依赖的调用并行写出来时，第二个参数只能靠编。② 并行结果靠顺序对应（Qwen）时，执行器必须按调用顺序回填，异步执行完成顺序不同也要重排。③ 有副作用的工具（下单、发邮件）并行调用可能重复执行，需要幂等键或干脆关掉并行。**验证**：构造“可并行 / 必须串行”两类测试题，分别统计调用数是否正确、依赖是否被违反；APIGen 造数据时就按 simple、multiple、parallel、parallel multiple 四种风格分开生成（§3.3），评测也应同样分桶（基准细节见 [Agent 评测](/notes/agent-eval/)）。

## 6. 工具调用 SFT 数据怎么造

**ToolLLM / ToolBench**（Qin et al., arXiv 2307.16789，ICLR 2024）。三步（§2）：

1. **收集 API。** 从 RapidAPI 爬到 10,853 个工具（53,190 个 API），过滤掉 404、内部错误等不可用的，保留 3,451 个工具、16,464 个 REST API，覆盖 49 个类别（§2.1）。
2. **生成指令。** 用 ChatGPT（gpt-3.5-turbo-16k）根据 API 文档和 3 条人写种子示例生成指令，并给出相关 API。分三类：单工具 I1；同类别多工具 I2；同 collection 多工具 I3，后两类从同一类别或 collection 里取 2–5 个工具、每个工具最多 3 个 API。去掉“相关 API”是幻觉的样本后，得到 87,413 / 84,815 / 25,251 条（§2.2）。
3. **标注解答路径。** 把每个 API 当作 ChatGPT 的 function，多轮对话里真实调用 API；另加两个函数 “Finish with Final Answer” 和 “Finish by Giving Up”（§2.3）。

**DFSDT**（depth-first search-based decision tree）。作者发现 ReAct 有两个问题：错误会传播（反复用错一个 API、编造 API），而且只探索一条路径。DFSDT 把轨迹组织成一棵树：模型可以沿当前路径继续，也可以调用 “Finish by Giving Up” 放弃当前节点，回到上层展开新的子节点；展开时把已有兄弟节点的信息放进提示，明确要求生成一个不同的节点。用深度优先而不是广度优先，是因为只要找到一条成功路径就够了，广度优先会花太多 API 调用（§2.3、附录 A.8）。只保留成功的路径，最终得到 126,486 条（指令，路径）对，真实 API 调用 469,585 次，平均推理轨迹 4.0 步（Table 1）。

**手算：为什么 DFSDT 比“多跑几次 ReAct”好**（教学构造）。一道题需要先调 A 再调 B。ReAct 第一步 70% 选错成 A′，选错后会在 A′ 上打转直到步数用完。重跑 3 次 ReAct，三次都失败的概率 $0.7^3=0.343$，通过率 65.7%，花 3 条完整轨迹的预算。DFSDT 在 A′ 返回无用结果后放弃、回溯，并被提示“换一个不同的动作”，第二个子节点只要不再选 A′ 就会走到 A。两者的差别在于：重跑是独立同分布的重复采样，同一个高概率错误会反复出现；回溯加“与兄弟不同”的提示，相当于不放回地采样。

**结论**（以下都是 ChatGPT 当评委的 ToolEval 指标：pass rate 是在预算内完成的比例；win rate 是与 ChatGPT-ReACT 的解答做成对比较的胜率；与人工标注的一致率分别为 87.1% 和 80.3%，§3.1）。在同等 API 花费下比较（ReACT@N 即重复跑 ReAct 直到花费与 DFSDT 持平），平均 pass rate：ReACT 35.3，ReACT@N 44.5，DFSDT 63.8，在更难的 I2、I3 上差距更大（Table 3）。在 LLaMA-2 7B 上用这些数据微调（位置插值把上下文从 4096 扩到 8192），ToolLLaMA + DFSDT 平均 pass 66.7、win 60.0；ChatGPT + DFSDT 为 64.8 / 64.3，GPT-4 + DFSDT 为 71.1 / 70.4；没训练过的 Vicuna、Alpaca 两项都是 0（Table 4）。

**APIGen**（Liu et al., arXiv 2406.18518，NeurIPS 2024 Datasets and Benchmarks）。思路是“生成后层层校验”（§3.2）：**格式检查**，即 JSON 能否解析、函数和参数是否存在；**执行检查**，即真的执行，过滤超时、类型错误、运行时错误；**语义检查**，由 LLM 判断执行结果是否与问题意图一致，调用次数和函数选择是否合理。数据源是 3,673 个可执行 API（3,539 个从 ToolBench 过滤得到的 REST API 和 134 个 Python 函数），覆盖 21 个类别，按 simple / multiple / parallel / parallel multiple 四种风格生成，公开 60,000 条（§4.1）。用它训练的 xLAM-7B（FC）在论文写作时的 BFCL 排名第 6，xLAM-1B 排第 24，超过 GPT-3.5-Turbo 与 Claude-3 Haiku（Table 2）。

**手算：校验漏斗**（教学构造）。生成 10,000 条，格式通过 90%，执行通过其中 80%，语义通过其中 70%：$10000\times0.9\times0.8\times0.7=5040$ 条。如果去掉执行检查，就会混进约 $9000\times0.2=1800$ 条“格式对但跑不通”的样本，模型学到的是“写得像”而不是“能执行”。

**边界。** ① ToolBench 的 API 会变：RapidAPI 上的服务会下线或改返回，论文自己也说这是用 ChatGPT 评测而不是固定标准答案的原因之一（§3.1），复现时数字会漂移。② 生成器即评委：ChatGPT 既造数据又当 ToolEval 评委，存在偏向自身风格的风险。③ 只保留成功路径，模型很少见到“工具报错后怎样恢复”的样本；可以有意保留“报错 → 修正参数 → 成功”的片段。④ 拒绝调用的样本（无关问题不该调工具）要单独构造，否则模型会过度调用。

**验证。** 按 I1/I2/I3 或 simple/parallel 分桶报告；做数据消融（去掉 DFSDT 路径只留 ReAct 路径、去掉执行检查）；对 held-out 工具（未见过的 API 和类别）单独评测，ToolLLM 的 Inst./Tool/Cat. 三级泛化就是这个设计（§3.2）。

## 7. 结构化输出与约束解码

微调和提示只能让模型**大概率**输出合法 JSON。约束解码让它**必然**合法：每一步生成前，根据当前已生成的前缀和目标格式，算出词表里哪些 token 合法，把其余 token 的 logit 设成 $-\infty$，再采样。

$$
\tilde p(v\mid y_{<t})=\frac{\exp(z_v)\,\mathbb 1[v\in\mathcal V_{\text{allowed}}(y_{<t})]}{\sum_{u\in\mathcal V_{\text{allowed}}(y_{<t})}\exp(z_u)}.
$$

**怎样算合法集合。** 正则和 JSON Schema（大多可化成正则）对应有限状态机。Outlines（Willard & Louf, arXiv 2307.09702）把生成重写为 FSM 状态之间的转移，并预先对词表建索引：每个 FSM 状态下哪些 token 能走通，事先算好，解码时查表即可。嵌套任意深的 JSON 或代码需要上下文无关文法，状态里要带栈（下推自动机）。XGrammar（Dong et al., arXiv 2411.15100，MLSys 2025）把 token 分成**与上下文无关**的（只看当前位置就能预先判定）和**与上下文相关**的（要看栈，运行时检查），前者预计算，后者用持久化栈加速，并与 GPU 推理重叠执行；摘要称比已有方案最多快 100 倍，端到端开销接近零。

**在推理框架里怎么用**（vLLM latest 文档，访问于 2026-10-06）。结构化输出支持 `choice`、`regex`、`json`、`grammar`、`structural_tag` 五种约束，OpenAI 兼容服务里用 `response_format={"type": "json_schema", ...}` 或 `extra_body={"structured_outputs": {...}}` 传入；后端有 xgrammar、guidance、outlines、lm-format-enforcer，默认 `auto` 自动选择。工具调用方面，指定函数（named function calling）时 vLLM 用结构化输出保证参数符合 schema；`tool_choice="required"` 保证至少生成一个调用；`auto` 模式下，只有满足 strict 相关条件（工具设 `strict: true` 等）且解析器支持 structural tag 时才约束调用格式，否则就从原始文本里解析，文档原话是参数可能偶尔不合法或违反 schema。

**手算：枚举约束**（教学构造）。schema 规定 `city` 只能取 `["北京","上海"]`。已生成 `{"city": "`，下一步四个候选 token 的 logit：北京 2.0，上海 1.0，广州 1.5，`"` 0.5。不约束时 softmax 为 (0.455, 0.167, 0.276, 0.102)，有 $0.276+0.102=0.378$ 的概率写出非法值。约束后只在 {北京, 上海} 上重新归一化：$e^{2}/(e^{2}+e^{1})=0.731$，$e^1/(e^2+e^1)=0.269$。

**边界。** ① 约束保证格式，不保证正确：上例里如果用户问的其实是广州，约束只会逼模型在两个错误答案里挑一个。② 约束会改变分布：逐 token 截断再归一化，不等于“在所有合法序列上按原模型概率采样”，模型原本想走的高概率路径一旦被截断，后续可能写出合法但别扭的内容；对推理模型，通常只约束最终答案段，不约束思考段（vLLM 对部分推理模型需要显式打开“在推理中启用”的开关）。③ 词表与格式边界不对齐：一个 token 可能同时跨越引号和字母，“哪些 token 合法”必须在字节级判断，这正是 XGrammar 要处理上下文相关 token 的原因。④ OpenAI 的 strict 要求所有字段 required，可选参数要改写成可空类型，schema 设计会跟着变。

**验证。** 对同一批请求比较三种设置：无约束、仅 JSON 模式、schema 约束，报告解析成功率、schema 合规率、**任务正确率**和每 token 延迟。只看解析率会高估约束的收益。语义 ID 生成式推荐里的前缀树约束解码是同一机制的另一种用法，见 [语义 ID](/notes/recsys-semantic-id/)。

## 8. 训练工具调用模型时的几条经验

这一节是工程总结，不对应单篇论文。

**损失掩码。** 只对助手写的 token（调用和最终回答）算损失，system 里的工具声明和 `<tool_response>` 里的返回都掩掉；多轮样本里每一轮助手输出都计入，不只算最后一轮（LoRA 等细节见 [多模态 SFT、LoRA 与 DPO](/notes/multimodal-sft-lora-dpo/)）。**负样本。** 显式构造“不该调用”“没有合适工具”“参数缺失需反问用户”三类样本，只训正例的模型会对什么都调工具。**工具描述扰动。** 打乱工具顺序、改写描述、加入无关工具，防止模型记住位置。**格式一致性。** 训练用的 chat template 必须与部署时的解析器一致（Qwen 的 `<tool_call>` 对应 vLLM 的 `hermes` 解析器），用第 4 节的渲染单测把关。**评测分层。** 分开报告调用判断、函数选择、参数精确匹配、可执行率和端到端成功率。

## 9. 面试常问

**ReAct 和 Function Calling 是什么关系？** ReAct 是一种轨迹组织方式（思考、动作、观测交替），Function Calling 是动作的一种序列化格式（JSON 调用加专门的结果角色）。现代 Agent 通常两者都用：思考写在助手文本或推理段里，动作用 `tool_calls` 表达。ReAct 的实验结论（thought 有助于决策、搜索失败难恢复、会陷入重复动作）在新格式下依然成立。

**Toolformer 的过滤准则为什么要和“调用但不给结果”比？** 如果只和“不调用”比，一个调用可能仅仅因为参数文本里含有后文的词就降低了损失，结果本身其实没用。取 $\min(L_i(\varepsilon),L_i(e(c_i,\varepsilon)))$ 把这种情况排除掉，要求**结果**本身至少再降 $\tau_f$。加权让远处 token 的权重为 0，保证调用只对邻近的 5 个 token 负责。

**DFSDT 和 ReAct@N 都多花钱，为什么 DFSDT 更好？** 重复跑 ReAct 是独立采样，高概率的错误动作会一次次被选中；DFSDT 回溯后提示“生成与已有子节点不同的动作”，相当于不放回地探索。Table 3 在同等花费下平均 pass rate 63.8 对 44.5，难题（I2、I3）上差距更大。它只用于造数据，推理时也能用，但成本高。

**约束解码能替代工具调用 SFT 吗？** 不能。约束只保证语法合法，管不了选哪个工具、填什么值、该不该调用；而且强约束一个没学过该格式的模型，会把它推到低概率区域，输出合法但语义差。实践是 SFT 让模型学会格式和决策，约束解码兜底保证可解析。

**并行调用有什么风险？** 有依赖的调用被并行写出时参数只能靠猜；有副作用的工具可能重复执行；结果按顺序配对时乱序回填会张冠李戴。对策是让模型只并行无依赖调用（训练数据里区分两类），执行器保序，副作用工具加幂等键或禁止并行。

## 闭卷验收

不看资料写出 ReAct 的扩充动作空间 $\hat{\mathcal A}=\mathcal A\cup\mathcal L$ 和 HotpotQA 的三种动作，说出 Table 1 里 ReAct 与 CoT 在两个数据集上谁高谁低、两种组合策略的切换条件，以及 Table 2 中 ReAct 的两类主要失败；写出 Toolformer 的 $L_i(z)$、$L_i^+$、$L_i^-$ 与过滤条件，算出 $\tilde w_t=\max(0,1-0.2t)$ 归一化后的权重，并对一个给定的负对数似然表判断调用是否保留，说出默认 $\tau_s,\tau_f,k,m$；默写一条 OpenAI 格式的 `tool_calls` 与 `tool` 消息，以及 Qwen 模板里 `<tools>`、`<tool_call>`、`<tool_response>` 的位置，说出 `arguments` 存成字符串在 Qwen2.5 模板下会出什么问题；对一次两路并行调用列出消息序列，指出哪些 token 计入损失，比较并行与串行的调用次数和提示总长；说出 ToolBench 的 API 数、类别数、样本数和 I1/I2/I3 的含义，讲清 DFSDT 怎样回溯、为什么用深度优先、ReACT@N 是什么对照；说出 APIGen 的三级校验；最后用一个枚举约束手算约束前后的概率，并说明约束解码保证什么、不保证什么。

**参考。** [ReAct](https://arxiv.org/abs/2210.03629)；[Toolformer](https://arxiv.org/abs/2302.04761)（[NeurIPS 2023](https://ai.meta.com/research/publications/toolformer-language-models-can-teach-themselves-to-use-tools/)）；[ToolLLM](https://arxiv.org/abs/2307.16789)（[ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html)，[ToolBench 仓库](https://github.com/OpenBMB/ToolBench)）；[APIGen](https://arxiv.org/abs/2406.18518)（[NeurIPS 2024 D&B](https://proceedings.neurips.cc/paper_files/paper/2024/hash/61cce86d180b1184949e58939c4f983d-Abstract.html)）；[OpenAI Function calling 指南](https://developers.openai.com/api/docs/guides/function-calling)；[Qwen 文档：Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)；[Qwen2.5-7B-Instruct tokenizer_config.json](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)；[Qwen3-8B tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json)；[vLLM：Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling.html)、[Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)；[Outlines / Efficient Guided Generation](https://arxiv.org/abs/2307.09702)；[XGrammar](https://arxiv.org/abs/2411.15100)。
