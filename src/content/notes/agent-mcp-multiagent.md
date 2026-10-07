---
title: MCP 协议与多智能体编排
date: '2026-10-06'
tags: [Agent, MCP, 多智能体, AutoGen, MetaGPT, CAMEL, JSON-RPC]
summary: 按官方规范讲 MCP 的角色、能力协商、三类原语、tools/list 与 tools/call 的消息形状和传输；再讲 AutoGen、MetaGPT、CAMEL、多智能体辩论的核心机制，手算通信与 token 成本，最后讨论多智能体什么时候不如单智能体。
draft: false
---

一个 Agent 要接很多外部能力：文件、数据库、搜索、内部 API。每接一个都写一套胶水代码，维护成本会随“应用数 × 工具数”增长。这一篇分两半讲。前一半是 **MCP**（Model Context Protocol）：它把“工具怎么被发现、怎么被调用、结果长什么样”定成一份 JSON-RPC 协议，全部以 modelcontextprotocol.io 的官方规范为准。后一半是**多智能体编排**：AutoGen、MetaGPT、CAMEL 和多智能体辩论各自的核心机制，以及什么时候拆成多个智能体反而更差。函数调用的消息格式和工具调用 SFT 数据见 [工具调用](/notes/agent-tool-calling/)，规划与记忆见 [规划与记忆](/notes/agent-planning-memory/)，本篇不重复。

阅读入口：[Agent 专题总览](/notes/agent-guide/)。数值算例为教学构造；论文数字注明表号或图号。MCP 部分依据**当前版本 2026-07-28**，并对照上一版 **2025-11-25**（两版差别很大，面试时要说清你讲的是哪一版）。

## 1. MCP 解决什么问题：角色与边界

**角色**（规范 *Architecture*）。MCP 是 client-host-server 架构：

| 角色 | 是什么 | 职责 |
|---|---|---|
| Host | LLM 应用本身（IDE、聊天客户端） | 创建、管理多个 client；执行安全策略与用户授权；协调 LLM 调用；**汇总各 client 的上下文** |
| Client | host 内部的连接器 | 与**恰好一个** server 一对一通信；在每个请求上附带协议版本和能力 |
| Server | 提供上下文和能力的服务 | 通过 resources、tools、prompts 三类原语暴露能力；可以是本地进程或远程服务 |

规范的设计原则里有一条值得背下来：server **不能读到完整对话，也不能“看进”其他 server**。完整历史留在 host，跨 server 的交互由 host 控制。所以 MCP server 拿到的只有一次请求的参数，它并不知道模型此前想了什么。

**为什么需要协议**（以下是本文的分析）。没有统一协议时，$M$ 个应用接 $N$ 个工具源，最坏要写 $M\times N$ 份适配；有了协议，每个应用实现一次 client、每个工具源实现一次 server，工作量变成 $M+N$。规范自己的类比是 LSP（语言服务器协议）：编辑器和语言支持之间也是靠一份协议解耦的。

**消息层**。所有消息都必须遵守 JSON-RPC 2.0，分三种：请求（带 `id`，且 MCP 规定 `id` 不能为 `null`）、响应（结果或错误，`id` 与请求相同）、通知（没有 `id`，接收方不回复）。

## 2. 版本与能力协商：从 initialize 握手到逐请求声明

这是两版规范差别最大的地方。

**2025-11-25 及更早（规范称为 legacy）：先握手。** 生命周期分三段：初始化、正常操作、关闭。初始化必须是双方的第一次交互，客户端发 `initialize`，携带协议版本、客户端能力和客户端信息（摘自规范 *Lifecycle*，删去了 icons 等字段）：

```json
{"jsonrpc":"2.0","id":1,"method":"initialize",
 "params":{"protocolVersion":"2025-11-25",
           "capabilities":{"roots":{"listChanged":true},"sampling":{},"elicitation":{"form":{},"url":{}}},
           "clientInfo":{"name":"ExampleClient","version":"1.0.0"}}}
```

服务端回复自己的版本、能力（如 `"tools":{"listChanged":true}`、`"resources":{"subscribe":true,"listChanged":true}`、`"prompts":{...}`、`"logging":{}`）和 `serverInfo`，可选 `instructions`。客户端再发一条通知 `notifications/initialized`，之后才进入正常操作。版本协商规则：客户端发自己支持的最新版本；服务端支持就原样返回，不支持就返回自己支持的另一个版本；客户端不支持服务端给的版本，就应当断开。之后双方只能使用协商成功的能力。

**2026-07-28（当前版，规范称为 modern）：没有握手。** 规范明确写“MCP 是无状态协议”，服务端**不能**依赖同一连接上之前的请求来确定版本、能力或客户端身份。每个请求在 `params._meta` 里自带：

| `_meta` 键 | 必需 | 含义 |
|---|---|---|
| `io.modelcontextprotocol/protocolVersion` | 是 | 本请求使用的协议版本，如 `"2026-07-28"` |
| `io.modelcontextprotocol/clientCapabilities` | 是 | 与本请求相关的客户端能力 |
| `io.modelcontextprotocol/clientInfo` | 否（SHOULD 带） | 客户端名称与版本 |

缺少必需字段的请求按 `-32602`（Invalid params）拒绝。服务端不支持请求里的版本时，返回 `-32022 UnsupportedProtocolVersion`，`data.supported` 列出它支持的版本，客户端换一个再重试。服务端必须实现 `server/discover`，它一次返回支持的版本、能力、`serverInfo` 和 `instructions`；客户端可以先调它，也可以直接发业务请求、遇到版本错误再处理。处理请求需要客户端没声明的能力时，服务端返回 `-32021 MissingRequiredClientCapability`。跨请求的状态（例如购物车）要由服务端返回一个显式句柄，模型在后续调用的参数里带上它。

**为什么这样改**（本文的分析）：带会话的协议要求同一客户端的请求落到持有会话的那台机器上，远程 server 横向扩容和负载均衡都更麻烦；逐请求自描述后，任何一台实例都能独立处理任何请求。代价是每个请求多带几十个字节的元数据（见第 4 节手算）。

**兼容**：规范给了双版本实现的探测规则。stdio 上先发 `server/discover`，收到的不是可识别的新版错误就回退到 `initialize`；HTTP 上先发新版请求，看 `400` 的响应体再决定。老客户端连只支持新版的服务端会直接失败，因为老客户端没有向前兼容的机制。

## 3. 三类原语：谁来决定用它

规范 *Server Features* 按“由谁控制”区分三类原语：

| 原语 | 控制方 | 典型形态 | 方法 |
|---|---|---|---|
| Prompts | 用户控制 | 斜杠命令、菜单项 | `prompts/list`、`prompts/get` |
| Resources | 应用控制 | 文件内容、git 历史，由 host 决定怎样放进上下文 | `resources/list`、`resources/read`、`resources/templates/list` |
| Tools | 模型控制 | API 调用、写文件，由模型决定何时调用 | `tools/list`、`tools/call` |

- **Resource** 用 URI 唯一标识（`file://`、`git://`、`https://` 或自定义 scheme）。`resources/read` 返回 `contents` 数组，每项要么是 `text`，要么是 base64 的 `blob`。模板用 RFC 6570 的 URI 模板（如 `file:///{path}`）。资源不存在时返回 `-32602`，不能返回空数组。
- **Prompt** 带参数列表，`prompts/get` 填好参数后返回一组 `{role, content}` 消息，content 可以是文本、图像、音频或嵌入资源。规范强调“用户控制”指的是由用户决定何时使用，内容仍由 server 定义。
- **Tool** 见下一节。

客户端一侧的能力有 sampling（server 请求 host 调一次 LLM）、elicitation（向用户索要信息）、roots（文件系统根目录）。当前版里 server 不能主动向客户端发起 JSON-RPC 请求，而是在回复里返回 `resultType: "input_required"`，把需要的 sampling、elicitation 或 roots 请求放进 `inputRequests`；客户端补齐后带着 `inputResponses` 重发原请求，规范称为多轮往返请求（MRTR）。

## 4. tools/list 与 tools/call：消息形状

以下为规范 *Tools* 页的示例（2026-07-28），为简洁省略了必需的 `_meta`，以及 `icons`、`title`。

**发现**：

```json
→ {"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"cursor":"optional-cursor-value"}}
← {"jsonrpc":"2.0","id":1,"result":{"resultType":"complete",
     "tools":[{"name":"get_weather",
               "description":"Get current weather information for a location",
               "inputSchema":{"type":"object",
                              "properties":{"location":{"type":"string","description":"City name or zip code"}},
                              "required":["location"]}}],
     "nextCursor":"next-page-cursor","ttlMs":300000,"cacheScope":"public"}}
```

**调用**：

```json
→ {"jsonrpc":"2.0","id":2,"method":"tools/call",
   "params":{"name":"get_weather","arguments":{"location":"New York"}}}
← {"jsonrpc":"2.0","id":2,"result":{"resultType":"complete",
   "content":[{"type":"text","text":"Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy"}],
   "isError":false}}
```

要点：① `inputSchema` 必须是合法的 JSON Schema 对象，没写 `$schema` 时默认 2020-12；无参数工具推荐写 `{"type":"object","additionalProperties":false}`。② 可选 `outputSchema`；声明了它，server 必须在 `structuredContent` 里返回符合 schema 的结构化结果，为兼容还应当把同样的 JSON 序列化后放进一个 text 块。③ `content` 可以是 text、image、audio、`resource_link` 或嵌入的 `resource`。④ `resultType` 是新版字段，老版本 server 不带它，客户端按 `"complete"` 处理。⑤ 工具名建议 1–128 个字符，只用字母、数字、`_`、`-`、`.`；名字只在单个 server 内唯一，host 聚合多个 server 时要自己加前缀消歧。⑥ 规范要求 server 按确定的顺序返回工具列表，理由之一就是提高把工具列表放进上下文时的 prompt cache 命中率。

**两种错误**。协议错误（未知工具、请求不符合 schema）走 JSON-RPC 的 `error`，例如 `{"code":-32602,"message":"Unknown tool: invalid_tool_name"}`。工具执行错误（API 失败、参数值不合法、业务逻辑错误）放在 `result` 里，并设 `isError: true`，文本写清原因，例如“出发日期必须在未来”。规范说客户端**应当**把执行错误交给模型，让它自己改参数重试；协议错误模型通常修不了。这个区分直接影响 Agent 的自我纠错能力。

**手算：一次 tools/call 往返的消息与 token**（教学构造）。先分清两件事：JSON-RPC 报文在 client 和 server 之间传，**不进模型上下文**；进上下文的是 host 转换后的工具定义、模型写出的调用和工具结果。下面按“4 个英文字符约 1 个 token”粗估，这是教学假设，真实值取决于分词器和 chat 模板。

| 项 | 字符数（紧凑 JSON，实测） | 约 token |
|---|---:|---:|
| 工具定义 `{name,title,description,inputSchema}` | 262 | 66 |
| `tools/call` 请求（不带 `_meta`） | 114 | 不进上下文 |
| 同上，加两项必需 `_meta`（`clientCapabilities` 为空对象） | 227 | 不进上下文 |
| 响应报文 | 187 | 不进上下文 |
| 其中结果文本 `content[0].text` | 72 | 18 |
| 模型写出的参数 `{"location":"New York"}` | 23 | 6 |

一次“问天气”需要两次 LLM 调用。设系统提示 200 token、用户问题 20 token；模型输出的工具调用连同模板标记共 15 token；工具结果套上模板后多 10 token；最终回答 30 token（均为假设）。

- 第 1 次调用：输入 $200+66+20=286$，输出 15（工具调用）。
- host 把调用转成 `tools/call` 发给 server，拿回结果。
- 第 2 次调用：输入 $286+15+18+10=329$，输出 30。
- 合计输入 $286+329=615$，输出 45。不调工具时只有 220 输入。

如果 host 挂了 20 个同样大小的工具，工具定义就是 $20\times66=1320$ token，两次调用的输入变成 $1540+1583=3123$，其中工具定义占了 $2640/3123\approx85\%$。结论：工具一多，**定义本身**就是主要成本，这就是为什么要按任务筛选暴露给模型的工具，并让工具列表保持稳定的顺序以命中前缀缓存（缓存与 prefill 成本见 [prefill 与 decode](/notes/prefill-decode-video-tokens/)）。另一个结论：新版每个请求多带的 `_meta` 只让报文从 114 字节涨到 227 字节，对模型 token 没有影响。

## 5. 传输方式

两版都定义了两种标准传输：

- **stdio**：客户端把 server 作为子进程启动，server 从 stdin 读、往 stdout 写；消息按换行分隔，消息内部**不能**有换行；stdout 上只能写合法的 MCP 消息，日志写 stderr。2025-11-25 版要求客户端尽量支持 stdio。
- **Streamable HTTP**：server 提供单一的 MCP 端点（如 `https://example.com/mcp`），客户端每条消息是一个 HTTP POST；对于请求，server 要么回一个 `application/json` 对象，要么开一个 SSE 流。HTTP 上要带 `MCP-Protocol-Version` 头。2025-11-25 版里 server 可以在初始化时用 `MCP-Session-Id` 头分配会话，客户端可以用 GET 打开一个 SSE 流来接收 server 主动发来的消息；当前版去掉了会话，回复改成请求范围内的 SSE 流，并把部分 body 字段镜像到 HTTP 头，方便网关路由。Streamable HTTP 替代了 2024-11-05 版的 HTTP+SSE 传输。

**安全**（规范 *Security* 与 *Tools*）：工具等同于任意代码执行，host 调用前必须取得用户明确同意；工具的 annotations 除非来自可信 server，否则一律视为不可信；本地 HTTP server 要校验 `Origin` 头防 DNS rebinding，并且只绑定 127.0.0.1。

**怎样验证一个 MCP 集成**：① 协议层用单测覆盖 `tools/list` 的分页、`tools/call` 的两类错误、版本不匹配时的重试；② 模型层离线统计工具选择准确率、参数 schema 校验通过率、`isError` 之后的重试成功率；③ 安全上用含注入指令的工具结果做红队测试，看模型会不会照着执行。评测基准见 [Agent 评测](/notes/agent-eval/)。

## 6. AutoGen：可对话的智能体与对话编程

AutoGen（Wu et al., arXiv 2308.08155，v2 于 2023-10-03 更新）的两个核心概念（§2）：

**可对话智能体**（conversable agent）。每个智能体有一个角色，能和其他智能体收发消息，根据收发过的消息维护自己的内部上下文。它的后端可以是 LLM、人或工具（代码执行、函数调用），也可以组合。统一接口只有三个：`send`、`receive`、`generate_reply`（Figure 2）。`ConversableAgent` 是最上层的抽象，内置两个子类：`AssistantAgent`（LLM 驱动，图中配置为 `human_input_mode="NEVER"`、不执行代码）和 `UserProxyAgent`（征求人类输入或执行代码与函数调用，图中为 `human_input_mode="ALWAYS"`）。

**对话编程**（conversation programming）。把多智能体应用拆成两件事：计算（智能体为了回复做了什么）和控制流（这些计算按什么顺序、在什么条件下发生）。关键机制是**自动回复**：智能体收到消息后自动调用 `generate_reply` 并回给发送方，直到满足终止条件。控制流因此由对话本身驱动，不需要额外的控制模块。终止条件可以用自然语言定（让助手在完成时回复 “TERMINATE”），也可以用 Python 定（最大自动回复次数、自定义回复函数）。动态多人对话由 `GroupChatManager` 实现：选出下一个发言者、让它回复、把回复广播给其他所有成员（附录中 A5 的描述）。

**论文里能核到的数字**：MATH 全测试集（5000 题）上 AutoGen 准确率 69.48%，GPT-4 为 55.18%（附录 D）。附录里还有一个 12 个任务的小规模试验：用 GPT-4 时，两智能体完成 9 个，四成员群聊（角色扮演式选发言者）完成 11 个；平均 LLM 调用次数分别为 6.8 和 4.5，终止失败分别为 3 次和 0 次（Table 5、Table 6）。样本只有 12 个任务，只能当方向性证据。

**边界**：自动回复如果没有可靠的终止条件，就会来回空转；工具执行由 `UserProxyAgent` 承担，安全边界要靠沙箱。

## 7. MetaGPT：把 SOP 写进提示

MetaGPT（Hong et al., arXiv 2308.00352，ICLR 2024）针对的问题是：简单串联多个 LLM 时，幻觉会逐级传递，导致逻辑不一致（摘要）。它的做法是把人类团队的**标准作业流程**（SOP）编码成提示序列。

- **角色分工**（§3.1）：五个角色，产品经理、架构师、项目经理、工程师、QA 工程师。每个角色有名字、profile、目标和约束，按 ReAct 风格行动。流程是顺序的：产品经理写 PRD（用户故事、需求池），架构师产出文件列表、数据结构和接口定义，项目经理分配任务，工程师写代码，QA 写测试。
- **结构化通信**（§3.2）：智能体之间不靠自由对话，而是交换**文档和图**，每个角色的输出都有固定 schema。论文用“传话游戏”说明自然语言多轮转述会失真。
- **发布订阅**（§3.2）：所有结构化消息发布到一个共享消息池，每个角色按自己的 profile 订阅相关信息，并且只有在所有前置依赖都到齐后才行动。这样避免了一对一通信使拓扑变复杂的问题。
- **可执行反馈**（§3.3）：工程师写完代码就运行测试，出错就结合历史执行记录调试，再继续。

**数字**：HumanEval 与 MBPP 的 Pass@1 分别为 85.9% 和 87.7%；去掉可执行反馈后分别下降 4.2 和 5.4 个点（Figure 4、§4.4）。自建的 SoftwareDev（70 个任务）上与 ChatDev 比：可执行性得分 3.75 对 2.25（满分 4），**token 用量 31,255 对 19,292**，但每行代码耗费的 token 是 124.3 对 248.9（Table 1）。角色消融（Table 3）：只有工程师时花费 0.915 美元、可执行性 1.0；四个角色齐全时花费 1.385 美元、可执行性 4.0。这组数字恰好说明多智能体的一般规律：总成本更高，换来的是质量。

## 8. CAMEL：角色扮演与 inception prompting

CAMEL（Li et al., arXiv 2303.17760，NeurIPS 2023）研究的是两个智能体怎样在没有人持续干预的情况下自主协作（§3）。

**流程**：人给一个初步想法和两个角色，例如“为股市开发交易机器人”，AI 助手是 Python 程序员，AI 用户是股票交易员。先由**任务细化器**（task specifier）把想法改写成具体任务，然后 AI 用户不断给指令，AI 助手给解答。记 $t$ 时刻的指令和解答为 $I_t,S_t$，消息集 $M_t=\{(I_i,S_i)\}_{i=0}^{t}$，则（式 (2)–(4)）：

$$
I_{t+1}=\mathcal U(M_t),\qquad S_{t+1}=\mathcal A(M_t,I_{t+1}),\qquad M_{t+1}\leftarrow M_t\cup\{(I_{t+1},S_{t+1})\}.
$$

**Inception prompting**（§3.2）：开场时给三份提示，即任务细化提示、助手系统提示、用户系统提示，之后两者自动互相提示直到终止。助手提示里的几条约束都对应一种失败：“Never flip roles! Never instruct me!”防止**角色反转**；要求每次以 `Solution:` 开头并给出具体实现，防止“我会去做……”这类**空头回复**；以 `Next request.` 结尾让对话继续。用户提示要求在任务完成时只回复 `<CAMEL_TASK_DONE>`，否则两个智能体可能无休止地互相道谢。

**终止条件**（§4.1）：用户连续 3 轮不给指令；助手开始下指令（视为角色反转）；出现结束 token；达到 token 上限；消息数达到 40 条。论文给的理由是：生成成本随对话长度**二次增长**，所以必须设上限。AI Society 数据集由 50 个助手角色、50 个用户角色、每对 10 个任务组成，共 25,000 段对话。CAMEL 的一个主要用途就是**合成指令数据**，这也是它和 Agent 训练的联系点。

## 9. 多智能体辩论

Du et al.（arXiv 2305.14325）：同一个模型开多个实例，各自先独立作答；然后把其他实例的回答拼进上下文，让每个实例参考后更新自己的答案，重复若干轮（§2.1）。辩论不保证收敛，但实验中通常会收敛到同一个答案；提示里让模型更“固执”，辩论会更长、结果更好（§2.2）。

Table 1（chatGPT，3 个智能体、2 轮辩论）：

| 方法 | 算术 (%) | GSM8K (%) | 国际象棋（ΔPS） |
|---|---:|---:|---:|
| 单智能体 | 67.0 ± 4.7 | 77.0 ± 4.2 | 91.4 ± 10.6 |
| 单智能体 + 反思 | 72.1 ± 4.5 | 75.0 ± 4.3 | 102.1 ± 11.9 |
| 多智能体多数投票 | 69.0 ± 4.6 | 81.0 ± 3.9 | 102.2 ± 6.2 |
| 多智能体辩论 | 81.8 ± 2.3 | 85.0 ± 3.5 | 122.9 ± 7.6 |

论文自己承认的局限（§5）：计算更贵；辩论变长后，模型往往只关注最近几条发言；智能体多了以后，拼接全部回答会超出上下文，作者改为先用 chatGPT 做摘要（§3.3）。注意这里的对照组在算力上并不对等：单智能体只生成一次，辩论生成了 $3\times3=9$ 次（若 2 轮辩论之外还有 1 轮初始作答）。下一节说明为什么这一点很要紧。

## 10. 手算：N 个智能体通信的消息数与成本

**拓扑与每轮消息数**（教学构造）。$N$ 个智能体，每轮每个智能体发一条长度为 $L$ token 的消息：

| 拓扑 | 每轮投递次数 | 每个智能体每轮读入 |
|---|---|---|
| 全连接（两两发送，如辩论） | $N(N-1)$ | $(N-1)L$ |
| 星形（1 个协调者 + $N-1$ 个工作者） | $2(N-1)$ | 工作者读 $L$；协调者读 $(N-1)L$ |
| 共享池 + 订阅（MetaGPT） | 发布 $N$ 次；投递次数等于订阅关系数 | 只读订阅的部分 |
| 顺序链（流水线） | $N-1$ | $L$ |

**全连接辩论的 token 账**（教学构造）。设 $N=4$，问题 $Q=200$ token，每条回答 $L=300$，初始作答 1 轮加辩论 $R=3$ 轮。简化：辩论轮里每个智能体的输入是问题加上全部 $N$ 条上一轮回答（含自己的），不累积更早的历史。

- 初始轮：输入 $NQ=800$。
- 每个辩论轮：输入 $N(Q+NL)=4\times(200+1200)=5600$，三轮共 16,800。
- 总输入 $800+16800=17600$；总输出 $(R+1)NL=4\times4\times300=4800$；LLM 调用 16 次；消息投递 $3\times N(N-1)=36$ 次。
- 单智能体一次 CoT：输入 200、输出 300。辩论的输入是它的 88 倍，输出是 16 倍。

把 $N$ 从 4 加到 8：每轮输入 $8\times(200+8\times300)=20800$，是 $N=4$ 时 5600 的约 3.7 倍。因为每轮输入近似 $N^2L$，智能体数翻倍，读入量接近翻两番。如果再保留完整历史，第 $t$ 轮的输入还要随 $t$ 线性增长，$T$ 轮累计约为 $mT(T+1)/2$（$m$ 为每轮新增 token），这就是 CAMEL 说的“成本随对话长度二次增长”。换成星形拓扑，同样 4 个智能体每轮只有 $2\times3=6$ 次投递。

**公平对照**：评价多智能体时，基线要**按同样的 token 预算**给单智能体，例如自一致性采样 16 次再投票，而不是只采一次。否则提升里分不清多少来自“多算了几倍”。

## 11. 什么时候多智能体不如单智能体

先列有研究支撑的结论，再写分析。

**有研究支撑的**：

- Wang et al.（arXiv 2402.18272）重新评测了辩论类方法：给单智能体一个强提示（带示例），就能在多种推理任务和骨干模型上达到最好的讨论框架几乎一样的表现；只有提示里**没有示例**时，多智能体讨论才更好（摘要）。
- MAST（Cemri et al., arXiv 2503.13657，NeurIPS 2025 Datasets and Benchmarks）：在 7 个多智能体框架上收集了 1600 多条带标注的轨迹，归纳出 14 种失败模式，分为三类：系统设计问题、智能体间失配、任务验证缺失；标注者一致性 $\kappa=0.88$（摘要）。引言写到，多智能体相对单智能体或 best-of-N 这类简单基线的增益“往往很小”。
- Kim et al.（arXiv 2512.08296，v3 于 2026-04-08 更新）在 6 个 Agent 基准、5 种架构、三家模型上做了 260 个受控配置，统一工具、提示和算力（摘要、§1）：单智能体成功率已超过约 45% 时，再加智能体收益为负；工具密集的任务会被协调开销拖累，因为每个智能体分到的 token 预算被切碎了；独立并行、没有汇总校验的架构把轨迹级错误放大 17.2 倍，中心化协调为 4.4 倍；相对单智能体的变化从可分解的金融推理 +80.8% 到顺序规划 −70.0% 不等。
- Anthropic 的多智能体研究系统工程博客（2025-06-13）：在他们的数据里，Agent 的 token 用量约为普通对话的 4 倍，多智能体约为 15 倍；在其内部研究评测上，Opus 4 主控加 Sonnet 4 子智能体比单个 Opus 4 高 90.2%。文章同时写到，需要所有智能体共享同一上下文、或智能体之间依赖很多的领域目前不适合多智能体，并举例说大多数编程任务里真正可并行的部分比研究任务少。

**分析**（本文推导，非论文结论）：

1. **成本**：第 10 节已经算过，全连接通信每轮输入约 $N^2L$。任务本身不能并行时，多出来的 token 换不来收益。
2. **错误传播**：设顺序链上每一环独立正确的概率为 $p=0.95$，5 个智能体串联且没有校验，整体正确率为 $0.95^5\approx0.774$。并行加投票能纠错：3 个独立、各自正确率 0.7 的投票者，多数正确的概率为 $0.7^3+3\times0.7^2\times0.3=0.784$。但同一个模型的多个实例错误高度相关，独立性假设不成立，实际增益会更小。
3. **通信损耗**：智能体之间只能通过消息传递信息，子智能体看不到主控的完整上下文，交接时容易丢约束。MetaGPT 改用结构化文档，CAMEL 加了一长串格式约束，都是在处理这个问题。

**经验法则**（分析）：任务可以拆成**互相独立、可以并行**的子任务，且单个上下文装不下时，才考虑多智能体；顺序依赖强、需要共享全局状态、或单智能体已经做得不错时，优先用单智能体加工具。多智能体系统的训练与 RL 见 [Agent RL 进阶](/notes/agent-rl-advanced/)，系统设计题怎样权衡见 [Agent 系统设计](/notes/agent-system-design/)。

## 12. 面试常问

**MCP 和 Function Calling 是什么关系？** Function Calling 是模型 API 层的格式：模型输出“调哪个函数、参数是什么”。MCP 是应用与工具服务之间的协议：工具怎样被发现（`tools/list`）、怎样被执行（`tools/call`）、结果和错误长什么样。host 把 MCP 工具定义转成模型的 tool schema，再把模型的调用转成 `tools/call`。两者在不同层，可以同时用。

**能力协商怎么做？** 要先问清版本。2025-11-25 及更早：`initialize` 请求和响应交换版本与能力，再发 `notifications/initialized`，之后只能用协商成功的能力。2026-07-28：没有握手，每个请求的 `_meta` 自带版本和客户端能力；server 能力通过 `server/discover` 获取；版本不支持时返回 `-32022`，客户端重试。

**工具执行出错，返回 JSON-RPC error 还是 isError？** 未知工具、请求格式错误属于协议错误，返回 `error`。参数值不合法、下游 API 失败返回 `result` 加 `isError: true`，并写清原因，让模型能自己修正后重试。

**AutoGen、MetaGPT、CAMEL 的核心区别？** AutoGen 提供通用的可对话智能体和自动回复机制，控制流由对话驱动。MetaGPT 用 SOP 固定流程，角色之间交换结构化文档，通过共享池加订阅分发。CAMEL 是两个角色的指令-解答式扮演，用 inception prompting 和终止条件防止角色反转与死循环，主要用来合成数据。

**你会怎样判断该不该上多智能体？** 先在同等 token 预算下跑单智能体基线（含自一致性），看任务能不能并行、单智能体成功率是否已经很高；上多智能体后按 MAST 的三类失败标注轨迹，并单独统计成本和延迟。

## 闭卷验收

不看资料说出 host、client、server 各自的职责，以及“server 看不到完整对话”的原则；写出 2025-11-25 的 `initialize` 三步握手和版本协商规则，再写出 2026-07-28 每个请求 `_meta` 里的两个必需字段、`server/discover` 的作用和 `-32022` 的重试流程；画出 prompts、resources、tools 三类原语的控制方和方法名；默写 `tools/list` 和 `tools/call` 的请求与响应，区分协议错误与 `isError`，说出 `structuredContent` 和 `outputSchema` 的关系；说出 stdio 与 Streamable HTTP 的分帧方式；对一次天气查询手算两次 LLM 调用的输入输出 token，并解释 20 个工具时为什么工具定义占约 85%；讲清 AutoGen 的 `send`/`receive`/`generate_reply` 与自动回复、MetaGPT 的五个角色与共享池订阅、CAMEL 的 $I_{t+1}=\mathcal U(M_t)$ 与五条终止条件、辩论 Table 1 的设置；手算 $N=4$ 全连接辩论的投递次数与 token，说明 $N$ 翻倍为何输入接近翻两番；最后用至少两项研究和一项分析回答“什么时候多智能体不如单智能体”。

**参考。** [MCP 规范 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28)（[Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)、[Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)、[Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)、[server/discover](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)、[Server Features](https://modelcontextprotocol.io/specification/2026-07-28/server)、[Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)、[Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)、[Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)、[Transports](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)）；[MCP 规范 2025-11-25：Lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)、[Transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)；[MCP 版本说明](https://modelcontextprotocol.io/specification/versioning)；[AutoGen](https://arxiv.org/abs/2308.08155)；[MetaGPT](https://arxiv.org/abs/2308.00352)；[CAMEL](https://arxiv.org/abs/2303.17760)；[Multiagent Debate](https://arxiv.org/abs/2305.14325)；[Rethinking the Bounds of LLM Reasoning](https://arxiv.org/abs/2402.18272)；[Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)；[Towards a Science of Scaling Agent Systems](https://arxiv.org/abs/2512.08296)；[Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)。
