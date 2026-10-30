---
title: 'learn-claude-code s01：Agent Loop，一个循环就够了'
date: '2026-10-29'
tags: [Agent, Harness, Agent Loop, learn-claude-code]
summary: 转载 learn-claude-code 第 1 课，补充配图、code.py 逐段导读和一次真实运行的 messages。模型只会输出命令，不会自己执行；一个 while True 循环把「调工具 → 执行 → 结果喂回」自动化，这就是最小的 agent harness 内核。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s01_agent_loop)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」和「实际跑一次」两节为本站补充。

> *"One loop & Bash is all you need"* — 一个工具 + 一个循环 = 一个 Agent。
>
> **Harness 层**: 循环 — 模型与真实世界的第一道连接。

## 问题

你提出了一个问题给大模型：“帮我读取下我的目录下有哪些文件，并且执行XXX.py”。

模型能输出一条 bash 命令，但输出完了就停了，它不会自己跑，也不会看到结果后继续推理。

你可以手动跑一遍，把输出粘贴回对话框，让它接着干。下一个命令出来，你再跑一遍、再贴回去。

每一个来回，你都在做中间层。而把它自动化，就是这一章要做的事。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s01-agent-loop/01-who-runs-the-command.jpg" target="_blank" rel="noopener" aria-label="查看原图：左边没有循环时，你在模型和终端之间来回复制粘贴；右边有了循环，harness 接过这个中间层，把 tool_use 交给终端执行，再把结果作为 tool_result 交回模型"><img src="/notes-assets/lcc-s01-agent-loop/01-who-runs-the-command.jpg" alt="左边没有循环时，你在模型和终端之间来回复制粘贴；右边有了循环，harness 接过这个中间层，把 tool_use 交给终端执行，再把结果作为 tool_result 交回模型" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s01-agent-loop/02-while-true.jpg" target="_blank" rel="noopener" aria-label="查看原图：messages 发给 LLM，判断响应里有没有 tool_use；有就执行 run_bash 并追加 tool_result，再回到 messages；没有就 return"><img src="/notes-assets/lcc-s01-agent-loop/02-while-true.jpg" alt="messages 发给 LLM，判断响应里有没有 tool_use；有就执行 run_bash 并追加 tool_result，再回到 messages；没有就 return" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

一个 `while True` 循环，模型调用工具就继续，不调用就停。循环直接检查响应里的内容块：

| 信号 | 含义 | 循环动作 |
|------|------|---------|
| 包含 `tool_use` block | 模型要求调用工具 | 执行 → 结果喂回去 → 继续 |
| 不包含 `tool_use` block | 模型没有调用工具 | 退出循环 |

## 工作原理

将这个过程翻译成代码。分步来看：

**第 1 步**：把用户的问题作为第一条消息。

```python
messages = [{"role": "user", "content": query}]
```

**第 2 步**：将消息和工具定义一起发给 LLM。

```python
response = client.messages.create(
    model=MODEL, system=SYSTEM, messages=messages,
    tools=TOOLS, max_tokens=8000,
)
```

**第 3 步**：追加模型回答，检查它是否调了工具。没调 → 结束。

```python
messages.append({"role": "assistant", "content": response.content})
tool_calls = [
    block for block in response.content if block.type == "tool_use"
]
if not tool_calls:
    return
```

只有实际存在的 `tool_use` block 才会进入执行阶段，因此不会追加空的工具结果消息。

**第 4 步**：执行模型要求的工具，收集结果。

```python
results = []
for block in tool_calls:
    output = run_bash(block.input["command"])
    results.append({
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": output,
    })
```

**第 5 步**：把工具结果作为新消息追加，回到第 2 步。

```python
messages.append({"role": "user", "content": results})
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s01-agent-loop/03-messages-one-round.jpg" target="_blank" rel="noopener" aria-label="查看原图：跑完一轮后 messages 里有四条消息：user 提问，assistant 的 tool_use（id=toolu_1），user 角色的 tool_result（tool_use_id=toolu_1，两个 id 必须对上），最后 assistant 给出文字回答"><img src="/notes-assets/lcc-s01-agent-loop/03-messages-one-round.jpg" alt="跑完一轮后 messages 里有四条消息：user 提问，assistant 的 tool_use（id=toolu_1），user 角色的 tool_result（tool_use_id=toolu_1，两个 id 必须对上），最后 assistant 给出文字回答" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

组装为一个完整函数：

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return

        results = []
        for block in tool_calls:
            output = run_bash(block.input["command"])
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        messages.append({"role": "user", "content": results})
```

三十多行，这就是最小可运行的 agent harness 内核。它为模型提供持续行动的最小运行框架：模型负责决策（要不要调工具、调哪个），harness 负责执行（调用工具，把结果作为新消息追加）。后面 16 个章节都在这个循环上叠加机制，循环本身始终不变。

## 代码导读：把 code.py 从头读一遍（本站补充）

上面的五步只是核心循环。完整的 [`s01_agent_loop/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s01_agent_loop/code.py) 有 153 行，我们按从上到下的顺序过一遍，每段只问一个问题：**它在 harness 里负责什么？**

### 第一段：连上模型

```python
load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]
```

从 `.env` 读出 key、模型名和接口地址，创建一个客户端。`override=True` 表示 `.env` 里的值优先于终端里已有的环境变量。

`ANTHROPIC_BASE_URL` 是一个很有用的口子：只要地址指向任何一个「长得像 Anthropic」的接口，代码一行不改就能换模型。后面我们会用到它。

### 第二段：告诉模型它是谁

```python
SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use bash to solve tasks. Act, don't explain."
)
```

system prompt 只有三件事：你在哪个目录、用的什么系统、用 bash 干活。

注意最后一句 **Act, don't explain**。没有这句，模型很容易先写一大段「我打算这样做……」，再不调用工具就结束了。这一句是在推它直接动手。

### 第三段：工具长什么样

```python
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]
```

模型看不到你的 Python 函数。它能看到的只有这三样：**名字、一句话描述、参数的 JSON Schema**。模型决定调哪个工具、传什么参数，全靠这几行字。

所以工具描述写得好不好，直接影响模型会不会用它。这一点到 s02 有 5 个工具时会更明显。

### 第四段：真正执行命令

```python
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, errors="replace", timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
```

短短十几行，其实做了五件事：

| 代码 | 作用 |
|---|---|
| `dangerous` 列表 | 按子串拦截几条明显危险的命令 |
| `timeout=120` | 命令卡住时不让整个 agent 一起卡住 |
| `stdout + stderr` | 报错信息也要给模型看，它才知道哪里错了 |
| `out[:50000]` | 输出太长就截断，免得一次撑爆上下文 |
| `"(no output)"` | 命令成功但没输出时，也给模型一个明确的回答 |

还有一个细节：**所有出错的情况都是 `return` 一个字符串，而不是抛异常。** 错误信息会作为 `tool_result` 交回给模型，模型看到 `Error: ...` 可以自己换个办法再试。

这个黑名单只是演示，很容易绕过，比如 `rm -rf ~` 就不在名单里。真正的权限控制在 s03。`timeout` 也有坑：超时后只杀掉了 shell，shell 启动的子进程可能还在跑，这个问题在[子进程那一课](/notes/agent-eng-subprocess/)里专门讲过。

### 第五段：入口和多轮对话

```python
history = []
while True:
    query = input("s01 >> ")
    ...
    history.append({"role": "user", "content": query})
    agent_loop(history)
    response_content = history[-1]["content"]
    ...
```

这里有**两层循环**，别搞混：

- **外层**是你和 agent 聊天：你问一句，它做完一件事，你再问下一句。
- **内层**是 `agent_loop`：为了完成你这一句，模型可能连着调好几次工具。

`history` 在外层循环外面创建，每一轮都往同一个列表里追加。`agent_loop` 拿到的也是这个列表本身，直接在上面 `append`。所以第二轮提问时，模型能看到第一轮的全部过程。**agent 的「记性」，说到底就是这个越来越长的列表。**

最后打印的是 `history[-1]`，也就是循环结束时模型那条不带工具调用的回复。

## 实际跑一次：messages 里到底有什么（本站补充）

光看代码不够直观，我们跑一次，把发给模型的 `messages` 原样打印出来。提问是：

```
Create a file called hello.py that prints "Hello, World!", then run it
```

模型一共被调用了两次：

```
第 1 次  msgs=1  -> stop_reason=tool_use   内容块: [tool_use]
第 2 次  msgs=3  -> stop_reason=end_turn   内容块: [text]
```

第 2 次调用时，`messages` 里是这三条（id 太长，截短了）：

```json
[
  {"role": "user", "content": "Create a file called hello.py that prints \"Hello, World!\", then run it"},
  {"role": "assistant", "content": [
    {"type": "tool_use", "id": "call_a969...", "name": "bash",
     "input": {"command": "cat > hello.py <<''PY''\nprint(\"Hello, World!\")\nPY\npython3 hello.py"}}
  ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "call_a969...", "content": "Hello, World!"}
  ]}
]
```

对照上面那张 messages 图：模型把「写文件」和「运行」拼成了一条命令，所以只用了一轮工具调用。`tool_result` 的 `tool_use_id` 和 `tool_use` 的 `id` 一模一样，模型靠它知道这条结果对应哪次调用。

### 换成 OpenAI 接口会怎样

这次运行我们用的模型只提供 OpenAI 格式的接口，所以中间加了一层翻译。同一轮对话，在两种协议里长这样：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s01-agent-loop/04-two-protocols.jpg" target="_blank" rel="noopener" aria-label="查看原图：同一轮对话在两种协议里的对应关系：Anthropic 的 tool_use 内容块对应 OpenAI 的 tool_calls；input 是字典，arguments 是 JSON 字符串；tool_result 放在 role user 里，OpenAI 用单独的 role tool；stop_reason tool_use 对应 finish_reason tool_calls"><img src="/notes-assets/lcc-s01-agent-loop/04-two-protocols.jpg" alt="同一轮对话在两种协议里的对应关系：Anthropic 的 tool_use 内容块对应 OpenAI 的 tool_calls；input 是字典，arguments 是 JSON 字符串；tool_result 放在 role user 里，OpenAI 用单独的 role tool；stop_reason tool_use 对应 finish_reason tool_calls" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

最容易踩坑的是两处：

1. **参数的类型不一样。** Anthropic 的 `input` 已经是字典，OpenAI 的 `arguments` 是一段 JSON **字符串**，要自己 `json.loads`。模型偶尔会生成不合法的 JSON，解析这一步本身就可能失败。
2. **工具结果放的位置不一样。** Anthropic 把 `tool_result` 放在 `role: user` 的消息里；OpenAI 要求单独一条 `role: tool` 的消息，并且必须紧跟在发起调用的那条 assistant 消息后面。

循环本身完全没变。变的只是消息怎么打包，这也说明 harness 的核心是那个循环，不是某一家的 API。

### 想一想

读完代码，试着回答这三个问题，答案都在上面的代码里：

1. 如果模型每次都调用工具，`agent_loop` 什么时候会停？有没有一个上限？

   <details class="qa"><summary>参考答案</summary>

   没有上限。`agent_loop` 是一个 `while True`，唯一的出口是模型某次回复里**没有** `tool_use`。只要模型一直调用工具，循环就一直转下去，一直花钱。

   真实的 harness 都会加一道保险：最大轮数、总 token 预算，或者超时。后面 s06 的子 Agent 就加了「最多 30 轮」，s17 有全局的 `MAX_TURNS`。

   </details>

2. 如果模型调用 `bash` 时忘了传 `command` 参数，会发生什么？程序会崩吗？

   <details class="qa"><summary>参考答案</summary>

   会崩。`block.input['command']` 在打印那一行就会先抛出 `KeyError`，这一行不在任何 `try` 里，异常一路冒到最外层，程序直接退出。

   模型其实很少漏参数，但「很少」不等于「不会」。后面的课会把工具执行包进 `try`，把异常变成一句 `Error: ...` 交还给模型，让它自己改。

   </details>

3. 第二轮提问时，`history` 里大概有几条消息？它会一直变长吗？

   <details class="qa"><summary>参考答案</summary>

   取决于第一轮调了几次工具。假设第一轮调了一次 `bash`，那第一轮结束时有 4 条：用户问题、assistant（tool_use）、user（tool_result）、assistant（最终回答）。第二轮提问时再加 1 条，就是 5 条。

   它会一直变长：每一轮的问题、工具调用、工具结果都留在 `history` 里，程序不退出就不会清空。这就是 s08 要解决的上下文压缩问题。

   </details>

## 试一下

> **安全提示**：代码会执行模型生成的 shell 命令。建议在一个临时测试目录中运行，避免影响你的项目文件。s03 会加入权限控制。

**准备**（首次运行）：

```sh
git clone https://github.com/shareAI-lab/learn-claude-code.git
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY 和 MODEL_ID
```

**运行**：

```sh
python s01_agent_loop/code.py
```

试试这些 prompt：

1. `Create a file called hello.py that prints "Hello, World!"`
2. `List all Python files in this directory`
3. `What is the current git branch?`

观察重点：模型什么时候调用工具（循环继续），什么时候不调用（循环结束）？

## 接下来

现在模型手里只有 bash 一个工具，读文件要 `cat`，写文件要 `echo ... >`，找个文件要 `find`，又丑又容易出错。

[s02 Tool Use](/notes/lcc-s02-tool-use/) → 给它 5 个真正的工具，会发生什么？模型会不会一次调用多个工具？几个工具同时跑会不会互相踩？
