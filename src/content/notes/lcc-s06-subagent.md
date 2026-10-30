---
title: 'learn-claude-code s06：Subagent，脏活交给子 Agent，父上下文只收结论'
date: '2026-10-30'
tags: [Agent, Harness, Subagent, Context, learn-claude-code]
summary: 转载 learn-claude-code 第 6 课，补充配图、code.py 逐段导读和一次真实运行。task 工具同步跑一个嵌套循环，子 Agent 用全新的 messages[]，结束后只把最终文本交回父循环；真实运行里子 Agent 的上下文涨到 85k 字符，父 Agent 只收到 1.7k。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s06_subagent)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> Subagent 从全新的 `messages[]` 开始。最终文本返回父循环，中间对话不会进入父上下文。
>
> **Harness 层**: 委派 — 在另一段对话上下文中处理一个明确的子任务。

## 问题

Agent 在修一个 bug。为了追踪调用链，它读取了许多文件；每次工具调用和结果都会留在父循环的 `messages[]` 中。调用链已经弄清以后，多数中间细节不再需要，却仍然占用上下文。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s06-subagent/01-fresh-messages.jpg" target="_blank" rel="noopener" aria-label="查看原图：父 Agent 的 messages 只有三行：用户问题、assistant 调用 task(prompt)、tool_result 等于最终文本；子 Agent 有自己的 messages，从 prompt 开始，经过 19 次工具调用，最后只有 final text 一条箭头回到父 Agent，其余消息被丢弃"><img src="/notes-assets/lcc-s06-subagent/01-fresh-messages.jpg" alt="父 Agent 的 messages 只有三行：用户问题、assistant 调用 task(prompt)、tool_result 等于最终文本；子 Agent 有自己的 messages，从 prompt 开始，经过 19 次工具调用，最后只有 final text 一条箭头回到父 Agent，其余消息被丢弃" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

调用 `task` 时，会同步运行一个使用全新 `messages[]` 的嵌套 Agent Loop。循环结束后，它的最终文本会成为父对话中的工具结果。

这里隔离的是消息，不是进程或文件系统。父 Agent 与子 Agent 共享 `WORKDIR`，写文件和命令仍会影响同一个工作区。子 Agent 拥有五个基础工具，但没有 `task`；它的工具调用与父 Agent 使用同一组权限和生命周期 Hooks。

## 工作原理

**run_subagent** 创建新的消息列表，运行嵌套循环，并返回最终文本：

```python
SUB_TOOLS = list(BASE_TOOLS)  # 不含 task 工具

def run_subagent(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]
        if not tool_calls:
            return extract_text(response.content) or "(no summary)"

        results = []
        for block in tool_calls:
            output = execute_tool(block, SUB_HANDLERS)
            results.append({... "content": output})
        messages.append({"role": "user", "content": results})

    return "Subagent stopped after 30 turns without a final answer."
```

主 Agent 调用时，跟调其他工具一样：

```python
TASK_TOOL = {
    "name": "task",
    "description": "Run a subagent with fresh conversation context and return its final text.",
    "input_schema": {
        "type": "object",
        "properties": {"prompt": {"type": "string"}},
        "required": ["prompt"],
    },
}

TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}
```

实际边界如下：

| 决策 | 选择 | 原因 |
|------|------|------|
| 对话 | 全新的 `messages[]` | 不把父对话复制给子 Agent |
| 执行 | 同一进程和 `WORKDIR` | 两个循环都能看到文件系统修改 |
| 返回值 | 只返回最终文本 | 子 Agent 的工具调用和结果不进入父消息列表 |
| 委派深度 | `SUB_TOOLS` 中没有 `task` | 本章只允许一层委派 |
| 工具策略 | 共享 Hooks | 父子循环使用相同的权限检查 |

父 Agent 与其他工具一样，通过 handler map 分发 `task`。子 Agent 使用 `SUB_SYSTEM`、`SUB_TOOLS` 和自己的局部 `messages` 列表。


## 代码导读：把 code.py 从头读一遍（本站补充）

完整的 [`s06_subagent/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s06_subagent/code.py) 有 385 行。前 260 行几乎都是 s04 留下来的老代码：五个基础工具、权限检查和 hooks（s05 的 todo_write 这一课没带过来），真正新增的只有 `run_subagent` 和 `task` 工具这几十行。

### 第一段：两份 system prompt

```python
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use task for focused exploration or a self-contained subtask."
)
SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Complete the given task, then return a concise final answer."
)
```

父 Agent 被告知「探索类、能独立完成的子任务，交给 task」。什么时候委派，**由模型自己判断**，代码里没有任何规则强制它委派。

子 Agent 被告知「做完以后，给一个简洁的最终答案」。这句话很重要：子 Agent 最后一次回复的文字，就是父 Agent 能拿到的**全部**内容。如果它最后只说一句 "Done."，父 Agent 就只知道 "Done."。

### 第二段：工具分成两份

```python
BASE_TOOLS = [bash, read_file, write_file, edit_file, glob]   # 简写
BASE_HANDLERS = {"bash": run_bash, ...}

SUB_TOOLS = list(BASE_TOOLS)
SUB_HANDLERS = dict(BASE_HANDLERS)

TOOLS = [*BASE_TOOLS, TASK_TOOL]
TOOL_HANDLERS = {**BASE_HANDLERS, "task": run_subagent}
```

之前几课里只有一份 `TOOLS`，这里拆成了「基础工具」和「父 Agent 额外多一个 `task`」。

子 Agent 的工具表里**没有 `task`**，所以它没办法再派子 Agent。这样就只有一层委派，不会出现「子 Agent 派孙 Agent，孙 Agent 再派……」无限套娃。

`list(...)` 和 `dict(...)` 各复制了一份。这样以后要单独给子 Agent 删掉某个工具（比如不让它写文件），改 `SUB_TOOLS` 就行，不会影响父 Agent。

### 第三段：`execute_tool`，父子共用一个工具执行器

```python
def execute_tool(block, handlers: dict) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        return str(blocked)

    handler = handlers.get(block.name)
    try:
        output = handler(**block.input) if handler else f"Unknown: {block.name}"
    except Exception as e:
        output = f"Error: {e}"

    trigger_hooks("PostToolUse", block, output)
    return str(output)
```

s05 里这几行写在 `agent_loop` 中间。现在有两个循环都要执行工具，就把它抽成一个函数，只多一个参数 `handlers`：父循环传 `TOOL_HANDLERS`，子循环传 `SUB_HANDLERS`。

这样，**权限检查、日志 hook 对子 Agent 同样生效**。子 Agent 想执行 `rm`，终端里照样会弹出 `Allow? [y/N]`。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s06-subagent/02-isolated-vs-shared.jpg" target="_blank" rel="noopener" aria-label="查看原图：父循环和子循环各有自己的 messages 列表，中间用 isolated 虚线隔开；两个循环都指向下方同一个 shared 框，里面是 WORKDIR 文件、hooks 和权限；子循环旁边标注没有 task 工具"><img src="/notes-assets/lcc-s06-subagent/02-isolated-vs-shared.jpg" alt="父循环和子循环各有自己的 messages 列表，中间用 isolated 虚线隔开；两个循环都指向下方同一个 shared 框，里面是 WORKDIR 文件、hooks 和权限；子循环旁边标注没有 task 工具" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

所以要分清楚：**隔离的只是消息**。文件系统、hooks、权限规则，父子用的是同一套。子 Agent 写的文件，父 Agent 下一步就能读到。

### 第四段：`run_subagent`，一个嵌套的 Agent 循环

```python
def run_subagent(prompt: str) -> str:
    print("\n\033[35m[Subagent started]\033[0m")
    messages = [{"role": "user", "content": prompt}]

    for _ in range(30):
        response = client.messages.create(
            model=MODEL, system=SUB_SYSTEM,
            messages=messages, tools=SUB_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            print("\033[35m[Subagent done]\033[0m")
            return extract_text(response.content) or "(no summary)"

        results = []
        for block in tool_calls:
            output = execute_tool(block, SUB_HANDLERS)
            print(f"  \033[90m[sub] {block.name}: {output[:100]}\033[0m")
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

    print("\033[35m[Subagent stopped]\033[0m")
    return "Subagent stopped after 30 turns without a final answer."
```

把它和下面的 `agent_loop` 并排看，骨架完全一样：调模型 → 没有工具调用就结束 → 有就执行、把结果塞回去。区别只有四处：

| | 父循环 `agent_loop` | 子循环 `run_subagent` |
|---|---|---|
| `messages` | 外面传进来的 `history`，跨轮保留 | 函数里的**局部变量**，函数返回就没了 |
| 轮数上限 | 没有，`while True` | 最多 30 轮 |
| 结束时 | 直接 `return` | 返回最后一次回复的文字 |
| 工具表 | `TOOLS`（含 task） | `SUB_TOOLS`（不含 task） |

`messages` 是局部变量，这一点就是整章的核心。子 Agent 读了多少文件、跑了多少命令，都只存在于这个局部列表里。`return` 之后，列表被 Python 回收，父 Agent 一个字都看不到。

另外注意 `run_subagent` 是被**同步**调用的：父 Agent 的 `for block in tool_calls` 走到 `task` 这一项时，会一直卡住，直到子 Agent 跑完。如果父 Agent 一次发了两个 `task`，它们是**一个接一个**跑的，不是并行。

### 第五段：`extract_text`，只取最后一条

```python
def extract_text(content) -> str:
    if not isinstance(content, list):
        return str(content)
    return "\n".join(
        getattr(block, "text", "")
        for block in content
        if getattr(block, "type", None) == "text"
    )
```

传进来的是 `response.content`，也就是子 Agent **最后一次**回复。前面几轮里模型顺手写的文字（比如「先看看 package.json」），不会被带回去。

如果最后一次回复连文字都没有，就返回 `"(no summary)"`。父 Agent 至少能知道「子 Agent 结束了，但什么也没说」，而不是收到一个空字符串。

### 第六段：30 轮上限之后

跑满 30 轮还没结束，子 Agent 返回一句 `"Subagent stopped after 30 turns without a final answer."`。

这里有个容易忽略的地方：**消息被丢掉了，但它对文件做的修改还在**。父 Agent 只收到「没跑完」这句话，不知道子 Agent 已经改了哪些文件。

## 实际跑一次（本站补充）

在 learn-claude-code 仓库根目录运行，提问 `Use a subtask to find what testing framework this project uses`。

父 Agent 的第一次回复就是调用 `task`，它自己写了给子 Agent 的 prompt：

```
Inspect the project at .../learn-claude-code and determine what testing framework(s)
it uses. Check package/config files and test files. Do not modify anything.
Report concise evidence with relevant file paths.
```

「Do not modify anything」是父 Agent 自己加的。子 Agent 拿不到原始对话，父 Agent 必须把要求写进 prompt。

接下来终端里是一长串 `[sub]` 输出。`log_hook` 只打印参数的前 60 个字符，所以每行都停在路径中间（下面把 `/Users/athebear/Documents/GitHub/bear-harness` 缩写成 `~/bh`，中间省略了 14 次工具调用）：

````
[Subagent started]
[HOOK] glob(['~/bh/learn-claude)
  [sub] glob: ~/bh/learn-claude-code/web/package.json
[HOOK] glob(['~/bh/learn-claude)
  [sub] glob: (no matches)
……
[HOOK] read_file(['~/bh/learn-claude)
  [sub] read_file: name: Test

on:
  push:
    branches: [main]
……
[HOOK] Stop: session used 19 tool calls
[Subagent done]
[HOOK] Stop: session used 1 tool calls
The project primarily uses **pytest**.

Evidence:
- `.github/workflows/test.yml` runs:
  ```bash
  python -m pytest tests -q
  ```
- Tests such as `tests/test_agents_smoke.py` and `tests/test_goal_loop.py` import and use `pytest`.
- Some tests subclass Python’s built-in `unittest.TestCase`, but pytest discovers and runs them.

The `web/` project does not appear to have a JavaScript testing framework configured.
````

两行 `Stop` 日志正好说明了隔离：同一个 `summary_hook`，在子 Agent 结束时数到 **19** 次工具调用，在父 Agent 结束时只数到 **1** 次，也就是那一次 `task`。

从代理日志看，这次一共发了 8 次请求：父 Agent 1 次，子 Agent 6 次，父 Agent 再 1 次。把每次请求里 `messages` 的大小量出来：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s06-subagent/03-context-size.jpg" target="_blank" rel="noopener" aria-label="查看原图：8 次请求按时间排列的 messages 字符数：父 Agent 第一次 95；子 Agent 六次分别是 285、1.7k、65k、73k、81k、85k；父 Agent 第二次 1.7k"><img src="/notes-assets/lcc-s06-subagent/03-context-size.jpg" alt="8 次请求按时间排列的 messages 字符数：父 Agent 第一次 95；子 Agent 六次分别是 285、1.7k、65k、73k、81k、85k；父 Agent 第二次 1.7k" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

sub 2 到 sub 3 那一下从 1.7k 跳到 65k，是因为子 Agent 跑了一条全仓库的 `grep`，输出正好顶到了 `run_bash` 的 50000 字符上限，被截断了。

如果没有子 Agent，这 85k 字符会**原封不动**留在父 Agent 的 `history` 里，之后每次提问都要带上。现在父 Agent 只收到了 1.7k 字符的结论。

## 想一想（本站补充）

1. 子 Agent 跑满 30 轮被叫停时，已经改过的文件怎么办？父 Agent 能知道它改了哪些文件吗？如果要让父 Agent 知道，你会让 `run_subagent` 多返回什么？
2. 父 Agent 在同一次回复里发了两个 `task`，它们是并行还是串行？如果改成并行，共享的 `WORKDIR` 和权限确认里的 `input()` 会出什么问题？
3. 子 Agent 读过的文件内容全都丢掉了。如果父 Agent 后面恰好需要其中某个文件，会发生什么？什么样的任务交给子 Agent 才划算？
4. 如果想允许两层委派（子 Agent 也能派孙 Agent），只把 `task` 加进 `SUB_TOOLS` 够吗？还需要加什么保护，才不会无限套娃？
5. 子 Agent 只返回**最后一次**回复的文字。如果它中间某轮写下了关键发现，最后一轮只说了一句 "Done."，父 Agent 拿到的是什么？可以怎么改？

## 试一下

```sh
cd learn-claude-code
python s06_subagent/code.py
```

试试这些 prompt：

1. `Use a subtask to find what testing framework this project uses`（子 Agent 去读文件，主 Agent 只收结论）
2. `Delegate: read all .py files in agents/ and summarize what each one does`
3. `Use a task to create s06_subagent/example/string_tools.py with a slugify(text: str) function, then verify it from the parent agent`

观察重点：是否出现 `[Subagent started]` / `[Subagent done]`？子 Agent 的工具调用是否以 `[sub] ...` 输出？父 Agent 是否只接收到 `task` 返回的最终文本？

## 接下来

Agent 现在能拆任务了。但每个任务需要的知识不一样：改前端组件需要知道 React 规范，写 SQL 需要知道表结构。这些知识全塞进 system prompt，上下文直接爆了。

[s07 Skill Loading](/notes/lcc-s07-skill-loading/) → 技能按需注入，不在 system prompt 里堆文档。用到的时候才加载，和读文件一样自然。
