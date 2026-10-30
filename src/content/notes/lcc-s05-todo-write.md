---
title: 'learn-claude-code s05：TodoWrite，没有计划的 Agent，做着做着就偏了'
date: '2026-10-30'
tags: [Agent, Harness, Planning, learn-claude-code]
summary: 转载 learn-claude-code 第 5 课，补充配图、code.py 逐段导读和一次真实运行。todo_write 不增加任何执行能力，只让模型先列清单再动手；harness 负责校验清单，连续三轮没更新就在工具结果后面追加提醒。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s05_todo_write)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"没有计划的 agent 走哪算哪"* — 先列步骤再动手，长任务更不容易漏项。
>
> **Harness 层**: 规划 — 让 Agent 在动手之前先想清楚。

## 问题

给 Agent 一个复杂任务："把所有 Python 文件改成 snake_case 命名，然后跑测试，修好失败。"

Agent 开始干活，改了 3 个文件，跑了个测试，发现 2 个失败，开始修。修着修着，它忘了最初是"改成 snake_case"，测试失败把注意力全吸走了。

对话越长越严重：工具结果不断填满上下文，系统提示的影响力被稀释。一个 10 步重构，做完 1-3 步就开始即兴发挥，因为 4-10 步已经被挤出注意力了。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s05-todo-write/01-todo-is-a-tool.jpg" target="_blank" rel="noopener" aria-label="查看原图：LLM 调用 todo_write 把清单交给 TodoManager；TodoManager 按最多 20 项、内容必填、只能一个 in_progress 三条规则校验，通过就渲染成带 [x] [>] [ ] 标记的清单作为 tool_result 返回，不通过就返回 Error"><img src="/notes-assets/lcc-s05-todo-write/01-todo-is-a-tool.jpg" alt="LLM 调用 todo_write 把清单交给 TodoManager；TodoManager 按最多 20 项、内容必填、只能一个 in_progress 三条规则校验，通过就渲染成带 [x] [>] [ ] 标记的清单作为 tool_result 返回，不通过就返回 Error" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

S05 保留 S04 的工具分发、权限检查和 Hooks，再加入 `todo_write` 与 reminder 计数器。`todo_write` 只更新计划状态，实际工作仍由原有工具完成。

新工具仍通过 `TOOL_HANDLERS[block.name]` 分发。连续三个工具调用轮次没有使用 `todo_write` 时，Harness 会把 reminder 追加到第三轮的工具结果中。

## 工作原理

**TodoManager** 持有内存中的任务列表，负责校验更新，并把渲染结果返回给模型。`run_todo_write` 同时把这份状态打印到终端：

```python
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, todos: list | str) -> str:
        # 替换当前列表前进行解析与校验
        validated = []
        ...
        self.items = validated
        return self.render()

    def render(self) -> str:
        # [ ] 待办, [>] 进行中, [x] 已完成
        ...


TODO = TodoManager()

def run_todo_write(todos: list | str) -> str:
    output = TODO.update(todos)
    print(output)
    return output
```

一次更新最多包含 20 项；每项都必须有非空的 `content`；同一时间只能有一个 `in_progress`。字符串输入可以是 JSON，也可以是 Python 列表表示，解析过程不使用 `eval`。

工具定义和其他 5 个工具一起加入 dispatch map：

```python
TOOLS = [
    {"name": "bash",       ...},
    {"name": "read_file",  ...},
    {"name": "write_file", ...},
    {"name": "edit_file",  ...},
    {"name": "glob",       ...},
    # s05: 新增一条
    {"name": "todo_write", "description": "Create and manage a task list ...",
     "input_schema": {
         "type": "object",
         "properties": {
             "todos": {
                 "type": "array",
                 "items": {
                     "type": "object",
                     "properties": {
                         "content": {"type": "string"},
                         "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                     },
                 },
             },
         },
     },
    },
]

TOOL_HANDLERS["todo_write"] = run_todo_write
```

**Reminder**：连续三个工具调用轮次没有使用 `todo_write` 时，reminder 会追加到第三轮的结果中，随后计数器清零：

```python
rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
if rounds_since_todo >= 3:
    results.append({
        "type": "text",
        "text": "<reminder>Update your todos.</reminder>",
    })
    rounds_since_todo = 0
```

Agent 收到任务后的典型流程：先调 `todo_write` 列出所有步骤（全 `pending`）→ 做一个步骤，改成 `in_progress` → 做完改成 `completed` → 看下一个 `pending` → 继续。

**关键洞察**：todo_write 不给 Agent 增加任何**执行能力**。它增加的是**规划能力**。

## 相对 s04 的变更

| 组件 | 之前 (s04) | 之后 (s05) |
|------|-----------|-----------|
| 工具数量 | 5 (bash, read, write, edit, glob) | 6 (+todo_write) |
| 规划能力 | 无 | 带状态的 TODO 列表 + reminder |
| SYSTEM 提示 | 通用提示 | 加入 "先计划再执行" 引导 |
| 循环 | 工具分发与 Hooks | 保留分发路径，加入 rounds_since_todo 和 reminder 注入 |

## 代码导读：把 code.py 从头读一遍（本站补充）

完整的 [`s05_todo_write/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s05_todo_write/code.py) 有 370 行，比 s04 多了不到 100 行，几乎都在 `TodoManager` 和循环里的计数器上。

### 第一段：计划是「劝」出来的

```python
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)
```

模型会不会先列计划，靠的是 system prompt 里这一句话。代码里**没有任何地方强制**模型第一步必须调用 `todo_write`。它可以完全不理会，直接开始干活。

后面的 reminder 也是同样的思路：harness 不替模型做计划，只在它忘了的时候提醒一下。

### 第二段：`TodoManager.update` 的三件事

```python
def update(self, todos: list | str) -> str:
    if isinstance(todos, str):
        try:
            todos = json.loads(todos)
        except json.JSONDecodeError:
            try:
                todos = ast.literal_eval(todos)
            except (SyntaxError, ValueError) as e:
                raise ValueError("todos must be a list or JSON array string") from e
    ...
    self.items = validated
    return self.render()
```

**1. 兼容模型的「手滑」。** schema 里写明了 `todos` 是一个数组，但模型有时会把整个数组当成一段字符串传进来。所以先试 `json.loads`，不行再试 `ast.literal_eval`。

为什么不用 `eval`？因为 `eval` 会执行字符串里的任意 Python 代码，而这段字符串是模型生成的。`ast.literal_eval` 只认列表、字典、字符串、数字这些「字面量」，遇到函数调用直接报错。

**2. 逐条校验。** 每一项必须是字典，`content` 去掉空格后不能为空，`status` 只能是三个值之一，整张表最多 20 项，`in_progress` 最多一个。任何一条不满足，就抛出 `ValueError`。

**3. 整张表替换。** 校验全部通过后，才执行 `self.items = validated`。中途出错的话，旧的清单原封不动。

### 第三段：schema 写了，代码为什么还要再查一遍

真实代码里的 schema 比上面正文里的简化版严格得多：

```python
{"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
 "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "maxItems": 20,
   "items": {"type": "object", "properties": {"content": {"type": "string", "minLength": 1},
     "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
   "required": ["content", "status"]}}}, "required": ["todos"]}},
```

`maxItems: 20`、`minLength: 1`、`enum`、`required`，跟 `update` 里的校验几乎一一对应。那为什么还要在代码里再查一遍？

因为 **schema 是写给模型看的说明，不是保证。** 模型通常会照着 schema 生成参数，但不是每次都照做。真正能拦住坏数据的，只有 harness 里的代码。

还有一条规则 schema 根本写不出来：「整张表里最多只有一个 `in_progress`」。这种跨多个元素的约束，只能靠代码检查。

### 第四段：为什么每次都传整张清单

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s05-todo-write/02-replace-not-patch.jpg" target="_blank" rel="noopener" aria-label="查看原图：第 1、3、7 轮各发一次完整清单，状态从 Inspect 进行中，到 Refactor 进行中，再到三项全部完成；同样的清单进来，就得到同样的状态"><img src="/notes-assets/lcc-s05-todo-write/02-replace-not-patch.jpg" alt="第 1、3、7 轮各发一次完整清单，状态从 Inspect 进行中，到 Refactor 进行中，再到三项全部完成；同样的清单进来，就得到同样的状态" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

`todo_write` 没有「把第 2 项改成已完成」这种操作，模型每次都要把**整张清单**重新发一遍，harness 整个替换掉。

这样设计有一个好处：**同一次调用执行两次，结果一模一样。** 网络重试、模型重复调用都不会把状态弄乱。这个性质叫幂等。代价是清单越长，每次调用要传的内容越多，所以才限制最多 20 项。

### 第五段：循环里的两处改动

```python
handler = TOOL_HANDLERS.get(block.name)
try:
    output = handler(**block.input) if handler else f"Unknown: {block.name}"
except Exception as e:
    output = f"Error: {e}"
...
if block.name == "todo_write":
    used_todo = True
...
rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
if rounds_since_todo >= 3:
    results.append({"type": "text",
                    "text": "<reminder>Update your todos.</reminder>"})
    rounds_since_todo = 0
```

**第一处：工具调用包进了 `try`。** 之前任何一个工具抛出异常，整个程序都会崩。现在不管哪个工具出错，都会变成一句 `Error: ...` 交给模型。

**第二处：reminder 计数器。** 它数的是「轮」，不是「次」：一轮里调了好几个工具，只要有一个是 `todo_write`，这一轮就算更新过。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s05-todo-write/03-reminder.jpg" target="_blank" rel="noopener" aria-label="查看原图：第 1 到 3 轮分别只调了 bash、read_file、write_file，计数器从 1 数到 3；第 3 轮的工具结果后面追加了 reminder，计数器清零；第 4 轮模型调用 todo_write"><img src="/notes-assets/lcc-s05-todo-write/03-reminder.jpg" alt="第 1 到 3 轮分别只调了 bash、read_file、write_file，计数器从 1 数到 3；第 3 轮的工具结果后面追加了 reminder，计数器清零；第 4 轮模型调用 todo_write" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

reminder 是一个 `text` 块，跟在这一轮所有 `tool_result` 的后面，放进**同一条** user 消息里。

`rounds_since_todo` 是 `agent_loop` 里的局部变量。每次你在终端里提一个新问题，`agent_loop` 重新开始，计数器也就从 0 重新数。

### 第六段：清单存在两个地方

清单在程序里有两份：

- **`TODO.items`：** harness 内存里的那份，经过了校验，是准确的状态。
- **messages 里的 `tool_result`：** 每次 `render()` 出来的文字，是模型看到的那份。

两份都只活在这一次运行里。程序一退出，清单就没了。

## 实际跑一次（本站补充）

把课程自带的 `example/hello.py` 复制到一个空目录：

```python
def greet(name):
    message = "Hello, " + name
    print(message)


greet("Claude")
```

提问 `Refactor hello.py: add type hints, docstrings, and a main guard, then run it`。一共调用了 8 次模型、7 次工具，过程是这样的（省略了一部分 `[HOOK]` 日志）：

```
[HOOK] todo_write(...)
## Current Tasks
[>] Inspect hello.py
[ ] Refactor with type hints, docstrings, and a main guard
[ ] Run hello.py and verify output
(0/3 completed)

[HOOK] read_file(['hello.py'])
## Current Tasks
[x] Inspect hello.py
[>] Refactor with type hints, docstrings, and a main guard
[ ] Run hello.py and verify output
(1/3 completed)

[HOOK] write_file(['hello.py', '"""Print a friendly greeting."""...'])
## Current Tasks
[x] Inspect hello.py
[x] Refactor with type hints, docstrings, and a main guard
[>] Run hello.py and verify output
(2/3 completed)

[HOOK] bash(['python3 hello.py'])
## Current Tasks
[x] Inspect hello.py
[x] Refactor with type hints, docstrings, and a main guard
[x] Run hello.py and verify output
(3/3 completed)

[HOOK] Stop: session used 7 tool calls
```

第一个工具调用就是 `todo_write`，列了 3 步。之后模型每做完一步，就调一次 `todo_write` 更新状态，做一步、更新一次，交替进行。因为它更新得很勤快，计数器从来没数到 3，这次运行里 reminder 一次都没有出现。

7 次工具调用里有 4 次是 `todo_write`，真正干活的只有 3 次。这就是「规划」的成本。

## 想一想（本站补充）

1. 如果模型完全不调用 `todo_write`，直接开始干活，harness 会做什么？它要连续多少轮不更新，才会收到第一次提醒？
2. 如果把 `todo_write` 改成「只传要改的那一项」，比如 `{"index": 1, "status": "completed"}`，能省多少传输内容？会多出哪些新问题？提示：想想模型记错了下标，或者同一个调用被执行了两次。
3. 为什么 reminder 要放在工具结果后面，而不是写进 system prompt 里？回到本文开头「问题」一节最后一段找线索。
4. 程序重启后，清单就没了。如果想让清单跨会话保留，你会把它存在哪里？什么时候读回来？

## 试一下

```sh
cd learn-claude-code
python s05_todo_write/code.py
```

试试这些 prompt：

1. `Refactor s05_todo_write/example/hello.py: add type hints, docstrings, and a main guard`（先列 3 步再执行）
2. `Create a Python package under s05_todo_write/example/demo_pkg with __init__.py, utils.py, and tests/test_utils.py`
3. `Review Python files under s05_todo_write/example and fix any style issues`

观察重点：第一次工具调用是不是 `todo_write`？TODO 列了几步？执行过程中状态有没有从 `pending` 变成 `in_progress` / `completed`？

## 接下来

Agent 能计划了。但如果一个任务太大，比如"重构整个认证模块"，光靠 TODO 列表不够。这个任务本身就是几十个小任务的集合，放在同一个对话里会被上下文淹没。

[s06 Subagent](/notes/lcc-s06-subagent/) → 把大任务拆成子任务，每个子任务派一个独立的 Agent。它们有自己的干净上下文，不会互相污染。
