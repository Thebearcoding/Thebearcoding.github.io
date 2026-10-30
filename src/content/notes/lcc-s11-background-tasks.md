---
title: 'learn-claude-code s11：Background Tasks，慢命令丢到后台，下一轮再收结果'
date: '2026-10-30'
tags: [Agent, Harness, Concurrency, learn-claude-code]
summary: 转载 learn-claude-code 第 11 课，补充配图、code.py 导读和一次真实运行。bash 多了一个 run_in_background 参数，后台线程跑命令，当场先回一个占位结果；跑完的结果放进队列，等下一次调用模型前才以 task_notification 的形式塞进对话。真实运行里命令在这一轮快结束时就跑完了，结果却一直等到用户问下一句才被看到。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s11_background_tasks)（MIT License，Copyright (c) 2024 shareAI Lab），手绘配图、「代码导读」「实际跑一次」「想一想」三节为本站补充，SVG 图示来自原仓库。

> *"慢操作放后台，Agent Loop 继续运行"* — 后台线程执行命令，后续轮次收集完成结果。
>
> **Harness 层**: 后台 — 异步执行, 不阻塞主循环。

## 问题

读取文件或运行 `git status` 通常很快，同步执行时等待并不明显。但安装依赖、执行完整测试或构建项目可能持续几分钟。在命令返回前，Harness 无法处理当前响应中的下一个工具调用，也不能进入下一轮。

如果后续工作并不依赖这个命令，继续等待就没有必要。例如，Agent 启动完整测试后，本来还可以检查文档或整理其他文件，但同步执行会让整个 Agent Loop 停在这次 Bash 调用上。

S11 要解决的问题是：让耗时的 Bash 命令在后台执行，使 Agent Loop 可以继续处理其他工作，并在后续轮次收集完成结果。

## 解决方案

<figure style="margin:1.5rem 0">
<img src="/notes-assets/lcc-s11-background-tasks/background-tasks-overview.svg" alt="Background Tasks Overview" loading="lazy" decoding="async" style="width:100%;height:auto;background:#fff;border-radius:8px;padding:8px;box-sizing:border-box" />
</figure>

本章把慢操作放入后台线程。当前工具调用先返回一个占位 `tool_result`，Agent Loop 可以继续运行；后续轮次开始时再收集已经完成的结果，以通知形式加入对话。

同步 vs 后台：

| | 同步 (s04) | 后台 (s11) |
|---|---|---|
| 慢操作 | 当前工具调用被阻塞 | 后台线程执行 |
| Agent Loop | 等待命令返回 | 收到占位结果后继续运行 |
| 结果 | 命令结束后返回 | 先返回 `bg_id`，后续轮次收集结果 |
| 判断标准 | — | bash 的 `run_in_background` 参数 |

## 工作原理

### should_run_background: 显式请求

模型通过 bash 工具的 `run_in_background` 参数请求后台执行。只有参数明确为 `true`，并且工具是 bash 时，才会进入后台执行路径。其他调用仍然同步执行。

```python
def should_run_background(tool_name: str, tool_input: dict) -> bool:
    return (
        tool_name == "bash"
        and tool_input.get("run_in_background") is True
    )
```

不再根据 `install`、`build` 或 `test` 等关键词猜测。是否进入后台由工具调用明确决定。

### BackgroundManager: 后台执行与生命周期

`BackgroundManager` 保存任务状态和完成队列。`start()` 先登记任务，再启动 daemon 线程，并立即返回 `bg_id`：

```python
class BackgroundManager:
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self._ready = []
        self._lock = threading.Lock()

    def start(self, block) -> str:
        # Register task, then run _run() in a daemon thread.
        ...

    def _run(self, task_id: str, command: str):
        output, exit_code = _run_bash_process(command)
        status = "completed" if exit_code == 0 else "failed"
        with self._lock:
            self.tasks[task_id]["status"] = status
            self.results[task_id] = _format_bash_result(output, exit_code)
            self._ready.append(task_id)
```

命令以非零状态退出或 worker 抛出异常时，任务会进入 `failed`。Shell 会在独立的进程组中启动；命令完成、超时，或 Agent 经正常路径、`SIGTERM` 退出时，运行时会停止原进程组。这只是生命周期清理，并不是沙箱；另建 session 的进程仍可能离开该进程组。

### collect_background_results: 通知收集

后续轮次开始时，`collect()` 从完成队列中取出结果，并格式化为 `<task_notification>` 通知：

```python
def collect_background_results() -> list[str]:
    return BACKGROUND.collect()
```

通知不复用原始 `tool_use_id`。原始 tool call 已经用占位 `tool_result` 回复了；后续收集完成结果时，会用 `task_notification` 格式把它作为独立事件加入对话。一个 `tool_use` 仍然只对应一个 `tool_result`。

### 循环中的集成

每次调用 LLM 前，Agent Loop 先收集已经完成的后台结果。`execute_tool()` 仍然在主线程执行 `PreToolUse`，然后再选择同步或后台执行：

```python
while True:
    inject_background_results(messages)
    response = client.messages.create(...)

def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)
    if should_run_background(block.name, block.input):
        task_id = start_background_task(block)
        output = f"[Background task {task_id} started]"
    else:
        output = call_tool(block)
    trigger_hooks("PostToolUse", block, output)
    return output
```

慢操作先返回一个带 `bg_id` 的占位 tool_result。后台结果不会主动唤醒 Agent；下一次进入 Agent Loop 时，`inject_background_results()` 才会收集已经完成的结果。

### 合起来跑

```
Turn 1:
  LLM → bash "npm install" (run_in_background=true)
  → start_background_task → bg_0001
  → tool_result: "[Background task bg_0001 started]..."
  → LLM: "OK, I'll check later. Let me also read the config."

Turn 2:
  LLM → read_file "package.json" (fast, sync)
  → tool_result: file content

Turn 3:
  → collect bg_0001 as <task_notification>
  → LLM sees: config file + install notification in one message
```

npm install 在后台运行时，Agent Loop 继续执行了 read_file。

## 本章新增了什么

| 组件 | S04 Kernel | S11 |
|------|-----------|-----------|
| 执行模型 | 全部同步 | 慢操作后台线程 + 通知注入 |
| bash schema | `command` | `command` + `run_in_background` |
| 新函数 | — | `should_run_background`, `start_background_task`, `collect_background_results`, `inject_background_results` |
| 新类型 | — | `BackgroundManager` |
| 通知格式 | — | `<task_notification>`（不复用 tool_use_id） |
| 循环行为 | 工具同步执行 | 显式后台执行，后续轮次收集完成结果 |
| 工具 | 5 | 5（bash schema 增加一个参数） |


## 代码导读：README 没讲到的部分（本站补充）

完整的 [`s11_background_tasks/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s11_background_tasks/code.py) 有 547 行，也是在 s04 的骨架上加的。改动集中在两处：`run_bash` 换成了能管理子进程的版本，再加上 `BackgroundManager` 和注入通知的几个函数。

### 第一段：bash 多一个参数，模型自己决定

```python
{"name": "bash", "description": "Run a shell command.",
 "input_schema": {"type": "object",
                  "properties": {
                      "command": {"type": "string"},
                      "run_in_background": {"type": "boolean"}},
                  "required": ["command"]}},
```

SYSTEM 里只加了一句提示：`Set run_in_background to true only for independent Bash commands.`，也就是「只有跟后续步骤无关的命令，才放到后台」。

要不要放到后台，harness 不猜，由模型通过参数明确告诉它。`run_bash(command, run_in_background=False)` 这个函数也接收这个参数，但完全不用它：走到 `run_bash` 的一定是同步执行，参数只是为了让 `handler(**block.input)` 不因为多了一个参数而报错。

### 第二段：每条命令一个进程组，结束时整组清理

```python
def _run_bash_process(command: str) -> tuple[str, int | None]:
    process = None
    try:
        popen_kwargs = {} if os.name == "nt" else {"start_new_session": True}
        process = subprocess.Popen(command, shell=True, cwd=WORKDIR,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors="replace", **popen_kwargs)
        with _shell_process_lock:
            _shell_processes.add(process)
        stdout, stderr = process.communicate(timeout=120)
        ...
    finally:
        if process is not None:
            _stop_process_group(process)
            ...
```

之前几课用的是 `subprocess.run`，这里换成了 `Popen`，多做了三件事：

1. `start_new_session=True`：让 shell 自成一个**进程组**。命令里启动的子进程（比如 `npm` 又拉起的 `node`）都在这个组里。
2. `finally` 里**无论如何**都调用 `_stop_process_group`：先发 `SIGTERM`，等 0.05 秒，再发 `SIGKILL`，整组一起停掉。这样命令里用 `&` 偷偷放到后台的进程，也不会在命令结束后继续跑。
3. 所有进程登记在 `_shell_processes` 里。程序正常退出（`atexit`）或收到 `SIGTERM` 时，把还在跑的进程组全部清掉。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s11-background-tasks/03-process-group.jpg" target="_blank" rel="noopener" aria-label="查看原图：每条命令在自己的进程组里运行，组里有 sh、npm、node；超时 120 秒、命令结束、Agent 退出或收到 SIGTERM 这三种情况，都会对整个进程组先发 SIGTERM 再发 SIGKILL；用 setsid 另开会话的子进程可以逃出这个组，所以这只是清理，不是沙箱"><img src="/notes-assets/lcc-s11-background-tasks/03-process-group.jpg" alt="每条命令在自己的进程组里运行，组里有 sh、npm、node；超时 120 秒、命令结束、Agent 退出或收到 SIGTERM 这三种情况，都会对整个进程组先发 SIGTERM 再发 SIGKILL；用 setsid 另开会话的子进程可以逃出这个组，所以这只是清理，不是沙箱" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

注意 `communicate(timeout=120)`：**后台命令也有 120 秒的超时**。放到后台只是不让主循环干等，不代表它能一直跑下去。

原文也特别说明了：这是生命周期清理，不是沙箱。命令里要是再用 `setsid` 开一个新会话，那个进程就不在这个组里了。

### 第三段：`BackgroundManager`，一把锁管三样东西

```python
class BackgroundManager:
    def __init__(self):
        self.tasks: dict[str, dict] = {}    # bg_id -> 命令、状态
        self.results: dict[str, str] = {}   # bg_id -> 输出
        self._ready: list[str] = []         # 跑完、还没收的 bg_id
        self._counter = 0
        self._lock = threading.Lock()
```

后台线程在写，主线程在读，所以这三样东西的每次读写都包在 `with self._lock:` 里。

`start` 的顺序是：先在锁里登记任务，再启动线程。如果反过来，线程可能在登记之前就跑完了，`_run` 里会找不到这个任务。线程启动失败的话，把刚登记的任务删掉，再把异常抛出去。

`_run` 里，退出码是 0 才算 `completed`，其他情况（非零退出码、超时、异常）都算 `failed`。

### 第四段：收结果时，只给 500 个字符

```python
def collect(self) -> list[str]:
    with self._lock:
        ready = []
        for task_id in self._ready:
            task = self.tasks.pop(task_id, None)
            result = self.results.pop(task_id, "")
            ...
        self._ready.clear()

    notifications = []
    for task_id, task, result in ready:
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{task_id}</task_id>\n"
            f"  <status>{task['status']}</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{result[:500]}</summary>\n"
            f"</task_notification>"
        )
```

两个细节：

- 在锁里只做「把数据拿出来」这一件事，拼字符串放到锁外面。锁拿着的时间越短越好。
- `result[:500]`：通知里只放输出的**前 500 个字符**，然后 `pop` 掉，完整输出就没了。跑一次完整测试，失败信息往往在几千行日志的最后面，模型是看不到的。

### 第五段：通知塞进哪条消息

```python
def inject_background_results(messages: list) -> int:
    notifications = collect_background_results()
    if not notifications:
        return 0
    blocks = [{"type": "text", "text": item} for item in notifications]
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, list):
            content.extend(blocks)
        else:
            messages[-1]["content"] = [{"type": "text", "text": str(content)}, *blocks]
    ...
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s11-background-tasks/02-tool-result-pairing.jpg" target="_blank" rel="noopener" aria-label="查看原图：assistant 发出 tool_use bash，紧跟着的 user 消息用同一个 id 回了占位结果，两者靠 id 配对；后面某一轮，用户的问题和 task_notification 作为两个 text 块放在同一条 user 消息里，通知没有 tool_use_id"><img src="/notes-assets/lcc-s11-background-tasks/02-tool-result-pairing.jpg" alt="assistant 发出 tool_use bash，紧跟着的 user 消息用同一个 id 回了占位结果，两者靠 id 配对；后面某一轮，用户的问题和 task_notification 作为两个 text 块放在同一条 user 消息里，通知没有 tool_use_id" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

每次调用模型之前，最后一条消息一定是 user 消息：要么是用户刚输入的问题，要么是一批 `tool_result`。通知就以 `text` 块的形式**追加到这条消息里**，而不是另起一条，避免出现两条连续的 user 消息。

通知里没有 `tool_use_id`。那个 `tool_use` 早在当时就用占位结果回复过了，一个调用只能对应一个结果。

### 第六段：权限检查在主线程，进后台之前

```python
def execute_tool(block) -> str:
    blocked = trigger_hooks("PreToolUse", block)
    if blocked is not None:
        return str(blocked)
    if should_run_background(block.name, block.input):
        ...
        output = f"[Background task {task_id} started] The result will be collected on a later turn."
    else:
        output = call_tool(block)
    trigger_hooks("PostToolUse", block, output)
    return output
```

`PreToolUse` 里的权限检查可能会调用 `input()` 问用户。这必须在主线程、命令还没放进后台的时候做完。要是等后台线程跑到一半再弹确认，它就会和主循环抢终端输入。

反过来，`PostToolUse` 拿到的 `output` 是那句占位文字，不是命令的真实输出。像「输出太大就警告」这种 hook，对后台命令就不起作用了。

## 实际跑一次（本站补充）

在一个有两个 Markdown 文件的空目录里，先说一句，等 12 秒，再问第二句：

````
s11 >> Run `sleep 5 && echo build-ok` in the background, then list all Markdown files in this directory.
[HOOK] bash(['sleep 5 && echo build-ok', True])
  [background] started bg_0001: sleep 5 && echo build-ok
[HOOK] glob(['**/*.md'])
[HOOK] Stop: session used 2 tool calls
Background task started (`bg_0001`).

Markdown files:
- `README.md`
- `docs/guide.md`

s11 >> Did the background command finish?
  [background] collected bg_0001: completed
[HOOK] Stop: session used 2 tool calls
Yes. The background command completed successfully with output:

```text
build-ok
```
````

对照代理日志的时间：

```
20:21:16 msgs=1 -> tool_use   ← 发出后台 bash，sleep 开始
20:21:19 msgs=3 -> tool_use   ← 发出 glob
20:21:21 msgs=5 -> end_turn   ← 第一轮结束
20:21:25 msgs=7 -> end_turn   ← 第二句话，带着通知
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s11-background-tasks/01-finished-early.jpg" target="_blank" rel="noopener" aria-label="查看原图：上面是 Agent 循环：发出后台 bash、glob、给出回答，然后这一轮结束等用户；下面是后台线程：sleep 5 从第一次回复开始，在第三次回复前后跑完，结果进入队列；直到用户问下一句，通知才被注入，模型在第四次回复里看到"><img src="/notes-assets/lcc-s11-background-tasks/01-finished-early.jpg" alt="上面是 Agent 循环：发出后台 bash、glob、给出回答，然后这一轮结束等用户；下面是后台线程：sleep 5 从第一次回复开始，在第三次回复前后跑完，结果进入队列；直到用户问下一句，通知才被注入，模型在第四次回复里看到" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

`sleep 5` 在 20:21:16 左右开始，20:21:21 左右就跑完了。但 20:21:19 那次请求发出时它还没完，20:21:21 模型直接给出回答，**这一轮就结束了**，之后再也没有调用模型。结果就一直躺在 `_ready` 队列里。

直到我问了第二句，`agent_loop` 开头的 `inject_background_results` 才把它收进来。这一次请求的最后一条消息是这样的：

```json
{
 "role": "user",
 "content": [
  {"type": "text", "text": "Did the background command finish?"},
  {"type": "text", "text": "<task_notification>\n  <task_id>bg_0001</task_id>\n  <status>completed</status>\n  <command>sleep 5 && echo build-ok</command>\n  <summary>build-ok</summary>\n</task_notification>"}
 ]
}
```

用户的问题和后台通知，并排放在同一条 user 消息里。

原文说「后台结果不会主动唤醒 Agent」，这次运行正好演示了它的后果：如果我不问第二句，模型永远不会知道 `build-ok` 已经出来了。

## 想一想（本站补充）

1. 这次运行里，后台命令跑完的时候这一轮已经结束了，结果一直等到用户下一句话。如果想让 Agent 在后台任务完成时**主动**继续工作，需要改哪里？会引入什么新问题？
2. 后台命令同样有 120 秒的超时。一次要跑 3 分钟的 `npm install` 放到后台，会发生什么？模型收到的通知是什么样的？
3. 通知只保留输出的前 500 个字符，完整输出被丢掉了。测试失败的关键信息在第 3000 个字符的位置，模型该怎么办？你会怎么改？（可以回想 s08 是怎么处理大结果的。）
4. `PostToolUse` hook 拿到的是占位文字。如果有一个审计 hook 要记录每条命令的真实输出，后台命令该怎么接进去？
5. 模型在同一次回复里放了两个后台命令，两个都在改同一个文件，会发生什么？harness 应该管这件事吗？
6. 为什么权限确认必须在主线程、放进后台之前完成？如果放到后台线程里去问用户，会出什么问题？

## 试一下

```sh
cd learn-claude-code
python s11_background_tasks/code.py
```

试试这些 prompt：

1. `Run pip list in the background and find all Python files in this directory`
2. `Run npm install (use run_in_background) and while waiting, read package.json`
3. `Run a short sleep in the background, then list all Markdown files`

观察重点：显式设置 `run_in_background` 后，命令有没有被送到后台？`bg_id` 是否返回？后续轮次有没有以 `<task_notification>` 格式收集完成结果？

## 接下来

后台任务解决了"慢操作不阻塞"。但如果想定时做某件事呢？比如"每天早上 9 点跑测试"、"每 5 分钟检查一次服务器状态"。

[s12 Cron Scheduler](/notes/lcc-s12-cron-scheduler/) → 给 Agent 装一个闹钟。
