---
title: 'learn-claude-code s04：Hooks，挂在循环上，不写进循环里'
date: '2026-10-30'
tags: [Agent, Harness, Hooks, learn-claude-code]
summary: 转载 learn-claude-code 第 4 课，补充配图、code.py 逐段导读和一次真实运行。把 s03 硬编码在循环里的权限检查搬到 PreToolUse hook 上，四个事件覆盖输入、执行前、执行后和退出，循环只负责调用 trigger_hooks()。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s04_hooks)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"挂在循环上, 不写进循环里"* — hook 在工具执行前后注入扩展逻辑。
>
> **Harness 层**: hook — 扩展点不侵入循环。

## 问题

s03 的 Agent 有权限检查了。但每次加一个新检查，比如"记录每次 bash 调用"、"操作后自动 git add"，都要修改 `agent_loop` 函数。

循环很快就变成了这样：

```python
def agent_loop(messages):
    while True:
        # ... LLM call ...
        for block in response.content:
            if block.type != "tool_use":
                continue
            log_to_file(block)          # 加一行
            check_permission(block)     # 加一行
            notify_slack(block)         # 又加一行
            output = execute(block)
            auto_git_add(block)         # 再加一行
            # ... 很快循环就认不出来了
```

你想扩展的是 Agent 的行为，但你改的却是循环本身。循环应该是一个稳定的核心，扩展应该挂在外面。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s04-hooks/01-four-hook-points.jpg" target="_blank" rel="noopener" aria-label="查看原图：user input 到 LLM 之间是 UserPromptSubmit；LLM 调工具前是 PreToolUse，工具执行完回到 LLM 前是 PostToolUse；LLM 退出到 return 之前是 Stop"><img src="/notes-assets/lcc-s04-hooks/01-four-hook-points.jpg" alt="user input 到 LLM 之间是 UserPromptSubmit；LLM 调工具前是 PreToolUse，工具执行完回到 LLM 前是 PostToolUse；LLM 退出到 return 之前是 Stop" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

s03 的循环和权限逻辑完全保留。唯一的变动是把 `check_permission()` 从循环体内移到了 hook 上，循环不再直接调用任何检查函数，改为 `trigger_hooks("PreToolUse", block)`，由注册表决定跑什么。

四个事件，覆盖一个完整的 agent cycle：

| 事件 | 触发时机 | 典型用途 |
|------|---------|---------|
| UserPromptSubmit | 用户输入提交后、进入 LLM 前 | 输入验证、注入上下文 |
| PreToolUse | 工具执行前 | 权限检查、日志记录 |
| PostToolUse | 工具执行后 | 副作用（自动 git add 等）、输出检查 |
| Stop | 循环即将退出时 | 收尾清理、决定是否继续循环 |

扩展通过 `register_hook()` 添加，循环只调用 `trigger_hooks()`。

## 工作原理

**hook 注册表**：一个字典，事件名映射到回调列表。

```python
HOOKS = {
    "UserPromptSubmit": [],
    "PreToolUse": [],
    "PostToolUse": [],
    "Stop": [],
}

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:   # 返回值 ≠ None → hook 说"停"
            return result
    return None
```

`PreToolUse` 返回非 `None` 时，本次工具执行被阻止；`Stop` 返回非 `None` 时，循环继续。`UserPromptSubmit` 和 `PostToolUse` 的返回值不参与控制流。

**UserPromptSubmit** 在用户输入提交后、进入 LLM 前触发。以下 hook 记录当前工作目录：

```python
def context_inject_hook(query: str) -> str | None:
    """Inject current working directory info into every prompt."""
    print(f"\033[90m[HOOK] UserPromptSubmit: working in {WORKDIR}\033[0m")
    return None   # return None = 不修改提示词，直接放行

register_hook("UserPromptSubmit", context_inject_hook)
```

在主循环中，用户输入后立即触发：

```python
query = input("s04 >> ")
trigger_hooks("UserPromptSubmit", query)   # ← 进入 LLM 之前
history.append({"role": "user", "content": query})
agent_loop(history)
```

**PreToolUse / PostToolUse**，工具执行前后的 hook。s03 的权限检查逻辑现在包装成 PreToolUse hook，再加一个日志 hook 和一个大输出提醒：

```python
# PreToolUse: 权限检查（s03 的逻辑，从循环移到 hook）
def permission_hook(block):
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                return "Permission denied by deny list"
    if block.name in ("read_file", "write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            choice = input("   Allow? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

# PreToolUse: 日志
def log_hook(block):
    print(f"[HOOK] {block.name}(...)")

# PostToolUse: 大文件提醒
def large_output_hook(block, output):
    if len(str(output)) > 100000:
        print(f"[HOOK] ⚠ Large output from {block.name}")

register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
```

**Stop** 在循环即将退出时触发。以下 hook 打印收尾统计：

```python
def summary_hook(messages: list) -> str | None:
    """Print a summary when the loop is about to stop."""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: session used {tool_count} tool calls\033[0m")
    return None   # return None = 允许终止, return string = 强制继续

register_hook("Stop", summary_hook)
```

在 agent_loop 中，退出前触发：

```python
tool_calls = [
    block for block in response.content if block.type == "tool_use"
]
if not tool_calls:
    force = trigger_hooks("Stop", messages)   # ← 退出之前
    if force:
        # hook 返回消息 → 注入消息并继续执行
        messages.append({"role": "user", "content": force})
        continue
    return
```

**循环里只改了一处**：s03 直接调用 `check_permission(block)`，s04 改为 `trigger_hooks("PreToolUse", block)`：

```python
for block in tool_calls:
    # s03: if not check_permission(block): ...
    # s04: hook 替代硬编码
    blocked = trigger_hooks("PreToolUse", block)
    if blocked:
        results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": str(blocked)})
        continue

    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**block.input) if handler else f"Unknown: {block.name}"

    trigger_hooks("PostToolUse", block, output)

    results.append({"type": "tool_result", "tool_use_id": block.id,
                    "content": output})
```

四个 hook 覆盖了 agent cycle 的关键节点：输入→执行前→执行后→退出。循环只负责调用 trigger_hooks()，具体逻辑全在 hook 回调里。

## 相对 s03 的变更

| 组件 | 之前 (s03) | 之后 (s04) |
|------|-----------|-----------|
| 扩展方式 | check_permission() 硬编码在循环里 | HOOKS 注册表 + trigger_hooks() |
| 新函数 | — | register_hook, trigger_hooks |
| hook 回调 | — | context_inject_hook, permission_hook, log_hook, large_output_hook, summary_hook |
| 循环 | 直接调用 check_permission() | 调用 trigger_hooks("PreToolUse", ...) |
| 退出控制 | 无 | trigger_hooks("Stop", ...) 可阻止退出 |
| 输入拦截 | 无 | trigger_hooks("UserPromptSubmit", ...) 可注入上下文 |

## 代码导读：把 code.py 从头读一遍（本站补充）

完整的 [`s04_hooks/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s04_hooks/code.py) 有 282 行。工具部分和 s03 一样，重点看 hook 系统怎么运转。

### 第一段：`trigger_hooks` 只听第一个开口的

```python
def trigger_hooks(event: str, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # A hook result blocks this tool call.
            return result
    return None
```

一个事件下面可以挂好几个 hook，按**注册的先后顺序**一个个调用。只要有一个返回了非 `None` 的值，就立刻把这个值交出去，**后面的 hook 一个都不再执行**。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s04-hooks/02-first-non-none-wins.jpg" target="_blank" rel="noopener" aria-label="查看原图：上面一排 hook A、B、C 都返回 None，全部执行完后继续；下面一排 hook B 返回 denied，在这里停下，hook C 被跳过"><img src="/notes-assets/lcc-s04-hooks/02-first-non-none-wins.jpg" alt="上面一排 hook A、B、C 都返回 None，全部执行完后继续；下面一排 hook B 返回 denied，在这里停下，hook C 被跳过" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

所以 **hook 的注册顺序是有意义的**。这一点在下面的实际运行里会看到效果。

### 第二段：同样是返回值，四个事件的意思不一样

| 事件 | 返回非 `None` 时 | 代码在哪里用到返回值 |
|---|---|---|
| `PreToolUse` | 阻止这次工具调用，返回的字符串直接作为 `tool_result` 交给模型 | `blocked = trigger_hooks("PreToolUse", block)` |
| `Stop` | 不退出，把返回的字符串当成一条新的 user 消息，继续循环 | `force = trigger_hooks("Stop", messages)` |
| `PostToolUse` | 没有影响 | 返回值被丢掉了 |
| `UserPromptSubmit` | 没有影响 | 返回值被丢掉了 |

注意 `PreToolUse` 比 s03 进步了一点：s03 被拒绝时模型只看到 `Permission denied.`，现在 hook 返回什么，模型就看到什么，比如 `Permission denied by deny list` 或 `Permission denied by user`，模型能分清是规则拦的还是你拒的。

还有一个名不副实的地方：`context_inject_hook` 的名字和注释都说「注入上下文」，但它只打印了一行日志就返回 `None`。主循环里 `trigger_hooks("UserPromptSubmit", query)` 的返回值也没有被接住。**现在的代码其实没法通过这个 hook 往提示词里加东西。**

### 第三段：Stop hook 能让循环停不下来

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s04-hooks/03-stop-hook.jpg" target="_blank" rel="noopener" aria-label="查看原图：LLM 的回复里没有 tool_use 时进入 Stop hooks；返回 None 就 return；返回字符串就追加成一条 user 消息，回到 LLM 继续"><img src="/notes-assets/lcc-s04-hooks/03-stop-hook.jpg" alt="LLM 的回复里没有 tool_use 时进入 Stop hooks；返回 None 就 return；返回字符串就追加成一条 user 消息，回到 LLM 继续" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

s01 里循环只有一个出口：模型不调工具了。s04 在这个出口前加了一道 Stop hook，它返回一个字符串，比如 `"测试还没跑，请先跑测试"`，循环就会把这句话当作用户的新消息发给模型，接着干。

这是一个很强的能力，Claude Code 的 Stop hook 就常用来做「没通过检查不许结束」。但它也打开了一个口子：**如果 Stop hook 每次都返回字符串，循环就永远不会结束。** 现在的代码里没有任何次数限制。

### 第四段：三个 PreToolUse 之外的小 hook

- **`log_hook`** 只打印参数的前两个值，最多 60 个字符。日志是给人看的，太长反而看不清。
- **`large_output_hook`** 在输出超过 10 万字符时提醒一下，但不修改输出，原样交给模型。
- **`summary_hook`** 数的是整个 `history` 里的 `tool_result`，所以统计的是**整个会话**用过的工具次数，不是这一轮的。

## 实际跑一次（本站补充）

提问 `Create a file test.txt containing hi, then delete it with rm`，两次审批都回答 `n`。输入是从管道传进去的，所以终端里不会显示 `n`，它就在 `Allow? [y/N]` 后面：

```
[HOOK] UserPromptSubmit: working in /.../sb4

[permission] Potentially destructive command
   Tool: bash({'command': "printf 'hi\\n' > test.txt && rm test.txt"})
   Allow? [y/N] [HOOK] write_file(['hi\n', '/.../sb4/test.txt'])

[permission] Potentially destructive command
   Tool: bash({'command': 'rm test.txt'})
   Allow? [y/N] [HOOK] Stop: session used 3 tool calls
Created `test.txt` containing `hi`, but deletion with `rm` was denied by the environment. The file still exists.
```

模型一共发起了三次工具调用：

1. 第一次把「写文件」和「删文件」拼成一条 bash 命令，因为里面有 `rm`，被拦了。
2. 模型换了个办法，先用 `write_file` 单独写文件，成功了。
3. 再单独执行 `rm test.txt`，又被拦了。

最后 `test.txt` 留在了目录里。

**数一数输出里的 `[HOOK]` 日志：** 三次工具调用，`PreToolUse` 的日志却只有 `write_file` 那一行。

## 想一想（本站补充）

1. 上面的输出里，为什么两次被拒绝的 `bash` 调用都没有留下 `[HOOK]` 日志？对照第一段讲的 `trigger_hooks` 和 `register_hook` 的调用顺序找原因。如果想让所有调用都留下日志，最小的改动是什么？

   <details class="qa"><summary>参考答案</summary>

   `permission_hook` 比 `log_hook` 先注册，`trigger_hooks` 遇到第一个返回非 `None` 的回调就直接返回。权限检查拒绝时返回了一个字符串，后面的 `log_hook` 根本没有机会执行，所以没有日志。

   最小的改动是把注册顺序对调：先 `register_hook("PreToolUse", log_hook)`，再注册 `permission_hook`。日志 hook 永远返回 `None`，不会挡住后面的权限检查。

   </details>

2. 模型第一次被拒后，把一条命令拆成了两步，先写文件，再删文件。这算不算一种「绕过」？如果第二步没有 `rm` 这个词，会发生什么？

   <details class="qa"><summary>参考答案</summary>

   算是绕过。用户拒绝的是「删掉这个文件」，模型换一种步骤达到同样的效果，意图没变。

   如果第二步里没有 `rm` 这个词，比如用 `find -delete` 或者 `python3 -c "import os; os.remove(...)"`，s03 的规则都匹配不到，命令会直接执行，用户连确认都看不到。这再次说明：按命令文本做匹配，拦不住有心的绕过。

   </details>

3. 如果要防止 Stop hook 让循环永远停不下来，你会把限制加在哪里，是 hook 里面还是 `agent_loop` 里面？为什么？

   <details class="qa"><summary>参考答案</summary>

   加在 `agent_loop` 里更稳妥。比如记一个计数器：Stop hook 连续要求继续超过 N 次，就不再听它的，直接返回。

   原因是：hook 是可以随便挂上去的扩展，你不能保证每个 hook 都写对了。限制放在循环里，不管挂了什么 hook、写得好不好，循环都一定能停下来。s17 就是这么做的：Stop hook 最多连续阻止 8 次。

   </details>

4. 想让 `UserPromptSubmit` 真的能往提示词里加内容，主循环里哪一行要改？改成什么样？

   <details class="qa"><summary>参考答案</summary>

   主循环里的 `trigger_hooks("UserPromptSubmit", query)` 这一行，返回值被直接丢掉了。改成接住返回值，并拼进用户消息：

   `extra = trigger_hooks("UserPromptSubmit", query)`，然后 `content = f"{query}\n\n{extra}" if extra else query`，再 `history.append({"role": "user", "content": content})`。

   要注意 `trigger_hooks` 只返回**第一个**非 `None` 的结果。如果有好几个 hook 都想注入内容，需要改成把所有结果收集起来再拼接。

   </details>

## 试一下

```sh
cd learn-claude-code
python s04_hooks/code.py
```

试试这些 prompt：

1. `Read the file README.md`（应该直接通过，观察 hook 日志）
2. `Create a file called test.txt`（通过后观察 PostToolUse 是否触发）
3. `Delete all temporary files in /tmp`（bash + rm 触发权限 hook）

观察重点：每次工具执行前，是否出现了 `[HOOK]` 日志？权限被拒时，是 hook 拦截的还是循环里硬编码的？

## 接下来

Agent 现在能安全执行操作了。但它有没有停下来想过"我应该先做什么，再做什么"？给它一个复杂任务，它是一上来就动手，还是先列个计划？

[s05 TodoWrite](/notes/lcc-s05-todo-write/) → 给 Agent 一个计划工具。先列清单，再做。
