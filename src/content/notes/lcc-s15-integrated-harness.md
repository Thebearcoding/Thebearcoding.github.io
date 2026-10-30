---
title: 'learn-claude-code s15：Integrated Harness，十几种机制挂在同一个循环上'
date: '2026-10-30'
tags: [Agent, Harness, Architecture, learn-claude-code]
summary: 转载 learn-claude-code 第 15 课，补充配图、代码地图和一次跨三分钟的真实运行。前 14 课的机制全部接进同一个 agent_loop，后台任务、cron 和团队事件都能自己唤醒一轮对话；代价是每次请求固定带上 26 个工具定义和 3k 字符的 system prompt，几种机制之间也开始互相影响。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s15_integrated_harness)（MIT License，Copyright (c) 2024 shareAI Lab），配图（部分据原仓库图示重绘）、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"多种机制，一个循环"* — 工具、权限、记忆、任务、团队、插件都挂在同一个 while True 上。
>
> **Harness 层**: 集成 — 把本章示例实际使用的机制放进同一个可运行系统。

## 问题

前面的章节把不同机制放在各自独立的示例中。本章把集成运行时需要的机制接到一起。

一个能长期工作的 coding agent 需要同时拥有：

- 工具分发和权限边界
- hooks 扩展点
- todo 计划和任务图
- 技能、记忆、系统 prompt 组装
- 压缩和错误恢复
- 后台任务和 cron 调度
- 团队、协议和 idle 任务认领
- 任务绑定的 worktree
- MCP 外部工具接入

S15 不再引入一个独立机制，而是展示现有机制从哪里进入模型循环，以及它们产生的事件如何回到同一段对话。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s15-integrated-harness/00-integrated-overview.jpg" target="_blank" rel="noopener" aria-label="查看原图：所有机制都挂在同一个循环上：准备阶段注入 cron 提示和后台通知、做压缩、把记忆、技能和 MCP 状态拼进 system prompt；调用模型时处理 429、529、max_tokens 和上下文过长；执行工具前后有 PreToolUse 和 PostToolUse，工具包括内置工具、mcp 工具和后台分发；没有工具调用时进入 Stop hooks 并提取记忆；循环外还有待办与任务图、子 Agent 与队友、worktree、cron 和后台线程"><img src="/notes-assets/lcc-s15-integrated-harness/00-integrated-overview.jpg" alt="所有机制都挂在同一个循环上：准备阶段注入 cron 提示和后台通知、做压缩、把记忆、技能和 MCP 状态拼进 system prompt；调用模型时处理 429、529、max_tokens 和上下文过长；执行工具前后有 PreToolUse 和 PostToolUse，工具包括内置工具、mcp 工具和后台分发；没有工具调用时进入 Stop hooks 并提取记忆；循环外还有待办与任务图、子 Agent 与队友、worktree、cron 和后台线程" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

S15 不再引入新机制，而是把前面各章的组件集成到同一个 harness：

```text
用户输入
  → UserPromptSubmit hooks
  → cron/background 通知注入
  → context compact
  → memory + skills + MCP 状态组装 system prompt
  → LLM
  → has tool_use block?
      否 → Stop hooks → 返回
      是 → PreToolUse hooks + permission
          → TOOL_HANDLERS / MCP handlers / background dispatch
          → PostToolUse hooks
          → tool_result / task_notification 回 messages
          → 下一轮
```

循环仍是同一个结构：调用模型，检查响应里是否出现 `tool_use` block，执行工具，再把结果追加回 `messages`。是否继续工具轮，由响应中有没有实际的 `tool_use` block 决定。

## 组件在循环中的位置

| 位置 | 组件 | 作用 |
|------|------|------|
| 用户输入前后 | `UserPromptSubmit` hooks | 记录、注入、审计用户输入 |
| LLM 前 | cron queue | 把定时触发的 prompt 注入 `messages` |
| LLM 前 | background notifications | 后台任务完成后以 `<task_notification>` 注入 |
| LLM 前 | compaction pipeline | 先压大输出，再裁历史，再压旧 tool_result，必要时摘要 |
| LLM 前 | memory / skills / MCP state | 组装 system prompt，让模型看到当前能力和长期上下文 |
| LLM 调用 | error recovery | 429/529 重试，`max_tokens` 升级，prompt too long 触发 reactive compact |
| 工具执行前 | `PreToolUse` hooks + permission | 拦截危险命令、写越界、破坏性 MCP 工具 |
| 工具分发 | `assemble_tool_pool` | 组装内置工具和 MCP 动态工具 |
| 工具执行时 | background dispatch | 显式标记的 bash 操作放入 daemon thread，主循环先返回占位结果 |
| 工具执行后 | `PostToolUse` hooks | 大输出告警、日志等后处理 |
| 返回循环 | tool_result | 每个 `tool_use` 对应一个 `tool_result`，再回到下一轮 |
| 本轮没有 tool_use / 停止时 | `Stop` hooks | 统计、清理、审计 |

## code.py 包含什么

### 工具与分发

内置工具池包含 26 个工具：

```text
bash, read_file, write_file, edit_file, glob
todo_write, task, load_skill, compact
create_task, update_task, list_tasks, get_task, claim_task, complete_task
schedule_cron, list_crons, cancel_cron
spawn_teammate, list_teammates, send_message
request_shutdown, request_plan, review_plan
create_worktree
connect_mcp
```

`assemble_tool_pool()` 每轮组装：

```text
BUILTIN_TOOLS + connected MCP tools
BUILTIN_HANDLERS + mcp__server__tool handlers
```

所以 `connect_mcp("docs")` 后，下一轮工具池里会出现 `mcp__docs__search`。

### 权限和 hooks

权限不写死在工具执行行里，而是作为 `PreToolUse` hook：

```python
blocked = trigger_hooks("PreToolUse", block)
if blocked:
    results.append(tool_result(block.id, blocked))
    continue
```

这样 permission、log、审计都可以挂在同一个 hook 点上。Lead、一次性 subagent 和队友的工具都会先经过 `PreToolUse`；允许执行的调用会在 handler 返回后触发 `PostToolUse`。

权限判断不会把 MCP server 自己写的 description 当成授权依据。宿主维护一组精确的已知只读工具名单，其他 MCP 工具都要询问用户。文件工具越过 `WORKDIR` 会直接拒绝，每条 bash 命令执行前都会询问。只有前台用户轮次可以弹出交互确认；异步轮次直接拒绝需要确认的操作，不和主 CLI 争抢输入。

### 计划与任务

S15 同时保留两层计划：

- `todo_write`：当前会话内的轻量计划，保存在内存中
- task graph：跨会话、可依赖、可认领的任务文件，写入 `.tasks/task_*.json`

前者帮助单个 Agent 不漂移；后者支撑团队协作。

两者目标相近，但实现不同：`todo_write` 整表替换当前会话清单，task record 则有稳定 ID 和单条生命周期更新。下面单独出现的 `task` 工具表示“一次性派发隔离 subagent”，不是 Task System。

集成宿主中的任务图仍采用两阶段构建：Lead 先创建所有任务节点，再使用 `create_task` 返回的运行时 ID 调用 `update_task`。队友只能列举、认领和完成任务，因此依赖结构由 Lead 在分发工作前确定。

### 子 agent 与团队

S15 有两种 delegation：

- `task`：一次性 subagent。独立 `messages[]`，中间过程丢弃，只返回最终摘要。
- `spawn_teammate`：持久队友线程。传入 ready `task_id` 时，运行时会在线程启动前完成认领；不传时，队友可以在 IDLE 中等待后续任务。没有 assignment 的队友不能使用文件或 Shell 工具。它按 `WORK → result → IDLE` 运行，不设固定的工具轮数上限；模型或分发失败会发出 `error`，线程清理会把未完成 assignment 释放回任务板。每次调用模型前都会先读取收件箱，因此直接消息和关机请求不会被连续的 tool-use 轮次饿死。idle 时先等待 `MessageBus` 消息，只在超时后扫描就绪 task，并以原子操作最多认领一个。

Lead 启动队友后结束当前轮次，不在模型循环里反复查询状态。队友事件进入 Lead 收件箱后，运行时会自动唤醒下一轮。

一次性 subagent 解决“上下文隔离”；持久队友解决“长期并行协作”。

### 记忆、技能和 prompt

S15 直接复用 s09 的 Memory runtime。每轮调用模型前，它读取 `.memory/MEMORY.md` 目录，根据当前请求选择相关记录，再把选中的正文交给 `assemble_system_prompt(context)`。本轮结束后，`extract_memories()` 提取可跨会话使用的信息；有新增记录时再运行 `consolidate_memories()`。

同一份 system prompt 还会加入身份、工具说明、workspace、skills catalog 和已连接的 MCP server。技能只放目录，完整内容通过 `load_skill(name)` 按需加载。

### 压缩和恢复

LLM 前先跑压缩管线：

```text
tool_result_budget → snip_compact → micro_compact → compact_history
```

`snip_compact` 在消息数量达到 50 条时归档完整历史，再裁掉中段消息，目标为包含归档标记在内的 30 条。这样为新增消息留出空间，避免立即再次归档，代价是直接保留在上下文中的近期消息更少。工具调用与结果保持配对，因此保留数量可能略高于目标。`micro_compact` 只在上下文超限时运行：它先保存较早且已读取的结果，再用恢复路径替换；最近 3 条保持完整，并在接近阈值 80% 时停止。如果未读取的新结果本身过大，S15 会先保留预览和完整输出路径，再考虑总结历史。

调用模型时再包一层恢复：

- 429：指数退避重试
- 529：指数退避，连续失败可切 fallback model
- `max_tokens`：先提高 max_tokens，再要求 continuation
- prompt too long：reactive compact 后重试

### 后台和 cron

bash 调用设置 `run_in_background=true` 后，主循环不再等待命令结束，而是先返回占位结果：

```text
should_run_background → start_background_task → placeholder tool_result
后台完成 → task_notification → 下一轮注入 messages
```

只有显式标记的 bash 调用会进入后台路径。命令非零退出或 worker 抛出异常时会发出 `failed` 通知。每条 Shell 命令都在独立进程组中运行；命令结束，或 Agent 经正常路径、`SIGTERM` 退出时，运行时会停止原进程组。另建 session 的进程可以离开这个进程组。

cron 调度器独立 daemon thread 每秒检查一次。durable 的一次性任务会先持久化为 `pending_delivery`，再进入队列，并保留到包含该 prompt 的模型调用成功；调用失败会放回队列，重启后也会再次入队，因此交付语义是至少一次。CLI 同时监听 `cron_queue`、Lead 收件箱和已经结束的后台任务，任一事件都能自动唤醒一轮 Agent。

### worktree 与 MCP

从 s13 继承的任务级 worktree 机制负责管理任务工作目录：

- pending 且未被认领的 task 可以留在主工作区，也可以通过 `create_worktree(name, task_id)` 绑定独立分支和目录
- 创建前会校验 task、名称、路径、分支和 Git registry；Git 命令失败后还会核对 registry 和分支状态，任何部分创建的 checkout 都保持未绑定并保留供人工恢复
- idle 队友以原子操作认领一个就绪 task，assignment 同时记录 `task_id` 和有效 `cwd`
- Lead 也可以把 ready `task_id` 直接传给 `spawn_teammate`，认领成功后才启动线程
- 队友所有文件工具都使用该 `cwd`；只有 task owner 能完成任务，assignment 会保留到当前模型轮次结束
- 移除保留在宿主侧的 `remove_worktree()` 函数中，模型不能调用。用户或宿主先检查任务所有权、assignment lease、后台工作和 Git 状态；破坏性移除需要另行取得用户确认

worktree 只改变工具的默认工作目录，用于分离 working copy，并不是安全沙箱。进程组清理也无法约束另建 session 的进程，因此删除保留为宿主操作。

认领或释放 task 会改变 assignment version，使旧的 plan approval 失效；普通 `send_message` 只传递消息，不会改变 task identity 或 plan 状态。

MCP 负责外部能力：

- `connect_mcp(name)` 连接 mock server
- `assemble_tool_pool()` 把 MCP 工具组装进工具池，并拒绝规范化后的名称冲突
- 工具名统一为 `mcp__server__tool`

## 相对 s14 的变化

| 范围 | s14 MCP | s15 Integrated Harness |
|------|---------|-------------------------|
| 内置工具 | 6 个 | 25 个 |
| 外部工具 | 已连接的 MCP 工具 | 沿用同一套动态 MCP 路径和宿主策略 |
| 本地机制 | S04 工具、hooks、权限和 MCP | todo、subagent、skills、compaction、memory、task graph、后台 bash、cron、teams 和 worktrees |
| 事件来源 | 用户输入和工具结果 | 用户输入、工具结果、cron prompt、后台通知和 team events |


## 代码地图：3330 行都放了什么（本站补充）

完整的 [`s15_integrated_harness/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s15_integrated_harness/code.py) 有 3330 行。它没有新机制，几乎每一段都能在前面某一课里找到原型。按代码里的分段注释排一下：

| 行号 | 分段 | 来自哪一课 |
|---|---|---|
| 140 | Task System | s10 |
| 431 | Task-bound Worktrees | s13 |
| 719 | Skill Loading | s07 |
| 795 | Prompt Assembly | s07 + s09 + s14 |
| 871 | Basic Tools | s02 |
| 1104–1772 | MessageBus、协议状态、任务分配、队友线程、Lead 团队工具 | s13 |
| 1773 | Hooks and Permission Checks | s03 + s04 |
| 1869 | Subagent Tool | s06 |
| 1962 | Context Compaction | s08 |
| 2228 | Error Recovery | 新整合（重试、降级、续写） |
| 2281 | Background Tasks | s11 |
| 2379 | Cron Scheduler | s12 |
| 2633 | MCP System | s14 |
| 2883 | Tool Definitions | 26 个内置工具 |
| 3090 | Agent Loop | s01 的那个循环 |

读这一课，最好的办法不是从头读到尾，而是拿着前面每一课的知识，去找它在这里**接在循环的哪个位置**。原文的「组件在循环中的位置」那张表就是索引。

（原文「相对 s14 的变化」表里写的是 25 个内置工具，正文和实际请求里都是 26 个。）

## 代码导读：集成后才出现的问题（本站补充）

### 第一段：一轮对话可以不由用户发起

```python
def async_event_loop(history: list, context: dict, session_state: dict):
    while True:
        time.sleep(1)
        with agent_lock:
            with cron_lock:
                fired = list(cron_queue)
            inbox = consume_lead_inbox(route_protocol=True)
            if not fired and not inbox and not has_pending_background():
                continue
            ...
            agent_loop(history, context, active_request)
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s15-integrated-harness/01-who-starts-a-turn.jpg" target="_blank" rel="noopener" aria-label="查看原图：能发起一轮对话的有四个来源：主线程的用户输入，以及后台任务完成、cron 到期、Lead 收件箱里的团队事件；后三个由每秒检查一次的 async_event_loop 处理；两边都要先拿到 agent_lock 才能运行 agent_loop；真实运行中 20:46:03 因后台任务完成自动开了一轮，20:48:00 因 cron 到期自动开了一轮"><img src="/notes-assets/lcc-s15-integrated-harness/01-who-starts-a-turn.jpg" alt="能发起一轮对话的有四个来源：主线程的用户输入，以及后台任务完成、cron 到期、Lead 收件箱里的团队事件；后三个由每秒检查一次的 async_event_loop 处理；两边都要先拿到 agent_lock 才能运行 agent_loop；真实运行中 20:46:03 因后台任务完成自动开了一轮，20:48:00 因 cron 到期自动开了一轮" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

主线程等用户输入，另一条线程每秒检查三件事：有没有到期的 cron、Lead 收件箱有没有消息、有没有跑完的后台任务。只要有一件，就拿锁跑一轮 `agent_loop`。

这修好了 s11 的一个问题：s11 的后台结果要等用户说下一句话才能被看到，在 s15 里，结果一出来，Agent 自己就会醒过来处理。

两边用的是同一把 `agent_lock`，和 s12 一样：同一时刻只能有一轮对话在改 `history`。

### 第二段：调用模型那一层，包了三种恢复

```python
def with_retry(fn, state: RecoveryState):
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0
            return result
        except Exception as e:
            ...
            if "ratelimit" in name or "429" in msg:
                time.sleep(retry_delay(attempt)); continue
            if "overloaded" in name or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529 and FALLBACK_MODEL:
                    state.current_model = FALLBACK_MODEL
                ...
                time.sleep(retry_delay(attempt)); continue
            raise
```

| 错误 | 处理方式 |
|---|---|
| 429 限流 | 指数退避后重试，`retry_delay` 每次翻倍，最多 32 秒，再加一点随机抖动 |
| 529 过载 | 同样退避重试；连续几次都是 529，且配置了 `FALLBACK_MODEL_ID`，就换成备用模型 |
| 回复被 `max_tokens` 截断 | 先把上限提高再请求一次；还不够，就把截断的回复留下，追加一句「继续」让模型接着写 |
| 上下文太长 | 用 s08 的 `reactive_compact` 压缩后重试，只试一次 |

随机抖动的作用是：很多客户端同时被限流时，不会在同一时刻一起重试，又一起被限流。

其他错误一律直接抛出。不认识的错误不重试，免得把一个本该立刻暴露的 bug 拖成好几分钟的等待。

### 第三段：system prompt 每次都在变

```python
def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"], PROMPT_SECTIONS["tools"], ...]
    sections.append(f"Current time: {datetime.now().isoformat(timespec='seconds')}")
    sections.append("Skills catalog:\n" + list_skills() + ...)
    if context.get("memory_catalog"): ...
    if context.get("memories"): ...
    if mcp_names: sections.append(f"Connected MCP servers: ...")
    return "\n\n".join(sections)
```

system prompt 每次调用模型前都重新拼一遍，里面有一行**精确到秒**的当前时间。下面的真实运行里，模型就是靠这行字，才把「1 分钟后」换算成了一个具体的 cron 表达式。

但这也意味着，两次请求的 system prompt 几乎不可能完全一样。s07 里说过，内容保持不变，服务端的 prompt cache 才能命中。时间戳精确到秒，system prompt 这一段就很难被缓存。这是一个「功能」和「成本」互相拉扯的例子：想让模型知道现在几点，又想让缓存命中，可以把时间放到更靠后、变化本来就多的位置，或者只精确到分钟。

### 第四段：机制之间开始互相影响

```python
# agent_loop 开头
if rounds_since_todo >= 3:
    messages.append({"role": "user", "content": "<reminder>Update your todos.</reminder>"})
    rounds_since_todo = 0
...
# 每执行一个工具
if block.name == "todo_write":
    rounds_since_todo = 0
else:
    rounds_since_todo += 1
```

s05 的 todo 提醒数的是「轮」：一轮里调了多少个工具都算一轮。s15 里同名的计数器，数的是**每一次工具调用**，而且是全局变量，用户换了问题也不清零。

单独看 s05，这个提醒很合理。和 MCP、后台任务、cron 放在一起以后，简单的查询也会被它催着去列待办。下面的真实运行里就出现了这种情况。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s15-integrated-harness/03-todo-counter.jpg" target="_blank" rel="noopener" aria-label="查看原图：上面是 s05：按轮计数，每轮里可以有好几个工具；下面是 s15 的真实运行：read_file 记 1，换了一个用户问题后 connect_mcp 记 2，mcp__docs__search 记 3，于是插入一条更新待办的提醒，模型为了一个简单查询调用了 todo_write；计数按工具调用累加，跨轮次不清零"><img src="/notes-assets/lcc-s15-integrated-harness/03-todo-counter.jpg" alt="上面是 s05：按轮计数，每轮里可以有好几个工具；下面是 s15 的真实运行：read_file 记 1，换了一个用户问题后 connect_mcp 记 2，mcp__docs__search 记 3，于是插入一条更新待办的提醒，模型为了一个简单查询调用了 todo_write；计数按工具调用累加，跨轮次不清零" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

## 实际跑一次（本站补充）

在一个只有 `README.md` 和课程 `skills/` 目录的空文件夹里，依次输入三句话，中间都隔了一段时间。每行前面是真实时间。

**第一句：后台命令和读文件一起做。**

```
20:45:39 s15 >> 在后台运行 `sleep 4 && echo deps-ok`，同时读一下 README.md 的标题。
20:45:43 > bash
20:45:43 [permission] shell command
20:45:43   sleep 4 && echo deps-ok
20:45:44   Allow? [y/N] [HOOK] bash
20:45:44   [background] bg_0001: sleep 4 && echo deps-ok
20:45:44 > read_file
20:45:44 # Demo project
20:45:53 [HOOK] Stop: 2 tool result(s)
20:46:01 后台任务已启动。README.md 的标题是 **Demo project**。
```

s15 规定每条 bash 都要用户确认，哪怕是放到后台的命令。我回答了 `y`。

`Stop` 在 20:45:53，回答到 20:46:01 才打印出来，中间这 8 秒是 s09 的记忆提取：每一轮结束后，都要多调用一次模型，判断有没有值得长期保存的信息。

**没人说话，Agent 自己醒了。**

```
20:46:03 s15 >> [HOOK] Stop: 2 tool result(s)
20:46:05 后台任务已完成，输出：`deps-ok`。
```

这一轮没有任何用户输入。它收到的最后一条消息是：

```
<task_notification>
  <task_id>bg_0001</task_id>
  <status>completed</status>
  <command>sleep 4 && echo deps-ok</command>
  <summary>deps-ok</summary>
</task_notification>
```

同样的场景在 s11 里，结果要一直等到用户问下一句才会被看到。

**第二句：连 MCP 查文档。**

```
20:46:20 > connect_mcp
20:46:20   [mcp] connected: docs -> ['search', 'get_version']
20:46:25 > mcp__docs__search
20:46:25 [docs] Found 3 results for 'agent loop'
20:46:36 > todo_write
20:46:36   [todo] updated 2 item(s)
20:46:41 > mcp__docs__search   (×2)
20:46:41 > mcp__docs__get_version
20:46:47 > todo_write
```

一个简单的查文档请求，中间却冒出来两次 `todo_write`。从 dump 里看，20:46:36 那次请求的末尾有一条单独的用户消息 `<reminder>Update your todos.</reminder>`：第一句里的 `read_file`、这一句的 `connect_mcp` 和第一次 `mcp__docs__search`，正好累计 3 次工具调用。模型收到提醒，只好列了一个两项的待办。

**第三句：1 分钟后提醒喝水。**

```
20:47:03 s15 >> 1 分钟后提醒我喝水。
20:47:14 > schedule_cron
20:47:14 Scheduled cron_107276: '48 20 30 10 *' -> 提醒用户：该喝水了。
20:47:22 好的，已设置 1 分钟后的喝水提醒。

20:48:00 s15 >>   [cron auto] 提醒用户：该喝水了。
20:48:00 s15 >>   [cron inject] 提醒用户：该喝水了。
20:48:06 💧 该喝水了！起来活动一下，喝杯水吧。
```

模型调用 `schedule_cron` 时传的参数是：

```json
{"cron": "48 20 30 10 *", "durable": true, "prompt": "提醒用户：该喝水了。", "recurring": false}
```

它从 system prompt 里的当前时间（20:47）算出了「10 月 30 日 20:48」，并且设成了一次性任务。20:48:00 准时触发，又是一轮没有用户输入的对话。触发之后，`.scheduled_tasks.json` 变回了 `[]`。

**每次请求的固定开销。**

从代理的 dump 里数，这三分钟一共发出 17 次请求：12 次是 Agent 本身的调用，5 次是每轮结束后的记忆提取。每次 Agent 调用都要带上：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s15-integrated-harness/02-fixed-cost.jpg" target="_blank" rel="noopener" aria-label="查看原图：每次请求在发送任何对话消息之前，都要先带上工具定义和 system prompt：连接 MCP 之前是 26 个工具、6.1k 字符的工具定义加 3.1k 的 system prompt；连接 docs 之后是 28 个工具、6.4k 字符；这次运行一共 12 次 Agent 调用和 5 次记忆提取"><img src="/notes-assets/lcc-s15-integrated-harness/02-fixed-cost.jpg" alt="每次请求在发送任何对话消息之前，都要先带上工具定义和 system prompt：连接 MCP 之前是 26 个工具、6.1k 字符的工具定义加 3.1k 的 system prompt；连接 docs 之后是 28 个工具、6.4k 字符；这次运行一共 12 次 Agent 调用和 5 次记忆提取" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

| | 工具数 | 工具定义 | system prompt |
|---|---|---|---|
| 连接 MCP 之前 | 26 | 6,103 字符 | 3,072 字符 |
| 连接 docs 之后 | 28 | 6,441 字符 | 3,101 字符 |

也就是说，对话还没开始，每次请求就已经有 9k 多字符了。一个「读 README 标题」这么简单的问题，模型也要先读完 26 个工具的说明。功能都集成进来了，但每个功能都在给每一次请求加成本。

## 想一想（本站补充）

1. 后台任务完成、cron 到期、团队事件都能自己发起一轮对话。如果用户正在输入一句很长的话，这时一个自动轮次开始了，会发生什么？`agent_lock` 能解决所有问题吗？
2. system prompt 里精确到秒的时间戳，会让这一段很难被缓存。你会怎么改，既让模型知道现在几点，又尽量不影响缓存？
3. 26 个工具每次都要发送。如果一个任务只需要读文件，有没有办法只给模型一部分工具？这样做有什么风险？
4. todo 提醒在 s05 里按「轮」计数，在 s15 里按「工具调用」计数，还跨轮次累加。你觉得哪一种更合理？集成多个机制时，怎么提前发现这类互相干扰？
5. 每轮结束都要多调用一次模型来提取记忆。在一个大部分是自动轮次（后台通知、cron 提醒）的会话里，这笔开销值得吗？你会怎么调整？
6. 529 连续失败时会切换到备用模型。切换以后，之前的对话历史原样交给另一个模型，可能出现哪些问题？什么时候应该切回来？

## 试一下

```sh
cd learn-claude-code
python s15_integrated_harness/code.py
```

可以试：

1. `检查这个仓库，告诉我哪些 Python 文件最重要。`
2. `从已连接的文档中查一下 agent loop 的相关说明。`
3. `请在独立的 worktree 中并行重构认证模块和登录页，修改前先把各自的计划给我看。`
4. `3 分钟后提醒我开会。`
5. `在后台安装依赖，同时继续阅读 README.md。`

观察重点：

- 工具调用前是否经过 hooks/permission
- `connect_mcp` 后下一轮是否出现 MCP 工具
- 设置 `run_in_background=true` 的 bash 调用是否返回 background placeholder
- 到点是不是自动提醒开会
- 队友是否提交 plan，并在 approval 前暂停
- idle 队友是否只原子认领一个就绪 task
- 队友所有文件工具是否都切换到已认领 task 的 `cwd`
- 完成任务后是否在本轮剩余工具调用中保持 task `cwd`，并在 IDLE 时释放

## 接下来

[s16 Workflow Runtime](/notes/lcc-s16-workflow-runtime/) 会在这个 host 中加入 `Workflow` 工具。Workflow 把固定的编排路径写在代码中，并记录运行进度，使同一次运行可以继续执行。
