---
title: 'learn-claude-code s13：Agent Teams，一个 Lead 带几个常驻队友'
date: '2026-10-30'
tags: [Agent, Harness, Multi-Agent, Concurrency, learn-claude-code]
summary: 转载 learn-claude-code 第 13 课，补充配图、code.py 导读和一次完整的团队运行。Lead 先提方案、等用户确认，再建任务、派队友；队友各自有独立的上下文和循环，结果和空闲通知通过文件收件箱回到 Lead，运行时把它们当作事件唤醒下一轮。真实运行里两个队友 80 秒内各自完成任务，但 Lead 没有照要求创建 worktree。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s13_agent_teams)（MIT License，Copyright (c) 2024 shareAI Lab），配图（部分据原仓库图示重绘）、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *“一个 Agent 装不下整项工作时，就让队友分头完成。”* — 持久队友、共享任务认领、可选 worktree 与协作协议。
>
> **Harness 层**：Team（团队）— 多个 Agent 如何分工、共享状态，同时接受 Lead 控制。

## 问题

假设我们让 Agent 重构整个后端，工作涉及配置加载、认证和测试。一个 Agent 可以依次处理，但总耗时更长，早期细节也会逐渐离开上下文。

这类工作适合并行，可用户通常只描述目标，不会替运行时设计团队：

```text
重构这个示例后端。清理配置加载、认证和测试，
保持现有接口，并确保测试通过。
```

Harness 需要回答一组相互关联的问题：

1. 谁判断并行是否有用，新增 Agent 又由谁确认？
2. 每个队友如何跨任务保留身份和上下文？
3. 结果如何自动返回 Lead，而不是让模型轮询收件箱？
4. 空闲队友能否直接接手 ready task，不再等待 Lead 逐项派发？
5. 并行修改可能冲突时，任务应该使用哪个工作目录？
6. 关机和计划审批如何成为可追踪、可执行的协议？

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s13-agent-teams/00-team-overview.jpg" target="_blank" rel="noopener" aria-label="查看原图：用户先确认团队方案，Lead 负责对话、建任务、派队友和审计划；中间的 MessageBus 用 .mailboxes 下的 jsonl 文件传递普通消息、结果、空闲通知和控制消息；三个队友各有自己的提示词、消息和工具，在 WORK 和 IDLE 之间切换；底部是共享任务板，空闲时先看收件箱再认领可做的任务；任务可以绑定 worktree，但它不是沙箱"><img src="/notes-assets/lcc-s13-agent-teams/00-team-overview.jpg" alt="用户先确认团队方案，Lead 负责对话、建任务、派队友和审计划；中间的 MessageBus 用 .mailboxes 下的 jsonl 文件传递普通消息、结果、空闲通知和控制消息；三个队友各有自己的提示词、消息和工具，在 WORK 和 IDLE 之间切换；底部是共享任务板，空闲时先看收件箱再认领可做的任务；任务可以绑定 worktree，但它不是沙箱" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

s13 复用 s10 的基础工具、Hooks、Permission 和 Task System，并增加一套由 Lead 管理的团队运行时：

- **Lead** 负责用户对话，提出分工方案并等待确认。
- **队友** 运行独立 Agent Loop，在 WORK 和 IDLE 之间切换。
- **MessageBus** 通过文件收件箱传递普通消息、结果和控制事件。
- **运行时投递** 消费 Lead 的收件箱，把团队事件注入下一轮对话。
- **共享任务板** 让空闲队友发现 ready task，并在锁内完成认领。
- **可选 worktree** 在需要时把任务绑定到另一个工作目录；未绑定任务仍使用仓库目录。
- **类型化协议和计划闸门** 显式记录关机与审批状态，并在计划获批前阻止修改型工具。

任务图继续采用 s10 的两阶段契约。Lead 先为所有节点调用 `create_task`，再使用返回的运行时 ID 调用 `update_task(addBlockedBy=...)`，最后才分配 ready task。只有 Lead 能使用 `update_task`；队友只能列举、认领和完成任务，团队运行期间不能改写任务图结构。

s11 的后台任务和 s12 的定时任务没有被带入本章。它们不参与队友通信、任务认领或计划审批。

这些机制都属于 Team 这一层。任务发现不需要另一套 Agent Loop，worktree 也不会产生另一种 Agent。

## 工作原理

### 1. Lead 先提出团队，再等待用户确认

启动队友会改变成本、并发度和可以修改工作区的角色集合。Lead 的系统提示词会把这条边界明确写出来：

```python
"When parallel work would help, first propose a small team with clear "
"responsibilities and wait for the user's confirmation. Do not call "
"spawn_teammate before the user confirms."
```

收到第一条需求后，Lead 只提出分工：

```text
我建议并行处理三个方向：
- config：清理配置加载
- auth：重构认证
- tests：补充回归测试

你确认后我再启动队友。
```

用户回复“开始吧”后，Lead 才能调用 `spawn_teammate`。Lead 会先创建任务，再把初始 `task_id` 传给队友。用户给出目标，Lead 设计团队，用户确认执行边界。

### 2. 每个队友拥有独立循环

s06 的 subagent 是一次性调用，队友则是持久执行单元：

| | s06 Subagent | s13 队友 |
|---|---|---|
| 生命周期 | 一次调用后结束 | `WORK → IDLE → WORK`，直到关机 |
| 上下文 | 只服务一个任务 | 跨任务保留 |
| 通信 | 返回一次结果 | 接收消息并发出事件 |
| 协作 | 单向委派 | 与 Lead 双向协作 |

`TeammateRuntime` 为每个队友保存独立的系统提示词、messages、工具和当前任务，再在线程中运行 WORK / IDLE 循环。队友工作时，Lead 可以继续协调其他任务。`lead` 和 `agent` 保留给运行时身份，但 `MessageBus` 仍允许把 `lead` 作为协调者收件箱。

`spawn_teammate` 在线程启动前认领初始任务。认领失败时不会启动队友。队友没有任务时，文件和 Shell 工具会要求它先认领任务，而不是回退到仓库目录。

### 3. MessageBus 把通信放在模型上下文之外

Lead 和队友不能共享同一个 messages 数组，否则一个队友的工具结果会进入另一个队友的推理上下文。`MessageBus` 为每个 Agent 提供 `.mailboxes/<name>.jsonl` 收件箱：

```python
class MessageBus:
    def send(self, from_agent, to_agent, content,
             msg_type="message", metadata=None):
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "metadata": metadata or {},
        }
        with self._changed:
            MAILBOX_DIR.mkdir(parents=True, exist_ok=True)
            with self._path(to_agent).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(msg, ensure_ascii=True) + "\n")
            self._changed.notify_all()

    def wait_for_messages(self, agent, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while not self.peek(agent):
                remaining = (None if deadline is None
                             else deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    return []
                self._changed.wait(remaining)
            return self._read_unlocked(agent)
```

锁会保护收件箱文件，避免队友并发读写。`Condition` 既能在消息到达时唤醒队友，也能支持 IDLE 状态下的短时等待。

### 4. 收件箱事件由运行时投递

`read_inbox()` 会读取并删除收件箱文件，因此 Lead 只保留一个消费者 `consume_lead_inbox()`：

```python
def consume_lead_inbox():
    messages = BUS.read_inbox("lead")
    for message in messages:
        if message["type"].endswith("_response"):
            match_response(...)
    return messages
```

CLI 主循环同时等待终端输入和 Lead 收件箱。新消息到达时，它会先消费收件箱，再发起一轮 Lead 调用：

```text
MessageBus → consume_lead_inbox
           → 更新协议状态
           → 把 [Team events] 注入 history
           → 启动新一轮 Lead 调用
```

Lead 启动队友后会结束当前轮次，不用反复调用 `list_teammates` 或 `get_task` 等待结果。队友事件到达时，运行时会自动唤醒下一轮。

`check_inbox` 不是模型工具。消息到达和消费属于运行时，模型只处理已经投递到上下文里的事件。

### 5. 结果与 IDLE 是两个事件

队友完成一项任务后，运行时按顺序发送两个事件：

```text
result:            "认证已重构，相关测试通过。"
idle_notification: "Waiting for more work."
```

`result` 回答“这项任务产出了什么”，`idle_notification` 回答“这个队友能否继续接任务”。一个含糊的“完成了”无法同时表达这两种状态。

空闲队友不会退出。直接消息或 ready task 会让它回到 WORK，`shutdown_request` 则会启动平滑关机握手。

### 6. IDLE 先看收件箱，再找 ready task

队友进入 IDLE 后优先处理消息，然后检查共享任务板：

```python
while True:
    inbox = BUS.wait_for_messages(name, IDLE_SCAN_INTERVAL)
    if inbox:
        should_stop = handle_messages(inbox)
        if should_stop or messages[-1]["role"] == "user":
            break
        continue

    task = claim_next_task(name)
    if task:
        messages.append({
            "role": "user",
            "content": f"[Auto-claimed task {task.id}] {task.subject}",
        })
        break
```

关机、计划审批和 Lead 的直接指令应该先于临时发现的工作。如果没有消息，也没有 ready task，队友会保持 IDLE。前置任务完成后，当前受阻的任务可能变为 ready。

### 7. 发现和认领分成两步，认领必须原子执行

扫描只负责找候选任务：

```python
def scan_unclaimed_tasks() -> list[Task]:
    return [
        task for task in list_tasks()
        if task.status == "pending"
        and task.owner is None
        and can_start(task.id)
    ]
```

候选列表只是某一时刻的快照。其他队友，甚至另一个使用同一任务目录的 Harness 进程，也可能看到同一任务。因此所有权变更必须放进 `claim_task()`，并由 `task_store_lock()` 同时取得进程内锁和文件锁：

```python
def claim_task(task_id: str, owner: str) -> str:
    with task_store_lock():
        task = load_task(task_id)
        if task.status != "pending" or task.owner is not None:
            return "Task is no longer available"
        if _owner_in_progress(owner):
            return "Owner must complete its current task first"
        if not can_start(task_id):
            return "Task is blocked"
        cwd, error = task_worktree_cwd(task)
        if error:
            return f"Cannot claim {task_id}: {error}"
        task.owner = owner
        task.status = "in_progress"
        save_task(task)
        teammate_assignments[owner] = {"task_id": task.id, "cwd": cwd}
        return f"Claimed {task.id}"
```

多个队友可以同时发现同一候选，但只有一个 claim 能把它推进到 `in_progress`。持有同一存储锁时，任务内容会先写入临时文件，再原子替换正式文件。队友完成当前任务后才能再认领下一项；worktree 绑定损坏时，认领会直接失败，不会回退到仓库目录。

### 8. 认领后的工作复用同一个 WORK 循环

认领成功后，运行时把任务 ID、标题和描述放进队友的 messages：

```text
任务板出现 ready task
  → IDLE 队友发现候选
  → claim_task 写入 owner 和 in_progress
  → 任务进入队友 messages
  → WORK
  → complete_task
  → result + idle_notification
  → IDLE
```

队友继续使用直接派发任务时的模型调用、文件工具、Shell、计划闸门、结果上报和关机协议。任务发现只是现有 WORK 循环的另一个入口。

### 9. 由任务选择工具的工作目录

`Task.worktree` 是可选字段：

```python
@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str]
    worktree: str | None = None
```

并行修改需要分开目录时，Lead 可以创建并绑定 worktree：

```python
create_worktree(name="auth-refactor", task_id="task_1a2b3c4d")
```

`create_worktree` 只提供给 Lead。它要求任务处于 pending、无人认领且尚未绑定，随后检查名称、路径、分支和 Git 注册信息，创建 checkout，最后才写入任务绑定。如果 Git 报告失败却已经留下分支或已注册的 checkout，运行时会报告 partial operation，让任务保持未绑定，并保留这些内容供人工恢复。队友只使用任务工具和文件工具。

认领任务时，运行时会把解析后的目录写入 `teammate_assignments`。该队友的 `bash`、`read_file`、`write_file`、`edit_file` 和 `glob` 都从 assignment 读取目录。没有绑定 worktree 的任务解析到 `WORKDIR`；没有认领任务的队友不能使用这些工作区工具：

```python
cwd, error = task_worktree_cwd(task)
if not error:
    teammate_assignments[owner] = {
        "task_id": task.id,
        "cwd": cwd,
    }
```

`complete_task(task_id, owner)` 会检查调用者是否拥有这个进行中的任务。成功完成只记录结果，不会马上清除 assignment；直到当前模型轮次结束，后续工具调用仍使用这个任务目录。队友回到 IDLE 时，运行时才释放 assignment。完成失败时也会保留目录，方便修正后重试。

进程重启后，`assignment_cwd()` 可以根据持久化任务中的 owner 和 worktree 绑定恢复进行中的 assignment。同一 owner 已转到新任务时，它也会替换本地的旧 lease。若绑定丢失或无效，它会直接失败，不会把操作悄悄切回仓库目录。

> Worktree 只分开 Git 工作目录和分支，不是安全沙箱。Shell 命令仍能访问父进程有权访问的路径和资源。

### 10. Worktree 移除由宿主负责

模型可以创建任务绑定的 worktree，但不能移除它。清理保留为宿主函数，让用户或宿主先检查任务所有权、assignment lease 和 Git 状态。这个函数会拒绝 pending 或 in-progress 绑定以及当前轮次仍在使用的 lease。未明确选择破坏性移除时，已跟踪、未跟踪和已忽略文件都会阻止清理。

`remove_worktree(name, discard_changes=True)` 只供已经另行取得用户明确确认的宿主调用。两种移除路径都会保留仓库里的 `wt/<name>` 分支，包括没有 upstream 的干净本地提交。移除成功后，任务绑定会被清空。

```text
干净 worktree   → 宿主可移除目录，保留 wt/<name> 分支
有改动 worktree → 由用户决定保留还是丢弃
待办/进行中任务 → 拒绝移除
```

任务完成与 worktree 清理也互相独立。`complete_task` 记录任务结果；队友回到 IDLE 后，用户或宿主才检查、合并、保留或移除 worktree。

### 11. 控制消息使用类型和 request_id

普通协作可以使用自由文本，关机和审批则不能依靠猜测消息意图。它们使用结构化消息：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s13-agent-teams/00-team-protocols.jpg" target="_blank" rel="noopener" aria-label="查看原图：两种协议共用一套状态机：关机由 Lead 发起，带 request_id，队友回复后 Lead 按 request_id 匹配；计划审批由 Lead 要求、队友提交、Lead 批准或驳回；状态都从 pending 变为 approved 或 rejected，记录在 pending_requests 里"><img src="/notes-assets/lcc-s13-agent-teams/00-team-protocols.jpg" alt="两种协议共用一套状态机：关机由 Lead 发起，带 request_id，队友回复后 Lead 按 request_id 匹配；计划审批由 Lead 要求、队友提交、Lead 批准或驳回；状态都从 pending 变为 approved 或 rejected，记录在 pending_requests 里" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

```python
@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    work_version: int | None = None
    task_id: str | None = None

pending_requests: dict[str, ProtocolState] = {}
```

关机路径如下：

```text
Lead 创建 pending 状态的关机请求
  → shutdown_request(request_id) 进入队友收件箱
  → 队友完成当前步骤
  → shutdown_response(request_id) 返回 Lead
  → request_id 找到原始请求
  → pending 变为 approved，队友循环退出
```

ID 把回复关联到请求，类型阻止不匹配的回复修改状态，状态则阻止同一回复重复生效。

### 12. 计划审批会约束执行

计划协议的方向相反：

```text
Lead → plan_request
队友 → plan_approval_request(request_id, plan)
Lead → plan_approval_response(request_id, approve, feedback)
```

如果 Lead 在启动队友前就知道必须先看计划，可以调用 `spawn_teammate(..., task_id=task.id, require_plan=True)`；运行时会先认领任务并打开闸门，再启动线程。对于已经运行的队友，也可以再用 `request_plan` 要求其提交计划。

工具分发层负责执行闸门：

```python
def _run_teammate_tool(name, block, handlers):
    gate = plan_gates.get(name, "not_required")
    if block.name in {"bash", "write_file", "edit_file"} and gate not in {
        "not_required", "approved"
    }:
        return f"Blocked: plan status is {gate}."
    try:
        return handlers[block.name](https://github.com/shareAI-lab/learn-claude-code/blob/main/s13_agent_teams/**block.input)
    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"
```

状态是 `required`、`pending` 或 `rejected` 时，队友可以读取文件、提交或修改计划，但不能运行 Shell 命令、写文件或编辑文件。提交计划时会记录队友当前的 task 和 work version；审批返回时两者仍然一致才会生效。认领或释放任务会改变 work version，使旧审批失效；普通消息不会改变任务身份或审批状态。

队友不会直接从后台线程读取用户输入。遇到需要用户确认的危险命令或工作区外路径时，工具会返回 permission 错误，由 Lead 与用户处理。

## 一次完整运行

```text
s13 >> 把后端重构拆到共享任务板，尽量并行完成配置、认证和测试。
       认证任务使用 worktree，保持现有接口，并确保测试通过。

Lead：我建议按 config、auth 和 tests 三个方向分工。
      是否启动团队？

s13 >> 开始吧

[task] config created
[task] auth created → worktree auth-refactor
[task] tests created
[claim] alice → config (cwd: repository)
[claim] bob → auth (cwd: .worktrees/auth-refactor)
[teammate] alice spawned
[teammate] bob spawned
[complete] auth
[bus] bob → lead (result) ...
[bus] bob → lead (idle_notification) ...
[wake: 2 team events → new turn]
Lead：我已收到认证任务的结果，接下来继续协调其余工作。
```

终端会显示用户请求、Lead 的团队方案、任务状态、认领结果、所选目录、结果、IDLE 切换和控制事件。用户不需要指定谁是 Lead，也不必提醒它检查收件箱。

## 相对 s10 的变化

| 组件 | s10 | s13 |
|---|---|---|
| Agent | 单个 Agent | 一个 Lead 加持久队友 |
| 用户流程 | 直接执行请求 | 先提团队方案，再确认启动 |
| 通信 | 无 | 文件收件箱加运行时投递 |
| 生命周期 | 一个循环 | 队友 `WORK / IDLE / shutdown` |
| 共享工作 | 单 Agent 使用任务工具 | IDLE 扫描加队友原子认领 |
| 工作目录 | 仓库 `WORKDIR` | 必须认领任务；任务可选 worktree |
| 结果上报 | 当前 Agent 输出 | 分开的 `result` 与 `idle_notification` |
| 控制 | 无 | 类型化关机与计划审批协议 |
| 执行约束 | 无团队约束 | 必需计划会锁住修改型工具 |


## 代码导读：README 没讲到的部分（本站补充）

完整的 [`s13_agent_teams/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s13_agent_teams/code.py) 有 1891 行，是前 12 课里最长的。它以 s10 的任务系统为基础，加上 worktree、收件箱、队友运行时和两种协议。原文已经把每个机制讲得很细，这里只补充几处原文没有展开的实现细节。

### 第一段：Lead 和队友拿到的工具不一样

```python
TOOLS = [*BASE_TOOLS, *TASK_TOOLS, *TEAM_TOOLS]          # Lead

TEAMMATE_TOOLS = [                                         # 队友
    *BASE_TOOLS,
    {"name": "send_message", ...},
    {"name": "submit_plan", ...},
    next(tool for tool in TASK_TOOLS if tool["name"] == "list_tasks"),
    next(tool for tool in TASK_TOOLS if tool["name"] == "claim_task"),
    next(tool for tool in TASK_TOOLS if tool["name"] == "complete_task"),
]
```

队友能读写文件、跑命令、看任务、认领和完成任务，还能给别人发消息、提交计划。但它**没有** `create_task`、`update_task`、`spawn_teammate`、`create_worktree`。

所以只有 Lead 能改任务图的结构、能招人、能开 worktree。权限不是靠 prompt 里写一句「你不能做什么」，而是工具表里根本就没有这个工具。

### 第二段：主循环同时等两件事

```python
def wait_for_cli_event() -> tuple[str, str | None]:
    prompt_visible = False
    while True:
        if BUS.peek("lead"):
            ...
            return "wake", None
        if not prompt_visible:
            print("s13 >> ", end="", flush=True)
            prompt_visible = True
        readable, _, _ = select.select([sys.stdin], [], [], 0.25)
        if readable:
            line = sys.stdin.readline()
            if line == "":
                return "quit", None
            return "user", line.rstrip("\n")
```

前面几课的主循环里，`input()` 会一直卡着等用户敲回车。这里换成了 `select.select`：最多等 0.25 秒看终端有没有输入，没有就去看一眼 Lead 的收件箱，然后再等 0.25 秒。

谁先来就处理谁：

- 用户输入了，就当普通问题处理；
- 收件箱有消息了，就把消息拼成一条 `[Team events]` 用户消息，再跑一轮 Lead。

```python
def format_team_events(msgs: list[dict]) -> str:
    lines = []
    for msg in msgs:
        ...
        lines.append(f"[{msg['type']}{suffix}] {msg['from']}: {msg['content']}")
    return "[Team events]\n" + "\n".join(lines)
```

这就是原文说的「运行时投递」：Lead 不需要调用任何「检查收件箱」的工具，消息一到，主循环就把它送进 Lead 的对话里。上一课 s11 的后台结果要等用户下一句话才能被看到，这里的团队事件可以**自己唤醒** Lead。

### 第三段：任务锁是「线程锁 + 文件锁」，还能重入

```python
@contextmanager
def task_store_lock():
    """Serialize task mutations across threads and host processes."""
    with task_lock:                                   # threading.RLock
        depth = getattr(_task_store_state, "depth", 0)
        if depth == 0:
            TASKS_DIR.mkdir(parents=True, exist_ok=True)
            handle = TASK_LOCK_PATH.open("a+", encoding="utf-8")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _task_store_state.handle = handle
        _task_store_state.depth = depth + 1
        try:
            yield
        finally:
            _task_store_state.depth -= 1
            if _task_store_state.depth == 0:
                ...
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s13-agent-teams/03-claim-lock.jpg" target="_blank" rel="noopener" aria-label="查看原图：两个队友扫描时都看到了同一个待认领任务，这只是一个快照；它们都去调用 claim_task，要先拿到线程锁加文件锁组成的 task_store_lock，在锁里重新检查状态和负责人；先拿到锁的 A 认领成功，B 收到任务已不可用"><img src="/notes-assets/lcc-s13-agent-teams/03-claim-lock.jpg" alt="两个队友扫描时都看到了同一个待认领任务，这只是一个快照；它们都去调用 claim_task，要先拿到线程锁加文件锁组成的 task_store_lock，在锁里重新检查状态和负责人；先拿到锁的 A 认领成功，B 收到任务已不可用" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

一把锁管两层：

- `threading.RLock` 管**同一个进程里的多个线程**（Lead 和各个队友）；
- `fcntl.flock` 锁住 `.tasks/.lock` 这个文件，管**不同进程**。原文说「另一个使用同一任务目录的 Harness 进程」，指的就是这种情况。

`depth` 计数让这把锁可以**重入**：`claim_task` 拿着锁去调用 `can_start`，`can_start` 里面再要一次锁，不会把自己锁死。只有最外层那一次才真正去加文件锁、解文件锁。

s10 的「想一想」里问过：两个 Agent 同时认领同一个任务会怎样？答案就在这里：扫描（`scan_unclaimed_tasks`）可以在锁外面做，看到的只是一个快照；真正改状态的 `claim_task` 必须在锁里，**重新读一遍**任务，确认它还是 `pending` 且没人认领，才写入。

### 第四段：队友的 WORK 和 IDLE

```python
def run(self):
    state = "continue"
    while state != "stop":
        if state == "idle" and not self.wait_for_work():
            break
        state = self.work()
```

`work()` 每次只调用一次模型，然后返回三种结果之一：

- `"continue"`：模型调用了工具，结果已经塞回去了，接着调用模型；
- `"idle"`：模型不再调用工具，说明这项任务做完了。这时先把最后一段文字作为 `result` 发给 Lead，再发一条 `idle_notification`；
- `"stop"`：收到关机请求，或者出错了。

进入 IDLE 后，`wait_for_work()` 每 2 秒醒一次：先看收件箱，有消息就处理；没有消息，就去任务板上找一个能做的任务**自动认领**。所以队友空下来以后不用 Lead 一个个派活。

`run()` 的 `finally` 里还会释放这个队友占用的任务目录，并从各个登记表里把它删掉。不管队友是正常关机还是中途崩溃，都会清理。

### 第五段：队友不能问用户，危险操作交给 Lead

```python
def _run_teammate_tool(name: str, block, handlers: dict) -> str:
    gate = plan_gates.get(name, "not_required")
    if block.name in {"bash", "write_file", "edit_file"}:
        if gate != "approved":
            if gate != "not_required":
                return (f"Blocked: plan status is {gate}. ...")
        blocked = check_permission(block, prompt_user=False)
        if blocked:
            return blocked
    ...
```

和 s12 的定时任务一样，队友跑在后台线程里，不能抢终端去问用户。`check_permission(block, prompt_user=False)` 遇到需要确认的命令，直接返回 `"Permission required: ask Lead to run this command."`。队友要么换一种做法，要么发消息请 Lead 来做。

这个函数里还能看到计划闸门：如果这个队友被要求先交计划，在计划批准之前，`bash`、`write_file`、`edit_file` 这三个会改动工作区的工具一律返回 `Blocked`，只有读文件和提交计划可以用。

## 实际跑一次（本站补充）

准备一个只有 `calc.py` 的小 git 仓库，里面有 `add`、`divide`、`mean` 三个函数，没有注释也没有测试。然后对 Lead 说：

```
把 calc.py 的改进拆到共享任务板并行完成：一项给每个函数补 docstring 和类型标注，
一项新建 tests/test_calc.py 写单元测试（包括除零和空列表）。测试任务使用 worktree。最后汇总结果。
```

**第一步，Lead 只提方案，不动手。**

```
建议使用 2 名协作者并行处理：

1. **文档与类型标注**
   - 修改 `calc.py`
   - 为每个函数补充 docstring 和类型标注

2. **单元测试**
   - 在独立 worktree 中新建 `tests/test_calc.py`
   - 覆盖正常行为、除零和空列表场景

确认后，我会先在共享任务板创建两项任务，再分派协作者并行执行，最后运行测试并汇总结果。请回复“确认”。
```

这一轮没有任何工具调用，完全照着 system prompt 里「先提方案，等用户确认」来做。

**第二步，我回复「开始吧」，Lead 建任务、派队友。**

```
20:30:11 > create_task      Created task_c0082e79: 为 calc.py 补充 docstring 和类型标注
20:30:16 > create_task      Created task_a3ce26cb: 新增 calc.py 单元测试
20:30:22 > spawn_teammate
  [claim] 为 calc.py 补充 docstring 和类型标注 -> in_progress (owner: calc-docs)
  [teammate] calc-docs spawned as 负责 calc.py 文档和类型
20:30:22 > spawn_teammate
  [claim] 新增 calc.py 单元测试 -> in_progress (owner: calc-tests)
  [teammate] calc-tests spawned as 负责 calc.py 单元测试
```

两个 `spawn_teammate` 在同一次回复里发出。每个队友在启动之前就认领好了自己的任务。

这里有个问题：方案里明明写着「在独立 worktree 中」，Lead 却**没有调用 `create_worktree`**。两个队友都在仓库目录里干活，`calc-tests` 中途还跑了一次 `git diff -- calc.py`，看到了另一个队友正在改的内容。prompt 只能**劝**模型用 worktree，最后用不用，还是模型自己决定。

**第三步，两个队友并行干活。**

从代理的 dump 里按 system prompt 区分请求来自谁，可以看到两个队友的请求交替出现：

```
20:30:35 calc-tests  write_file tests/test_calc.py（import pytest）
20:30:37 calc-docs   write_file calc.py
20:30:42 calc-tests  bash: python -m pytest -q
20:30:52 calc-tests  bash: python3 -m pytest -q   → No module named pytest
20:30:56 calc-tests  write_file tests/test_calc.py（改用 import unittest）
20:31:03 calc-tests  bash: /opt/homebrew/bin/python3 -m unittest discover -s tests -v
20:31:05 calc-docs   edit_file calc.py
20:31:06 calc-tests  bash: rm -rf __pycache__ ...   → Permission required: ask Lead to run this command.
20:31:09 calc-tests  complete_task task_a3ce26cb
```

`calc-tests` 先写了 pytest 版本的测试，发现环境里没装 pytest，就自己改成了标准库的 `unittest`。它最后想用 `rm -rf` 清理缓存目录，被队友的权限规则直接拒绝了，`calc-docs` 后来也碰到了同样的拒绝。

**第四步，结果回到 Lead，Lead 被唤醒。**

```
  [bus] calc-tests -> lead: (result) 已完成并标记任务 `task_a3ce26cb`。……
  [bus] calc-tests -> lead: (idle_notification) Waiting for more work.

[wake: 2 team event(s) -> new turn]
```

Lead 这一轮收到的用户消息是这样的：

```
[Team events]
[result] calc-tests: 已完成并标记任务 `task_a3ce26cb`。

修改：
- 新建 `tests/test_calc.py`
- 使用标准库 `unittest`
- 覆盖 `add`、`divide`、`mean` 的正常行为
- 覆盖 `divide(1, 0)` 除零异常
- 覆盖 `mean([])` 空列表异常
- 未修改 `calc.py`
……
[idle_notification] calc-tests: Waiting for more work.
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s13-agent-teams/01-lead-sees.jpg" target="_blank" rel="noopener" aria-label="查看原图：两个队友各自的 messages 很长，里面全是读写和命令；它们只把结果写进 lead 的收件箱文件，Lead 的 messages 很短，只收到一条 Team events，里面是测试队友的结果和空闲通知；工具结果不会进入 Lead 的上下文"><img src="/notes-assets/lcc-s13-agent-teams/01-lead-sees.jpg" alt="两个队友各自的 messages 很长，里面全是读写和命令；它们只把结果写进 lead 的收件箱文件，Lead 的 messages 很短，只收到一条 Team events，里面是测试队友的结果和空闲通知；工具结果不会进入 Lead 的上下文" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

队友读过的文件、跑过的命令，Lead 一样都看不到，它只收到这份总结。这一点和 s06 的子 Agent 一样。不同的是，队友做完以后不会消失，而是进入 IDLE，还可以接着接活。

Lead 收到结果后调用了 `request_shutdown` 让 `calc-tests` 下线：

```
  [bus] lead -> calc-tests: (shutdown_request) Finish the current step and shut down.
Shutdown requested from calc-tests (req_518231)
  [bus] calc-tests -> lead: (shutdown_response) Shutdown acknowledged.
  [teammate] calc-tests finished
  [protocol] req_518231 -> approved
```

`req_518231` 就是原文讲的 `request_id`：回复里带着同一个 ID，Lead 一侧才把这条请求从 `pending` 改成 `approved`。

**第五步，Lead 自己验证。**

`calc-docs` 在 20:31:41 完成任务。Lead 再次被唤醒，读了两个文件，自己跑了一遍测试：

```
> bash
Error: TypeError: run_agent_bash() got an unexpected keyword argument 'timeout'
> bash
test_add (test_calc.CalcTests) ... ok
test_divide (test_calc.CalcTests) ... ok
test_divide_by_zero (test_calc.CalcTests) ... ok
test_mean (test_calc.CalcTests) ... ok
test_mean_of_empty_list (test_calc.CalcTests) ... ok
```

第一次调用 `bash` 时，模型多传了一个 schema 里没有的 `timeout` 参数，Python 函数不接受，报了 `TypeError`。这个错误被 `try` 接住，变成一句 `Error: ...` 交给模型，模型去掉这个参数又试了一次就成功了。（s02 的「想一想」第 1 题问过参数名写错会怎样；这里是类似的真实例子，只不过是多传了一个参数。s13 的 `execute_tool` 把这一行包进了 `try`，所以程序没有崩。）

最后 Lead 关掉 `calc-docs`，输出汇总，运行时打印 `[all teammates shut down]`。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s13-agent-teams/02-real-timeline.jpg" target="_blank" rel="noopener" aria-label="查看原图：真实运行的时间线：Lead 在 20:29:28 提出方案等待确认，20:30:22 建两个任务、派两个队友；测试队友到 20:31:09 完成，中途发现没装 pytest 改用 unittest；文档队友到 20:31:41 完成；Lead 每次收到两个事件就被唤醒，最后在 20:32:10 全部关机"><img src="/notes-assets/lcc-s13-agent-teams/02-real-timeline.jpg" alt="真实运行的时间线：Lead 在 20:29:28 提出方案等待确认，20:30:22 建两个任务、派两个队友；测试队友到 20:31:09 完成，中途发现没装 pytest 改用 unittest；文档队友到 20:31:41 完成；Lead 每次收到两个事件就被唤醒，最后在 20:32:10 全部关机" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

从派出队友到两个任务都完成，一共用了 79 秒。整个过程中我只说了两句话：一句需求，一句「开始吧」。

## 想一想（本站补充）

1. 这次 Lead 答应了用 worktree，最后却没有创建。如果「测试任务必须在独立目录里做」是硬性要求，应该靠 prompt 劝，还是在代码里检查？你会在哪个函数里加这个检查？

   <details class="qa"><summary>参考答案</summary>

   硬性要求应该在代码里检查，不能只靠 prompt 劝。这次运行就证明了：模型答应了也可能不做。

   可以给 `Task` 加一个字段，比如 `require_worktree`，由 Lead 建任务时声明；`claim_task` 或 `spawn_teammate` 发现这个任务要求 worktree 却还没绑定，就直接拒绝认领、拒绝启动。

   </details>

2. 两个队友在同一个目录里改代码，`calc-tests` 跑测试的时候，`calc-docs` 可能正改到一半。这次没出问题，什么情况下会出问题？worktree 能解决哪部分，解决不了哪部分？

   <details class="qa"><summary>参考答案</summary>

   `calc-tests` 跑测试的时候，`calc.py` 可能正被改到一半，甚至有语法错误，测试失败；测试队友可能以为是代码的问题，动手去「修」别人负责的文件。两个人同时编辑同一个文件，后写的会覆盖先写的。

   worktree 能解决的是：每个人有自己的一份代码，互相看不到对方改到一半的状态。解决不了的是：最后合并时的冲突；测试队友测的是**旧**代码（另一个人的修改不在它的目录里）；以及数据库、端口这类共享的外部资源。

   </details>

3. 队友想执行 `rm` 时只能收到「ask Lead」。Lead 收到请求后应该直接帮它执行吗？这样会不会变成一个绕过权限检查的后门？

   <details class="qa"><summary>参考答案</summary>

   不应该直接执行。在 s13 里，Lead 跑在主线程，它执行 `bash` 时权限检查仍然会弹出确认问用户，所以不算完全绕过。

   但如果 Lead 对队友的请求一律自动照办，或者以后改成自动批准，就真成了后门：队友只要把危险命令转给 Lead，就能绕开自己的限制。正确的做法是 Lead 把请求转给用户确认，或者直接拒绝。

   </details>

4. `task_store_lock` 同时用了线程锁和文件锁。如果只用 `fcntl.flock`、不用线程锁，同一个进程里的两个队友还能互相挡住吗？如果只用线程锁呢？

   <details class="qa"><summary>参考答案</summary>

   只用 `fcntl.flock`：同一进程里两个线程各自打开锁文件再加锁，在 macOS 和 Linux 上，两次独立的 `open` 会互相排斥，所以能挡住。但同一个线程嵌套调用时（`claim_task` 里再调 `can_start`），第二次加锁会把自己锁死，这就是 `depth` 计数要解决的事。

   只用线程锁：同一进程里没问题，另一个进程完全挡不住。

   </details>

5. 队友空闲时会自动认领任务板上能做的任务。假设 Lead 刚建好一个任务，还没来得及给它加依赖，就被某个空闲队友抢先认领了，会发生什么？代码里有没有防这件事？

   <details class="qa"><summary>参考答案</summary>

   有可能。空闲队友每 2 秒扫描一次「待开始、没人认领、依赖都满足」的任务，刚建好还没加依赖的任务，恰好符合条件。被抢先认领以后，`update_task` 要求任务待开始且无人认领，Lead 就再也加不上依赖了。

   代码里没有专门防这件事，只靠 system prompt 让 Lead「先建好所有节点、连好边，再派活」，但空闲队友不会等。可以让新建的任务默认处于不可认领的草稿状态，Lead 把图建完再统一发布。

   </details>

6. Lead 每收到一批团队事件就跑一轮。如果有 10 个队友，几乎同时完成，Lead 会被唤醒几次？每次看到的是什么？

   <details class="qa"><summary>参考答案</summary>

   不一定是 10 次。每次 Lead 被唤醒，`consume_lead_inbox` 会把当时收件箱里**所有**消息一次读走。如果 10 个结果几乎同时到达，可能被合并进一两条 `[Team events]`；如果是 Lead 正空闲时一个一个到，每来一批就唤醒一次。

   每次看到的，是上次消费之后积攒下来的全部事件，按到达顺序排列。

   </details>

## 试一下

```sh
cd learn-claude-code
python s13_agent_teams/code.py
```

输入一个自然需求：

```text
把后端重构拆到共享任务板，在依赖允许时并行完成配置、认证和测试。
认证任务使用 worktree，保持现有接口，并在最后汇总结果。
```

Lead 提出团队方案后回复：

```text
开始吧
```

观察 `.tasks/` 如何从 `pending` 进入 `in_progress` 和 `completed`，`.mailboxes/` 如何投递 `result` 与 `idle_notification`，以及 `.worktrees/` 是否只为绑定的任务创建。还可以检查直接消息是否先于任务板扫描，以及 `complete_task` 失败后队友的工作目录是否保持不变。

## 接下来

Lead 和队友目前只能调用直接写在 `code.py` 里的工具。接入 Jira、部署平台或知识库时，Harness 还要为每个外部系统分别编写工具定义和调用逻辑；外部系统增加或修改工具，也要跟着修改课程代码。

[s14 MCP Tools](/notes/lcc-s14-mcp-plugin/) → 通过统一的发现与调用协议，在运行时连接外部服务并把它们的工具加入工具池。
