---
title: 'learn-claude-code s10：Task System，把任务和依赖写到磁盘上'
date: '2026-10-30'
tags: [Agent, Harness, Task, Planning, learn-claude-code]
summary: 转载 learn-claude-code 第 10 课，补充配图、code.py 导读和一次跨三个进程的真实运行。每个任务是 .tasks/ 下的一个 JSON 文件，blockedBy 记录依赖，claim 和 complete 两个动作推动状态；先建节点再连边，连边前检查自依赖和环。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s10_task_system)（MIT License，Copyright (c) 2024 shareAI Lab），配图（部分据原仓库图示重绘）、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"大目标拆成小任务, 排好序, 持久化"* — 文件持久化的任务图, 多 agent 协作的基础。
>
> **Harness 层**: 任务 — 持久化的目标, 可恢复的进度。

## 问题

s05 的 TodoWrite 让 Agent 记录当前任务的执行步骤。清单中的每一项只有内容和状态，用来提醒 Agent 接下来还要做什么。

当项目被拆成创建数据库表、编写 API 和添加测试三个任务时，Harness 还需要知道它们之间的关系：数据库表完成后才能编写 API，API 接口确定后才能添加测试。每个任务还要记录由谁负责。

TodoWrite 没有记录这些依赖和分工。它可以显示“编写 API”仍未完成，但 Harness 无法据此判断这个任务是否可以开始。

本章加入 Task System。每个任务都有独立的 ID 和状态，`blockedBy` 记录前置任务，`owner` 记录负责执行的 Agent。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s10-task-system/00-task-overview.jpg" target="_blank" rel="noopener" aria-label="查看原图：Agent 循环的工具分发里，除了 5 个基础工具，多了 create_task、update_task、list_tasks、get_task、claim_task、complete_task 六个任务工具；任务存成 .tasks 下的 JSON 文件，字段有 id、subject、description、status、owner、blockedBy；状态从 pending 经 claim 到 in_progress，再经 complete 到 completed"><img src="/notes-assets/lcc-s10-task-system/00-task-overview.jpg" alt="Agent 循环的工具分发里，除了 5 个基础工具，多了 create_task、update_task、list_tasks、get_task、claim_task、complete_task 六个任务工具；任务存成 .tasks 下的 JSON 文件，字段有 id、subject、description、status、owner、blockedBy；状态从 pending 经 claim 到 in_progress，再经 complete 到 completed" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

代码保留 S04 的五个基础工具、Permission、Hooks 和统一 `execute_tool`，再加入 6 个任务工具、`.tasks/` 目录持久化和 `blockedBy` 依赖检查。

TodoWrite vs Task System：

| | TodoWrite (s05) | Task System (s10) |
|---|---|---|
| 定位 | 当前任务的执行清单 | 可恢复的任务系统 |
| 存储 | 进程内 / 会话状态 | `.tasks/{id}.json` |
| 依赖 | 无 | `blockedBy` 依赖图 |
| 生命周期 | 当前会话 / 当前任务 | 跨会话保留 |
| 分工 | 不负责任务认领 | `owner` / claim |
| 状态 | pending / in_progress / completed | pending / in_progress / completed |
| 粒度 | Agent 自己的步骤 | 可被认领、追踪、解锁的任务 |
| 更新契约 | 整表替换 | 对单条记录执行创建、读取、更新、列举 |

## 工作原理

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s10-task-system/00-task-dag.jpg" target="_blank" rel="noopener" aria-label="查看原图：任务依赖图：schema 已完成；endpoints 进行中，依赖 schema；docs 依赖 schema，可以开始；tests 依赖 endpoints，被阻塞；deploy 依赖 tests 和 docs，被阻塞"><img src="/notes-assets/lcc-s10-task-system/00-task-dag.jpg" alt="任务依赖图：schema 已完成；endpoints 进行中，依赖 schema；docs 依赖 schema，可以开始；tests 依赖 endpoints，被阻塞；deploy 依赖 tests 和 docs，被阻塞" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

### Task: 数据结构

每个任务是一个 JSON 文件，存于 `.tasks/` 目录：

```python
@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # 负责当前任务的 Agent
    blockedBy: list[str] # 依赖的任务 ID 列表
```

ID 使用 `task_` 加 8 位随机十六进制字符生成。创建文件时使用排他写入；如果 ID 已存在，就重新生成。

`TaskStore` 负责校验任务 ID 和读写 JSON 文件，`TASKS = TaskStore(TASKS_DIR)` 是本章使用的任务存储。

### create_task: 创建任务

```python
def create_task(subject: str, description: str = "") -> Task:
    return TASKS.create(subject, description)
```

`TaskStore.create` 检查 subject，分配随机 ID，再把任务写入 `.tasks/{id}.json`。新任务的 `blockedBy` 固定为空，工具结果会把运行时生成的 ID 返回给模型。

### update_task: 使用返回的 ID 添加依赖

```python
def update_task(task_id: str, addBlockedBy: list[str]) -> Task:
    return TASKS.update_dependencies(task_id, addBlockedBy)
```

任务图采用两阶段构建：先创建所有节点，再使用 `create_task` 返回的 ID 调用 `update_task` 添加边。模型可能在一条回复里同时发出多个工具调用，而这些同级调用在任何工具结果产生前就已经确定，因此某个 `create_task` 无法直接使用另一个调用刚生成的 ID。

`update_task` 会先校验整次修改，再统一保存。目标任务和依赖必须存在，目标必须仍为 pending 且无人认领，并且不能形成自依赖或环。重复添加已有依赖是安全的，不会产生重复边。

### can_start: 依赖检查

一个任务只能在它的 `blockedBy` **全部 completed** 之后才能开始：

```python
def can_start(task_id: str) -> bool:
    return not incomplete_dependencies(load_task(task_id))
```

`incomplete_dependencies` 读取每个前置任务。只要有一个不是 completed，或者对应文件已经不存在，任务就不能认领。

### claim_task: 认领任务

Agent 开始做一个任务时，调用 `claim_task`：设置 `owner`，状态从 `pending` → `in_progress`。`owner` 字段记录谁认领了这个任务：

```python
def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    dependencies = incomplete_dependencies(task)
    if dependencies:
        return f"Blocked by: {dependencies}"
    task.owner = owner
    task.status = "in_progress"
    TASKS.save(task)
    return f"Claimed {task_id} ({task.subject})"
```

如果任务不是 pending，或者依赖没有完成，就拒绝认领。S10 只处理顺序执行的状态更新。

### complete_task: 完成与解锁

任务做完后，设为 `completed`。同时扫描所有其他任务，找出**刚刚被解锁**的下游任务：

```python
def complete_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    if task.owner != owner:
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    ready_before = {t.id for t in list_tasks()
                    if t.status == "pending" and t.blockedBy
                    and can_start(t.id)}
    task.status = "completed"
    TASKS.save(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy
                 and t.id not in ready_before
                 and can_start(t.id)]
    msg = f"Completed {task_id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
    return msg
```

完成 "schema" 后，"endpoints" 和 "docs" 的 `can_start` 返回 True，它们可以开始。

### get_task: 查看完整细节

`list_tasks` 只显示一行摘要。`get_task` 返回完整的任务 JSON，包括 description 和依赖细节。跨会话恢复时，Agent 需要读取完整描述才能继续工作：

```python
def get_task(task_id: str) -> str:
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)
```

### 状态机: 两个动作，三个状态

```
pending ──claim──→ in_progress ──complete──→ completed
```

这里的 `claim` / `complete` 是动作，`pending` / `in_progress` / `completed` 是状态：

- **claim_task**: `pending` → `in_progress`。设置 owner，开始工作。
- **complete_task**: `in_progress` → `completed`。把任务标记为完成，并解锁下游。

### 合起来跑

```python
# 第一阶段：创建所有节点并取得运行时 ID
schema = create_task("setup database schema")
endpoints = create_task("create API endpoints")
tests = create_task("write tests")
docs = create_task("write docs")

# 第二阶段：使用返回的 ID 建立依赖边
update_task(endpoints.id, addBlockedBy=[schema.id])
update_task(tests.id, addBlockedBy=[endpoints.id])
update_task(docs.id, addBlockedBy=[schema.id])

# Agent 认领第一个可做的任务
claim_task(schema.id)       # ✓ Claimed (无依赖)
complete_task(schema.id)    # ✓ Completed → 解锁 endpoints, docs

claim_task(endpoints.id)    # ✓ Claimed (schema 已完成)
complete_task(endpoints.id) # ✓ Completed → 解锁 tests

claim_task(docs.id)         # ✓ Claimed (schema 已完成)
complete_task(docs.id)      # ✓ Completed

claim_task(tests.id)        # ✓ Claimed (endpoints 已完成)
complete_task(tests.id)     # ✓ Completed
```

每个 `create_task` 写一个 JSON 文件，`update_task`、`claim_task` 和 `complete_task` 更新文件。跨会话时，`.tasks/` 目录还在，Agent 读文件就能恢复进度。


## 代码导读：README 没讲到的部分（本站补充）

完整的 [`s10_task_system/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s10_task_system/code.py) 有 602 行。和 s06 到 s09 一样，它也是在 s04 的骨架（五个基础工具、权限、hooks）上加的。s05 的 `todo_write` 没有带过来，任务清单由这一整套任务工具来管。原文已经把每个函数做什么讲清楚了，这里补充几个「为什么这么写」。

### 第一段：任务 ID 是模型传进来的，先验格式

```python
TASK_ID_PATTERN = re.compile(r"^task_[0-9a-f]{8}$")

def _path(self, task_id: str, create_root: bool = False) -> Path:
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(f"Invalid task ID: {task_id!r}")
    root = self._root(create=create_root)
    path = (root / f"{task_id}.json").resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid task ID: {task_id!r}")
    return path
```

`claim_task`、`update_task` 这些工具的参数都是模型给的 `task_id`，代码直接拿它去拼文件名。所以第一步就是用正则卡死格式：只能是 `task_` 加 8 位小写十六进制。`../../.env` 这种东西在这一步就会被拒绝。

后面的 `is_relative_to` 是第二道保险。按理说正则通过以后不可能跑出目录，但「按理说」不等于「一定」，多查一次没有坏处。

### 第二段：创建任务用「排他写入」

```python
for _ in range(100):
    task = Task(id=f"task_{secrets.token_hex(4)}", ...)
    try:
        with self._path(task.id, create_root=True).open("x", encoding="utf-8") as handle:
            json.dump(asdict(task), handle, indent=2)
        return task
    except FileExistsError:
        continue
raise RuntimeError("Could not allocate a unique task ID")
```

`open(..., "x")` 的意思是：文件已经存在就报错，不存在才创建。这个「检查是否存在 + 创建」是操作系统一次完成的，不会出现两个进程都觉得「这个 ID 没人用」、然后互相覆盖的情况。

8 位十六进制有 40 多亿种可能，撞上的概率很小。真撞上了，就换一个 ID 再试，最多试 100 次。

### 第三段：读文件时也不全信

```python
def load(self, task_id: str) -> Task:
    data = json.loads(self._path(task_id).read_text(encoding="utf-8"))
    task = Task(**data)
    if task.id != task_id:
        raise ValueError(f"Task file ID does not match {task_id}")
    if task.status not in ("pending", "in_progress", "completed"):
        raise ValueError(f"Invalid task status: {task.status}")
    return task
```

`.tasks/` 是普通文件夹，模型有 `write_file` 和 `bash`，完全可以绕过任务工具直接改 JSON。所以读回来以后，要检查文件里的 `id` 和文件名对得上、`status` 是三个合法值之一。不合法就报错，交给 `execute_tool` 变成一句 `Error: ...`。

### 第四段：环检测，从新依赖出发往回走

```python
def _depends_on(self, task_id: str, target_id: str) -> bool:
    """Return whether task_id transitively depends on target_id."""
    pending = [task_id]
    visited = set()
    while pending:
        current = pending.pop()
        if current == target_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        pending.extend(self.load(current).blockedBy)
    return False
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s10-task-system/02-update-checks.jpg" target="_blank" rel="noopener" aria-label="查看原图：已有依赖链 A 被 B 依赖、B 被 C 依赖，这时要给 A 加上依赖 C，会形成环，被拒绝；update_task 保存前依次检查：目标任务待开始且无人认领、不依赖自己、依赖的任务文件存在、从依赖出发走不回目标任务，全部通过才保存一次"><img src="/notes-assets/lcc-s10-task-system/02-update-checks.jpg" alt="已有依赖链 A 被 B 依赖、B 被 C 依赖，这时要给 A 加上依赖 C，会形成环，被拒绝；update_task 保存前依次检查：目标任务待开始且无人认领、不依赖自己、依赖的任务文件存在、从依赖出发走不回目标任务，全部通过才保存一次" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

要给 A 加一条「A 依赖 C」，就从 C 出发，沿着 `blockedBy` 一路往上找。如果能走到 A，说明 C 本来就（间接地）依赖 A，再加这条边就成环了：A 等 C，C 等 B，B 又等 A，谁也开始不了。

这是一个用栈实现的深度优先搜索，`visited` 防止在菱形依赖（两条路通到同一个任务）上重复走。

`update_dependencies` 的结构也值得注意：先用一个 `for` 循环把所有新依赖**全部检查完**，有一个不合格就整体报错；全部通过后，才一次性 `extend` 并 `save`。不会出现「加了一半、另一半失败」的状态。

### 第五段：为什么一定要「先建节点，再连边」

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s10-task-system/01-nodes-then-edges.jpg" target="_blank" rel="noopener" aria-label="查看原图：第一次回复里并行发出四个 create_task，结果回来后才有 task_99f5 等 ID；第二次回复里的三个 update_task 才能用上这些 ID 建立依赖"><img src="/notes-assets/lcc-s10-task-system/01-nodes-then-edges.jpg" alt="第一次回复里并行发出四个 create_task，结果回来后才有 task_99f5 等 ID；第二次回复里的三个 update_task 才能用上这些 ID 建立依赖" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

原文提到了「两阶段构建」，这里说得更直白一点。模型在**一次回复**里发出的多个工具调用，是它在看到任何结果之前就一起写好的。`create_task(endpoints)` 写出来的时候，`create_task(schema)` 还没执行，schema 的 ID 根本不存在。

所以 `create_task` 干脆不接受 `blockedBy` 参数，SYSTEM 里也明确写了：先建好所有节点，拿到 ID 以后再用 `update_task` 连边。

### 第六段：依赖找不到，按「没完成」处理

```python
def incomplete_dependencies(task: Task) -> list[str]:
    incomplete = []
    for dependency in task.blockedBy:
        try:
            if load_task(dependency).status != "completed":
                incomplete.append(dependency)
        except (FileNotFoundError, ValueError):
            incomplete.append(dependency)
    return incomplete
```

前置任务的文件被删了，或者内容坏了，代码不会当它「不存在所以不用管」，而是当它**没完成**，下游任务继续被挡着。宁可卡住，也不要在前置条件不明的时候放行。

### 第七段：`complete_task` 不检查活干没干完

```python
def complete_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    if task.owner != owner:
        return f"Task {task_id} is owned by {task.owner}, not {owner}"
    ...
```

它只检查两件事：任务确实在进行中，而且是你认领的。至于数据库表到底建没建好，代码一概不知。**什么时候算完成，完全由模型自己决定。**

`ready_before` 那一步是为了让返回信息更准确：只报告**这一次**新解锁的任务，之前就已经可以开始的不重复报告。

## 实际跑一次（本站补充）

在一个空目录里分三次启动程序，中间都退出过。

**第一次运行**，用原文的第 1 个 prompt 建任务：

```
[HOOK] create_task(['Setup database schema', 'Design and implement the database)
  [create] Setup database schema
[HOOK] create_task(['Create API endpoints', 'Implement API endpoints after the )
  [create] Create API endpoints
[HOOK] create_task(['Write tests', 'Write tests after the API endpoints are imp)
  [create] Write tests
[HOOK] create_task(['Write docs', 'Write documentation after the database schem)
  [create] Write docs
[HOOK] update_task(['task_96249b80', ['task_99f597f1']])
  [update] Create API endpoints blockedBy: task_99f597f1
[HOOK] update_task(['task_97d42687', ['task_96249b80']])
  [update] Write tests blockedBy: task_96249b80
[HOOK] update_task(['task_7aeddc5d', ['task_99f597f1']])
  [update] Write docs blockedBy: task_99f597f1
```

代理日志显示，第一次回复里有 4 个 `tool_use`（四个 `create_task`），第二次回复里有 3 个（三个 `update_task`）。模型完全按照「先建节点、再连边」来做。`.tasks/` 里出现了 4 个 JSON 文件，比如：

```json
{
  "id": "task_96249b80",
  "subject": "Create API endpoints",
  "description": "Implement API endpoints after the database schema is ready.",
  "status": "pending",
  "owner": null,
  "blockedBy": [
    "task_99f597f1"
  ]
}
```

**第二次运行**是一个新进程，`history` 为空。先问「列出所有任务」，模型调用 `list_tasks`，从磁盘上读回了全部 4 个任务和依赖关系。

然后我故意让它去认领被挡住的任务：

```
s10 >> Claim the write tests task
[HOOK] claim_task(['task_97d42687'])
Unable to claim **Write tests** (`task_97d42687`) because it is blocked by **Create API endpoints** (`task_96249b80`).
```

接着让它「认领第一个能做的任务并完成它」。这一次它不只是改了状态，还真的去干活了：

```
[HOOK] claim_task(['task_99f597f1'])
  [claim] Setup database schema -> in_progress (owner: agent)
[HOOK] glob(['**/*'])
[HOOK] bash(['pwd && ls -la'])
[HOOK] bash(['find .tasks -type f -maxdepth 3 -print -exec cat {} \\;'])
[HOOK] get_task(['task_99f597f1'])
[HOOK] write_file(['db/schema.sql', "-- Task service database schema (SQLite 3)

[permission] Potentially destructive command
   Tool: bash({'command': 'set -eu\ncommand -v sqlite3 >/dev/null\nsqlite3 /tmp/task-schema-test.sqlite < db/schema.sql\nsqlite3 /tmp/task-schema-test.sqlite ".schema"\nrm /tmp/task-schema-test.sqlite'})
   Allow? [y/N] [HOOK] read_file(['db/schema.sql'])
[HOOK] complete_task(['task_99f597f1'])
  [complete] Setup database schema
  [unblocked] Write docs, Create API endpoints
```

它写了一份 `db/schema.sql`，还想用 sqlite 建一个临时数据库验证一下。验证命令最后有一句 `rm`，触发了 s03 的权限确认。

这里有个小插曲：我是用管道一次性把所有输入喂进去的，权限确认的 `input()` 把排在后面的下一句提问当成了回答。那句话不是 `y`，所以命令被拒绝了。模型没能跑验证，改成读一遍文件，然后就调用了 `complete_task`。这正好说明了上面第七段：**任务标成完成，不代表做完的东西验证过。**

**第三次运行**，再列一次任务：

```
Now unblocked:

- `task_7aeddc5d` — **Write docs**
- `task_96249b80` — **Create API endpoints**

Still blocked:

- `task_97d42687` — **Write tests** (blocked by `task_96249b80`)
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s10-task-system/03-progress-on-disk.jpg" target="_blank" rel="noopener" aria-label="查看原图：三次运行共用 .tasks 目录：第一次建 4 个任务和 3 条依赖；第二次认领 tests 被拒绝，认领并完成 schema 后解锁 docs 和 endpoints；第三次列任务，docs 和 endpoints 可以开始，tests 仍被阻塞"><img src="/notes-assets/lcc-s10-task-system/03-progress-on-disk.jpg" alt="三次运行共用 .tasks 目录：第一次建 4 个任务和 3 条依赖；第二次认领 tests 被拒绝，认领并完成 schema 后解锁 docs 和 endpoints；第三次列任务，docs 和 endpoints 可以开始，tests 仍被阻塞" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

三个进程之间没有共享任何内存，进度全靠 `.tasks/` 里的 4 个文件接力。

## 想一想（本站补充）

1. `complete_task` 不检查活是不是真的干完了。如果模型没写 schema 就直接调用 `complete_task`，下游任务会怎样？你会在哪里加验证，验证什么？

   <details class="qa"><summary>参考答案</summary>

   下游任务照样被解锁，后面的人会在一个根本不存在的 schema 上写 API。

   验证可以加在 `complete_task` 里：任务描述里写好验收条件（比如「`pytest tests/db` 通过」），完成前必须提交证据或者由 harness 自己跑一遍；也可以做成一个 hook；或者像 s17 那样，让一个独立的判断器根据对话里的结果来决定。关键是验证**可以检查的结果**，而不是模型的一句「做完了」。

   </details>

2. 进程在 `claim_task` 之后、`complete_task` 之前崩溃了，任务就一直停在 `in_progress`，`owner` 是 `agent`。下次启动时，谁能把它接着做完？现在的代码允许吗？

   <details class="qa"><summary>参考答案</summary>

   现在的代码其实允许：所有进程的 `owner` 都叫 `"agent"`，新进程调用 `complete_task` 时 owner 能对上，可以直接把任务标成完成。但它不知道上一个进程做到了哪一步，也分不清「上一个进程崩了」和「另一个进程正在做」。

   更好的做法是给认领加一个**租约**：记录认领时间，做的过程中定期续约，超时没续约就视为放弃，任务回到 `pending`，谁都可以重新认领。

   </details>

3. 如果两个 Agent 进程**同时**认领同一个任务，两边读到的都是 `pending`，会发生什么？s10 的代码哪里没有防这件事？（s13 会讲到）

   <details class="qa"><summary>参考答案</summary>

   两边都先读到 `pending`、没人认领，都改成 `in_progress` 再写回去，后写的覆盖先写的。两个 Agent 都以为任务归自己，会同时去做同一件事。

   s10 的 `claim_task` 是「读 → 检查 → 写」三步，中间没有锁，不是原子操作。s13 用 `task_store_lock()` 把这三步包起来，并在锁里重新读一遍任务。

   </details>

4. 为什么依赖只能在任务「待开始且无人认领」的时候修改？如果允许给一个进行中的任务追加依赖，会出现什么奇怪的状态？

   <details class="qa"><summary>参考答案</summary>

   任务已经在做了，却又多出一个还没完成的前置任务，状态就自相矛盾：它到底是在进行中，还是被阻塞了？正在做的人要不要停下来？

   只允许在任务还没开始、也没人认领的时候改依赖，图的结构一旦有人开工就固定下来，就不会出现这种两难。

   </details>

5. 依赖的任务文件找不到时，代码把它当成「没完成」。这样做有什么好处？什么情况下会让人困扰？

   <details class="qa"><summary>参考答案</summary>

   好处是「前提不明就不放行」：宁可卡住，也不在前置条件不清楚的时候开工，不会在错误的基础上越做越多。

   困扰在于：不小心删了一个任务文件，下游会永远卡住，报的还是 `Blocked by: [task_xxx]`，一个已经不存在的 ID，很难排查。可以把「找不到」和「没完成」区分开，找不到时明确报错，提示需要人工处理。

   </details>

6. 这次运行里，权限确认的 `input()` 把用户的下一句话当成了回答。真实的产品里，「回答权限确认」和「输入下一个问题」应该怎么分开？

   <details class="qa"><summary>参考答案</summary>

   两件事要走不同的通道。图形界面里通常是一个单独的确认弹窗或按钮，和输入框分开；命令行里，可以在弹确认之前清空已经缓冲的输入，并且明确提示「下面这个回答只针对权限确认」。

   更彻底的做法是把权限请求做成带 ID 的结构化事件（像 s13 的协议那样），由界面专门展示、专门收集回答，不和普通输入共用一个 `input()`。

   </details>

## 试一下

```sh
cd learn-claude-code
python s10_task_system/code.py
```

试试这些 prompt：

1. `Create tasks: setup database schema, create API endpoints (depends on schema), write tests (depends on endpoints), write docs (depends on schema)`
2. `List all tasks and their statuses`
3. `Claim the first unblocked task and complete it`
4. `List tasks again — which ones are now unblocked?`

观察重点：`.tasks/` 目录下是否生成了 JSON 文件？完成任务后，被阻塞的任务是否解锁？

## 接下来

任务图有了，但全量测试、安装依赖和部署等命令可能需要很长时间。同步执行这些命令时，Agent Loop 会一直停在当前工具调用上，只有命令结束后才能继续处理其他工作。

[s11 Background Tasks](/notes/lcc-s11-background-tasks/) → 把慢操作放到后台。Agent 可以继续处理其他任务，后台执行完成后再接收通知。
