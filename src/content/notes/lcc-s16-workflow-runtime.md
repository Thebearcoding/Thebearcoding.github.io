---
title: 'learn-claude-code s16：Workflow Runtime，把固定流程写成能续跑的脚本'
date: '2026-10-30'
tags: [Agent, Harness, Workflow, Multi-Agent, learn-claude-code]
summary: 转载 learn-claude-code 第 16 课，补充配图、code.py 导读和三次运行（demo、resume 和真实模型）。模型只能按名字调用宿主注册好的 workflow，脚本里用 agent、parallel、pipeline 编排子 Agent，每次调用的结果按内容哈希记进 journal，续跑时没变的调用直接用缓存。真实运行里 9 个子 Agent 用了 18 秒，三个维度各自报出了同一个 SQL 注入。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s16_workflow_runtime)（MIT License，Copyright (c) 2024 shareAI Lab），配图（部分据原仓库图示重绘）、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"一次 tool_use，跑完一整套编排"* — `Workflow` 工具启动一个可恢复的脚本运行时，协调多次 agent 调用。
>
> **Harness 层**: 编排 — 在单 agent 循环之上，执行保存好的多 agent 脚本。

从 s01 到 s15，每一轮都由模型决定调用哪些工具。工具结果进入 `messages[]` 后，模型再根据更新后的上下文决定下一步。当后续路径取决于上一步发现了什么时，这种方式很合适。

有些任务会重复一套固定流程。例如代码审查可以同时检查多个维度，再逐条验证发现、合并重复项并按严重程度排序。执行前已经知道步骤及其先后关系，这时宿主需要三样东西：

- **并行**，别一个一个串着等；
- **稳定的结果结构**，即使每个 agent 的回答会变化；
- **可恢复**，跑到一半断了，已经做完的部分别从头再来。

如果这套编排只存在于对话历史里，步骤顺序和检查点也只存在于历史里。保存好的 workflow 把固定流程写进代码，并在 journal 中记录已经完成的调用。

## 计划写在代码里，不是靠聊天一轮轮凑

在 harness 的工具池里加入一个 `Workflow` 工具。宿主注册由 `agent() / parallel() / pipeline() / phase()` 组成的可信脚本。模型只提供保存好的 workflow 名称、参数和可选的续跑 run ID，不会提交可执行代码或元数据。

workflow 以一次 `tool_use` 进入主循环。脚本运行时，runtime 会发出生命周期和进度事件，并把每一步写进磁盘上的 journal。脚本结束后，这次调用返回启动信息、结果和任务状态。脚本里的中间结果存在变量里，不会塞进对话历史。下次用 `resume_from_run_id` 重启时，没改过的 `agent()` 会直接使用 journal 中的结果。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s16-workflow-runtime/00-workflow-overview.jpg" target="_blank" rel="noopener" aria-label="查看原图：Agent 循环的工具分发里有一个 Workflow 工具，参数是名字、参数和可选的续跑 ID；它从宿主的 WORKFLOWS 注册表里找到 review-changes，交给 workflow 运行时；运行时用 phase、agent、parallel、pipeline、log 编排多个子 Agent，并在 .runtime 目录下写快照、journal 和输出文件；整个过程的事件从 task_started 到 task_notification，最后作为一条 tool_result 回到 messages"><img src="/notes-assets/lcc-s16-workflow-runtime/00-workflow-overview.jpg" alt="Agent 循环的工具分发里有一个 Workflow 工具，参数是名字、参数和可选的续跑 ID；它从宿主的 WORKFLOWS 注册表里找到 review-changes，交给 workflow 运行时；运行时用 phase、agent、parallel、pipeline、log 编排多个子 Agent，并在 .runtime 目录下写快照、journal 和输出文件；整个过程的事件从 task_started 到 task_notification，最后作为一条 tool_result 回到 messages" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

```python
SAMPLE_META = {"name": "review-changes", "description": "审查代码改动", "phases": ["Review", "Verify"]}

async def sample_workflow(ctx, args):
    ctx.phase("Review")
    results = await ctx.pipeline(DIMENSIONS, audit, verify)   # 每个维度独立走 审计 → 验证
    confirmed = [f for r in results if r for f in r["confirmed"]]
    ctx.log(f"确认了 {len(confirmed)} 个真实问题")
    return {"confirmed": confirmed}
```

## Workflow 工具：一次调用，完成整次运行

`Workflow` 会加入 s15 宿主已有的工具池。用户可以要求运行一个保存好的 workflow，模型也可以在任务匹配已知编排时选择这个工具。适配器会用名称查询宿主管理的 `WORKFLOWS` registry，再把可信的元数据和函数交给运行时；s15 的其他工具仍在同一个循环里可用。

模型可见的 schema 只接受 `name`、`args` 和 `resume_from_run_id`。名称未知或参数格式错误时，适配器会返回错误工具结果，不会让宿主循环退出。随后运行时校验已经注册的元数据、经过权限检查、注册本地 workflow 任务，并在执行脚本前发出 `async_launched`。进度事件和最终的 `task_notification` 随后到达；调用返回可写入 JSON 的启动信息、结果和任务状态。

```python
WORKFLOW_TOOL = {
    "name": "Workflow",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "args": {"type": "object"},
            "resume_from_run_id": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}

async def run_workflow(name, args=None, resume_from_run_id=None):
    meta, script_fn = WORKFLOWS[name]
    out = await WorkflowTool().call(
        meta, script_fn,
        args=args,
        resume_from_run_id=resume_from_run_id,
    )
    return {"launched": out["launched"], "result": out["result"],
            "task": serialize_task(out["task"])}
```

## Workflow 元数据：启动前先校验

每个保存好的 workflow 都会注册一份可信元数据，包含 `name`、`description` 和可选的 `phases`。运行时会在执行 workflow 代码前校验它：`name` 和 `description` 用来标识任务，`phases` 给进度显示分组命名。这些字段属于宿主 registry，不是模型输入。

注册内容不合法时，运行时会在启动前抛出 `WorkflowInputError`。这和 s12 校验 cron 表达式是一个思路：保存好的 workflow 有问题，就不要等到执行时才发现。

运行时会把 `meta.name` 用在本地产物文件名中，因此还要求它是 1-64 个字符的安全 slug，只能包含字母、数字、`.`、`_`、`-`。

```python
def validate_meta(meta):
    if not isinstance(meta, dict):
        raise WorkflowInputError("meta 必须是对象字面量")
    if not meta.get("name") or not meta.get("description"):
        raise WorkflowInputError("meta 必须包含 name 和 description")
    if not isinstance(meta["name"], str) or not WORKFLOW_NAME_RE.fullmatch(meta["name"]):
        raise WorkflowInputError("meta.name 必须是 1-64 字符的安全 slug")
    if "phases" in meta and (
        not isinstance(meta["phases"], list)
        or not all(isinstance(p, str) and p for p in meta["phases"])
    ):
        raise WorkflowInputError("meta.phases 必须包含非空字符串")
    return meta
```

## 编排原语

脚本收到一个只暴露少量编排原语的 `ExecutionState`，本身不直接读写文件，也不运行 shell。默认交互模式把 `agent()` 接到与宿主相同的真实 API client；每个子 agent 只读取 workflow 参数中提供的内容。`demo` 和单元测试使用 `MockAgentRunner`，便于重复观察事件和 journal。

| 原语 | 作用 |
|------|------|
| `agent(prompt, {schema, label, phase})` | 派一个子 agent 干活 |
| `parallel(thunks)` | **等齐屏障**：所有任务并行跑完，一起等结果回来 |
| `pipeline(items, *stages)` | 每个 item 分阶段跑，**不等齐**，跑完一个往下走一个 |
| `phase(title)` | 标记当前进度阶段（更新进度条） |
| `log(message)` | 打一行进度日志 |
| `workflow(name, args)` | 嵌套子工作流（只支持一层） |

每个 item 都要独立经过相同步骤时，可以使用 `pipeline`。item A 跑到第 3 阶段时，item B 可能还在第 1 阶段；下一步必须同时使用上一阶段全部结果时，再使用 `parallel` 等待所有调用完成。

某个分支失败时，两种原语都会先等待其他已启动的分支结束，再抛出错误。
通过 `asyncio.to_thread()` 发出的模型请求，不会因为等待它的协程被取消就停止。
等各分支结束后再关闭 journal，成功调用的结果才能保存并用于 resume，
最终任务通知也才能出现在最后一条进度事件之后。

```python
async def parallel(self, thunks):
    results = await asyncio.gather(
        *[thunk() for thunk in thunks], return_exceptions=True
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return results
```

```python
async def pipeline(self, items, *stages):
    async def run_item(item, idx):
        value = item
        for stage in stages:
            value = await stage(value, item, idx)
        return value
    return await self.parallel([
        lambda it=it, i=i: run_item(it, i) for i, it in enumerate(items)
    ])
```

## 结构化输出：别让子 agent 回来写散文

`agent({schema})` 会要求子 agent 只返回匹配 schema 的 JSON 对象。运行时解析并校验结果，不符合时重试一次。这样下游代码拿到的是对象，不必再从自然语言中提取字段。

s05 就说过，工具的参数不能全信；这里是同一个道理反过来：子 agent 的输出也不能全信。加一层校验，不对就给一次机会重试，把不确定性挡在编排层外面。

```python
run = await asyncio.to_thread(self.runner.run, prompt, schema, label)
result = run.value
if schema is not None:
    ok, err = SimpleJsonSchema(schema).validate(result)
    if not ok:                                       # 提醒一次重试，再不对就报错
        retry = await asyncio.to_thread(
            self.runner.run, prompt + "\n\n返回合法的 JSON。", schema, label
        )
        result = retry.value
        ok, err = SimpleJsonSchema(schema).validate(result)
        if not ok:
            raise WorkflowInputError(f"agent({{schema}}) 输出不合法: {err}")
```

## 任务状态和进度事件

`LocalWorkflowTask` 维护状态和 token 用量，向外发一条 SDK 风格的事件流：`task_started` → 一串 `task_progress`（包含阶段切换、子 agent 启动和日志输出）→ 最后一个 `task_notification`（完成或失败，带输出文件、agent 数和 token 数）。

演示会按顺序打印这些事件，并在最终通知后返回任务状态。

```python
class LocalWorkflowTask:
    def progress_event(self, ptype, **data):         # 阶段/子agent/日志
        self.progress.append({"type": ptype, **data})
        print(f"  进度   {ptype} ...")
```

## 存储：快照 + journal，断了能续

运行时把每次运行的数据存在 `s16_workflow_runtime/.runtime/`：快照 `<runId>.json`、输出 `<runId>.output.json`、journal `<runId>.journal.jsonl` 和协调文件 `<runId>.lock`。每次新运行都会在打开 journal 前，用排他式文件创建预留新的 `runId`。整次执行和最终持久化期间都持有 run lock，另一个进程不能同时 resume 同一次运行。快照记录 workflow 名称、参数和任务状态；resume 会先验证已保存的快照和 journal，再改动原有的成功产物。

journal 是断点续跑的核心，它一条一条记下来每个 `agent()` 的结果：

```python
class WorkflowJournal:
    def record(self, key, value):
        self._f.write(json.dumps({"key": key, "value": value}) + "\n")
        self._f.flush()
        self.cache[key] = value
```

## resume：用 runId 续跑，没改的直接用缓存

带着 `resume_from_run_id` 再次调用 workflow 时，脚本会重新执行，但每个 `agent()` 都会计算一个确定的语义 key：key 在 journal 里有记录，就直接返回缓存结果；只有改过的调用以及依赖它的后续步骤才会真的运行。

这里有个关键点：key 不能依赖并发顺序。`parallel` 和 `pipeline` 里 agent 完成的顺序是不确定的，用"第几个完成"当 key，两次跑缓存就对错位了。所以 key 是根据调用内容（类型、标签、prompt、schema）算的稳定哈希，不是一个会竞争的计数器：

```python
def key(self, kind, label, prompt, schema):
    basis = f"{kind}|{label}|{prompt}|{json.dumps(schema, sort_keys=True)}"
    return f"{kind}-{_stable_hash(basis) % 10**10:010d}"

# agent() 内部：
cached = self.journal.cached(key)
if cached is not MISS:
    self.task.progress_event("workflow_agent", label=label, status="cached")
    return cached
```

## 稳定调用键

续跑时，运行时需要把当前 `agent()` 与 journal 中的旧调用对应起来。稳定哈希让同一份 workflow 和同样的参数产生相同的调用 key。真实模型的回答可以变化；只要调用内容没有变化，resume 就直接使用 journal 中已经保存的结果。

## 跑起来看看

示例 workflow `review-changes` 用 `pipeline` 让每个审查维度独立走“审计 → 验证”。默认交互模式使用真实 API，并从 `args.changes` 读取待审查内容；`demo` 使用固定 runner 数据来展示 pipeline、结构校验、journal 和续跑。

```python
async def sample_workflow(ctx, args):
    ctx.phase("Review")
    changes = args.get("changes", "")

    async def audit(_v, dimension, _i):
        out = await ctx.agent(f"检查这段变更里有没有{dimension}相关的问题：\n{changes}",
                              schema=FINDINGS_SCHEMA, label=f"audit:{dimension}", phase="Review")
        return {"dimension": dimension, "findings": out["findings"]}

    async def verify(audited, dimension, _i):
        ctx.phase("Verify")
        verdicts = await ctx.parallel([                       # 每条发现独立做对抗性验证
            (lambda f=f: ctx.agent(f"根据变更内容验证这条 finding：\n{changes}\n\n{f}",
                                   schema=VERDICT_SCHEMA, label=f"verify:{dimension}:{f['title']}"))
            for f in audited["findings"]])
        return {"dimension": dimension,
                "confirmed": [f for f, v in zip(audited["findings"], verdicts) if v and v["isReal"]]}

    results = await ctx.pipeline(DIMENSIONS, audit, verify)
    ...
```

## 相对 s15 的变更

| | s15 Agent Harness 集成 | s16 Workflow Runtime |
|--|-----------|---------------------|
| 循环 | 单个、模型驱动 | 主循环不变；工具背后执行脚本编排 |
| 谁决定下一步 | 模型逐轮决定 | 脚本预先写好编排流程 |
| 多 agent | s06 子 agent，一次性派出去 | 通过 agent-runner 边界执行脚本化、可续跑的调用 |
| 新增机制 | — | 编排原语、宿主 registry 与工具适配器、任务生命周期、进度事件、journal/续跑、结构化输出 |

s16 不替换主循环，它只是在工具层暴露 `Workflow`，背后启动一个本地 workflow 运行时：一份保存好的脚本通过 agent-runner 边界协调 N 次调用。s06 的子 agent 是模型临场派一次；s16 把编排写成可续跑的宿主代码。


## 代码导读：README 没讲到的部分（本站补充）

完整的 [`s16_workflow_runtime/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s16_workflow_runtime/code.py) 有 899 行。它不是从头再写一个宿主，而是在 s15 的集成宿主上加一个 `Workflow` 工具。原文已经讲得很完整，这里补充几个容易忽略的地方。

### 第一段：模型只能「点菜」，不能「写菜谱」

```python
WORKFLOW_TOOL = {
    "name": "Workflow",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "args": {"type": "object"},
            "resume_from_run_id": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    },
}
```

模型能传的只有三样东西：workflow 的名字、参数、要续跑的 runId。workflow 本身是宿主事先注册在 `WORKFLOWS` 里的 Python 函数，模型改不了，也不能临时写一个新的交给 harness 执行。

这是一条信任边界。如果允许模型提交脚本，等于让它绕过所有工具的权限检查，直接在宿主进程里跑代码。`"additionalProperties": False` 也是同样的意思：多传任何字段，schema 这一关就不通过。

### 第二段：中间结果放在变量里，不放进对话

```python
async def sample_workflow(ctx, args):
    ctx.phase("Review")
    results = await ctx.pipeline(DIMENSIONS, audit, verify)
    confirmed = [f for r in results if r for f in r["confirmed"]]
    ...
    return {"confirmed": confirmed}
```

这个 workflow 会调用好几个子 Agent，每个子 Agent 的回答都存在脚本的 Python 变量里。主模型看到的只有最后 `return` 的那个字典，加上启动信息和任务状态，作为**一条** `tool_result` 回到对话里。

这和 s06 的子 Agent 一样，都是「脏活在外面干，只把结论带回来」。区别在于：s06 里由模型临场决定派几个、怎么派；这里的编排顺序是写死在代码里的。

### 第三段：子 Agent 的调用在线程里跑

```python
run = await asyncio.to_thread(self.runner.run, prompt, schema, label)
```

workflow 脚本是 `async` 函数，`parallel` 用的是 `asyncio.gather`。但 Anthropic 的 `client.messages.create` 是同步调用，直接在协程里调用会把整个事件循环卡住，几个「并行」的子 Agent 实际上就变成一个接一个了。

`asyncio.to_thread` 把每次模型调用放进单独的线程，协程只负责等它返回，几个调用就能真正同时发出去。原文提到的「取消协程不会停掉已经发出去的请求」，也是因为请求实际是在线程里跑的。

### 第四段：token 预算和 runId 预留

```python
class Budget:
    """budget.total / spent() / remaining(). Once spent reaches total, agent()
    ..."""

# agent() 开头
if self.budget.remaining() <= 0:
    raise WorkflowInputError("token budget exceeded")
```

workflow 一次可能派出几十个子 Agent。`args` 里可以带一个 `budget`，每次调用之后把用掉的 token 记上，超过预算就不再派新的子 Agent。真实运行里每个子 Agent 只用了两三百个 token，但一个写错的循环，很可能派出几百个。

```python
def reserve_run_id(meta) -> str:
    """Reserve a fresh run identity before any journal can be truncated."""
    STORE.mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        run_id = validate_run_id(create_run_id(meta))
        snapshot_path = STORE / f"{run_id}.json"
        try:
            fd = os.open(snapshot_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        ...
```

`O_CREAT | O_EXCL` 和 s10 创建任务时的 `open(..., "x")` 是同一件事：文件不存在才创建，已经存在就失败，由操作系统保证不会有两个进程拿到同一个 runId。

### 第五段：为什么不能用 Python 自带的 `hash()`

```python
def _stable_hash(s: str) -> int:
    """Process-stable hash (Python's hash() is salted per process, which would
    break resume keys across `run` and `resume`)."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s16-workflow-runtime/02-journal-keys.jpg" target="_blank" rel="noopener" aria-label="查看原图：每次 agent 调用用类型、标签、提示词和 schema 算一个稳定哈希，得到 agent-4165874729 这样的键，和结果一起写进 journal；续跑时每个调用按自己的键查 journal，11 个调用全部命中缓存，agents=0 tokens=0；如果按完成顺序编号，每次并行完成的顺序都不一样，就会对错"><img src="/notes-assets/lcc-s16-workflow-runtime/02-journal-keys.jpg" alt="每次 agent 调用用类型、标签、提示词和 schema 算一个稳定哈希，得到 agent-4165874729 这样的键，和结果一起写进 journal；续跑时每个调用按自己的键查 journal，11 个调用全部命中缓存，agents=0 tokens=0；如果按完成顺序编号，每次并行完成的顺序都不一样，就会对错" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

原文讲了 key 要根据调用内容算，不能按完成顺序编号。这里还有一个小坑：Python 的 `hash("abc")` 在每次启动进程时会加一个随机盐，同一个字符串在两次运行里算出来的值不一样。`run` 和 `resume` 是两个不同的进程，如果用 `hash()`，续跑时一个缓存都对不上。所以这里改用 `sha256`。

## 实际跑一次（本站补充）

### 第一次：demo 和 resume

`demo` 用固定的假数据代替真实模型，结果每次都一样：

```
  event      async_launched     runId=wf_review-changes_3f66e85d0eeee4c7 ...
  event      task_started       workflow=review-changes phases=Review,Verify resume=False
  progress   workflow_phase   title=Review
  progress   workflow_agent   label=audit:correctness phase=Review status=done
  progress   workflow_phase   title=Verify
  progress   workflow_agent   label=audit:security phase=Review status=done
  progress   workflow_agent   label=audit:performance phase=Review status=done
  progress   workflow_agent   label=audit:style phase=Review status=done
  progress   workflow_agent   label=verify:correctness:audit:correctness #1 phase=Verify status=done
  ……
  progress   workflow_log     message=confirmed 6 real finding(s)
  event      task_notification  status=completed agents=11 tokens=883 ...
```

注意第 4、5 行：`audit:correctness` 一做完，马上就进入了 `Verify` 阶段，这时其他三个维度还在审计。这就是 `pipeline` 「不等齐」的效果。

`.runtime/` 下的 journal 有 11 行，每行对应一次 `agent()` 调用：

```json
{"key": "agent-4165874729", "value": {"findings": [{"title": "audit:correctness #1", "severity": "low"}]}}
{"key": "agent-6350648910", "value": {"findings": [{"title": "audit:security #1", "severity": "low"}, ...]}}
……
{"key": "agent-0285516561", "value": {"isReal": false, "reason": "could not reproduce"}}
```

接着运行 `resume`，11 个调用全部显示 `status=cached`，最后一行是：

```
  event      task_notification  status=completed agents=0 tokens=0 ...
```

一个子 Agent 都没有真的调用，结果和第一次一模一样。

### 第二次：真实模型

准备一个有 SQL 注入问题的 `changes.py`：

```python
def load_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query).fetchone()
```

然后对主模型说：`读取 changes.py，把它的内容作为 changes 参数，运行保存好的 review-changes workflow 审查它，最后汇总确认的问题。`

```
20:52:31 > read_file
20:52:50 > Workflow
20:52:50   event      task_started       workflow=review-changes phases=Review,Verify resume=False
20:52:59   progress   workflow_agent   label=audit:security phase=Review status=done
20:52:59   progress   workflow_phase   title=Verify
20:52:59   progress   workflow_agent   label=audit:correctness phase=Review status=done
20:53:00   progress   workflow_agent   label=audit:style phase=Review status=done
20:53:02   progress   workflow_agent   label=audit:performance phase=Review status=done
20:53:06   progress   workflow_agent   label=verify:security:SQL injection via direct interpolation of user_id into query ...
20:53:06   progress   workflow_agent   label=verify:correctness:SQL injection vulnerability from interpolating user_id directly into the query ...
20:53:06   progress   workflow_agent   label=verify:style:Directly interpolating user_id into the SQL query enables SQL injection; ...
20:53:08   progress   workflow_agent   label=verify:performance:Interpolating user_id creates distinct SQL text per request, ...
20:53:08   progress   workflow_agent   label=verify:performance:SELECT * may fetch unnecessary columns, ...
20:53:08   progress   workflow_log     message=confirmed 4 real finding(s)
20:53:08   event      task_notification  status=completed agents=9 tokens=2579 ...
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s16-workflow-runtime/01-pipeline-timing.jpg" target="_blank" rel="noopener" aria-label="查看原图：真实运行中四个维度各自先审计再验证：security 在 20:52:59 审计完就开始验证，这时 performance 还在审计，到 20:53:02 才完成；整个 workflow 一共 9 个子 Agent、2579 个 token、18 秒"><img src="/notes-assets/lcc-s16-workflow-runtime/01-pipeline-timing.jpg" alt="真实运行中四个维度各自先审计再验证：security 在 20:52:59 审计完就开始验证，这时 performance 还在审计，到 20:53:02 才完成；整个 workflow 一共 9 个子 Agent、2579 个 token、18 秒" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

4 个审计调用几乎同时发出，9 秒后陆续返回；验证阶段又是并行的，整个 workflow 从开始到结束一共 18 秒。主模型这边只多了一条 2,592 字符的 `tool_result`，9 个子 Agent 的提示词和回答都没有进入它的上下文。

但结果里有个问题。`output.json` 里确认的 4 条是：

```json
{"confirmed": [
  {"dimension": "correctness", "title": "SQL injection vulnerability from interpolating user_id directly into the query", "severity": "high"},
  {"dimension": "security", "title": "SQL injection via direct interpolation of user_id into query", "severity": "high"},
  {"dimension": "style", "severity": "high", "title": "Directly interpolating user_id into the SQL query enables SQL injection; use a parameterized query"},
  {"dimension": "performance", "title": "Interpolating user_id creates distinct SQL text per request, potentially reducing query-plan cache reuse compared with a parameterized query", "severity": "low"}
]}
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s16-workflow-runtime/03-duplicates.jpg" target="_blank" rel="noopener" aria-label="查看原图：correctness、security、style 三个维度都报出了同一个 SQL 注入，performance 报了查询计划缓存和 SELECT 星号两条；验证阶段确认了三条 SQL 注入和查询计划缓存，否决了 SELECT 星号；workflow 报告确认 4 条，主模型汇总成 2 个问题；示例 workflow 里没有去重这一步"><img src="/notes-assets/lcc-s16-workflow-runtime/03-duplicates.jpg" alt="correctness、security、style 三个维度都报出了同一个 SQL 注入，performance 报了查询计划缓存和 SELECT 星号两条；验证阶段确认了三条 SQL 注入和查询计划缓存，否决了 SELECT 星号；workflow 报告确认 4 条，主模型汇总成 2 个问题；示例 workflow 里没有去重这一步" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

前三条说的是**同一件事**。每个维度的审计 Agent 只看到自己的问题（「检查有没有 correctness 相关的问题」），看不到别的维度报了什么，于是 SQL 注入被报了三次，验证阶段又逐条确认了三次。原文开头提到代码审查需要「合并重复项并按严重程度排序」，但这个示例 workflow 里没有写去重这一步。

最后是主模型在汇总时自己发现了重复：

```
审查完成，共确认 **4 条问题**：

1. **高危｜安全性/正确性**
   `user_id` 被直接插入 SQL ……存在 **SQL 注入漏洞**。该问题也分别在代码风格维度被确认，实质上属于同一个根因。

2. **低危｜性能**
   每个不同的 `user_id` 都会生成不同的 SQL 文本，可能降低数据库查询计划缓存的复用率。
```

固定流程的好处是可预测、可续跑；坏处是流程里没写的步骤，就真的不会发生。

## 想一想（本站补充）

1. 这次真实运行里，同一个 SQL 注入被报了三次。如果要在 workflow 里加一步去重，你会放在 `pipeline` 之后，还是 `audit` 和 `verify` 之间？用代码比较 title，还是再派一个子 Agent 来判断？各有什么代价？

   <details class="qa"><summary>参考答案</summary>

   放在 `pipeline` 之后最简单：拿到所有确认的结果，统一去重。但验证阶段已经为每个重复项各付了一次调用。

   放在 `audit` 和 `verify` 之间能省掉重复的验证，但需要先等**所有**审计都完成才能去重，`pipeline` 不等齐的好处就没了，要改成先 `parallel` 再统一验证。

   用代码比较 title 很便宜，但措辞稍有不同就认不出来，这次三条标题就各不相同。派一个子 Agent 来判断「哪些说的是同一件事」更准，代价是多一次调用，而且结果需要用 schema 约束。

   </details>

2. 改了 `audit` 的 prompt 以后再 `resume`，哪些调用会命中缓存，哪些会重新执行？依赖它的 `verify` 呢？

   <details class="qa"><summary>参考答案</summary>

   key 是根据调用内容算的，包括 prompt。改了 `audit` 的 prompt，4 个审计调用的 key 全变了，全部重新执行。

   `verify` 的 prompt 里包含了审计找到的那条 finding 的内容。新的审计如果返回的措辞变了，`verify` 的 key 也跟着变，同样重新执行；如果新审计恰好返回了一模一样的 finding，`verify` 还会命中缓存。

   </details>

3. 为什么只允许模型调用宿主注册好的 workflow，而不让模型自己提交一段编排脚本？如果非要支持，至少要加哪些限制？

   <details class="qa"><summary>参考答案</summary>

   让模型提交脚本，就等于让它在宿主进程里执行任意代码：读写任何文件、访问网络，所有工具的权限检查都被绕开了。

   如果非要支持，至少要：放进隔离的进程或沙箱里跑，不给文件和网络权限；只允许使用几个固定的编排原语，而不是任意 Python；执行前给用户看并确认；再加上 token 预算和时间上限。

   </details>

4. `parallel` 里有一个分支失败了，运行时会先等其他分支跑完再抛错。为什么不立刻取消其他分支？

   <details class="qa"><summary>参考答案</summary>

   已经发出去的模型请求跑在线程里，取消等待它的协程，并不能让请求停下来，它还会继续执行、继续花 token。

   等所有分支都结束，成功的那些结果才能写进 journal，下次 `resume` 时可以直接复用，不用再花一遍钱；同时也保证最后的任务通知一定出现在所有进度事件之后，状态是一致的。

   </details>

5. workflow 一次派出 9 个子 Agent，主模型只收到 2.6k 字符的结果。这和 s06 的子 Agent、s13 的队友相比，分别适合什么样的任务？

   <details class="qa"><summary>参考答案</summary>

   workflow：步骤事先就知道、需要并行、需要可重复和续跑的流程，比如代码审查、批量处理一堆文件。

   s06 的子 Agent：临时的一次性探索，怎么做由模型当场决定，做完就扔。

   s13 的队友：长时间、多个任务、需要来回沟通的协作，比如几个人分头改不同的模块。

   </details>

6. 真实模型每次的回答都不一样。同一个 workflow，`resume` 用的是第一次运行时缓存下来的回答。什么情况下这是你想要的，什么情况下反而是问题？

   <details class="qa"><summary>参考答案</summary>

   想要的情况：一个大批量任务跑到一半断了，续跑时已经完成的部分不用重来；或者想复现上次的结果，方便排查问题。

   成问题的情况：你就是想「重新审一遍」，看看有没有新的意见，缓存却返回了上次的回答；或者 prompt 里只写了文件名，没写文件内容，代码改了而 key 没变，缓存的结论就过时了。这个示例把变更内容直接放进了 prompt，内容一变 key 就变，避开了后一种情况。

   </details>

## 试一下

```bash
python s16_workflow_runtime/code.py          # 主模型和 Workflow 子 agent 都使用真实 API
python s16_workflow_runtime/code.py demo     # 运行确定性的 review-changes 测试数据并观察事件流
python s16_workflow_runtime/code.py resume   # 用上次的 runId 续跑，每个 agent() 都命中 journal 缓存
```

默认命令里，可以先让模型读取改动，再把内容放进 `args.changes` 并运行保存好的 `review-changes` workflow。主模型和 workflow 子 agent 都使用真实 API。`demo` 命令使用固定 runner 数据，便于重复观察生命周期和续跑；续跑命中全部缓存时显示 `agents=0 tokens=0`。

## 接下来

[s17 Goal Loop](/notes/lcc-s17-goal-loop/) 会使用一个更小、独立的循环检查既定目标是否已经达成，并据此决定是否还需要下一轮。
