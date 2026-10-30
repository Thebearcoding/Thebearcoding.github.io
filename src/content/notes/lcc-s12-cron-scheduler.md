---
title: 'learn-claude-code s12：Cron Scheduler，给 Agent 装一个闹钟'
date: '2026-10-30'
tags: [Agent, Harness, Scheduling, Concurrency, learn-claude-code]
summary: 转载 learn-claude-code 第 12 课，补充配图、code.py 导读和一次跑了三分钟的真实运行。调度线程每秒对一次表，到点的任务进队列；队列线程等 Agent 空闲、拿到 agent_lock 后，把 [Scheduled] prompt 当作用户消息交给 Agent Loop。真实运行里任务刚创建 1 秒就触发了第一次，之后每个整分钟准时触发，用户的新问题要排在定时任务后面。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s12_cron_scheduler)（MIT License，Copyright (c) 2024 shareAI Lab），配图（部分据原仓库图示重绘）、「代码导读」「实际跑一次」「想一想」三节为本站补充。

## 问题

S11 解决的是命令开始后的执行方式：耗时的 Bash 命令可以在后台运行。但它不会记录某项工作应该在什么时间开始，也没有组件持续检查当前时间。

对于“每天早上 9 点跑测试”或“每 30 分钟检查 CI 状态”这样的请求，如果只依靠当前的 Agent Loop，用户仍要在每次到点后重新发送 prompt。Harness 需要保存执行时间，到点后把对应的 prompt 加入待执行队列，再在 Agent 空闲时交给 Agent Loop。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s12-cron-scheduler/00-cron-overview.jpg" target="_blank" rel="noopener" aria-label="查看原图：调度线程每秒检查一次到期任务，匹配后先存盘再放进 cron_queue；Agent 空闲时接收，[Scheduled] 提示被注入到 messages；cron 表达式有分、时、日、月、星期五个字段；持久任务存在 .scheduled_tasks.json，会话任务只在内存里；进程退出调度器就停了"><img src="/notes-assets/lcc-s12-cron-scheduler/00-cron-overview.jpg" alt="调度线程每秒检查一次到期任务，匹配后先存盘再放进 cron_queue；Agent 空闲时接收，[Scheduled] 提示被注入到 messages；cron 表达式有分、时、日、月、星期五个字段；持久任务存在 .scheduled_tasks.json，会话任务只在内存里；进程退出调度器就停了" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

假设 Agent 注册了下面这项任务：

```text
cron:   0 9 * * *
prompt: run tests
```

调度线程在本地时间 09:00 匹配到这项任务，把 `[Scheduled] run tests` 放进 `cron_queue`。队列处理线程等到 Agent 空闲后启动一轮 Agent Loop，模型随后可以调用 Bash 执行测试。

S12 的代码保留 S04 的五个基础工具和 Hooks，再增加 `schedule_cron`、`list_crons`、`cancel_cron`。它不包含 S11 的后台命令，因为这里传递的是一条待执行的 prompt，而不是某个后台命令的执行结果。

## 工作原理

### CronJob 保存什么

```python
@dataclass
class CronJob:
    id: str
    cron: str
    prompt: str
    recurring: bool
    durable: bool
    pending_delivery: bool = False
    last_fired: str | None = None
```

`cron` 决定何时触发，`prompt` 是触发后交给 Agent 的任务。`pending_delivery` 表示任务已经到期但尚未被模型接收，`last_fired` 防止同一分钟重复入队。

### 五段式 Cron 表达式

```text
分钟  小时  日  月  星期
  *    *   *   *   *      每分钟
  0    9   *   *   *      每天 09:00
 */5   *   *   *   *      每 5 分钟
  0    9   *   *  1-5     工作日 09:00
```

本章支持 `*`、`*/N`、`N`、`N-M` 和 `N,M,...`。`schedule_job()` 会在保存任务前调用 `validate_cron()`，拒绝字段数量或取值范围不正确的表达式。

### 到期后先入队

调度线程每秒读取一次本地时间。表达式匹配且任务在当前分钟尚未触发时，`_enqueue_due_job()` 先保存 `pending_delivery` 和 `last_fired`，再把任务放进内存队列：

```python
def poll_due_jobs(moment: datetime):
    minute_marker = moment.strftime("%Y-%m-%d %H:%M")
    with cron_lock:
        for job in list(scheduled_jobs.values()):
            if job.pending_delivery or job.last_fired == minute_marker:
                continue
            if cron_matches(job.cron, moment):
                _enqueue_due_job(job, minute_marker)
```

持久化失败时，`_enqueue_due_job()` 会恢复原来的状态，不会把只存在于内存中的任务暴露给队列处理线程。

### Agent 空闲后再交付

`queue_processor_loop()` 不负责判断时间。它只检查队列，并用 `agent_lock` 避免定时任务与用户正在进行的回合同时修改会话：

```python
def queue_processor_loop(stop_event=RUNTIME_STOP):
    while not stop_event.wait(0.2):
        if not has_cron_queue() or not agent_lock.acquire(blocking=False):
            continue
        try:
            if has_cron_queue():
                run_agent_turn_locked()
        finally:
            agent_lock.release()
```

Agent Loop 从队列取出到期任务，并把它们作为新的用户消息追加：

```python
fired = consume_cron_queue()
for job in fired:
    messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
```

模型调用失败时，这些消息会从当前会话中移除，任务重新放回队列。模型成功接收后，一次性任务会被删除，周期任务则清除 `pending_delivery`，等待下一次匹配。

### 持久化边界

| 模式 | 保存位置 | 进程重启后 |
|---|---|---|
| `durable=True` | `.scheduled_tasks.json` | 重新加载 |
| `durable=False` | 内存 | 消失 |

`.scheduled_tasks.json` 使用临时文件和 `os.replace()` 更新。文件损坏时，启动日志会报告错误，不会静默忽略。

这里采用至少一次交付：进程若在模型接收 prompt 后、确认状态写回前退出，同一任务可能在重启后再次交付。

### 运行边界

- 调度器使用 Agent 进程的本地时间。
- Agent 进程关闭后，调度线程也会停止；`durable` 只保留任务定义。
- 重启时只恢复任务，不补跑停机期间错过的时间点。
- 定时回合运行在队列处理线程中。需要交互确认的工具调用会被拒绝，不会与主终端同时读取输入。
- 调度线程和队列处理线程只在运行 CLI 时启动，导入 `code.py` 不会启动后台线程。

需要在 Agent 关闭时仍按时执行任务，应使用系统的 crontab、systemd timer 或其他外部调度服务。


## 代码导读：README 没讲到的部分（本站补充）

完整的 [`s12_cron_scheduler/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s12_cron_scheduler/code.py) 有 792 行，同样是在 s04 的骨架上加的，没有带上 s11 的后台命令。新代码分三块：cron 表达式的解析和校验、任务的持久化和交付、两条后台线程。

### 第一段：三条线程，一把锁

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s12-cron-scheduler/01-three-threads.jpg" target="_blank" rel="noopener" aria-label="查看原图：调度线程每秒检查一次到期任务，放进 cron_queue；队列线程每 0.2 秒看一次队列，并尝试拿 agent_lock；主线程读取用户输入，也要拿同一把锁；拿到锁的一方才能运行 agent_loop，处理 [Scheduled] run date 这样的消息"><img src="/notes-assets/lcc-s12-cron-scheduler/01-three-threads.jpg" alt="调度线程每秒检查一次到期任务，放进 cron_queue；队列线程每 0.2 秒看一次队列，并尝试拿 agent_lock；主线程读取用户输入，也要拿同一把锁；拿到锁的一方才能运行 agent_loop，处理 [Scheduled] run date 这样的消息" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

程序启动后一共有三条线程在跑：

| 线程 | 做什么 | 多久一次 |
|---|---|---|
| `cron-scheduler` | `poll_due_jobs(datetime.now())`，到期的任务放进 `cron_queue` | 每 1 秒 |
| `cron-queue-processor` | 队列不空，就**试着**拿 `agent_lock`，拿到了跑一轮 Agent | 每 0.2 秒 |
| 主线程 | `input()` 等用户输入，然后 `with agent_lock:` 跑一轮 Agent | 用户输入时 |

关键在 `agent_lock`。用户的提问和定时任务都要往**同一个** `session_history` 里追加消息，同一时刻只能有一方在跑。

队列线程用的是 `agent_lock.acquire(blocking=False)`：拿不到锁就直接放弃，0.2 秒后再试，不会一直卡着。主线程用的是 `with agent_lock:`，会**一直等**到锁释放为止。

另外，这两条线程都是 `daemon=True`，而且只在 `__main__` 里启动。`import code` 只会定义函数，不会在后台偷偷跑起一个调度器。

### 第二段：定时回合里不能弹权限确认

```python
def request_permission(block, reason: str) -> str | None:
    if threading.current_thread() is not threading.main_thread():
        return "Permission denied: scheduled turns cannot request interactive approval"
    ...
    choice = input("   Allow? [y/N] ").strip().lower()
```

定时触发的那一轮跑在队列线程里。这时主线程很可能正卡在 `input("s12 >> ")` 上等用户输入，两边同时读终端，用户敲的字不知道会进到哪一边去。（s10 的真实运行里，就出现过权限确认把下一句问题当成回答的情况。）

所以这里直接规定：不在主线程，就不问用户，一律拒绝。需要确认的危险操作，定时任务做不了。

### 第三段：cron 表达式怎么匹配

```python
def cron_matches(cron_expr: str, moment: datetime) -> bool:
    ...
    minute, hour, day, month, weekday = fields
    cron_weekday = (moment.weekday() + 1) % 7
    ...
    if day == "*" and weekday == "*":
        return True
    if day == "*":
        return weekday_matches
    if weekday == "*":
        return day_matches
    return day_matches or weekday_matches
```

两个容易错的地方：

1. **星期几的编号不一样。** Python 的 `weekday()` 星期一是 0，cron 是星期日为 0。`(weekday() + 1) % 7` 就是在两者之间换算。
2. **「日」和「星期」同时指定时，是「或」。** 比如 `0 9 1 * 1` 的意思是「每月 1 号，**或者**每个星期一的 9 点」，不是「恰好是星期一的 1 号」。这是标准 cron 的规则，代码照着实现了。

`validate_cron` 在保存任务**之前**检查每个字段的取值范围：分钟 0–59、小时 0–23、日 1–31、月 1–12、星期 0–6。模型给了一个 `61 * * * *`，在 `schedule_cron` 这一步就会报错，而不是保存下来、永远不触发。

### 第四段：改状态之前先存盘，存盘失败就回滚

```python
def _enqueue_due_job(job: CronJob, minute_marker: str | None = None):
    old_pending = job.pending_delivery
    old_last_fired = job.last_fired
    job.pending_delivery = True
    if minute_marker is not None:
        job.last_fired = minute_marker
    try:
        if job.durable:
            save_durable_jobs()
    except Exception:
        job.pending_delivery = old_pending
        job.last_fired = old_last_fired
        raise
    cron_queue.append(job)
```

这个模式在 s12 里出现了很多次（`schedule_job`、`cancel_job`、`acknowledge_cron_jobs` 都是）：

1. 记下旧值；
2. 改内存里的状态；
3. 写盘；
4. 写盘失败，把内存改回旧值，再把异常抛出去。

这样内存和磁盘始终是一致的。`cron_queue.append(job)` 放在最后，只有存盘成功了，任务才会进队列。

`save_durable_jobs` 本身也很小心：先写到一个临时文件，再用 `os.replace` 换过去。`os.replace` 是原子操作，就算写到一半程序崩了，`.scheduled_tasks.json` 要么是旧的完整内容，要么是新的完整内容，不会是半个 JSON。

### 第五段：交付和确认分成两步

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s12-cron-scheduler/03-at-least-once.jpg" target="_blank" rel="noopener" aria-label="查看原图：任务到期后先把 pending_delivery 设为 true 并存盘，进入 cron_queue，再作为 [Scheduled] 消息追加，然后调用模型；成功就确认，周期任务清掉 pending_delivery，一次性任务直接删除；出错就删掉这些消息、放回队列；如果在调用模型和确认之间崩溃，重启后会再交付一次"><img src="/notes-assets/lcc-s12-cron-scheduler/03-at-least-once.jpg" alt="任务到期后先把 pending_delivery 设为 true 并存盘，进入 cron_queue，再作为 [Scheduled] 消息追加，然后调用模型；成功就确认，周期任务清掉 pending_delivery，一次性任务直接删除；出错就删掉这些消息、放回队列；如果在调用模型和确认之间崩溃，重启后会再交付一次" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

```python
def agent_loop(messages: list, context: dict | None = None):
    fired = consume_cron_queue()
    scheduled_start = len(messages)
    for job in fired:
        messages.append({"role": "user", "content": f"[Scheduled] {job.prompt}"})
    waiting_for_ack = list(fired)
    while True:
        try:
            response = client.messages.create(...)
        except Exception as error:
            if waiting_for_ack:
                del messages[scheduled_start:]
                restore_cron_jobs(waiting_for_ack)
            ...
            return context
        messages.append({"role": "assistant", "content": response.content})
        if waiting_for_ack:
            acknowledge_cron_jobs(waiting_for_ack)
            waiting_for_ack = []
```

「交给模型」和「确认已交付」是分开的两步。模型调用失败，就把刚加进去的 `[Scheduled]` 消息删掉，任务放回队列，下次再试。只有模型**成功回复**之后，才算交付完成。

如果程序恰好在「模型回复了」和「确认写盘」之间崩溃，重启后 `pending_delivery` 还是 `true`，任务会被**再交付一次**。原文说这是「至少一次」交付：宁可重复，也不要漏掉。所以定时任务的 prompt 最好设计成**执行两次也没关系**的，这就是 s05 里讲过的幂等。

## 实际跑一次（本站补充）

在空目录里启动程序，注册一个每分钟执行一次的持久任务，然后让进程一直开着。每一行前面是真实时间：

```
20:23:35 s12 >> [HOOK] UserPromptSubmit: ...
20:23:39 [HOOK] schedule_cron(['* * * * *', True])
20:23:39   [cron] scheduled cron_d09be83e: * * * * * -> run date
20:23:40   [cron] due cron_d09be83e: run date
20:23:41 [HOOK] Stop: session used 1 tool calls
20:23:41 Scheduled **“run date”** to run every minute. The job is durable and will persist after restart.
20:23:42 s12 >>   [cron] delivered cron_d09be83e: run date
20:23:45 [HOOK] bash(['date'])
20:23:49 Fri Oct 30 20:23:45 CST 2026
20:24:00   [cron] due cron_d09be83e: run date
20:24:00   [cron] delivered cron_d09be83e: run date
20:24:02 [HOOK] bash(['date'])
20:24:05 Fri Oct 30 20:24:02 CST 2026
20:25:00   [cron] due cron_d09be83e: run date
20:25:01   [cron] delivered cron_d09be83e: run date
20:25:10 [HOOK] bash(['date'])
20:25:19 Fri Oct 30 20:25:10 CST 2026
20:26:00   [cron] due cron_d09be83e: run date
20:26:00   [cron] delivered cron_d09be83e: run date
20:26:02 [HOOK] bash(['date'])
20:26:05 Fri Oct 30 20:26:02 CST 2026
20:26:05 [HOOK] UserPromptSubmit: ...
20:26:08 [HOOK] list_crons([])
20:26:13 - **Job ID:** `cron_d09be83e`
……
20:26:27 [HOOK] cancel_cron(['cron_d09be83e'])
20:26:27   [cron] cancelled cron_d09be83e
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s12-cron-scheduler/02-real-timeline.jpg" target="_blank" rel="noopener" aria-label="查看原图：真实运行的时间线：20:23:39 用户这一轮创建任务，20:23:40 同一分钟内就到期，等用户这一轮结束后在 20:23:42 交付；20:24、20:25、20:26 整点各触发一次 bash date；20:26:05 用户的 List 请求要等定时回合释放锁；整个过程中 session_history 从 1 条涨到 27 条"><img src="/notes-assets/lcc-s12-cron-scheduler/02-real-timeline.jpg" alt="真实运行的时间线：20:23:39 用户这一轮创建任务，20:23:40 同一分钟内就到期，等用户这一轮结束后在 20:23:42 交付；20:24、20:25、20:26 整点各触发一次 bash date；20:26:05 用户的 List 请求要等定时回合释放锁；整个过程中 session_history 从 1 条涨到 27 条" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

有几个地方值得细看：

**1. 刚创建就触发了一次。**任务在 20:23:39 创建，20:23:40 调度线程就发现它到期了。`* * * * *` 匹配**当前这一分钟**，而这个任务还从来没有触发过（`last_fired` 为空）。如果你在 9 点 0 分 30 秒创建一个 `0 9 * * *` 的任务，它也会马上执行一次。

**2. 到期了，但要等锁。**20:23:40 到期时，用户创建任务的那一轮还没结束，锁在主线程手里。队列线程拿不到锁，每 0.2 秒试一次，直到 20:23:42 用户那一轮结束，才把 `[Scheduled] run date` 交给模型。

**3. 用户也要排队。**20:26:00 那次定时任务还在跑的时候，我输入了「List all cron jobs.」。主线程在 `with agent_lock:` 上等着，20:26:05 定时回合结束后，才开始处理我的问题。

**4. 定时回合和用户共用一段对话。**代理日志里，每次请求的消息数从 1 一路涨到 27。每触发一次，`session_history` 里就多 4 条消息：`[Scheduled]` 提示、`tool_use`、`tool_result` 和回复。`Stop` 日志里的「session used N tool calls」也是一路累加的。这个任务如果每分钟跑一次、跑上一整天，就会往对话里塞进 5000 多条消息。

`.scheduled_tasks.json` 在第一次触发之后是这样的：

```json
[
  {
    "id": "cron_d09be83e",
    "cron": "* * * * *",
    "prompt": "run date",
    "recurring": true,
    "durable": true,
    "pending_delivery": false,
    "last_fired": "2026-10-30 20:23"
  }
]
```

取消之后，文件变成了 `[]`。

## 想一想（本站补充）

1. 这次运行里，任务一创建就触发了一次。如果你不希望这样，应该在哪里改？改成「从下一个匹配的时间点开始」，需要记录什么？
2. 每次定时触发都往 `session_history` 里追加消息。一个每 5 分钟跑一次的任务，一天会让上下文涨多少？你会怎么处理？（可以想想 s06 的子 Agent 和 s08 的压缩。）
3. 定时回合不能弹权限确认，危险操作会被直接拒绝。如果一个定时任务确实需要删除旧日志，有什么办法既能自动执行，又不完全放开权限？
4. 「至少一次」交付意味着同一个任务可能被执行两次。`run date` 执行两次没关系，什么样的定时任务执行两次会出问题？怎么把它改成可以安全重复执行的？
5. 程序关掉 3 个小时再打开，这 3 小时里错过的触发会补跑吗？什么场景需要补跑，什么场景绝对不能补跑？
6. 用户正在和 Agent 做一个很长的任务（几十轮工具调用），这时一个定时任务到期了。它要等多久？有没有更好的做法？

## 试一下

```sh
cd learn-claude-code
python s12_cron_scheduler/code.py
```

可以依次输入：

1. `Schedule "run date" every 2 minutes and keep it after restart.`
2. `List all cron jobs.`
3. `Cancel the cron job you just created.`

运行时可以查看 `.scheduled_tasks.json`，并观察到期后出现的 `[Scheduled] run date` 消息。测试一分钟级任务时，Agent 进程需要保持运行。

## 接下来

调度器可以在指定时间启动一轮 Agent Loop，但这一轮仍由一个 Agent 处理。面对需要同时调查多个模块、并行修改并汇总结果的任务，Harness 还需要把工作分给多个 Agent，并收集各自的执行结果。

[s13 Agent Teams](/notes/lcc-s13-agent-teams/) → Lead 分配任务，队友独立执行，再通过收件箱返回结果。
