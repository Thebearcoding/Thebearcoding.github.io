---
title: Agent 执行一条命令，背后发生了什么？子进程、超时与进程组
date: '2026-10-27'
tags: [Agent, Harness, 子进程, 进程组, 信号, 操作系统]
summary: 手写 Agent 的 run_command 工具。从最朴素的 subprocess.run 出发，亲手跑出两个真实的坑：超时后孙进程变成孤儿继续运行，以及管道被孙进程占着导致 harness 卡死。再用进程组和「先 SIGTERM 后 SIGKILL」修好它，最后接进一个最小的 Agent 循环。
draft: false
---

你对 Agent 说「帮我跑一下测试」，模型回了一行：

```json
{"tool": "run_command", "cmd": "pytest -x"}
```

然后呢？模型只会生成文字，它从来不会真的去执行任何东西。**真正去跑这条命令、等它结束、把结果塞回给模型的，是 harness，也就是我们自己写的代码。**

这一课就来写这个 `run_command` 工具。它看起来只是「调一下 subprocess」，但我会带你亲手跑出两个真实的坑。这两个坑都和操作系统里的**进程**有关，搞懂它们，面试里「孤儿进程」「信号」「管道」这几道八股也就顺带会了。

本课代码在 Python 3.9 和 macOS 上测过，Linux 上行为相同（Windows 的进程模型不同，这里不讨论）。专题入口：[学习路线](/notes/agent-eng-roadmap/)。

## 1. 先看全貌：一条命令走过的路

<figure style="margin:1.5rem 0">
<a href="/notes-assets/agent-eng-subprocess/01-one-tool-call.jpg" target="_blank" rel="noopener" aria-label="查看原图：一次工具调用：模型输出 JSON，harness 解析后启动 sh -c，sh 再启动 pytest；输出和退出码经管道流回 harness，截断后作为工具结果交还给模型"><img src="/notes-assets/agent-eng-subprocess/01-one-tool-call.jpg" alt="一次工具调用：模型输出 JSON，harness 解析后启动 sh -c，sh 再启动 pytest；输出和退出码经管道流回 harness，截断后作为工具结果交还给模型" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

注意最右边的「孙进程」。harness 启动的通常是一个 shell（`sh -c "..."`），shell 再去启动真正的命令。命令自己可能还会再开进程，比如 `pytest -n 4` 会开 4 个 worker。**一条命令往往不是一个进程，而是一棵进程树。** 后面两个坑都出在这棵树上。

## 2. 第一版：能跑，但会卡死

最朴素的写法，一行就够：

```python
import subprocess

def run_command(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout + r.stderr
```

问题是**它会无限等下去**。模型完全可能写出这些命令：

- `python3`：进入交互模式，一直等你输入；
- `npm run dev`：启动开发服务器，本来就不会自己退出；
- `pytest`：某个测试里有死循环。

这时整个 Agent 就卡住了，用户只能看着它发呆。所以第一个要求是：**必须有超时。**

## 3. 加上超时：坑一，孙进程变成孤儿

`subprocess.run` 本来就支持 `timeout` 参数，超时后会把子进程杀掉，然后抛出 `TimeoutExpired`。看起来问题解决了，我们来验证一下。用 `sleep 33 & sleep 33` 模拟「命令又开了别的进程」：

```python
import subprocess

try:
    subprocess.run("sleep 33 & sleep 33", shell=True,
                   capture_output=True, timeout=1)
except subprocess.TimeoutExpired:
    print("run() raised TimeoutExpired")
```

跑完以后，在终端里查一下还有没有 `sleep 33`：

```text
$ python3 demo.py
run() raised TimeoutExpired
$ pgrep -fl "sleep 33"
87733 sleep 33
87734 sleep 33
```

**超时报了，可两个 `sleep` 都还活着。** 为什么？

`subprocess.run` 杀掉的只是它直接启动的那个进程，也就是 shell。shell 启动的两个 `sleep` 是孙进程，没人管它们。父进程死了，它们就变成了**孤儿进程**，被系统的 1 号进程（Linux 上是 init 或 systemd，macOS 上是 launchd）收养，然后继续运行。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/agent-eng-subprocess/02-orphan-vs-group.jpg" target="_blank" rel="noopener" aria-label="查看原图：左：proc.kill() 只杀掉 sh -c，两个 sleep 成了孤儿继续运行；右：os.killpg() 杀掉整个进程组，sh -c 和两个 sleep 一起结束"><img src="/notes-assets/agent-eng-subprocess/02-orphan-vs-group.jpg" alt="左：proc.kill() 只杀掉 sh -c，两个 sleep 成了孤儿继续运行；右：os.killpg() 杀掉整个进程组，sh -c 和两个 sleep 一起结束" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

放到 Agent 场景里，后果很具体：模型启动了一个开发服务器，超时后「被杀掉」了，其实服务器还占着端口。模型下一步再启动，就报「端口已被占用」。它不知道原因，于是开始乱改代码。

## 4. 坑二：管道被占着，harness 自己卡死

`subprocess.run` 在超时后只等子进程退出，所以它自己不会卡住，只是留下了孤儿。可我们写 harness 时通常会用更灵活的 `Popen`，在超时后还想把**已经产生的输出**读出来给模型看。这时就会掉进第二个坑：

```python
import subprocess, time

proc = subprocess.Popen("sleep 32 & sleep 32", shell=True,
                        stdout=subprocess.PIPE, text=True)
try:
    proc.communicate(timeout=1)
except subprocess.TimeoutExpired:
    proc.kill()
    t = time.monotonic()
    proc.communicate()   # 想把剩下的输出读完
    print("communicate returned after", round(time.monotonic() - t, 1), "s")
```

我实测的现象是：**第二次 `communicate()` 一直不返回**，4 秒后它还卡着。直到我手动 `pkill` 掉两个 `sleep`，它才打印出 `communicate returned after 3.0 s`。如果我不去杀，它会一直等到两个 `sleep` 自己结束，也就是 30 多秒后。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/agent-eng-subprocess/03-pipe-eof.jpg" target="_blank" rel="noopener" aria-label="查看原图：管道示意：harness 在读端等待 EOF；写端分出三根管子，sh -c 已被杀，两个 sleep 仍拿着写端，所以 EOF 迟迟不来"><img src="/notes-assets/agent-eng-subprocess/03-pipe-eof.jpg" alt="管道示意：harness 在读端等待 EOF；写端分出三根管子，sh -c 已被杀，两个 sleep 仍拿着写端，所以 EOF 迟迟不来" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

原因在**管道**。`stdout=subprocess.PIPE` 会创建一根管道：子进程往写端写，我们从读端读。`communicate()` 要一直读到「文件结束」（EOF）才返回，而 **EOF 只有在所有拿着写端的进程都关掉它之后才会出现**。孙进程是从 shell 那里继承了这个写端的，shell 死了，孙进程还拿着，所以读的一方就永远等不到 EOF。

> 一句话总结：**只杀子进程，孙进程会变成孤儿；孙进程拿着管道，你的读取就会卡死。** 两个坑的根源一样，都是只处理了进程树的一个节点。

## 5. 修复：进程组 + 先礼后兵

解决办法是**把整棵树放进同一个进程组，超时就杀整个组**。

**进程组是什么？** 操作系统可以把一组进程编成一个组，给它们发信号时一次全部发到。在终端按 Ctrl+C 能同时停掉一条管道命令里的所有进程（比如 `cat log | grep err | sort`），靠的就是进程组。

在 Python 里，`Popen(..., start_new_session=True)` 会让子进程调用 `setsid()`，开一个新的会话和新的进程组，组号就等于子进程的 pid。之后它启动的孙进程默认都在这个组里，`os.killpg(proc.pid, sig)` 就能一次把整个组杀掉。

**杀的时候还要讲个顺序：先发 SIGTERM，等一会儿，还不退出再发 SIGKILL。** 这几个信号的区别是面试常考点：

| 信号 | 编号 | 含义 | 程序能拦截吗 |
|---|---|---|---|
| SIGINT | 2 | 中断，终端里按 Ctrl+C 就是它 | 能 |
| SIGTERM | 15 | 请你退出，`kill` 命令默认发的就是它 | 能，可以先收尾再退出 |
| SIGKILL | 9 | 立刻结束 | **不能**，由内核直接执行 |

先发 SIGTERM，是给程序一个收尾的机会，比如把写了一半的文件存完、删掉临时文件。我实测了一个会拦截 SIGTERM 的命令：

```text
>>> run_command('trap "echo got TERM; exit 3" TERM; echo ready; while true; do sleep 0.1; done', timeout=1)
{'exit_code': 3, 'timed_out': True, 'output': 'ready\nTerminated: 15\ngot TERM\n', 'seconds': 1.0}
```

它收到 SIGTERM 后打印了 `got TERM`，然后以退出码 3 自己退出了。如果一上来就发 SIGKILL，它连这句话都来不及说。

## 6. 完整代码

现在把超时、进程组、输出处理拼起来。建议你**逐行手敲**，存成 `run_command.py`：

```python title="run_command.py"
import os
import signal
import subprocess
import time

MAX_OUTPUT = 2000  # 返回给模型的最多字符数


def truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    """保留开头和结尾，中间用一行说明代替。"""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - limit
    return text[:half] + f"\n...[省略 {dropped} 个字符]...\n" + text[-half:]


def run_command(cmd: str, timeout: float = 10.0) -> dict:
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # 把 stderr 合并进 stdout，模型能看到报错
        text=True,
        start_new_session=True,     # 新开一个进程组，超时时能整组杀掉
    )
    timed_out = False
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)       # 先礼貌地请它退出
        try:
            output, _ = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)   # 不听话就强制结束
            output, _ = proc.communicate()
    return {
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "output": truncate(output),
        "seconds": round(time.monotonic() - start, 2),
    }
```

除了进程组，还有四个设计值得注意，它们都是为了**让模型看得懂结果**：

1. **stderr 合并进 stdout。** 报错信息通常在 stderr。模型看不到报错，就不知道该怎么修。
2. **返回 `exit_code`。** 很多命令失败时什么也不打印，只靠退出码表示失败。另外，退出码为负数表示进程是被信号杀死的：`-15` 是被 SIGTERM，`-9` 是被 SIGKILL。
3. **返回 `timed_out`。** 「超时被杀」和「正常跑完但没有输出」，对模型来说是完全不同的两件事，必须明确告诉它。
4. **截断时保留头尾。** 输出太长会挤爆上下文窗口（第 4 课会细讲）。而报错和测试总结通常在**最后**，所以截断时要保留结尾，不能只保留开头。

跑一下看看：

```python
print(run_command("echo hello && ls /nope"))
print(run_command("echo start; sleep 30", timeout=1))
print(run_command("sleep 30 & sleep 30", timeout=1))
```

```text
{'exit_code': 1, 'timed_out': False, 'output': 'hello\nls: /nope: No such file or directory\n', 'seconds': 0.01}
{'exit_code': -15, 'timed_out': True, 'output': 'start\n', 'seconds': 1.0}
{'exit_code': -15, 'timed_out': True, 'output': '', 'seconds': 1.0}
```

第二行里，超时前已经输出的 `start` 被保留了下来。跑完第三行后，再用 `pgrep -fl "sleep 30"` 查，**一个都不剩**。

> **诚实地说一句：** 进程组并不是牢笼。命令自己也可以调用 `setsid` 逃出这个组，比如有些守护进程就会这么做。真正可靠的隔离要靠容器和 cgroup，那是第 8 课的内容。

## 7. 接进一个最小的 Agent 循环

工具写好了，把它接进循环。这里先用一个**脚本化的假模型**，它按顺序吐出预先写好的回复，这样不用 API key，也不花钱，就能把循环调通。存成 `mini_loop.py`：

```python title="mini_loop.py"
import json

from run_command import run_command


class ScriptedLM:
    """假模型：按顺序吐出预先写好的回复，用来在没有 API key 时调试循环。"""

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages):
        return self.replies.pop(0)


SYSTEM = """你可以调用工具 run_command 执行 shell 命令。
需要调用时，只输出一行 JSON：{"tool": "run_command", "cmd": "..."}
得到答案后，输出：{"final": "..."}"""


def agent(lm, task, max_steps=5):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    for step in range(max_steps):
        reply = lm.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        try:
            action = json.loads(reply)
        except json.JSONDecodeError:
            messages.append({"role": "user", "content": "格式错误：请只输出一行 JSON。"})
            continue
        if "final" in action:
            return action["final"]
        result = run_command(action["cmd"])
        print(f"[step {step}] $ {action['cmd']} -> exit {result['exit_code']}")
        messages.append({"role": "user", "content": "工具结果：" + json.dumps(result, ensure_ascii=False)})
    return "达到最大步数，任务未完成"


if __name__ == "__main__":
    lm = ScriptedLM([
        '{"tool": "run_command", "cmd": "ls *.py"}',
        '我先看看文件',  # 故意写错格式，看循环怎么处理
        '{"tool": "run_command", "cmd": "wc -l run_command.py"}',
        '{"final": "看完了，答案在上面两次工具结果里"}',
    ])
    print(agent(lm, "这个目录里有哪些 Python 文件？run_command.py 有多少行？"))
```

```text
[step 0] $ ls *.py -> exit 0
[step 2] $ wc -l run_command.py -> exit 0
看完了，答案在上面两次工具结果里
```

第 1 步模型「说错了话」，没有输出 JSON。循环没有崩，而是把「格式错误」作为一条消息告诉它，让它重来。这种**把错误变成观察结果**的做法，是 harness 的基本功，第 2 课会专门讲。

你可能注意到了：假模型的最终回答是写死的，它**根本没看**工具结果。这正是假模型的边界：它能测「循环的机制对不对」，比如解析、执行、追加消息、步数上限；但测不了「模型会不会用工具」。后者要接真模型才能测。CS329Z 的作业就是这样把测试分成两类的：不花钱的确定性测试，和调用真模型的 live 测试。

## 8. 动手

不做完这三个练习，不算学会：

1. **加 `cwd` 参数**，让命令能在指定目录下执行。想一想：如果模型传了一个不存在的目录，应该返回什么给它？
2. **拦截危险命令。** 在执行之前检查，遇到 `rm -rf /` 这类命令就直接拒绝，并告诉模型被拒绝的原因。再想想，为什么光靠字符串匹配是拦不住的？（提示：`rm -r -f /`、`eval`、base64 编码……）这就是第 8 课要上沙箱的原因。
3. **把 `ScriptedLM` 换成真模型**，任意一个兼容 OpenAI 接口的 API 都行，让它真的回答「这个目录里有哪些 Python 文件」。观察它在格式出错时会不会自己改正。

## 9. 面试追问

**Q：孤儿进程和僵尸进程有什么区别？**
孤儿进程是**父进程先死了**，子进程还活着，于是被 1 号进程收养，继续正常运行，本身无害。我们第 3 节遇到的就是孤儿：它没害处，只是不该继续跑。僵尸进程是**子进程已经结束了，但父进程还没调用 `wait()` 读取它的退出状态**，于是它在进程表里留下一个条目。它不占 CPU 和内存，但会占用 pid，积累多了就分配不出新的 pid。`Popen` 的 `communicate()` 和 `wait()` 都会回收子进程，所以我们的代码不会留下僵尸。

**Q：为什么先发 SIGTERM，再发 SIGKILL？**
SIGTERM 可以被程序拦截，给它一个收尾的机会，比如保存文件、删除临时文件、释放锁。SIGKILL 不能被拦截，由内核直接结束进程，程序来不及做任何清理。所以先礼后兵：先发 SIGTERM，等两秒还没退出，再发 SIGKILL。

**Q：`shell=True` 不是有命令注入风险吗？为什么还用？**
在普通 Web 后端里，把用户输入拼进 `shell=True` 的命令确实是注入漏洞。但在 Agent 里，命令本来就是模型写出来、并且打算完整执行的，管道、`&&`、通配符这些 shell 特性它都需要。所以这里真正的风险不是「注入」，而是「**模型本身就可能写出危险命令**」。防线不在 `shell=True` 这个参数上，而在权限确认和沙箱上。

**Q：管道为什么会卡住？**
读的一方要等到 EOF 才会结束，而 EOF 要等**所有**持有写端的进程都关闭它之后才会出现。子进程启动的孙进程会继承这个写端，所以只杀子进程，写端还开着，读的一方就会一直等下去。

## 10. 这一课对应的八股

- **操作系统**：进程与进程树、进程组与会话、信号（SIGINT、SIGTERM、SIGKILL）、孤儿进程与僵尸进程、管道与文件描述符的继承。
- **CS329Z 作业 1**：终端工具（terminal）和沙箱（sandbox）相关的题目。
- **下一课**：工具协议。模型说的话，怎样稳定地被解析成「调用哪个工具、传什么参数」？
