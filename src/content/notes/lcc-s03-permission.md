---
title: 'learn-claude-code s03：Permission，执行前做权限判断'
date: '2026-10-30'
tags: [Agent, Harness, Permission, learn-claude-code]
summary: 转载 learn-claude-code 第 3 课，补充配图、code.py 逐段导读和三次真实运行。工具执行前插入三道闸门：硬拒绝、规则匹配、用户审批；也亲手跑出了它的漏洞：规则只认字符串，换个写法就能绕过去。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s03_permission)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> *"工具执行前先做权限判断"* — 权限管线决定哪些操作需要审批。
>
> **Harness 层**: 权限 — 在工具执行前加一道门。

## 问题

s02 的 Agent 有 5 个工具。file tools 受 `safe_path` 保护，但 bash 不受限制。让它"清理一下项目"，可能执行 `rm -rf /`。

安全边界由代码负责，判断发生在工具执行之前。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s03-permission/01-three-gates.jpg" target="_blank" rel="noopener" aria-label="查看原图：tool_use 依次经过 deny list（只查 bash）、rules、ask user 三道闸门；命中 deny list 直接 blocked；规则命中后问用户，y 执行，N 返回 Permission denied.；规则没命中直接执行；三种结果最后都变成 tool_result"><img src="/notes-assets/lcc-s03-permission/01-three-gates.jpg" alt="tool_use 依次经过 deny list（只查 bash）、rules、ask user 三道闸门；命中 deny list 直接 blocked；规则命中后问用户，y 执行，N 返回 Permission denied.；规则没命中直接执行；三种结果最后都变成 tool_result" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

s02 的循环完全保留。唯一的变动是在工具执行前插入 `check_permission()`。每个工具调用依次经过三道闸门：硬拒绝优先，软询问次之，都没命中就放行。

三道闸门对应三种决策：

| 闸门 | 作用 | 命中后 |
|------|------|--------|
| 1. 拒绝列表 | 永远禁止的操作（`rm -rf /`、`sudo`） | 直接拒绝，不执行 |
| 2. 规则匹配 | 取决于上下文的操作（读/写工作区外、`rm` 文件） | 交给闸门 3 |
| 3. 用户审批 | 闸门 2 命中后，暂停等用户确认 | 用户决定允许或拒绝 |

三道都没命中 → 直接执行。大部分日常操作走这条路。

## 工作原理

**闸门 1**：一张硬拒绝表，先查，命中就返回阻止信息。这张表使用简单字符串匹配来说明权限闸门的位置，不能视为完整的安全边界。

```python
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None
```

**闸门 2**负责规则匹配，用来描述"什么时候需要问用户"。每条规则指定工具和检查条件。

```python
import re

DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)

def contains_destructive_command(command: str) -> bool:
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))

PERMISSION_RULES = [
    {
        "tools": ["read_file", "write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Access outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: contains_destructive_command(args.get("command", "")) or any(
            kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]
        ),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None
```

**闸门 3**：规则命中后，暂停等用户输入。

```python
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"
```

**三道闸门串在一起**，插在工具执行之前：

```python
def check_permission(block) -> bool:
    # 闸门 1: 硬拒绝
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    # 闸门 2 + 3: 规则匹配 → 用户审批
    reason = check_rules(block.name, block.input)
    if reason:
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False

    return True

# 在 agent_loop 中——s02 的循环只加了一行：
for block in tool_calls:
    if not check_permission(block):           # ← 新增
        results.append({... "content": "Permission denied."})
        continue
    output = TOOL_HANDLERS[block.name](**block.input)  # s02 原有
    results.append(...)
```

## 相对 s02 的变更

| 组件 | 之前 (s02) | 之后 (s03) |
|------|-----------|-----------|
| 安全模型 | 无（信任模型） | 三道闸门权限管线 |
| 新函数 | — | check_deny_list, check_rules, ask_user, check_permission |
| 循环 | 直接执行所有工具 | 执行前插入 check_permission() |

## 代码导读：把 code.py 从头读一遍（本站补充）

完整的 [`s03_permission/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s03_permission/code.py) 有 271 行。跟 s02 对比着读，除了三道闸门，还有三处容易漏看的改动。

### 改动一：system prompt 让模型别自己拦

```python
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
    "Use tools to solve tasks. Do not ask for approval yourself. "
    "For destructive operations, still request the tool call; "
    "the harness will approve or deny before execution."
)
```

s02 的最后一句是 "Act, don't explain."，s03 换成了一段话，意思是：**危险操作你照样发起调用，批不批由 harness 决定。**

为什么要专门写这句？因为模型自己也有安全意识。没有这句，它遇到 `rm` 可能会在聊天里问「你确定要删吗？」，然后不调用工具就结束了。这样权限检查就发生在模型的文字里，harness 管不着，也记不下来。这句话把「谁来做决定」划清楚了：**模型负责提出动作，harness 负责审批。**

### 改动二：文件工具不再直接拦截

s02 的 `run_read`、`run_write`、`run_edit` 都先过 `safe_path`，跑出工作区就报错。s03 把这一步删了，换成直接 `resolve()`：

```python
# s02
lines = safe_path(path).read_text(encoding="utf-8").splitlines()
# s03
lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()
```

工作区外的路径现在交给闸门 2 的第一条规则处理：**不是直接拒绝，而是问你。** 这个改动是合理的，因为 agent 确实会有正当理由读工作区外的文件，比如 `~/.gitconfig`。一刀切地拒绝，反而让它没法干活。

只有 `run_glob` 还保留着过滤，匹配结果里跑出工作区的仍然会被丢掉。

顺便留意一个小细节：这条规则的提示文字写的是 `"Writing outside workspace"`，但它对 `read_file` 也生效。读文件时你看到的提示也是「写」，这是提示文字不准确，逻辑本身没问题。

### 改动三：被拒绝的调用也要有回复

```python
for block in tool_calls:
    print(f"\033[36m> {block.name}\033[0m")
    if not check_permission(block):
        results.append({"type": "tool_result", "tool_use_id": block.id,
                        "content": "Permission denied."})
        continue
    ...
```

注意被拒绝时并不是直接 `continue` 跳过，而是**先补一条 `tool_result`，再 `continue`**。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s03-permission/03-every-call-answered.jpg" target="_blank" rel="noopener" aria-label="查看原图：assistant 发出两个 tool_use，第一个 bash rm 被拒绝，第二个 read_file 正常执行；user 消息里两个 tool_result 都在，第一个内容是 Permission denied."><img src="/notes-assets/lcc-s03-permission/03-every-call-answered.jpg" alt="assistant 发出两个 tool_use，第一个 bash rm 被拒绝，第二个 read_file 正常执行；user 消息里两个 tool_result 都在，第一个内容是 Permission denied." width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

还记得 s01 那张 messages 图吗？每个 `tool_use` 的 `id` 都必须有一个 `tool_use_id` 对得上的 `tool_result`。少一条，下一次调用 API 就会直接报错。所以「拒绝」在协议层面也是一种回复，只不过内容是 `Permission denied.`。

### 两种规则写法

闸门 2 判断 bash 命令危不危险，用了两种办法，满足任意一种就算命中：

| 写法 | 能抓到 | 抓不到 |
|---|---|---|
| 正则 `DESTRUCTIVE_COMMAND_WORD` | 出现在命令开头，或跟在 `;` `&` `\|` `(` 换行后面的 `rm`、`del`，不分大小写 | 写在别处的 `rm`，比如 `xargs rm` |
| 子串 `"rm "` `"> /etc/"` `"chmod 777"` | 命令里任何位置出现这几段文字 | 不包含这几段文字的危险命令 |

两种都是**看命令的文字**，而不是看命令会做什么。正则比子串精确一点，比如 `model`、`delimiter` 这种单词里的 `del` 不会被误判。但只要危险操作不用 `rm` 这个词，两种都抓不到。下面的实际运行就会碰到这种情况。

### 闸门 3 会卡住整个程序

`ask_user` 里是一个 `input()`。程序会停在这里，什么都不做，直到你在终端里敲下回答。

在命令行里这样没问题。但如果 harness 是一个网页或 IDE 插件，就没有终端让你敲 `y`。这时审批要改成：harness 发一个「待审批」事件给界面，暂停这一次工具调用，等界面把结果传回来再继续。原理一样，实现会复杂很多。

## 实际跑一次（本站补充）

在一个只有 `test.txt` 的空目录里，跑了三次。

**第一次：** 提问 `Delete the file test.txt`，审批时回答 `n`。输入是从管道传进去的，所以 `n` 不会显示在 `Allow? [y/N]` 后面：

```
> bash

[permission] Potentially destructive command
   Tool: bash({'command': 'rm -f -- test.txt'})
   Allow? [y/N] Unable to delete `test.txt`: permission denied.
```

闸门 2 抓到了 `rm`，问我，我拒绝了。`test.txt` 还在。

**第二次：** 提问 `Run sudo ls /root`。

```
> bash

[blocked] Blocked: 'sudo' is on the deny list
Permission denied.
```

闸门 1 直接拦下，连问都没问。

**第三次：** 让模型用 `find . -name test.txt -delete` 删文件。

```
> bash
(no output)
```

没有任何提示，直接执行了。**`test.txt` 被删掉了。**

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s03-permission/02-rules-see-text.jpg" target="_blank" rel="noopener" aria-label="查看原图：三条命令的结果：rm -f test.txt 需要审批；sudo ls /root 被直接拦截；find . -name test.txt -delete 直接运行，文件被删除"><img src="/notes-assets/lcc-s03-permission/02-rules-see-text.jpg" alt="三条命令的结果：rm -f test.txt 需要审批；sudo ls /root 被直接拦截；find . -name test.txt -delete 直接运行，文件被删除" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

三次做的是同一类事，结果完全不同。规则检查的是命令的**文字**，`find -delete` 里没有 `rm`，也没有 `sudo`，所以三道闸门都放行了。这就是原文说的「不能视为完整的安全边界」。

## 想一想（本站补充）

1. 第三次运行里，怎样才能拦住 `find -delete`？如果你的办法是「把 `-delete` 也加进规则」，那下一个绕过的写法是什么？有没有不靠列举危险命令的办法？

   <details class="qa"><summary>参考答案</summary>

   把 `-delete` 加进规则，能拦住这一种写法，但删文件的办法太多了：`python3 -c "import os; os.remove('test.txt')"`、`perl -e 'unlink "test.txt"'`、`mv test.txt /tmp/`、`> test.txt` 清空内容……列举危险命令永远列不完。

   不靠列举的办法有两类：一是**白名单**，只允许已知安全的命令，其他都要确认；二是**操作系统沙箱**，让进程根本没有权限动不该动的文件。真实系统通常两者都用，再加上 git 或快照保证改错了能恢复。

   </details>

2. 被拒绝时，模型只看到 `Permission denied.`，不知道是被哪道闸门、因为什么拦下的。把原因也告诉模型，会有什么好处和坏处？

   <details class="qa"><summary>参考答案</summary>

   好处：模型知道是哪条规则拦的，能换一种合规的做法，或者如实告诉用户「这个操作需要你确认」，不会盲目重试。

   坏处：等于把规则的细节告诉了模型。它可能学会怎么绕开，比如知道是 `rm` 这个词被拦了，就换成 `find -delete`。所以一般会告诉模型「被拒绝了、为什么需要权限」，但不把匹配规则原样暴露出来。

   </details>

3. 如果模型被拒绝后，换了一种写法去做同一件事，这算不算在绕过权限？harness 能发现这种情况吗？

   <details class="qa"><summary>参考答案</summary>

   算。用户或规则拒绝的是**这件事**，不是这几个字符。换一种写法做同一件事，本质上就是绕过。

   harness 很难发现，因为它只能看到命令文本，看不出两条不同的命令其实是同一个意图。能做的有：在 system prompt 里明确要求「被拒绝后不要换方法做同一件事，先问用户」；记录被拒绝过的操作，类似操作一律要确认；最根本的还是靠沙箱，按实际效果而不是按文本来限制。

   </details>

4. 闸门 1 和闸门 2 的顺序能对调吗？对调之后，`sudo rm -rf /` 会发生什么？

   <details class="qa"><summary>参考答案</summary>

   不能随便对调。闸门 1 是「永远不允许」，闸门 2 是「可能允许，要问一下」。先查闸门 1，绝对禁止的事情连问都不会问。

   对调以后，`sudo rm -rf /` 会先命中闸门 2 的 `rm` 规则，弹出 `Allow? [y/N]`。如果用户一时手快按了 `y`（确认弹得多了，人就会习惯性地按 `y`，这叫审批疲劳），而代码在批准后直接放行，这条命令就真的执行了。原则是：**不要去问一个你无论如何都不会允许的问题。**

   </details>

## 试一下

```sh
cd learn-claude-code
python s03_permission/code.py
```

试试这些 prompt：

1. `Create a file called test.txt in the current directory`（应该直接通过）
2. `Delete the file test.txt`（bash + rm 会触发闸门 2）
3. `What files are in the current directory?`（只读，全部通过）
4. `Try to write a file to /etc/something`（写工作区外，触发闸门 2）
5. 在 Windows 上，`del test.txt` 和 `DEL test.txt` 会触发闸门 2，而 `model`、`delimiter` 和 `echo del test.txt` 不会。

观察重点：哪些操作直接通过？哪些需要你确认？哪些被直接拒绝？

## 接下来

当前权限检查每次都在循环里硬编码 `check_permission()`。如果我想在每次工具执行前后加日志？如果想在某些操作后自动触发 git commit？这些扩展逻辑散落在 loop 里，循环很快就会膨胀。

[s04 Hooks](/notes/lcc-s04-hooks/) → 给循环加钩子，扩展逻辑挂在钩子上，循环保持干净。
