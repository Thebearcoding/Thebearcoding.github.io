---
title: 'learn-claude-code s02：Tool Use，多加一个工具，只加一行'
date: '2026-10-29'
tags: [Agent, Harness, Tool Use, learn-claude-code]
summary: 转载 learn-claude-code 第 2 课，补充配图、code.py 逐段导读和一次真实运行。从 1 个 bash 扩展到 5 个工具，循环一行不改，只是把硬编码的 run_bash 换成按名字查表的 TOOL_HANDLERS。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s02_tool_use)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」和「实际跑一次」两节为本站补充。上一课：[s01 Agent Loop](/notes/lcc-s01-agent-loop/)。

> *"加一个工具, 只加一个 handler"* — 循环不用动, 新工具注册进 dispatch map 就行。
>
> **Harness 层**: 工具分发 — 扩展模型能触达的边界。

## 只有 bash 一个工具

s01 的 Agent 只有一个 bash 工具。读文件要 `cat`，写文件要 `echo "..." > file.py`，改文件要 `sed`。

模型想的是"读这个文件"，却要拼出 `cat path/to/file`。多了一层翻译，浪费 token，还容易拼错。

## Bash 已经能做这些事，为什么还要加工具？

这些工具没有给 Agent 增加新的原始能力——同样的操作 Bash 都能完成。它们改善的是模型与 Harness 之间的接口。调用带有 `path` 参数的 `read_file` 时，模型不必把意图翻译成 shell 语法，因此能减少引号和转义错误。

Harness 看到的也不再是一整条不透明的命令字符串，而是具名操作和结构化参数。这样后续策略可以更精确：s03 能按工具和路径决定放行或拒绝，s04 能在统一 hook 点记录操作。s02 先搭好这些章节依赖的分发结构。

## 全局视角：工具分发

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s02-tool-use/01-dispatch-map.jpg" target="_blank" rel="noopener" aria-label="查看原图：tool_use 块带着 name 和 input，按 name 在 TOOL_HANDLERS 表里查到 run_read，用 **input 展开参数调用，返回值作为 tool_result 交回；加一个工具只需一份 schema 加一行映射"><img src="/notes-assets/lcc-s02-tool-use/01-dispatch-map.jpg" alt="tool_use 块带着 name 和 input，按 name 在 TOOL_HANDLERS 表里查到 run_read，用 **input 展开参数调用，返回值作为 tool_result 交回；加一个工具只需一份 schema 加一行映射" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

s01 的循环完全保留（LLM 调用、`tool_use` block 判断、消息追加）。唯一的变动在工具执行那 1 行：`run_bash()` 替换为 `TOOL_HANDLERS[block.name]()` 查表分发。

给 Agent 加一个工具只需要做两件事：

1. **定义工具**：在 `TOOLS` 数组里加一条描述
2. **注册处理函数**：在 `TOOL_HANDLERS` 字典里加一个映射

## 从 1 个工具到 5 个工具

s01 只有一个 bash：

```python
TOOLS = [{"name": "bash", ...}]

def run_bash(command): ...
```

s02 加到 5 个，每个工具都是独立定义：

```python
TOOLS = [
    {"name": "bash",       "description": "Run a shell command.", ...},
    {"name": "read_file",  "description": "Read file contents.",  ...},
    {"name": "write_file", "description": "Write content to file.", ...},
    {"name": "edit_file",  "description": "Replace text in file once.", ...},
    {"name": "glob",       "description": "Find files by pattern.", ...},
]
```

每个工具有自己的实现函数：

```python
def run_read(path, limit=None):
    lines = safe_path(path).read_text(encoding="utf-8").splitlines()
    if limit:
        lines = lines[:limit]
    return "\n".join(lines)

def run_write(path, content):
    safe_path(path).write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} bytes to {path}"

def run_edit(path, old_text, new_text):
    text = safe_path(path).read_text(encoding="utf-8")
    if old_text not in text:
        return "Error: text not found"
    safe_path(path).write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
    return f"Edited {path}"

def run_glob(pattern):
    import glob as g
    matches = sorted(set(g.glob(
        pattern, root_dir=WORKDIR, recursive=True)))
    shown = matches[:200]
    if len(matches) > 200:
        shown.append("... (more matches omitted; narrow the pattern)")
    return "\n".join(shown)
```

## 工具分发

```python
TOOL_HANDLERS = {
    "bash":       run_bash,
    "read_file":  run_read,
    "write_file": run_write,
    "edit_file":  run_edit,
    "glob":       run_glob,
}

# 循环里只改了一行——从硬编码 run_bash 变成查表：
for block in tool_calls:
    handler = TOOL_HANDLERS[block.name]    # 查表
    output = handler(**block.input)         # 调用
    results.append(...)
```

加一个工具 = 在 `TOOLS` 数组加一条 + 在 `TOOL_HANDLERS` 字典加一行。循环不变。

## 多个工具调用

模型经常一次返回多个 tool_use："读一下 a.py 和 b.py，然后列出所有 .py 文件"。

这些调用按照 `response.content` 中的原始顺序逐个执行。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s02-tool-use/03-two-calls-one-reply.jpg" target="_blank" rel="noopener" aria-label="查看原图：一条 assistant 消息里有两个 tool_use，harness 按顺序依次执行，两个 tool_result 放进同一条 user 消息一起交回"><img src="/notes-assets/lcc-s02-tool-use/03-two-calls-one-reply.jpg" alt="一条 assistant 消息里有两个 tool_use，harness 按顺序依次执行，两个 tool_result 放进同一条 user 消息一起交回" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

## 速查

| 概念 | 一句话 |
|------|--------|
| TOOL_HANDLERS | 工具名 → 处理函数的字典。加工具 = 加一行映射 |
| 工具定义 | 告诉模型"我能做什么"的 JSON schema |
| 多工具调用 | 模型可一次返回多个 tool_use，并按原始顺序逐个执行 |
| 循环不变 | s01 的 `while True` 循环一行都没改 |

## 相对 s01 的变更

| 组件 | 之前 (s01) | 之后 (s02) |
|------|-----------|-----------|
| 工具数量 | 1 (bash) | 5 (+read, write, edit, glob) |
| 工具执行 | 硬编码 `run_bash()` | TOOL_HANDLERS 查表分发 |
| 路径安全 | 无 | safe_path 校验（仅 file tools） |
| 循环 | `while True` + `tool_use` block | 与 s01 完全一致 |

## 代码导读：把 code.py 从头读一遍（本站补充）

上面的代码片段为了好读做了简化。完整的 [`s02_tool_use/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s02_tool_use/code.py) 有 208 行，`run_bash` 和循环跟 s01 一样，这里只看新加的部分。

### 第一段：工作区的边界

```python
WORKDIR = Path.cwd()

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path
```

`WORKDIR` 在程序启动时就定死了，就是你运行程序时所在的目录。之后所有文件工具都只能碰这个目录里的东西。

`safe_path` 只有两步：

1. **`resolve()` 把路径算成真实的绝对路径。** `notes/../../etc/passwd` 里的 `..` 会被算掉，符号链接也会被追到它真正指向的地方。
2. **`is_relative_to` 检查结果还在不在 `WORKDIR` 里面。** 不在就报错。

还有一种情况容易漏掉：模型直接传一个绝对路径 `/etc/passwd`。在 `pathlib` 里，`WORKDIR / "/etc/passwd"` 的结果就是 `/etc/passwd`，前面的 `WORKDIR` 会被丢掉。不过它照样过不了第 2 步，所以也会被拦下。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s02-tool-use/02-safe-path.jpg" target="_blank" rel="noopener" aria-label="查看原图：WORKDIR 是一个虚线边界，read_file 读 notes/b.md 在边界内，放行；读 ../../etc/passwd 经 resolve 后跑出边界，报错；bash 不经过 safe_path，可以直接穿过边界"><img src="/notes-assets/lcc-s02-tool-use/02-safe-path.jpg" alt="WORKDIR 是一个虚线边界，read_file 读 notes/b.md 在边界内，放行；读 ../../etc/passwd 经 resolve 后跑出边界，报错；bash 不经过 safe_path，可以直接穿过边界" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

图右下角那个红框是这一课留下的洞：**`bash` 不经过 `safe_path`。** 模型想读 `/etc/passwd`，换成 `bash` 执行 `cat /etc/passwd` 就行。所以 `safe_path` 只能算「防手滑」，不是真正的安全边界，这正是下一课 s03 要解决的。

### 第二段：四个新工具，同一个约定

四个函数长得都一样：

```python
def run_xxx(...) -> str:
    try:
        ...                      # 真正干活
        return "成功时的说明"
    except Exception as e:
        return f"Error: {e}"
```

**不管成功还是失败，都返回一个字符串，不往外抛异常。** 这跟 s01 的 `run_bash` 是同一个约定：错误信息会作为 `tool_result` 交给模型，由模型决定下一步怎么办。

每个工具还各有一个小心思：

| 工具 | 细节 | 为什么 |
|---|---|---|
| `run_read` | 截断后追加 `... (N more lines)` | 让模型知道文件没读完，需要的话再读 |
| `run_write` | 先 `mkdir(parents=True)` | 模型常常直接写 `src/utils/a.py`，目录还不存在 |
| `run_edit` | `replace(old, new, 1)`，找不到就报错 | 只改一处；找不到时明确告诉模型，而不是假装成功 |
| `run_glob` | 过滤掉跑出 `WORKDIR` 的结果，最多 200 条 | `glob` 里的 `**` 和 `..` 也可能逃出工作区；结果太多时提醒模型缩小范围 |

### 第三段：Schema 里的必填和选填

```python
{"name": "read_file", "description": "Read file contents.",
 "input_schema": {"type": "object",
                  "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                  "required": ["path"]}},
```

`path` 在 `required` 里，`limit` 不在。对应到 Python 函数，就是 `run_read(path, limit=None)`：必填参数没有默认值，选填参数有默认值。

**Schema 里参数的名字必须和函数的参数名一字不差**，因为下一段会用 `**block.input` 把模型给的参数直接展开传进函数。

### 第四段：分发

```python
for block in tool_calls:
    print(f"\033[33m> {block.name}\033[0m")
    handler = TOOL_HANDLERS.get(block.name)
    output = handler(**block.input) if handler else f"Unknown: {block.name}"
    print(str(output)[:200])
    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
```

比教程正文里的简化版多了一个保护：用 `.get` 查表。模型如果编出一个不存在的工具名，比如 `delete_file`，不会抛 `KeyError`，而是返回 `Unknown: delete_file` 给模型看。

`handler(**block.input)` 是整课最关键的一行。模型给的参数是一个字典 `{"path": "a.py"}`，`**` 把它展开成 `run_read(path="a.py")`。这一行把模型的输出直接变成了一次 Python 函数调用。

## 实际跑一次（本站补充）

在一个放了 `a.py` 和 `b.py` 的目录里提问：

```
Read a.py and b.py, and list all .py files. Do all three in one step.
```

模型第一次就返回了三个 `tool_use`：

```
msgs=1  -> stop_reason=tool_use  内容块: [tool_use, tool_use, tool_use]
           read_file {"path": "a.py"}
           read_file {"path": "b.py"}
           glob      {"pattern": "**/*.py"}
msgs=3  -> stop_reason=end_turn  内容块: [text]
```

注意第二次调用时 `msgs=3`，不是 5。三个工具结果被放进了**同一条** user 消息，所以只多了两条消息：一条 assistant（三个 `tool_use`），一条 user（三个 `tool_result`）。

## 想一想（本站补充）

做真实项目时，工具系统光「能跑」是不够的，还要考虑校验、重试和幂等。拿 s02 的代码对照下面几个问题，看看它做了哪些，没做哪些：

1. **校验：** 如果模型调用 `read_file` 时把参数名写成了 `{"file": "a.py"}`，会发生什么？注意 `handler(**block.input)` 这一行在不在 `try` 里面。
2. **重试：** `run_edit` 返回 `Error: text not found` 之后，谁来重试？harness 有没有写重试的代码？
3. **幂等：** 同一个 `edit_file` 调用连续执行两次，第二次会怎样？`write_file` 呢？
4. **歧义：** 如果 `old_text` 在文件里出现了两次，`run_edit` 改的是哪一处？模型知道吗？

## 试一下

```sh
cd learn-claude-code
python s02_tool_use/code.py
```

试试这些 prompt：

1. `Read the file README.md and tell me what this project is about`
2. `Create a file called test.py that prints "hello", then read it back`
3. `Find all Python files in this directory`
4. `Read both README.md and requirements.txt, then create a summary file`

观察重点：模型什么时候只调一个工具，什么时候一次调多个？多个工具调用的顺序和结果是否正确？

## 接下来

现在 Agent 有 5 个专用工具。file tools 受 `safe_path` 保护，但 bash 不受限制，`rm -rf /` 还是能跑。

[s03 Permission](https://github.com/shareAI-lab/learn-claude-code/tree/main/s03_permission) → 在工具执行之前加一道门：这个操作安全吗？需要用户批准吗？
