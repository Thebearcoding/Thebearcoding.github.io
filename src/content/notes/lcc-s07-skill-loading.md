---
title: 'learn-claude-code s07：Skill Loading，先给目录，用到再翻全文'
date: '2026-10-30'
tags: [Agent, Harness, Skill, Context, learn-claude-code]
summary: 转载 learn-claude-code 第 7 课，补充配图、code.py 逐段导读和一次真实运行。启动时只把技能的名字和一句话描述放进 system prompt，模型需要时调用 load_skill 把整份 SKILL.md 作为工具结果拿回来；真实运行里目录只占 1.3k 字符，四份技能全文有 16.5k。
draft: false
---

> 本文转载自 [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code/tree/main/s07_skill_loading)（MIT License，Copyright (c) 2024 shareAI Lab），配图、「代码导读」「实际跑一次」「想一想」三节为本站补充。

> system prompt 保存技能目录；`load_skill` 返回完整的 `SKILL.md`。
>
> **Harness 层**：知识加载 — 让模型先知道有哪些技能，再按名称读取内容。

## 问题

假设某个项目有一套 React 组件规范、一份 SQL 风格指南和一份 API 设计文档。我们希望 Agent 在开发过程中遵守这些规范，最直接的做法就是把它们全部放进 system prompt：

```python
SYSTEM = (
    f"You are a coding agent. "
    + open("docs/react-style.md").read()
    + open("docs/sql-style.md").read()
    + open("docs/api-design.md").read()
)
```

这种做法能让 Agent 读到所有规范，但问题在于，三份文档被固定放进了 system prompt，无法根据当前任务只选择需要的那一份。每次调用 LLM 时，三份文档的全文都会一起发送给模型。当前任务只修改 React 组件时，实际需要的只有 React 组件规范；SQL 风格指南和 API 设计文档与任务无关，却仍然占用输入 token 和上下文窗口，留给代码、对话和工具结果的空间也会变少。

## 解决方案

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s07-skill-loading/01-catalog-then-load.jpg" target="_blank" rel="noopener" aria-label="查看原图：启动时扫描 skills 目录下的四份 SKILL.md，只把名字和描述写进 system prompt；模型调用 load_skill('code-review') 后，整份 SKILL.md 作为 tool_result 追加到 messages"><img src="/notes-assets/lcc-s07-skill-loading/01-catalog-then-load.jpg" alt="启动时扫描 skills 目录下的四份 SKILL.md，只把名字和描述写进 system prompt；模型调用 load_skill('code-review') 后，整份 SKILL.md 作为 tool_result 追加到 messages" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

启动时，`SkillLoader` 扫描 `skills/*/SKILL.md`，读取 YAML frontmatter 中的 `name` 和 `description`，并把这份目录加入 system prompt。模型需要完整说明时，调用 `load_skill(name)`；返回的 `SKILL.md` 作为 `tool_result` 追加到消息列表。

| 内容 | 进入模型的位置 | 何时加入 |
|------|----------------|----------|
| 技能名称和描述 | system prompt | 启动时 |
| 完整 `SKILL.md` | `tool_result` | 调用 `load_skill` 时 |

## 工作原理

每个技能是一个包含 `SKILL.md` 的目录：

```text
skills/
  agent-builder/SKILL.md
  code-review/SKILL.md
  mcp-builder/SKILL.md
  pdf/SKILL.md
```

### 扫描技能

```python
class SkillLoader:
    def scan(self):
        self.skills.clear()
        skills_root = self.skills_dir.resolve()
        for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
            if (not manifest.is_file()
                    or not manifest.resolve().is_relative_to(skills_root)):
                continue
            content = manifest.read_text(encoding="utf-8")
            metadata, body = self.parse_frontmatter(content)
            raw_name = metadata.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest.parent.name
            raw_description = metadata.get("description")
            description = (raw_description.strip()
                           if isinstance(raw_description, str) else "")
            description = description or body.split("\n", 1)[0]
            description = " ".join(str(description).lstrip("# ").split())
            self.skills[name] = {
                "name": name,
                "description": description,
                "content": content,
            }
```

`catalog()` 只输出名称和描述：

```text
- code-review: Perform thorough code reviews...
- pdf: Process PDF files...
```

### 组装 system prompt

```python
ENVIRONMENT_PROMPT = (
    "Windows: the bash tool runs through cmd.exe; use cmd.exe syntax, not Unix "
    "Bash or PowerShell syntax, and prefer dedicated file tools for file operations"
    if os.name == "nt"
    else "Unix-like: the bash tool runs the system shell"
)

def build_system_prompt() -> str:
    return (
        f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
        "Use tools to solve tasks. "
        "Act, don't explain.\n\n"
        f"Skills available:\n{SKILL_LOADER.catalog()}\n\n"
        "Use load_skill to read the full instructions when a skill applies."
    )
```

固定的 Agent 指令和扫描得到的技能目录在这里组成实际传给模型的 system prompt。

### 加载完整内容

```python
def load(self, name: str) -> str:
    skill = self.skills.get(name)
    if skill:
        return skill["content"]
    available = ", ".join(self.skills) or "none"
    return f"Error: Unknown skill '{name}'. Available: {available}"
```

`name` 用于查询启动时建立的注册表，不会被当作文件路径。工具返回后，原有 Agent Loop 会把内容作为新的 `tool_result` 消息追加。


## 代码导读：把 code.py 从头读一遍（本站补充）

完整的 [`s07_skill_loading/code.py`](https://github.com/shareAI-lab/learn-claude-code/blob/main/s07_skill_loading/code.py) 有 401 行。注意它是在 s05 的基础上加的，**s06 的 `task` 工具没有带过来**。每一课只演示一个机制，不是一路叠加。

新增的部分全在文件开头：`SkillLoader` 类、`build_system_prompt()`，以及工具表里的一行 `load_skill`。

### 第一段：技能目录只在启动时扫一次

```python
WORKDIR = Path.cwd()
SKILLS_DIR = WORKDIR / "skills"
...
SKILL_LOADER = SkillLoader(SKILLS_DIR)   # __init__ 里调用 self.scan()
...
SYSTEM = build_system_prompt()
```

这三行都是**模块级**代码，`import` 的时候就执行完了。由此可以推出两件事：

1. `SKILLS_DIR` 跟着**当前目录**走。在仓库根目录启动，才能找到 `skills/`；换个目录启动，目录就是空的，system prompt 里只会写 `(no skills found)`。
2. `SYSTEM` 是一个固定的字符串。程序运行过程中你再新建一个技能，模型是看不到的，要重启才行。

### 第二段：`parse_frontmatter`，宽容地解析

```python
@staticmethod
def parse_frontmatter(text: str) -> tuple[dict, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1)
         if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        return {}, text
    frontmatter = "".join(lines[1:closing_index])
    body = "".join(lines[closing_index + 1:]).strip()
    try:
        metadata = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, body
```

SKILL.md 开头两行 `---` 之间是 YAML，写着 `name` 和 `description`。这个函数的特点是**怎么都不报错**：

- 没有 `---` 开头，或者找不到结尾的 `---` → 当作没有 frontmatter，整篇都是正文。
- YAML 写错了 → 当作空字典。
- YAML 解析出来不是字典（比如只写了一个字符串）→ 也当作空字典。

一份写坏的 SKILL.md 不应该让整个 Agent 起不来，所以这里选择「降级」而不是「报错」。

另外，用的是 `yaml.safe_load` 而不是 `yaml.load`。后者遇到特殊标签时可以构造任意 Python 对象，读别人写的文件时不安全。

### 第三段：`scan`，每个字段都有兜底

```python
for manifest in sorted(self.skills_dir.glob("*/SKILL.md")):
    if (not manifest.is_file()
            or not manifest.resolve().is_relative_to(skills_root)):
        continue
    ...
    name = name or manifest.parent.name
    ...
    description = description or body.split("\n", 1)[0]
    description = " ".join(str(description).lstrip("# ").split())
```

- 没写 `name`，就用**文件夹名**。
- 没写 `description`，就用**正文第一行**，再去掉开头的 `#` 和多余空白。所以一份只有 `# PDF 处理` 标题的 SKILL.md，描述就是「PDF 处理」。
- `resolve().is_relative_to(skills_root)`：如果某个技能文件夹其实是一个指向别处的符号链接，就跳过。和 s02 的 `safe_path` 是同一个思路。
- `sorted(...)`：保证每次启动，目录的顺序都一样。这样 system prompt 每次都一字不差，对后面要讲的 prompt caching 很友好。

### 第四段：模型只能靠 description 做决定

```python
def build_system_prompt() -> str:
    return (
        f"You are a coding agent at {WORKDIR}. Environment: {ENVIRONMENT_PROMPT}. "
        "Use tools to solve tasks. "
        "Act, don't explain.\n\n"
        f"Skills available:\n{SKILL_LOADER.catalog()}\n\n"
        "Use load_skill to read the full instructions when a skill applies."
    )
```

模型决定「要不要加载某个技能」的时候，手上只有这一行描述。看看 `code-review` 是怎么写的：

```yaml
description: Perform thorough code reviews with security, performance, and maintainability analysis. Use when user asks to review code, check for bugs, or audit a codebase.
```

前半句说**它是什么**，后半句说**什么时候用**。`agent-builder` 甚至直接列了一串 `Keywords:`。这些描述其实是写给模型看的「触发条件」。描述写得含糊，技能就可能该用的时候没被用上。

### 第五段：`load` 按名字查表，不是按路径读文件

```python
def load(self, name: str) -> str:
    skill = self.skills.get(name)
    if skill:
        return skill["content"]
    available = ", ".join(self.skills) or "none"
    return f"Error: Unknown skill '{name}'. Available: {available}"
```

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s07-skill-loading/03-name-lookup.jpg" target="_blank" rel="noopener" aria-label="查看原图：load_skill 的参数只用来查 self.skills 字典：传 code-review 拿到 SKILL.md 全文；传 ../../.env 查不到，返回 Unknown skill 错误和可用列表，那个文件根本不会被打开"><img src="/notes-assets/lcc-s07-skill-loading/03-name-lookup.jpg" alt="load_skill 的参数只用来查 self.skills 字典：传 code-review 拿到 SKILL.md 全文；传 ../../.env 查不到，返回 Unknown skill 错误和可用列表，那个文件根本不会被打开" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

`name` 只是字典的键。模型就算传进来 `../../.env`，也只会得到一句「找不到」，不会去读任何文件。启动时扫描好的内容才是唯一的来源。

报错信息里附上了 `Available: ...`。模型拼错名字时，看到这一句通常就能自己改正，再调用一次。

另外，返回的是 `content`，也就是**包括 frontmatter 在内**的整个文件。

### 第六段：加载的内容放在哪里

`load_skill` 和其他工具一样，返回值变成 `tool_result`，追加在 `messages` 末尾。为什么不把加载的技能直接拼进 system prompt？

- **system prompt 保持不变。** 每次请求的开头都一样，服务端的 prompt cache 才能命中。要是每加载一个技能就改一次 system prompt，前面缓存的内容就全部作废了。
- **历史记录里能看出加载的时机。** 什么时候、因为什么加载了哪个技能，都留在对话里。

代价是：加载过的技能会**一直留在 `messages` 里**，之后每次请求都要带上。下一课的上下文压缩，就是来解决这类问题的。

## 实际跑一次（本站补充）

在仓库根目录运行，提问 `Review s05_todo_write/example/hello.py and load the relevant skill first`：

```
[HOOK] load_skill(['code-review'])
[HOOK] read_file(['/Users/athebear/Documents/GitHub/bear-harness/learn-claude)
[HOOK] Stop: session used 2 tool calls
## Code Review: `s05_todo_write/example/hello.py`

### Summary
The script is clear and works for string inputs. No security or performance concerns were found.

### Improvements

1. **Import-time side effect** (`line 6`)
   Importing this module immediately prints `"Hello, Claude"`, which can interfere with tests or reuse.
……
### Verdict
- [ ] Ready to merge
- [x] Needs minor changes
- [ ] Needs major revision
```

模型先调用了 `load_skill('code-review')`，然后才去读文件。最后的输出带着 `Summary`、`Positive Notes`、`Verdict` 这几个小节，还有三个勾选框。这些都是 `code-review/SKILL.md` 里规定的格式，说明模型确实照着技能文档在做。

从代理的 dump 里量一下大小：

| | 字符数 |
|---|---|
| system prompt（固定指令 + 4 条技能目录） | 1,322 |
| 4 份 SKILL.md 全文加起来 | 16,478 |
| 加载 `code-review` 后，messages 增加了 | 约 4,700 |

<figure style="margin:1.5rem 0">
<a href="/notes-assets/lcc-s07-skill-loading/02-request-cost.jpg" target="_blank" rel="noopener" aria-label="查看原图：三次请求各自携带的字符数：把四份技能全部放进 system prompt，每次都是 16.5k；只放目录再按需加载，每次是 1.3k 目录，加载后的第二、三次请求多出 4.7k 的 code-review"><img src="/notes-assets/lcc-s07-skill-loading/02-request-cost.jpg" alt="三次请求各自携带的字符数：把四份技能全部放进 system prompt，每次都是 16.5k；只放目录再按需加载，每次是 1.3k 目录，加载后的第二、三次请求多出 4.7k 的 code-review" width="1400" height="788" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</figure>

这次一共发了 3 次请求。如果把 4 份技能全塞进 system prompt，每次都要多带 16k 字符，其中 3 份和这次代码审查毫无关系。

顺便一提，你现在用的 Claude Code 也是这么做的：可用的技能以「名字 + 一句话描述」列给模型，模型觉得用得上时，再调用 Skill 工具加载全文。

## 想一想（本站补充）

1. 进到 `s07_skill_loading/` 目录里再运行 `python code.py`，system prompt 里的技能目录会变成什么？为什么？
2. 程序运行中新建一个 `skills/foo/SKILL.md`，模型能用上吗？如果要支持「热加载」，你会改哪里？改完以后，对 prompt caching 有什么影响？
3. 两个技能文件夹的 frontmatter 里写了同一个 `name`，`self.skills` 里最后剩下哪一个？模型能发现吗？
4. 加载过的技能会一直留在 `messages` 里。如果一次长会话里先后加载了五六个技能，会出现什么问题？
5. 如果某个技能的描述写成 `Useful helper`，会发生什么？你会怎么验证一条 description 写得好不好？

## 试一下

```sh
cd learn-claude-code
python s07_skill_loading/code.py
```

试试这些 prompt：

1. `What skills are available?`
2. `Load the code-review skill and follow its instructions`
3. `Review README.md and load the relevant skill first`

观察 system prompt 中是否只有技能目录，以及调用 `load_skill` 后是否出现完整的 `SKILL.md` 内容。

## 接下来

随着工具调用增加，`messages[]` 会积累较早的文件内容和工具结果。

[s08 Context Compact](/notes/lcc-s08-context-compact/) → 缩短较早的消息，为后续调用保留上下文空间。
