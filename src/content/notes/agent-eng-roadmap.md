---
title: Agent 工程实战：从零写一个 Harness 的学习路线
date: '2026-10-27'
tags: [Agent, Harness, 学习路线, 后端基础, Python]
summary: 不先啃 Java 后端，而是从零手写一个 Agent Harness，让它一步步长成能持久化、能流式输出、能部署、能评测、能自我改进的服务，后端知识在需要的地方顺手学。
draft: false
---

这个专题记录我从零手写一个 Agent Harness 的过程。Harness 就是套在大模型外面的那层代码，负责调用模型、执行工具、管理上下文、处理权限和保存会话。Claude Code、Codex、OpenHands 的核心都是一个 harness。

我的情况是：Agent 算法懂一些，**后端基础几乎为零**。常见的建议是先补 Java 后端，做一个点评或外卖项目，再学 Agent。我不打算这么走，原因很简单：Agent 本身就用 Python 写，harness 需要的后端知识（子进程、并发、数据库、HTTP、容器）全都能在写 harness 的过程中遇到。与其学两套东西，不如让 harness 一点点长成一个后端服务，每遇到一个后端知识点就停下来把它学透。

> 这个专题是我和 AI 一起整理的。每一课的代码我都在本地跑过，文中贴出的输出是真实运行结果。

## 1. 一张图看全路线

<figure style="margin:1.5rem 0">
<svg viewBox="0 0 720 300" role="img" aria-label="学习路线：最小 Harness、变成服务、评测集、自我改进四个阶段，每个阶段下面列出顺带学到的后端知识" style="width:100%;height:auto;font-family:var(--font-heading)">
<style>.ae-box{fill:var(--surface);stroke:var(--border);stroke-width:1.5}.ae-hot{fill:#F6E3D9;stroke:var(--accent);stroke-width:1.5}.ae-t{fill:var(--text);font-size:15px;font-weight:500}.ae-s{fill:var(--text-dim);font-size:12.5px}.ae-a{stroke:var(--accent);stroke-width:2;fill:none;marker-end:url(#ae-arrow)}</style>
<defs><marker id="ae-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--accent)"/></marker></defs>
<rect class="ae-hot" x="10" y="20" width="160" height="70" rx="12"/><text class="ae-t" x="90" y="50" text-anchor="middle">① 最小 Harness</text><text class="ae-s" x="90" y="72" text-anchor="middle">循环 · 工具 · 上下文</text>
<rect class="ae-box" x="190" y="20" width="160" height="70" rx="12"/><text class="ae-t" x="270" y="50" text-anchor="middle">② 变成服务</text><text class="ae-s" x="270" y="72" text-anchor="middle">持久化 · API · 部署</text>
<rect class="ae-box" x="370" y="20" width="160" height="70" rx="12"/><text class="ae-t" x="450" y="50" text-anchor="middle">③ 评测集</text><text class="ae-s" x="450" y="72" text-anchor="middle">20 道题 · 自动判分</text>
<rect class="ae-box" x="550" y="20" width="160" height="70" rx="12"/><text class="ae-t" x="630" y="50" text-anchor="middle">④ 自我改进</text><text class="ae-s" x="630" y="72" text-anchor="middle">从轨迹沉淀 skill</text>
<path class="ae-a" d="M170 55H188"/><path class="ae-a" d="M350 55H368"/><path class="ae-a" d="M530 55H548"/>
<text class="ae-s" x="10" y="130">顺带学到的后端知识 ↓</text>
<text class="ae-s" x="20" y="160">· 子进程与信号</text><text class="ae-s" x="20" y="182">· 管道与超时</text><text class="ae-s" x="20" y="204">· JSON 协议与校验</text><text class="ae-s" x="20" y="226">· token 与截断</text>
<text class="ae-s" x="200" y="160">· SQLite 与事务</text><text class="ae-s" x="200" y="182">· asyncio 并发</text><text class="ae-s" x="200" y="204">· FastAPI 与 HTTP</text><text class="ae-s" x="200" y="226">· SSE 流式输出</text><text class="ae-s" x="200" y="248">· Docker 沙箱</text>
<text class="ae-s" x="380" y="160">· 测试与回放</text><text class="ae-s" x="380" y="182">· pass@k 与方差</text><text class="ae-s" x="380" y="204">· LLM 当裁判</text>
<text class="ae-s" x="560" y="160">· 检索与索引</text><text class="ae-s" x="560" y="182">· A/B 对比实验</text>
<text class="ae-s" x="10" y="288">每一步都要能跑、能测，并且自己能讲清楚为什么这么设计。</text>
</svg>
</figure>

四个阶段是串起来的：**没有能跑的 harness，就谈不上评测；没有评测，「自我改进」就没法衡量**。所以顺序不能乱。

## 2. 每一课学什么

| 课 | 主题 | 顺带学的后端或八股 | 状态 |
|---|---|---|---|
| 1 | [Agent 执行一条命令，背后发生了什么](/notes/agent-eng-subprocess/) | 进程、进程组、信号、管道、孤儿进程 | ✅ |
| 2 | 工具协议：让模型稳定地说出「我要调用什么」 | JSON、schema 校验、解析失败重试 | 写作中 |
| 3 | Agent 循环：步数上限、重复动作检测与取消 | 状态机、异常处理 | 计划 |
| 4 | 上下文管理：截断、压缩与 token 计数 | 分词、缓存 | 计划 |
| 5 | 会话持久化：用 SQLite 保存和恢复对话 | SQL、索引、事务 | 计划 |
| 6 | 并行执行工具 | asyncio、线程池、GIL | 计划 |
| 7 | 把 Harness 变成服务 | FastAPI、HTTP、SSE | 计划 |
| 8 | 沙箱：把工具关进容器 | Docker、namespace、cgroup | 计划 |
| 9 | 做一个 20 道题的小评测集 | 测试、回放、统计 | 计划 |
| 10 | 自我改进：从轨迹里沉淀 skill | 检索、对比实验 | 计划 |

第 1–4 课基本对应 Stanford CS329Z 作业 1「Building an Agentic Harness」的内容。那门课是 2026 年秋季新开的，讲义公开在 [cs329z.stanford.edu](https://cs329z.stanford.edu/)，作业仓库也是公开的（MIT 协议）。我会对照它的测试来检查自己的实现，但不会公开作业答案。

## 3. 参考材料

- **入门**：Datawhale 的 [Hello Agents](https://datawhalechina.github.io/hello-agents/) 第 4 章（ReAct、Plan-and-Solve、Reflection）和第 7 章（自己写一个 Agent 框架）。
- **课程**：CS329Z 第 1、4、6 讲打基础，第 7（记忆）、9（优化）讲对应第 10 课的自我改进。
- **源码**：生产级 harness 看 BaoCode、DeepSeek Harness、Pi，自我改进看 Hermes-agent 的学习闭环。
- **八股**：操作系统、计算机网络、MySQL、Redis 这几块，用图解风格的资料每天刷 30–60 分钟。每一课末尾也会列出这一课对应的八股知识点。

## 4. 怎么用这个专题

1. **手敲，不复制粘贴。** 每一课的代码都不长，敲一遍才会发现自己哪里没懂。
2. **先跑反例。** 每一课都会先给一个「看起来对，其实有坑」的版本，自己跑一遍，亲眼看到坑，再看修复。
3. **做完「动手」再往下读。** 每一课末尾有 2–3 个小改动，是检验自己是否真的会了的最低标准。
4. **面试追问要能脱稿答。** 答不出来，就回到对应的小节重看。
