# RAAT — 面向开源仓库的多Agent深度审计工具
> 通过多Agent编排，融合官方文档、Issue讨论、PR diff三源证据，对开源项目做技术问题深度审计。
> 数据底座：LangChain全量历史 issue/PR/评论 + 官方文档，14.3 万向量点。

---

## 一、要解决的问题

单 LLM 问一个仓库的「已知问题」只会给泛泛而谈的答案——真正有价值的信息散落在三处：
官方文档、issue 讨论、PR diff，单一检索抓不全。RAAT 把这件事做成一个多 Agent 流水线：

```
用户问题
   │
   ▼
┌─────────┐    ┌─────────────────────────────────────┐    ┌─────────┐
│ Planner │───▶│ Supervisor（动态调度）               │───▶│ Reflect │
│         │    │  ├ doc_worker   → 官方文档检索        │    │ 冲突识别 │
│         │    │  ├ issue_worker → issue 讨论          │    │ 证据分级 │
│         │    │  └ code_worker  → PR diff 代码证据    │    └────┬────┘
└─────────┘    │  第1轮并行 → 证据不足则动态追问        │         ▼
               └─────────────────────────────────────┘    ┌─────────────┐
                                                          │ Aggregator  │
                                                          │ 结构化报告   │
                                                          └─────────────┘
```

- **Planner**：把开放问题拆成 `doc/issue/code` 三类子任务
- **Supervisor**：第 1 轮并行派发全部子任务，读各 worker 产出判断证据充分性，不足则动态追加追问（有界轮数）
- **Workers**：doc / issue / code 三个 worker 各自手写 ReAct 工具循环，按类型分发工具（职责隔离），每步 function-calling 可观测
- **Reflect**：确定性证据分级（合并PR > 文档 > 未合并PR > 已关闭issue > 评论）+ 语义冲突识别与可信度裁决
- **Aggregator**：来源去重 + 生成带证据等级的结构化审计报告

核心设计：**可观测（每步落库）+ 反思（识别矛盾）+ 动态调度（证据不足再查）**。

## 二、技术栈与设计取舍

| 层 | 选型 | 说明 |
|---|---|---|
| 编排 | LangGraph | `StateGraph` + `stream(updates)` 实时节点流 |
| 检索 | Qdrant + rank_bm25 | **BM25 + ANN 混合检索 + RRF 融合** |
| 向量 | sentence-transformers/all-MiniLM-L6-v2 (384d) | 本地 CPU 推理，14.3 万点 |
| 数据 | GitHub GraphQL 全量拉取 | 绕开 REST 1 万条硬上限，断点续传 |
| 编排/检索模型 | 分层（可在 config 切换） | 编排与反思走强推理模型、检索 Worker 走低成本快速模型 |
| 记忆 | SQLite | 长期（跨会话审计记忆）+ 短期（本轮来源去重） |
| 可观测 | LangSmith + 自建 agent_traces | 开发期图级/span 追踪（key-gated）+ 产品内结构化审计日志 |
| MCP | fastmcp server | 检索层暴露为 MCP server，任何 MCP 客户端可调用 |

**关键设计决策**：
- **Qdrant payload 只存元数据、全文放 SQLite**：QdrantLocal 写大 payload 是慢的根源，分离后写入快数倍；向量库管检索、关系库管文本，各司其职。
- **确定性 uuid5 point_id**：同一记录永远生成同一 id，索引构建幂等、断点续传。
- **BM25 与向量同源对齐**：同一 chunk 函数 + 同一 uuid5，两路结果在 point_id 上对齐，才能 RRF 融合。
- **低信号机器人评论过滤**（Dosu 等自动化 triage）：查询时按 author + 文本标记过滤，防止污染检索结果。

## 三、快速开始

```bash
conda create -n tdas python=3.11 && conda activate tdas
pip install -r requirements.txt
cp .env.example .env   # 填 LLM API key / GITHUB_TOKEN

# 1) 拉数据（GraphQL 全量，断点续传）＋ 建索引
python -m backend.scripts.ingest_repo
python -m backend.scripts.ingest_docs
python -m backend.scripts.build_index
python -m backend.scripts.build_bm25

# 2) CLI 审计（planner → supervisor[三 worker] → reflect → aggregator）
python -m backend.scripts.audit "LangChain 的 memory 模块在生产环境有哪些已知问题？"

# 3) Streamlit UI（实时展示各 agent 节点进度 + 报告 + 轨迹）
streamlit run backend/app/ui/app.py

# 4) MCP server（把检索层暴露给任何 MCP 客户端）
python -m backend.app.mcp_server --transport sse --port 8010   # 常驻服务
python -m backend.app.mcp_server                                # 或 stdio
```

**可选：MCP 客户端接入**（以 SSE 常驻服务为例，实测通过）：
```bash
# 启动常驻服务（预热模型）
python -m backend.app.mcp_server --transport sse --port 8010
# 注册到 MCP 客户端（如 Claude Code / Cursor）
claude mcp add --transport sse raat-retrieval http://127.0.0.1:8010/sse
```
之后即可直接调用 `search_docs / search_issues / search_prs / get_issue_detail` 检索 LangChain 历史数据。

## 四、验证结果（真实数据）

对 LangChain 仓库跑「memory 模块生产问题」审计：

```
计划: [doc, issue, code] 三子任务
元数据: worker_outputs=7（3 静态 + 4 动态追问）· total_steps=24 · covered_types=[code,issue] · unique_sources=74
```

产出质量（节选）：
- 官方文档明确推荐 LangGraph 记忆体系，旧 `langchain.memory` 已随 legacy chains 移入 `langchain-classic`
- `ConversationSummaryBufferMemory` 的 `moving_summary_buffer` 摘要会不断膨胀最终突破 `max_token_limit`（issue #17888）
- token 裁剪在持久化历史（Redis/Mongo）下失效——`pop()` 只在内存/file 实现（issue #5053 等一簇）
- `trim_messages` 未转发 `bind_tools` 参数导致低估 token（PR #35519）

反思层评估集（已知 ground truth）**3/3 PASS**：2 个真矛盾正确检出并给出可信度裁决，1 个互补样本正确不误报。

## 五、项目结构

```
backend/
├── app/
│   ├── agents/       # LangGraph 编排（planner/supervisor/worker/reflect/aggregator）
│   ├── rag/          # 检索层（retriever 混合检索 / indexer 分块入库 / bm25 / embedder）
│   ├── ingest/       # 数据拉取（GraphQL 全量 issues/PRs/评论 + 官方文档）
│   ├── ui/           # Streamlit 前端（实时展示 agent 节点进度）
│   ├── mcp_server.py # 检索层 MCP server（SSE/stdio，供任何 MCP 客户端调用）
│   ├── config.py     # 配置（模型/仓库/路径/网络镜像）
│   ├── db.py         # SQLite（长期记忆 + agent_traces 可观测轨迹 + 全文）
│   └── llm.py        # 统一 LLM 客户端（OpenAI 兼容，可切 provider）
└── scripts/          # ingest_repo / ingest_docs / build_index / build_bm25 / audit / eval_reflect / search_demo
```

## 六、限制与后续工作

- **code worker 的证据来源**：来自 PR diff与discussion，未做 AST/符号级代码检索（后续可接入代码索引）。
- **Qdrant 嵌入式（QdrantLocal）**：本地模式便于复现，非生产级；规模化需换 Qdrant 服务端。
- **数据规模**：14.3 万点（LangChain 全量历史 + 官方文档），足以评测与复现。
- **单仓验证**：当前验证集基于LangChain仓库，后续计划扩充多仓库批量自动化评测。
