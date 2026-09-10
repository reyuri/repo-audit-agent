# RAAT Demo 录屏脚本（约 5 分钟）

> 目标：5 分钟完整演示。节奏：**先讲痛点 → 架构 → 真实审计 → 反思 → 可观测**。
> 录屏工具推荐 OBS（1080p，或系统自带 Win+G）。录前确保 `data/` 索引就绪。

---

## 0. 准备（录前，不计入时长）

```bash
conda activate tdas
streamlit run backend/app/ui/app.py
```

浏览器打开 http://localhost:8501 。把示例问题换成下面这个（问题本身就在 UI 默认值里）：
**"LangChain 的 memory 模块在生产环境里有哪些已知 bug、性能陷阱和已修复的问题？"**

> 提示：可先跑一遍完整审计让结果留在 UI 里，录屏时点按钮重跑会实时刷新，或直接展示已缓存报告 —— 取决于你要突出「实时进度」还是「报告质量」。**推荐录实时**，进度动画是卖点。

---

## 1. 开场（30s）—— 痛点

> "大家好，我来演示 RAAT：一个面向开源仓库的多 Agent 深度审计工具。
> 背景是——单个 LLM 问一个开源仓库'有哪些已知问题'，只会给泛泛而谈的答案，
> 因为代码里真正有价值的答案藏在 issue 讨论、PR diff 和官方文档三处，单一检索抓不全。
> 所以我把它做成一个多 Agent 流水线。"

**画面**：停在这个开场话术页面，或者切到 README 架构图。

## 2. 数据底座（30s）—— 规模与获取

> "数据底座是 LangChain 全量历史 issue/PR/评论 + 官方文档，共 14.3 万向量点。
> 关键细节：GitHub REST 接口有 1 万条硬上限，所以我用 GraphQL 游标分页全量拉取，
> 断点续传、幂等重建。"

**画面**：鼠标点到侧边栏「数据底座」统计（Qdrant 向量点 143,468、全文文本 xxx）。

## 3. 架构（40s）—— 一张图讲清

> "编排用 LangGraph。流水线是 planner 拆解 → supervisor 并行调度三个 worker →
> reflect 反思 → aggregator 汇总。三个 worker 各有职责：文档检索、issue 讨论、PR diff。
> 每个 worker 只被分发职责内的工具——控制面和执行面分离。"

**画面**：打开 README 的 ASCII 架构图，或口头配合白板。

## 4. 实时审计（2min）—— 核心演示，实拍实说

> "现在点运行审计。你们注意看左侧进度——planner 先把问题拆成三个子任务，
> supervisor 并行派发。等 worker 陆续回来，supervisor 会判断证据够不够，
> 不够就动态追加追问。这里是真正的 agent 行为，不是一次性 RAG。"

**画面**：点「🚀 运行审计」。**等进度条走**，重点解说：
- Planner 拆出 doc/issue/code 三行子任务
- Supervisor 状态行显示「已并行产出 N 个 worker 结论」
- 如果出现追加轮，强调「这就是动态调度——它自己觉得证据不足，又去查了」

> 等待期间可以说：「期间每个工具调用都写进了 agent_traces，等会给你们看。」

## 5. 报告解读（60s）—— 讲发现 + 反思

> "审计完成。看发现部分，每条都带证据等级：强证据是合并 PR 或多方印证，弱证据只有评论。
> 比如官方文档明确推荐 LangGraph 记忆体系、旧 memory 模块已移入 legacy 包；
> 而 issue #17888 显示 ConversationSummaryBufferMemory 的摘要会不断膨胀突破 token 上限——
> 这就是'官方推荐 vs 实际失效'的张力。"

**画面**：展开 1-2 条 strong 证据的 finding，指来源链接（可点开 GitHub）。
> "反思层我把各 worker 结论喂给强推理模型做矛盾检测。这轮没检出冲突——
> 因为结论是互补的。但我用已知矛盾样本做过评估，3/3 通过：能分辨真矛盾（限制可靠 vs 失效）
> 和看似矛盾（已弃用 vs 还在用）。"

**画面**：如果 UI 显示冲突会更好；没有就切到 `eval_reflect.py` 输出。

## 6. 可观测（60s）—— 双层杀手锏

> "可观测我做了两层。开发期我用 **LangSmith** —— 你们看，整张图在这里：
> supervisor 派发、每个 worker 的 ReAct 循环、每次 LLM 调用是子 span，token 和耗时
> 全有。这是行业标准工具，LangGraph 自动接入，我额外给 LLM 客户端挂了 span。
> 而产品层我自己写了 **agent_traces** —— 每次 planner 规划、工具调用、supervisor 决策、
> reflect 反思都落 SQLite，UI 直接可查，能复盘'慢在哪、谁调了什么'。两个消费方不同：
> LangSmith 服务开发者调 bug，agent_traces 服务产品使用者。多 Agent 不是黑盒。"

**画面**：先切到 LangSmith 网页（smith.langchain.com，project=raat，展开整张图 DAG），
指 supervisor/reflect 节点和 LLM span；再切回 Streamlit 的 Agent 轨迹表（agent/node/tool/平均耗时）。

## 6.5 MCP 能力（40s）—— 加分项

> "再补一个能力：我们这 14.3 万点的检索层，我包装成了 **MCP server**。任何 MCP 客户端——
> Claude、Cursor、IDE——都能直接搜 LangChain 的文档、issue、PR。也就是说，同一套
> BM25+ANN+RRF 检索有两个身份：驱动我的多 Agent 审计，也作为标准工具服务外部 agent。
> 这让我理解了 MCP 的协议模型：server 提供工具、client 消费工具，两边通过 JSON-RPC 通信。"

**画面**：终端启动 `python -m backend.app.mcp_server --transport sse --port 8010`（预热 ~30-60s），
再用 Claude Code（`claude mcp list` 显示 ✔ Connected）跑一个 prompt 让它调 search_issues，
展示它检索出真实 issue #17888。强调「自己写 server 并接入 Claude Code」而非「只调用别人工具」。
> 注：若录屏时间紧，可先在 `claude -p --allowedTools "mcp__raat-retrieval__*"` 下跑好检索，录屏时展示结果。

## 7. 收尾（20s）—— 边界 + 工程细节

> "两个边界：code worker 的代码证据来自 PR diff 和讨论，没做 AST 符号级检索，这是未来工作；
> Qdrant 用的是嵌入式本地模式，生产会换服务端。工程上的细节——确定性 uuid5 让索引幂等可续传、
> 机器人评论过滤避免噪音污染检索、BM25 和向量在同一批 chunk 上对齐才能做 RRF 融合。"

**画面**：README「限制与后续工作」小节。
