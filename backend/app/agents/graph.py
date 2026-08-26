"""审计图：planner → supervisor → reflect → aggregator。

- planner（Pro）：开放问题 → 结构化子任务 [{id, type: doc|issue|code, query}]。
- supervisor（Pro）：动态调度。第 1 轮并行派发全部子任务（doc/issue/code 三类
  worker，各带自己的工具子集），之后每轮读当前产出判断证据是否充分、动态追加追问。
- reflect（Pro）：冲突识别 + 来源溯源 + 证据分级（反思机制，详见 reflect.py）。
- aggregator：合并发现 → 结构化报告（含来源去重 + 证据等级 + 冲突清单）。

D4 相比 D3：doc_worker 单 worker 线性流 → supervisor + 三 worker 并行 + 动态追问
+ 短期 Memory 去重。D5 再叠 reflect 反思层。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .aggregator import aggregator_node
from .planner import planner_node
from .reflect import reflect_node
from .state import AgentState
from .supervisor import supervisor_node


def build_graph(llm, retriever, db, repo: str):
    g = StateGraph(AgentState)
    g.add_node("planner", lambda s: planner_node(s, llm, db, repo))
    g.add_node("supervisor", lambda s: supervisor_node(s, llm, retriever, db, repo))
    g.add_node("reflect", lambda s: reflect_node(s, llm, db, repo))
    g.add_node("aggregator", lambda s: aggregator_node(s))

    g.add_edge(START, "planner")
    g.add_edge("planner", "supervisor")
    g.add_edge("supervisor", "reflect")
    g.add_edge("reflect", "aggregator")
    g.add_edge("aggregator", END)
    return g.compile()
