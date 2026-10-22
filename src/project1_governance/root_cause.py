"""Four-layer root-cause knowledge graph with Bayesian Top-K attribution.

Graph topology (expert-confirmed prior, NOT learned from data)
    operation(42) -> KPP(160) -> KQC(80) -> defect mode(30)

The structure is fixed by FMEA / 5M1E review. Structure learning would need
orders of magnitude more failure cases than a shop floor can supply, so the
project puts expert knowledge in the structure and leaves only CPT parameters
to the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from ..config import (
    CPT_CONCENTRATION,
    DEFECT_MODE_COUNT,
    KPP_COUNT,
    KQC_COUNT,
    OPERATION_COUNT,
)

CAUSE_NAMES = [
    "进给速度漂移", "刀具磨损超限", "装夹压紧力不足", "主轴热伸长",
    "切削液浓度偏低", "毛坯余量不均", "编程抬刀不足", "测头未校准",
    "上工序基准误差累积", "加工顺序不合理",
]

KQC_NAMES = [
    "平面度", "垂直度", "平行度", "位置度", "同轴度",
    "轮廓度", "圆跳动", "表面粗糙度", "壁厚", "孔径",
]

DETECTION_ACTIONS = {
    "进给速度漂移": ("调取该工序 CNC 采集的进给实际值曲线，比对程序指令值", 5),
    "刀具磨损超限": ("查刀具寿命记录的已用分钟数与额定值，检查刃口状态", 10),
    "装夹压紧力不足": ("复核夹具压紧点分布与压紧力设定，检查定位面贴合", 120),
    "主轴热伸长": ("查主轴连续运行小时数与温升记录", 20),
    "切削液浓度偏低": ("现场抽检切削液浓度与流量压力", 15),
    "毛坯余量不均": ("查来料炉批号与毛坯检测报告余量分布", 30),
    "编程抬刀不足": ("回看 CAM 刀轨抬刀段与安全平面设置", 25),
    "测头未校准": ("查测头最近一次校准日期与标准件复测记录", 15),
    "上工序基准误差累积": ("追溯上工序检测报告基准体系与实测偏差", 40),
    "加工顺序不合理": ("评审工艺路线工序排布与去应力安排", 60),
}


@dataclass
class RootCauseGraph:
    graph: nx.DiGraph
    op_nodes: List[str]
    kpp_nodes: List[str]
    kqc_nodes: List[str]
    defect_nodes: List[str]
    cpt_defect_given_kqc: np.ndarray
    cpt_kqc_given_kpp: np.ndarray
    cpt_kpp_given_op: np.ndarray
    prior_op: np.ndarray
    case_count: int = 0
    # boolean CPT masks: which (child, parent) pairs the causal graph actually
    # wires up. Random Dirichlet rows would otherwise hand every defect a little
    # posterior mass on every parameter, and the Top-K ranking would be noise.
    support_defect_kqc: Optional[np.ndarray] = None
    support_kqc_kpp: Optional[np.ndarray] = None
    support_kpp_op: Optional[np.ndarray] = None


@dataclass
class AttributionResult:
    defect: str
    candidates: pd.DataFrame
    evidence_chain: Dict[str, List[str]]
    top_k: int
    posterior_sum: float


def _build_support(graph: nx.DiGraph, child_nodes: Sequence[str],
                   parent_nodes: Sequence[str]) -> np.ndarray:
    """Boolean mask over CPT rows: which parents the graph actually wires to a child.

    Edges are directed cause -> effect, so the source of an edge is the *parent*
    and the target is the *child*.
    """
    child_index = {n: i for i, n in enumerate(child_nodes)}
    parent_index = {n: i for i, n in enumerate(parent_nodes)}
    support = np.zeros((len(child_nodes), len(parent_nodes)), dtype=bool)
    for parent, child in graph.edges():
        ci, pi = child_index.get(child), parent_index.get(parent)
        if ci is not None and pi is not None:
            support[ci, pi] = True
    return support


def _mixture_from_shares(shares: np.ndarray) -> np.ndarray:
    """Blend a within-support distribution with a uniform floor.

    The floor keeps a cause the graph does not wire up at a small but non-zero
    probability, so it can be out-ranked by evidence but never ruled out by
    construction. Rows stay normalised because the two parts sum back to one.
    """
    return shares * CPT_CONCENTRATION + (1.0 - CPT_CONCENTRATION) / shares.shape[1]


def _structure_aware_cpt(rng: np.random.Generator, support: np.ndarray) -> np.ndarray:
    """Row-normalised CPT whose mass sits on the causes the graph connects.

    A plain Dirichlet row treats every cause as a candidate, which spreads the
    posterior so thinly that Top-K is indistinguishable from guessing. Putting
    ``CPT_CONCENTRATION`` of the mass on the wired-up parents restores a ranking
    the graph's topology can actually justify.
    """
    n_rows, n_cols = support.shape
    shares = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        parents = np.flatnonzero(support[i])
        if len(parents) == 0:
            shares[i] = 1.0 / n_cols
            continue
        # A random split of the concentrated mass stands in for CPT rows a shop
        # floor has not yet populated with confirmed cases.
        shares[i, parents] = rng.dirichlet(np.ones(len(parents)) * 2.0)
    return _mixture_from_shares(shares)


def build_graph(rng: np.random.Generator, defect_cases: Optional[pd.DataFrame] = None) -> RootCauseGraph:
    graph = nx.DiGraph()

    op_nodes = [f"OP-{i:02d}" for i in range(1, OPERATION_COUNT + 1)]
    kpp_nodes = [f"KPP-{i:03d}" for i in range(1, KPP_COUNT + 1)]
    kqc_nodes = [f"KQC-{i:03d}" for i in range(1, KQC_COUNT + 1)]
    defect_nodes = [f"DM-{i:02d}" for i in range(1, DEFECT_MODE_COUNT + 1)]

    for n in op_nodes:
        graph.add_node(n, layer="operation", label="工序")
    for n in kpp_nodes:
        graph.add_node(n, layer="kpp", label="关键工艺参数")
    for n in kqc_nodes:
        graph.add_node(n, layer="kqc", label="关键质量特性")
    for n in defect_nodes:
        graph.add_node(n, layer="defect", label="缺陷模式")

    rng_state = np.random.default_rng(11)

    for idx, kpp in enumerate(kpp_nodes):
        src = op_nodes[idx % len(op_nodes)]
        graph.add_edge(src, kpp, relation="包含参数")
        extra = op_nodes[(idx * 7 + 3) % len(op_nodes)]
        if extra != src:
            graph.add_edge(extra, kpp, relation="影响")

    for idx, kqc in enumerate(kqc_nodes):
        graph.add_edge(kpp_nodes[(idx * 3) % len(kpp_nodes)], kqc, relation="决定")
        graph.add_edge(kpp_nodes[(idx * 5 + 11) % len(kpp_nodes)], kqc, relation="决定")

    for idx, defect in enumerate(defect_nodes):
        graph.add_edge(kqc_nodes[idx % len(kqc_nodes)], defect, relation="表征")
        graph.add_edge(kqc_nodes[(idx * 2 + 5) % len(kqc_nodes)], defect, relation="表征")

    support_kpp_op = _build_support(graph, kpp_nodes, op_nodes)
    support_kqc_kpp = _build_support(graph, kqc_nodes, kpp_nodes)
    support_defect_kqc = _build_support(graph, defect_nodes, kqc_nodes)

    cpt_defect_given_kqc = _structure_aware_cpt(rng_state, support_defect_kqc)
    cpt_kqc_given_kpp = _structure_aware_cpt(rng_state, support_kqc_kpp)
    cpt_kpp_given_op = _structure_aware_cpt(rng_state, support_kpp_op)
    prior_op = rng_state.dirichlet(np.ones(OPERATION_COUNT) * 4)

    g = RootCauseGraph(
        graph=graph,
        op_nodes=op_nodes,
        kpp_nodes=kpp_nodes,
        kqc_nodes=kqc_nodes,
        defect_nodes=defect_nodes,
        cpt_defect_given_kqc=cpt_defect_given_kqc,
        cpt_kqc_given_kpp=cpt_kqc_given_kpp,
        cpt_kpp_given_op=cpt_kpp_given_op,
        prior_op=prior_op,
        support_defect_kqc=support_defect_kqc,
        support_kqc_kpp=support_kqc_kpp,
        support_kpp_op=support_kpp_op,
    )

    if defect_cases is not None and len(defect_cases) > 0:
        fit_cpt_from_cases(g, defect_cases, alpha=3.0)

    return g


def fit_cpt_from_cases(graph: RootCauseGraph, defect_cases: pd.DataFrame, alpha: float = 3.0) -> None:
    """Refresh CPT rows from confirmed CAPA cases.

    Laplace smoothing alone would lift every one of the 80 causes toward uniform
    and drown the two the defect is actually wired to, so the evidence is
    projected onto the admissible support before it is normalised: confirmed cases
    re-weight the causes the graph admits, they do not create new ones.
    """
    kqc_index = {n: i for i, n in enumerate(graph.kqc_nodes)}
    defect_index = {n: i for i, n in enumerate(graph.defect_nodes)}
    support = graph.support_defect_kqc

    counts = np.full_like(graph.cpt_defect_given_kqc, alpha)
    for _, row in defect_cases.iterrows():
        d = defect_index.get(row.get("defect_mode"))
        if d is None:
            continue
        admissible = np.flatnonzero(support[d]) if support is not None else None
        causes = row.get("confirmed_causes") or []
        for cause in causes:
            kqc = _cause_to_kqc(cause, graph.kqc_nodes, admissible)
            counts[d, kqc_index[kqc]] += 1.0

    if support is not None:
        counts = counts * support
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    graph.cpt_defect_given_kqc = _mixture_from_shares(shares)
    graph.case_count = int(len(defect_cases))


def _cause_to_kqc(cause: str, kqc_nodes: Sequence[str],
                  admissible: Optional[Sequence[int]] = None) -> str:
    """Map a confirmed cause onto a KQC node, restricted to admissible ones."""
    h = sum(ord(c) for c in str(cause))
    if admissible is not None and len(admissible):
        return kqc_nodes[int(admissible[h % len(admissible)])]
    return kqc_nodes[h % len(kqc_nodes)]


def infer(graph: RootCauseGraph, defect: str, evidence_strength: float = 1.0, top_k: int = 5) -> AttributionResult:
    """Bayesian inversion: rank Key Process Parameters given an observed defect."""
    if defect not in graph.defect_nodes:
        defect = graph.defect_nodes[0]
    d = graph.defect_nodes.index(defect)

    likelihood = graph.cpt_defect_given_kqc[d].copy()
    likelihood = likelihood ** evidence_strength
    # P(kqc): how strongly each key characteristic is implicated across every
    # defect mode. Note cpt_kqc_given_kpp is stored row-normalised as P(kpp|kqc),
    # so averaging its rows would return a constant -- the prior has to come from
    # the defect side of the graph instead, and it must be KQC-shaped (KQC_COUNT)
    # to broadcast against the likelihood.
    prior_kqc = graph.cpt_defect_given_kqc.sum(axis=0)
    prior_kqc = prior_kqc / (prior_kqc.sum() + 1e-12)

    joint = likelihood * prior_kqc
    posterior_kqc = joint / (joint.sum() + 1e-12)

    posterior_kpp = graph.cpt_kqc_given_kpp.T @ posterior_kqc
    posterior_kpp = posterior_kpp / (posterior_kpp.sum() + 1e-12)

    posterior_op = graph.cpt_kpp_given_op.T @ posterior_kpp
    posterior_op = posterior_op / (posterior_op.sum() + 1e-12)

    order = np.argsort(-posterior_kpp)[:top_k]
    rows = []
    for rank, kpp_i in enumerate(order, start=1):
        cause_name = CAUSE_NAMES[kpp_i % len(CAUSE_NAMES)]
        action, minutes = DETECTION_ACTIONS[cause_name]
        # Walk the chain back through the layers that actually carry the mass, so
        # the evidence shown is the path the posterior travelled and not a
        # placeholder assembled from the parameter index.
        kqc_parents = _parents_of(graph.support_kqc_kpp, kpp_i, axis=0)
        kqc_i = int(kqc_parents[np.argmax(posterior_kqc[kqc_parents])])
        op_parents = _parents_of(graph.support_kpp_op, kpp_i, axis=1)
        op_i = int(op_parents[np.argmax(posterior_op[op_parents])])
        rows.append(
            {
                "排名": rank,
                "候选根因": cause_name,
                "关键工艺参数": graph.kpp_nodes[kpp_i],
                "关联工序": graph.op_nodes[op_i],
                "后验概率": f"{posterior_kpp[kpp_i] * 100:.1f}%",
                "证据链": (f"{defect} ← {graph.kqc_nodes[kqc_i]} ← "
                        f"{graph.kpp_nodes[kpp_i]} ← {graph.op_nodes[op_i]}"),
                "建议动作": action,
                "预计耗时(min)": minutes,
                "信息价值": f"{posterior_kpp[kpp_i] / max(minutes, 1) * 1000:.2f}",
            }
        )

    chain: Dict[str, List[str]] = {}
    for node in graph.defect_nodes[: min(5, DEFECT_MODE_COUNT)]:
        chain[node] = _trace_chain(graph, node)

    return AttributionResult(
        defect=defect,
        candidates=pd.DataFrame(rows),
        evidence_chain=chain,
        top_k=top_k,
        posterior_sum=float(posterior_kpp[order].sum()),
    )


def _parents_of(support: Optional[np.ndarray], child: int, axis: int) -> np.ndarray:
    """Indices of the parents the graph wires to `child`.

    `axis=0` means the support table is indexed [parent, child] (KQC -> KPP),
    `axis=1` means [child, parent] (KPP -> OP).
    """
    if support is None:
        return np.array([0])
    parents = np.flatnonzero(support[:, child] if axis == 0 else support[child])
    return parents if len(parents) else np.array([0])


def _trace_chain(graph: RootCauseGraph, defect: str) -> List[str]:
    path = [defect]
    current = defect
    for _ in range(3):
        preds = list(graph.graph.predecessors(current))
        if not preds:
            break
        current = preds[0]
        path.append(current)
    return path[::-1]


def mechanism_cross_check(observed_load_delta: float, predicted_load_delta: float, tol: float = 0.25) -> Dict[str, object]:
    """Compare measured spindle-load signature against the cutting-force / deflection model."""
    error = abs(observed_load_delta - predicted_load_delta) / max(abs(predicted_load_delta), 1e-6)
    consistent = bool(error <= tol)
    return {
        "observed": round(observed_load_delta, 4),
        "predicted": round(predicted_load_delta, 4),
        "relative_error": round(float(error), 4),
        "consistent": consistent,
        "verdict": "确认为参数漂移(工艺性原因)" if consistent else "签名不吻合, 转向操作/装夹因素排查",
    }
