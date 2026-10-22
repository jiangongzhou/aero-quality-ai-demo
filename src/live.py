"""Live inference layer behind the Streamlit panel.

``app.py`` must not retrain, but it *should* be allowed to run the models that
``run_all.py`` produced. This module is that bridge: every function here executes
a real forward pass -- GAT encoder + union-find merge + parameter head, Bayesian
inversion over the causal CPTs, or Shapley pricing with redundancy discount --
on the same objects and seeds the offline pipeline used.

Reproducibility comes from ``outputs/run_meta.parquet``: the panel reads the
``*_seed`` columns recorded by the last ``run_all.py`` invocation and rebuilds
from them, so a visitor clicking through the panel reproduces the parquet on
disk bit for bit instead of getting a "similar looking" second run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common.data_valuation import ValuationResult, redundancy_report, run_valuation
from .config import (
    CAUSAL_SEED,
    NC_IOU_THRESHOLD,
    NC_MERGE_THRESHOLD,
    OUTPUT_DIR,
    RANDOM_SEED,
    SHAPLEY_BUDGET,
    VALUATION_SEED,
    WEIGHT_DIR,
)
from .project1_governance.root_cause import (
    AttributionResult,
    RootCauseGraph,
    build_graph,
    infer,
    mechanism_cross_check,
)
from .project2_nc.brep_graph import build_corpus_graphs, build_part_graph, union_find_merge
from .project2_nc.gat_model import (
    FEED_MAX,
    SPINDLE_MAX,
    GATSegmenter,
    _graph_stats,
    _majority_label,
    softmax,
)

META_PATH = OUTPUT_DIR / "run_meta.parquet"


# ---------------------------------------------------------------------------
# run manifest
# ---------------------------------------------------------------------------


def load_meta() -> Dict[str, object]:
    """Last run's parameters, falling back to the shipped demo defaults."""
    defaults: Dict[str, object] = {
        "run_scale": "demo",
        "part_limit": 80,
        "defect_limit": 150,
        "random_seed": RANDOM_SEED,
        "causal_seed": CAUSAL_SEED,
        "valuation_seed": VALUATION_SEED,
        "gat_seed": RANDOM_SEED % 97,
        "defect_mode": "",
        "gat_mode": "inference",
        "weights_path": str(WEIGHT_DIR / "gat_weights.npz"),
        "elapsed_s": 0.0,
    }
    if not META_PATH.exists():
        return defaults
    frame = pd.read_parquet(META_PATH)
    if frame.empty:
        return defaults
    row = frame.iloc[0].to_dict()
    for key, value in row.items():
        defaults[key] = value
    return defaults


# ---------------------------------------------------------------------------
# project 1 -- Bayesian root-cause attribution
# ---------------------------------------------------------------------------


@dataclass
class CausalBundle:
    """Rebuilt causal graph plus everything needed to re-run attribution."""

    graph: RootCauseGraph
    defect_modes: List[str]
    case_count: int
    build_s: float


def _normalise_cases(cases: pd.DataFrame) -> pd.DataFrame:
    """Undo the parquet round-trip on the ``confirmed_causes`` column.

    Parquet stores the per-case cause list as an Arrow list, so pandas hands it
    back as an ndarray. ``fit_cpt_from_cases`` iterates with ``or []`` to absorb
    missing values, and an ndarray is neither falsy nor iterable-truthy in that
    context -- it raises instead. Converting back to plain lists keeps the fit
    path identical to the in-memory one.
    """
    if "confirmed_causes" not in cases.columns:
        return cases
    cases = cases.copy()
    cases["confirmed_causes"] = [
        [] if value is None or (isinstance(value, float) and np.isnan(value))
        else [str(v) for v in np.atleast_1d(np.asarray(value, dtype=object)).tolist()]
        for value in cases["confirmed_causes"]
    ]
    return cases


def load_causal_bundle() -> CausalBundle:
    """Rebuild the four-layer causal graph exactly as ``run_all.py`` left it.

    ``build_graph`` seeds every CPT row from a Dirichlet draw, so it has to be
    re-entered with the recorded ``causal_seed`` and the same case window; the
    shared pipeline rng is deliberately *not* reused because its state depends
    on how much the stages before it consumed.
    """
    meta = load_meta()
    started = time.perf_counter()
    cases = _normalise_cases(pd.read_parquet(OUTPUT_DIR / "defect_cases.parquet"))
    rng = np.random.default_rng(int(meta["causal_seed"]))
    graph = build_graph(rng, cases)
    bundle = CausalBundle(
        graph=graph,
        defect_modes=list(graph.defect_nodes),
        case_count=graph.case_count,
        build_s=time.perf_counter() - started,
    )
    return bundle


def diagnose(
    bundle: CausalBundle,
    defect: str,
    evidence_strength: float = 1.0,
    top_k: int = 5,
) -> Tuple[AttributionResult, float]:
    """Run Bayesian inversion for one defect mode and report the wall-clock cost."""
    started = time.perf_counter()
    result = infer(bundle.graph, defect, evidence_strength=evidence_strength, top_k=top_k)
    return result, time.perf_counter() - started


def cross_check(observed: float, predicted: float) -> Dict[str, object]:
    """Physics/mechanism cross-validation against the given tolerance."""
    return mechanism_cross_check(observed_load_delta=observed, predicted_load_delta=predicted)


# ---------------------------------------------------------------------------
# project 2 -- GAT machining-region inference
# ---------------------------------------------------------------------------


@dataclass
class GatBundle:
    """Cached weights plus the deterministically rebuilt B-rep corpus."""

    model: GATSegmenter
    corpus: object
    part_codes: List[str]
    mode: str
    weights_path: str
    load_s: float


def load_gat_bundle() -> GatBundle:
    """Load cached GAT weights and rebuild the part graphs they were fit on.

    The corpus is rebuilt here rather than dumped because ``build_part_graph``
    is cheap, pure NumPy, and fully determined by ``gat_seed`` -- so the panel
    gets real embeddings instead of a lookup table of pre-baked predictions.
    """
    meta = load_meta()
    started = time.perf_counter()
    parts = pd.read_parquet(OUTPUT_DIR / "parts.parquet")
    seed = int(meta["gat_seed"])
    corpus = build_corpus_graphs(parts, np.random.default_rng(seed), limit=int(meta["part_limit"]))

    weights_path = Path(str(meta["weights_path"])) if meta["weights_path"] else WEIGHT_DIR / "gat_weights.npz"
    node_dim = corpus.graphs[0].node_features.shape[1]
    edge_dim = corpus.graphs[0].edge_features.shape[1]
    model = GATSegmenter(node_dim, edge_dim, seed=seed)

    mode = "inference"
    if not weights_path.exists() or not model.load(weights_path):
        weights_path = WEIGHT_DIR / "gat_weights.npz"
        mode = "inference" if (weights_path.exists() and model.load(weights_path)) else "untrained"

    return GatBundle(
        model=model,
        corpus=corpus,
        part_codes=[g.part_code for g in corpus.graphs],
        mode=mode,
        weights_path=str(weights_path),
        load_s=time.perf_counter() - started,
    )


def _graph_by_code(bundle: GatBundle, part_code: str):
    for g in bundle.corpus.graphs:
        if g.part_code == part_code:
            return g
    raise KeyError(part_code)


def predict_part(
    bundle: GatBundle,
    part_code: str,
    merge_threshold: float = NC_MERGE_THRESHOLD,
) -> Dict[str, object]:
    """Full forward pass for one part: encode -> edge probs -> merge -> parameters.

    Returns the per-region table plus the intermediate tensors the panel plots,
    and the wall-clock time so the visitor can see it really ran.
    """
    started = time.perf_counter()
    graph = _graph_by_code(bundle, part_code)

    h = bundle.model.encode(graph, training=False)
    logits = bundle.model.edge_logits(h, graph)
    probs = softmax(logits)
    pred_edges = (probs[:, 1] >= NC_IOU_THRESHOLD).astype(int)

    labels, n_regions = union_find_merge(graph.edge_index, probs[:, 1], graph.n_faces, threshold=merge_threshold)
    gt_labels, _ = union_find_merge(graph.edge_index, graph.edge_label, graph.n_faces)

    pooled, counts = GATSegmenter.pool(h, labels)
    stats = _graph_stats(graph)
    pred_params = bundle.model.region_params(pooled, graph.material, GATSegmenter.pool(stats, labels)[0])
    truth_labels = _majority_label(labels, graph.region_id, int(labels.max()) + 1)

    rows = [
        {
            "零件号": graph.part_code,
            "类别": graph.category,
            "材料": graph.material,
            "预测区域": f"R{k:02d}",
            "面数": int(counts[k]),
            "预测转速": round(float(pred_params[k, 0] * SPINDLE_MAX), 0),
            "参考转速": round(float(graph.region_params[truth_labels[k], 0]), 0),
            "预测进给": round(float(pred_params[k, 1] * FEED_MAX), 0),
            "参考进给": round(float(graph.region_params[truth_labels[k], 1]), 0),
        }
        for k in range(len(pooled))
    ]

    elapsed = time.perf_counter() - started
    return {
        "table": pd.DataFrame(rows),
        "part_code": graph.part_code,
        "category": graph.category,
        "material": graph.material,
        "n_faces": int(graph.n_faces),
        "n_edges": int(graph.edge_index.shape[1]),
        "n_regions": int(n_regions),
        "edge_confidence": probs[:, 1],
        "pred_edges": pred_edges,
        "true_edges": graph.edge_label,
        "pred_labels": labels,
        "true_labels": gt_labels,
        "elapsed_s": elapsed,
    }


def summarise_prediction(pred: Dict[str, object]) -> Dict[str, float]:
    """Edge-level and parameter-level accuracy for a single predicted part."""
    table = pred["table"]
    pe = np.asarray(pred["pred_edges"])
    te = np.asarray(pred["true_edges"])
    pl = np.asarray(pred["pred_labels"])
    gl = np.asarray(pred["true_labels"])

    tp = int(((pe == 1) & (te == 1)).sum())
    fp = int(((pe == 1) & (te == 0)).sum())
    fn = int(((pe == 0) & (te == 1)).sum())
    iou = _pairwise_iou(pl, gl)

    sp = table["预测转速"].to_numpy(float)
    sr = table["参考转速"].to_numpy(float)
    fd = table["预测进给"].to_numpy(float)
    fr = table["参考进给"].to_numpy(float)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "边界准确率": float(np.mean(pe == te)),
        "边界精确率": round(precision, 4),
        "边界召回率": round(recall, 4),
        "区域IoU": round(float(iou), 4),
        "预测区域数": float(pred["n_regions"]),
        "转速MAPE": round(float(np.mean(np.abs(sp - sr) / np.maximum(np.abs(sr), 1e-6))), 4),
        "进给MAPE": round(float(np.mean(np.abs(fd - fr) / np.maximum(np.abs(fr), 1e-6))), 4),
        "推理耗时ms": round(float(pred["elapsed_s"]) * 1000, 1),
    }


def _pairwise_iou(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Mean best-overlap IoU between predicted and ground-truth region partitions."""
    preds = np.unique(pred_labels)
    gts = np.unique(gt_labels)
    best = []
    for p in preds:
        mask_p = pred_labels == p
        scores = []
        for g in gts:
            mask_g = gt_labels == g
            inter = float((mask_p & mask_g).sum())
            union = float((mask_p | mask_g).sum())
            scores.append(inter / union if union else 0.0)
        best.append(max(scores) if scores else 0.0)
    return float(np.mean(best)) if best else 0.0


# ---------------------------------------------------------------------------
# project 1 -- Shapley data valuation
# ---------------------------------------------------------------------------


def load_valuation_pool() -> np.ndarray:
    """The mined route embedding the offline run priced."""
    return pd.read_parquet(OUTPUT_DIR / "route_embedding.parquet").to_numpy(float)


def price_pool(pool: np.ndarray, budget: int) -> Tuple[ValuationResult, float]:
    """Re-price the whole pool for a new budget: LOO -> Shapley -> greedy selection."""
    meta = load_meta()
    started = time.perf_counter()
    result = run_valuation(pool, budget=budget, rng=np.random.default_rng(int(meta["valuation_seed"])))
    return result, time.perf_counter() - started


def valuation_frame(result: ValuationResult) -> pd.DataFrame:
    """Per-sample pricing in the same shape as ``outputs/valuation.parquet``."""
    mask = np.zeros(len(result.aggregated), dtype=bool)
    mask[result.selected] = True
    return pd.DataFrame({
        "样本编号": np.arange(len(result.aggregated)),
        "Shapley值": result.values,
        "冗余折价后价值": result.aggregated,
        "是否入选": mask,
    })


def valuation_report(result: ValuationResult, pool: np.ndarray) -> pd.DataFrame:
    return redundancy_report(result, pool)
