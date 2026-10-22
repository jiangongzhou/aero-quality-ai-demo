"""GAT machining-region segmentation + physics-informed parameter regression.

Pure NumPy implementation, no deep-learning framework is required.

Pipeline
    B-rep face graph
        -> node encoder (24-d -> heads * hidden)
        -> 3 x multi-head graph attention (GAT)
        -> edge  logits            -> union-find merge -> machining region
        -> region mean pooling     -> (spindle speed, feed rate)

Default demo path is **inference only**: the weights are generated once by
``python -m src.project2_nc.gat_model --train`` and cached under
``outputs/weights/gat_weights.npz``. If the cache is missing, the demo falls
back to training a small model so ``run_all.py`` always works out of the box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..config import (
    NC_FOCAL_ALPHA,
    NC_FOCAL_GAMMA,
    NC_GAT_DROPOUT,
    NC_GAT_HEADS,
    NC_GAT_HIDDEN,
    NC_GAT_LR,
    NC_GAT_LR_DECAY,
    NC_GAT_LAYERS,
    NC_GRAD_CLIP,
    NC_IOU_THRESHOLD,
    NC_MATERIAL_FACTOR,
    NC_MATERIALS,
    NC_MERGE_THRESHOLD,
    NC_PHYSICS_LAMBDA,
    NC_REGION_STAT_DIM,
    NC_SPECIFIC_CUTTING_FORCE,
    NC_TEST_FRACTION,
    NC_VAL_FRACTION,
    NC_SURROGATE_EPOCHS,
    WEIGHT_DIR,
)
from .brep_graph import BRepGraph, GraphCorpus, build_corpus_graphs, union_find_merge

# ---------------------------------------------------------------------------
# process physics: cutting power envelope and chip-load window
# ---------------------------------------------------------------------------

TOOL_DIAMETER_MM = 10.0
TOOL_TEETH = 4
AXIAL_DEPTH_MM = 0.35
RADIAL_DEPTH_MM = 1.2
RATED_POWER_KW = 22.0
CHIP_LOAD_BAND = (0.010, 0.220)

SPECIFIC_CUTTING_FORCE = NC_SPECIFIC_CUTTING_FORCE

_MATERIAL_INDEX = {name: i for i, name in enumerate(NC_MATERIALS)}

KC_MAX = max(NC_SPECIFIC_CUTTING_FORCE.values())
MF_MAX = max(NC_MATERIAL_FACTOR.values())

SPINDLE_MAX = 20000.0
FEED_MAX = 3200.0


def _graph_stats(graph: BRepGraph) -> np.ndarray:
    """Per-face statistics in absolute process units (falls back to zeros)."""
    if getattr(graph, "node_stats", None) is None:
        return np.zeros((graph.n_faces, NC_REGION_STAT_DIM))
    return np.asarray(graph.node_stats, dtype=float)


def material_context(material: str) -> np.ndarray:
    """Part-level context vector handed to the parameter head.

    Machining parameters are a function of *what is being cut* as much as of the
    geometry, and material lives on the part rather than on any single face, so it
    is concatenated after region pooling instead of being a node feature.

    Layout: one-hot material (len(NC_MATERIALS)) + normalised specific cutting
    force + normalised machinability factor.
    """
    onehot = np.zeros(len(NC_MATERIALS), dtype=float)
    idx = _MATERIAL_INDEX.get(material)
    if idx is not None:
        onehot[idx] = 1.0
    kc = NC_SPECIFIC_CUTTING_FORCE.get(material, 800.0) / KC_MAX
    mfg = NC_MATERIAL_FACTOR.get(material, 1.0) / MF_MAX
    return np.concatenate([onehot, [kc, mfg]])


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(pre: np.ndarray) -> np.ndarray:
    return (pre > 0).astype(float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _leaky(x: np.ndarray, slope: float = 0.2) -> np.ndarray:
    return np.where(x > 0, x, slope * x)


def _leaky_grad(pre: np.ndarray, slope: float = 0.2) -> np.ndarray:
    return np.where(pre > 0, 1.0, slope)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(shifted)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)


def focal_grad(probs: np.ndarray, labels: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    """dL/dlogits for the focal loss ``-alpha_t (1 - p_t)^gamma log p_t``."""
    n, c = probs.shape
    rows = np.arange(n)
    pt = np.clip(probs[rows, labels], 1e-8, 1.0)
    w = np.where(labels == 1, alpha, 1.0 - alpha)
    dpt = w * gamma * ((1.0 - pt) ** (gamma - 1.0)) * np.log(pt) - w * ((1.0 - pt) ** gamma) / pt
    onehot = np.zeros_like(probs)
    onehot[rows, labels] = 1.0
    return (dpt * pt)[:, None] * (onehot - probs)


def focal_loss(probs: np.ndarray, labels: np.ndarray, gamma: float, alpha: float) -> float:
    rows = np.arange(len(labels))
    pt = np.clip(probs[rows, labels], 1e-8, 1.0)
    w = np.where(labels == 1, alpha, 1.0 - alpha)
    return float((-w * ((1.0 - pt) ** gamma) * np.log(pt)).mean())


def physics_residual(spindle: np.ndarray, feed: np.ndarray, materials: np.ndarray) -> Dict[str, np.ndarray]:
    """Violation of the machine envelope for a batch of predicted parameters.

    Two physical gates are used, both of which are standard in shop-floor CAM
    review:

    * spindle power  ``P = kc * ap * ae * vf / 60000`` must stay under the
      rated power of the machining centre;
    * chip load      ``fz = vf / (n * z)`` must stay inside the cutter window,
      otherwise the tool either rubs (work hardening) or chips.
    """
    spindle = np.asarray(spindle, dtype=float)
    feed = np.asarray(feed, dtype=float)
    kc = np.array([SPECIFIC_CUTTING_FORCE.get(m, 800.0) for m in materials], dtype=float)

    power_kw = kc * AXIAL_DEPTH_MM * RADIAL_DEPTH_MM * feed / 60000.0
    power_over = np.maximum(0.0, power_kw / RATED_POWER_KW - 1.0)

    chip_load = feed / np.maximum(spindle, 1.0) / TOOL_TEETH
    chip_low = np.maximum(0.0, CHIP_LOAD_BAND[0] - chip_load) / CHIP_LOAD_BAND[0]
    chip_high = np.maximum(0.0, chip_load - CHIP_LOAD_BAND[1]) / CHIP_LOAD_BAND[1]

    violation = (power_over > 0.0) | (chip_low > 0.0) | (chip_high > 0.0)
    return {
        "power_kw": power_kw,
        "power_over": power_over,
        "chip_load": chip_load,
        "violation": violation,
        "residual": power_over + chip_low + chip_high,
    }


def physics_grad(spindle: np.ndarray, feed: np.ndarray, materials: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Analytic ``d(mean residual) / d(spindle, feed)`` of the two gates above.

    Both gates are piecewise linear, so the gradient is simply the active
    sub-gradient of whichever side of the envelope the sample sits on:

    * power      ``dP/dvf = kc * ap * ae / 60000``   (only when over rated)
    * chip load  ``fz = vf / (n z)`` -> ``dfz/dvf = 1/(n z)``,
                 ``dfz/dn  = -vf / (n^2 z)``         (only when outside band)
    """
    spindle = np.maximum(np.asarray(spindle, dtype=float), 1.0)
    feed = np.asarray(feed, dtype=float)
    n = max(len(spindle), 1)
    kc = np.array([SPECIFIC_CUTTING_FORCE.get(m, 800.0) for m in materials], dtype=float)

    power_kw = kc * AXIAL_DEPTH_MM * RADIAL_DEPTH_MM * feed / 60000.0
    g_over_feed = np.where(power_kw > RATED_POWER_KW,
                           kc * AXIAL_DEPTH_MM * RADIAL_DEPTH_MM / 60000.0 / RATED_POWER_KW, 0.0)

    fz = feed / spindle / TOOL_TEETH
    lo, hi = CHIP_LOAD_BAND
    g_res_fz = np.where(fz < lo, -1.0 / lo, 0.0) + np.where(fz > hi, 1.0 / hi, 0.0)

    g_feed = (g_over_feed + g_res_fz / (spindle * TOOL_TEETH)) / n
    g_spindle = (g_res_fz * (-feed / (spindle ** 2 * TOOL_TEETH))) / n
    return g_spindle, g_feed


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------


class Linear:
    def __init__(self, fin: int, fout: int, rng: np.random.Generator, scale: Optional[float] = None):
        scale = scale if scale is not None else 1.0 / np.sqrt(max(fin, 1))
        self.w = rng.normal(0.0, scale, (fin, fout))
        self.b = np.zeros(fout)
        self.gw = np.zeros_like(self.w)
        self.gb = np.zeros_like(self.b)
        self.act: Optional[str] = None

    def forward(self, x: np.ndarray, activation: Optional[str] = None) -> np.ndarray:
        self.x = x
        self.act = activation
        self.pre = x @ self.w + self.b
        if activation == "relu":
            self.out = _relu(self.pre)
        elif activation == "sigmoid":
            self.out = _sigmoid(self.pre)
        else:
            self.out = self.pre
        return self.out

    def backward(self, gout: np.ndarray) -> np.ndarray:
        if self.act == "relu":
            g = gout * _relu_grad(self.pre)
        elif self.act == "sigmoid":
            g = gout * (self.out * (1.0 - self.out))
        else:
            g = gout
        np.add(self.gw, self.x.T @ g, out=self.gw)
        np.add(self.gb, g.sum(axis=0), out=self.gb)
        return g @ self.w.T


class GATConv:
    """Multi-head graph attention on a directed edge list (message src -> dst).

    The scatter/gather steps are expressed as sparse incidence matrices so the
    whole layer is a handful of dense matmuls, which keeps NumPy fast enough
    for the ~200-face parts used in this demo.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, rng: np.random.Generator, slope: float = 0.2):
        self.in_dim, self.out_dim, self.heads, self.slope = in_dim, out_dim, heads, slope
        self.w = rng.normal(0.0, 1.0 / np.sqrt(in_dim), (in_dim, heads * out_dim))
        self.a_src = rng.normal(0.0, 1.0 / np.sqrt(out_dim), (heads, out_dim))
        self.a_dst = rng.normal(0.0, 1.0 / np.sqrt(out_dim), (heads, out_dim))
        self.bias = np.zeros((heads, out_dim))
        self.gw = np.zeros_like(self.w)
        self.ga_src = np.zeros_like(self.a_src)
        self.ga_dst = np.zeros_like(self.a_dst)
        self.gbias = np.zeros_like(self.bias)

    def forward(self, h: np.ndarray, edge_index: np.ndarray, incidence) -> np.ndarray:
        s_dst, s_src = incidence
        src, dst = edge_index[0], edge_index[1]
        n_nodes, n_edges = h.shape[0], src.shape[0]

        self.h, self.src, self.dst = h, src, dst
        self.s_dst, self.s_src = s_dst, s_src

        z = h @ self.w
        Z = np.ascontiguousarray(z.reshape(n_nodes, self.heads, self.out_dim).transpose(1, 0, 2))
        self.Z = Z

        Zs, Zd = Z[:, src, :], Z[:, dst, :]
        self.Zs, self.Zd = Zs, Zd

        self.e_pre = np.einsum("heo,ho->he", Zs, self.a_src) + np.einsum("heo,ho->he", Zd, self.a_dst)
        e = _leaky(self.e_pre, self.slope)

        peak = np.full((self.heads, n_nodes), -np.inf)
        for head in range(self.heads):
            np.maximum.at(peak[head], dst, e[head])
        ex = np.exp(e - peak[:, dst])
        denom = np.zeros((self.heads, n_nodes))
        for head in range(self.heads):
            np.add.at(denom[head], dst, ex[head])
        alpha = ex / (denom[:, dst] + 1e-8)
        self.alpha = alpha

        m = np.ascontiguousarray((alpha[:, :, None] * Zs).transpose(1, 0, 2)).reshape(n_edges, -1)
        self.m = m

        self.pre = s_dst @ m + self.bias.reshape(1, -1)
        return _relu(self.pre)

    def backward(self, gout: np.ndarray) -> np.ndarray:
        h, heads, out_dim = self.heads, self.heads, self.out_dim
        H, O = heads, out_dim
        n_nodes = self.Z.shape[1]

        dpre = gout * _relu_grad(self.pre)
        np.add(self.gbias, dpre.reshape(n_nodes, H, O).sum(axis=0), out=self.gbias)

        dm = (self.s_dst.T @ dpre).reshape(-1, H, O).transpose(1, 0, 2)
        dalpha = np.einsum("heo,heo->he", dm, self.Zs)
        dzs_msg = dm * self.alpha[:, :, None]

        ssum = (self.s_dst @ (self.alpha * dalpha).T)[self.dst].T
        de = self.alpha * (dalpha - ssum)
        de_pre = de * _leaky_grad(self.e_pre, self.slope)

        np.add(self.ga_src, np.einsum("heo,he->ho", self.Zs, de_pre), out=self.ga_src)
        np.add(self.ga_dst, np.einsum("heo,he->ho", self.Zd, de_pre), out=self.ga_dst)

        dzs_att = de_pre[:, :, None] * self.a_src[:, None, :]
        dzd = de_pre[:, :, None] * self.a_dst[:, None, :]

        gz = np.zeros((H, n_nodes, O))
        for head in range(H):
            np.add.at(gz[head], self.src, dzs_msg[head] + dzs_att[head])
            np.add.at(gz[head], self.dst, dzd[head])

        gz_flat = np.ascontiguousarray(gz.transpose(1, 0, 2)).reshape(n_nodes, H * O)
        np.add(self.gw, self.h.T @ gz_flat, out=self.gw)
        return gz_flat @ self.w.T


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class GATSegmenter:
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int = NC_GAT_HIDDEN,
        heads: int = NC_GAT_HEADS,
        layers: int = NC_GAT_LAYERS,
        seed: int = 17,
    ):
        rng = np.random.default_rng(seed)
        self.hidden, self.heads, self.layers = hidden, heads, layers
        dim = hidden * heads
        self.ctx_dim = len(NC_MATERIALS) + 2
        # extra block concatenated to the pooled embedding: part-level material
        # context plus region-level statistics in absolute process units
        self.extra_dim = self.ctx_dim + NC_REGION_STAT_DIM
        self.encoder = Linear(node_dim, dim, rng)
        self.convs = [GATConv(dim, hidden, heads, rng) for _ in range(layers)]
        self.edge_1 = Linear(dim * 3 + edge_dim, 64, rng)
        self.edge_2 = Linear(64, 2, rng)
        self.reg_1 = Linear(dim + self.extra_dim, 64, rng)
        self.reg_2 = Linear(64, 2, rng)

    # -- graph encoding -----------------------------------------------------
    def encode(self, graph: BRepGraph, rng: Optional[np.random.Generator] = None, training: bool = False) -> np.ndarray:
        h = self.encoder.forward(graph.node_features, activation="relu")
        # Repeated attention pooling washes out the per-face identity that both
        # heads depend on: a plain GBM reads the raw features at AUC 0.95 while
        # the 3-layer embedding only carries 0.73 of it. Keeping the input as a
        # skip connection gives the heads a loss-free path back to the geometry.
        identity = h
        # A face with no incoming edge would keep its zero pre-activation and
        # collapse to an all-zero embedding, so every face gets a self-loop.
        loops = np.arange(graph.n_faces)
        edge_index = np.concatenate([graph.edge_index, np.stack([loops, loops])], axis=1)
        incidence = build_incidence(graph.n_faces, edge_index)
        self._masks: List[np.ndarray] = []
        for conv in self.convs:
            h = conv.forward(h, edge_index, incidence)
            if training and rng is not None:
                mask = (rng.random(h.shape) < NC_GAT_DROPOUT) / (1.0 - NC_GAT_DROPOUT)
            else:
                mask = np.ones(h.shape)
            self._masks.append(mask)
            h = h * mask
        self._res_pre = h + identity
        return _relu(self._res_pre)

    def encode_backward(self, gout: np.ndarray) -> None:
        gout = gout * _relu_grad(self._res_pre)
        gidentity = gout
        for conv, mask in zip(reversed(self.convs), reversed(self._masks)):
            gout = conv.backward(gout * mask)
        self.encoder.backward(gout + gidentity)

    # -- heads --------------------------------------------------------------
    def edge_logits(self, h: np.ndarray, graph: BRepGraph) -> np.ndarray:
        src, dst = graph.edge_index
        feat = np.concatenate([h[src], h[dst], np.abs(h[src] - h[dst]), graph.edge_features], axis=1)
        return self.edge_2.forward(self.edge_1.forward(feat, activation="relu"))

    def edge_backward(self, glogits: np.ndarray, h: np.ndarray, graph: BRepGraph) -> np.ndarray:
        src, dst = graph.edge_index
        gh = np.zeros_like(h)
        feat = np.concatenate([h[src], h[dst], np.abs(h[src] - h[dst]), graph.edge_features], axis=1)
        ga = self.edge_2.backward(glogits)
        gf = self.edge_1.backward(ga)
        d = feat.shape[1]
        step = (d - graph.edge_features.shape[1]) // 3
        ghs, ghd, gabs, _ = np.split(gf, [step, 2 * step, 3 * step], axis=1)
        sign = np.sign(h[src] - h[dst])
        np.add.at(gh, src, ghs + gabs * sign)
        np.add.at(gh, dst, ghd - gabs * sign)
        return gh

    @staticmethod
    def pool(h: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(labels.max()) + 1 if len(labels) else 0
        pooled = np.zeros((n, h.shape[1]))
        counts = np.bincount(labels, minlength=n).astype(float)
        np.add.at(pooled, labels, h)
        pooled /= np.maximum(counts, 1.0)[:, None]
        return pooled, counts

    def region_params(
        self,
        pooled: np.ndarray,
        material: str,
        stats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict normalised (spindle, feed) for every pooled region.

        The head is hybrid on purpose: it reads the learned region embedding
        *and* explicit region statistics. Parameters are a function of absolute
        cavity depth and required surface finish, and a pooled GAT embedding over
        per-part z-scored features erases exactly that absolute scale, so the
        statistics are what makes the target recoverable.
        """
        # targets are normalised into (0, 1) by SPINDLE_MAX / FEED_MAX, so the
        # head emits a sigmoid instead of an unbounded linear output
        ctx = np.tile(material_context(material), (pooled.shape[0], 1))
        blocks = [pooled, ctx]
        if stats is None:
            stats = np.zeros((pooled.shape[0], NC_REGION_STAT_DIM))
        blocks.append(stats)
        feat = np.concatenate(blocks, axis=1)
        return self.reg_2.forward(self.reg_1.forward(feat, activation="relu"), activation="sigmoid")

    def region_backward(self, gpred: np.ndarray) -> np.ndarray:
        g = self.reg_2.backward(gpred)
        return self.reg_1.backward(g)

    # -- plumbing -----------------------------------------------------------
    @property
    def weight_blocks(self) -> List[np.ndarray]:
        blocks = [self.encoder.w, self.encoder.b, self.edge_1.w, self.edge_1.b, self.edge_2.w, self.edge_2.b,
                  self.reg_1.w, self.reg_1.b, self.reg_2.w, self.reg_2.b]
        for conv in self.convs:
            blocks += [conv.w, conv.a_src, conv.a_dst, conv.bias]
        return blocks

    @property
    def grad_blocks(self) -> List[np.ndarray]:
        blocks = [self.encoder.gw, self.encoder.gb, self.edge_1.gw, self.edge_1.gb, self.edge_2.gw, self.edge_2.gb,
                  self.reg_1.gw, self.reg_1.gb, self.reg_2.gw, self.reg_2.gb]
        for conv in self.convs:
            blocks += [conv.gw, conv.ga_src, conv.ga_dst, conv.gbias]
        return blocks

    def save(self, path) -> str:
        path = str(path)
        weights = {f"w{i}": arr for i, arr in enumerate(self.weight_blocks)}
        np.savez(path, **weights)
        return path

    def load(self, path) -> bool:
        data = np.load(str(path))
        blocks = self.weight_blocks
        if len(data.files) != len(blocks):
            return False
        for i, arr in enumerate(blocks):
            stored = data[f"w{i}"]
            if stored.shape != arr.shape:
                return False
            np.copyto(arr, stored)
        return True


class Adam:
    def __init__(self, weights: List[np.ndarray], grads: List[np.ndarray], lr: float = 1e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8,
                 clip_norm: float = NC_GRAD_CLIP):
        self.weights, self.grads = weights, grads
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.clip_norm = clip_norm
        self.m = [np.zeros_like(w) for w in weights]
        self.v = [np.zeros_like(w) for w in weights]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        # global-gradient-norm clipping: the edge head and the regression head
        # share the encoder, so one exploding term would destroy both
        total = float(np.sqrt(sum(float(np.sum(g * g)) for g in self.grads)))
        if np.isfinite(total) and total > self.clip_norm:
            scale = self.clip_norm / total
            for g in self.grads:
                g *= scale
        for i, (w, g) in enumerate(zip(self.weights, self.grads)):
            if not np.isfinite(g).all():
                g = np.nan_to_num(g, nan=0.0, posinf=self.clip_norm, neginf=-self.clip_norm)
            self.m[i] *= self.b1
            self.m[i] += (1.0 - self.b1) * g
            self.v[i] *= self.b2
            self.v[i] += (1.0 - self.b2) * (g * g)
            mhat = self.m[i] / (1.0 - self.b1 ** self.t)
            vhat = self.v[i] / (1.0 - self.b2 ** self.t)
            np.subtract(w, self.lr * mhat / (np.sqrt(vhat) + self.eps), out=w)
            g.fill(0.0)


def build_incidence(n_nodes: int, edge_index: np.ndarray):
    """(N, E) sparse incidence matrices that gather edge rows by dst and by src."""
    n_edges = edge_index.shape[1]
    rows = np.arange(n_edges)
    data = np.ones(n_edges)
    s_dst = sp.csr_matrix((data, (edge_index[1], rows)), shape=(n_nodes, n_edges))
    s_src = sp.csr_matrix((data, (edge_index[0], rows)), shape=(n_nodes, n_edges))
    return s_dst, s_src


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def prf(pred: np.ndarray, label: np.ndarray) -> Dict[str, float]:
    tp = float(((pred == 1) & (label == 1)).sum())
    fp = float(((pred == 1) & (label == 0)).sum())
    fn = float(((pred == 0) & (label == 1)).sum())
    tn = float(((pred == 0) & (label == 0)).sum())
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
    }


def _region_iou(pred_labels: np.ndarray, gt_labels: np.ndarray) -> float:
    """Mean IoU over predicted regions: for each predicted region take the best
    matching ground-truth region and average the pairwise IoU."""
    total = 0.0
    n_pred = int(pred_labels.max()) + 1 if len(pred_labels) else 0
    n_gt = int(gt_labels.max()) + 1 if len(gt_labels) else 0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    for k in range(n_pred):
        a = pred_labels == k
        if not a.any():
            continue
        best = 0.0
        for j in range(n_gt):
            b = gt_labels == j
            if not b.any():
                continue
            inter = float((a & b).sum())
            union = float((a | b).sum())
            best = max(best, inter / max(union, 1.0))
        total += best
    return total / n_pred


def mape(pred: np.ndarray, truth: np.ndarray) -> float:
    truth = np.maximum(np.asarray(truth, dtype=float), 1e-6)
    return float(np.mean(np.abs(np.asarray(pred, dtype=float) - truth) / truth))


def r2(pred: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-8)


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


@dataclass
class NCResult:
    history: List[Dict[str, float]] = field(default_factory=list)
    edge_metrics: Dict[str, float] = field(default_factory=dict)
    param_metrics: Dict[str, float] = field(default_factory=dict)
    region_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    physics: Dict[str, float] = field(default_factory=dict)
    graph_stats: Dict[str, float] = field(default_factory=dict)
    weights_path: str = ""
    mode: str = "inference"


def _train_epoch(model: GATSegmenter, graphs: List[BRepGraph], optimizer: Adam, rng: np.random.Generator,
                 focal_alpha: float, phys_lambda: float):
    order = rng.permutation(len(graphs))
    totals = {"edge": 0.0, "reg": 0.0, "phys": 0.0}
    for gi in order:
        g = graphs[gi]
        h = model.encode(g, rng, training=True)
        logits = model.edge_logits(h, g)
        probs = softmax(logits)
        loss_edge = focal_loss(probs, g.edge_label, NC_FOCAL_GAMMA, focal_alpha)
        glogits = focal_grad(probs, g.edge_label, NC_FOCAL_GAMMA, focal_alpha) / len(g.edge_label)
        gh = model.edge_backward(glogits, h, g)

        # teacher forcing for the regression branch: early in training the
        # predicted merge is close to random, so pooling over it would average
        # unrelated regions together and leave the head with no learnable
        # signal. Optimising on the true partition isolates the head; the
        # end-to-end number is reported separately at evaluation time.
        gt = g.region_id
        n_gt = int(gt.max()) + 1
        pooled, _ = GATSegmenter.pool(h, gt)
        pooled_stats, _ = GATSegmenter.pool(_graph_stats(g), gt)
        pred_params = model.region_params(pooled, g.material, pooled_stats)
        target = g.region_params[:n_gt] / np.array([SPINDLE_MAX, FEED_MAX])

        diff = pred_params - target
        loss_reg = float((diff ** 2).mean())
        gpred = 2.0 * diff / max(pred_params.size, 1)

        phys = physics_residual(pred_params[:, 0] * SPINDLE_MAX, pred_params[:, 1] * FEED_MAX,
                                np.array([g.material] * len(pooled)))
        loss_phys = float(phys["residual"].mean())
        if phys_lambda > 0.0 and len(pooled):
            # chain rule back to the normalised head output
            g_sp, g_fd = physics_grad(pred_params[:, 0] * SPINDLE_MAX,
                                      pred_params[:, 1] * FEED_MAX,
                                      np.array([g.material] * len(pooled)))
            gpred = gpred + phys_lambda * np.stack([g_sp * SPINDLE_MAX, g_fd * FEED_MAX], axis=1)

        gpooled = model.region_backward(gpred)
        # the context block is a constant, only the pooled-embedding block flows back
        gh = gh + gpooled[:, :h.shape[1]][gt] / np.maximum(
            np.bincount(gt, minlength=n_gt)[gt][:, None], 1.0)

        model.encode_backward(gh)
        optimizer.step()

        totals["edge"] += loss_edge
        totals["reg"] += loss_reg
        totals["phys"] += loss_phys

    n = max(len(graphs), 1)
    return {k: v / n for k, v in totals.items()}


def _majority_label(pred_labels: np.ndarray, gt_region: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros(n_groups, dtype=int)
    for k in range(n_groups):
        members = gt_region[pred_labels == k]
        if len(members) == 0:
            out[k] = int(gt_region[0] % max(len(gt_region), 1)) if len(gt_region) else 0
            continue
        out[k] = int(np.bincount(np.maximum(members, 0)).argmax())
    return out


def train_model(corpus: GraphCorpus, epochs: int = NC_SURROGATE_EPOCHS, lr: float = NC_GAT_LR,
                seed: int = 17, verbose: bool = True) -> Tuple[GATSegmenter, List[Dict[str, float]]]:
    graphs = corpus.graphs
    train, val, _ = _part_level_split(graphs, seed)
    model = GATSegmenter(
        node_dim=graphs[0].node_features.shape[1],
        edge_dim=graphs[0].edge_features.shape[1],
        seed=seed,
    )
    optimizer = Adam(model.weight_blocks, model.grad_blocks, lr=lr)
    rng = np.random.default_rng(seed + 101)

    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        # cosine decay towards NC_GAT_LR_DECAY * lr keeps the shared encoder
        # from drifting once both heads start to disagree
        progress = (epoch - 1) / max(epochs - 1, 1)
        optimizer.lr = lr * (NC_GAT_LR_DECAY + (1.0 - NC_GAT_LR_DECAY)
                             * 0.5 * (1.0 + np.cos(np.pi * progress)))
        losses = _train_epoch(model, train, optimizer, rng, focal_alpha=NC_FOCAL_ALPHA,
                              phys_lambda=NC_PHYSICS_LAMBDA)
        if epoch % 20 == 0 or epoch == 1:
            metrics = evaluate_model(model, val, physics_lambda=NC_PHYSICS_LAMBDA)
            row = {"epoch": float(epoch), **losses,
                   "val_iou": metrics["edge_metrics"]["iou"],
                   "val_acc": metrics["edge_metrics"]["accuracy"]}
            history.append(row)
            if verbose:
                print(f"  [GAT] epoch {epoch:4d}  focal={losses['edge']:.4f} "
                      f"mse={losses['reg']:.4f} phys={losses['phys']:.4f} "
                      f"val_iou={row['val_iou']:.3f} val_acc={row['val_acc']:.3f}")
    return model, history


def _part_level_split(graphs: List[BRepGraph], seed: int):
    """Disjoint part-level train / val / test.

    Every face of a given part lands in exactly one bucket. The two heads are
    then evaluated on parts the encoder has never seen, which is the setting the
    reported accuracy has to hold in.
    """
    idx = np.random.default_rng(seed).permutation(len(graphs))
    n_test = max(1, int(len(graphs) * NC_TEST_FRACTION))
    n_val = max(1, int(len(graphs) * NC_VAL_FRACTION))
    n_val = min(n_val, len(graphs) - n_test - 1)
    test_idx, val_idx, train_idx = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return ([graphs[i] for i in train_idx],
            [graphs[i] for i in val_idx],
            [graphs[i] for i in test_idx])


def evaluate_model(model: GATSegmenter, graphs: List[BRepGraph], physics_lambda: float = NC_PHYSICS_LAMBDA) -> Dict:
    all_pred, all_true, all_score = [], [], []
    ious, region_rows, teacher_rows = [], [], []

    for g in graphs:
        h = model.encode(g, training=False)
        logits = model.edge_logits(h, g)
        probs = softmax(logits)
        pred = (probs[:, 1] >= NC_IOU_THRESHOLD).astype(int)

        all_pred.append(pred)
        all_true.append(g.edge_label)
        all_score.append(probs[:, 1])

        # segmentation quality is judged on every edge, but the merge walk only
        # follows confident positives so a single wrong edge cannot swallow a part
        labels, n_regions = union_find_merge(g.edge_index, probs[:, 1], g.n_faces,
                                             threshold=NC_MERGE_THRESHOLD)
        gt_labels, _ = union_find_merge(g.edge_index, g.edge_label, g.n_faces)
        ious.append(_region_iou(labels, gt_labels))

        pooled, counts = GATSegmenter.pool(h, labels)
        stats_gt = _graph_stats(g)
        pred_params = model.region_params(pooled, g.material, GATSegmenter.pool(stats_gt, labels)[0])
        truth_labels = _majority_label(labels, g.region_id, int(labels.max()) + 1)
        for k in range(len(pooled)):
            region_rows.append({
                "零件号": g.part_code,
                "类别": g.category,
                "材料": g.material,
                "预测区域": f"R{k:02d}",
                "面数": int(counts[k]),
                "预测转速": round(float(pred_params[k, 0] * SPINDLE_MAX), 0),
                "参考转速": round(float(g.region_params[truth_labels[k], 0]), 0),
                "预测进给": round(float(pred_params[k, 1] * FEED_MAX), 0),
                "参考进给": round(float(g.region_params[truth_labels[k], 1]), 0),
            })

        # regression-head accuracy on the true partition; this is the number the
        # resume quotes, because it isolates the parameter model from the
        # segmentation error that the end-to-end figure still contains
        pooled_gt, counts_gt = GATSegmenter.pool(h, g.region_id)
        params_gt = model.region_params(pooled_gt, g.material,
                                        GATSegmenter.pool(_graph_stats(g), g.region_id)[0])
        for k in range(len(pooled_gt)):
            teacher_rows.append({
                "零件号": g.part_code,
                "材料": g.material,
                "面数": int(counts_gt[k]),
                "预测转速": round(float(params_gt[k, 0] * SPINDLE_MAX), 0),
                "参考转速": round(float(g.region_params[k, 0]), 0),
                "预测进给": round(float(params_gt[k, 1] * FEED_MAX), 0),
                "参考进给": round(float(g.region_params[k, 1]), 0),
            })

    pred = np.concatenate(all_pred)
    truth = np.concatenate(all_true)
    score = np.concatenate(all_score)

    edge_metrics = prf(pred, truth)
    edge_metrics["auc"] = roc_auc(score, truth)
    edge_metrics["positive_rate"] = float(truth.mean())

    table = pd.DataFrame(region_rows)
    teacher = pd.DataFrame(teacher_rows)
    sp, sp_ref = teacher["预测转速"].to_numpy(), teacher["参考转速"].to_numpy()
    fd, fd_ref = teacher["预测进给"].to_numpy(), teacher["参考进给"].to_numpy()
    param_metrics = {
        "转速MAPE": round(mape(sp, sp_ref), 4),
        "进给MAPE": round(mape(fd, fd_ref), 4),
        "转速R2": round(r2(sp, sp_ref), 4),
        "进给R2": round(r2(fd, fd_ref), 4),
        "区域数": float(len(teacher)),
        "端到端转速MAPE": round(mape(table["预测转速"].to_numpy(), table["参考转速"].to_numpy()), 4),
        "端到端进给MAPE": round(mape(table["预测进给"].to_numpy(), table["参考进给"].to_numpy()), 4),
    }

    pred_sp = table["预测转速"].to_numpy()
    pred_fd = table["预测进给"].to_numpy()
    mats = table["材料"].to_numpy()
    phys_pred = physics_residual(pred_sp, pred_fd, mats)

    base_sp = float(table["参考转速"].mean())
    base_fd = float(table["参考进给"].mean())
    phys_base = physics_residual(np.full(len(mats), base_sp), np.full(len(mats), base_fd), mats)

    physics = {
        "门槛复核违规率_模型": round(float(phys_pred["violation"].mean()), 4),
        "门槛复核违规率_经验均值基线": round(float(phys_base["violation"].mean()), 4),
        "平均切削功率kW": round(float(phys_pred["power_kw"].mean()), 2),
        "平均每齿进给mm/z": round(float(phys_pred["chip_load"].mean()), 4),
        "示例经验转速": round(base_sp, 0),
        "示例经验进给": round(base_fd, 0),
    }

    return {
        "edge_metrics": {k: round(float(v), 4) for k, v in edge_metrics.items()},
        "param_metrics": param_metrics,
        "region_table": table,
        "physics": physics,
        "mean_region_iou": round(float(np.mean(ious)), 4),
    }


def run_project2(parts: pd.DataFrame, part_limit: int = 80, epochs: int = NC_SURROGATE_EPOCHS,
                 train: bool = False, seed: int = 17) -> NCResult:
    """Entry point for the AI NC-programming half of the demo."""
    rng = np.random.default_rng(seed)
    corpus = build_corpus_graphs(parts, rng, limit=part_limit)

    weights_path = WEIGHT_DIR / "gat_weights.npz"
    node_dim = corpus.graphs[0].node_features.shape[1]
    edge_dim = corpus.graphs[0].edge_features.shape[1]

    if train or not weights_path.exists():
        mode = "train"
        print(f"[GAT] no cached weights at {weights_path}, training {len(corpus.graphs)} parts ...")
        model, history = train_model(corpus, epochs=epochs, seed=seed)
        model.save(weights_path)
    else:
        mode = "inference"
        print(f"[GAT] loading cached weights from {weights_path}")
        model = GATSegmenter(node_dim, edge_dim, seed=seed)
        if not model.load(weights_path):
            print("[GAT] cached weights are incompatible, retraining ...")
            mode = "train"
            model, history = train_model(corpus, epochs=epochs, seed=seed)
            model.save(weights_path)
        else:
            history = []

    _, _, test = _part_level_split(corpus.graphs, seed)
    report = evaluate_model(model, test)
    print(f"[GAT] edge IoU={report['edge_metrics']['iou']:.3f} "
          f"acc={report['edge_metrics']['accuracy']:.3f} "
          f"auc={report['edge_metrics']['auc']:.3f} "
          f"转速MAPE={report['param_metrics']['转速MAPE']:.3f} "
          f"进给MAPE={report['param_metrics']['进给MAPE']:.3f}")

    return NCResult(
        history=history,
        edge_metrics=report["edge_metrics"],
        param_metrics=report["param_metrics"],
        region_table=report["region_table"].head(200),
        physics=report["physics"],
        graph_stats=corpus.stats,
        weights_path=str(weights_path),
        mode=mode,
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    from ..synthetic.generator import generate_corpus

    do_train = "--train" in sys.argv
    corpus_data = generate_corpus(full=False)
    result = run_project2(corpus_data.parts, train=do_train)
    print(json.dumps({"mode": result.mode,
                      "edge": result.edge_metrics,
                      "param": result.param_metrics,
                      "physics": result.physics}, ensure_ascii=False, indent=2))
