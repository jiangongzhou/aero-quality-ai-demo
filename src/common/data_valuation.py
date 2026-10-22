"""Sample valuation: leave-one-out gain, Shapley fair allocation, redundancy discount.

Stage 1  leave-one-out   : marginal gain of each sample on a surrogate metric
Stage 2  Shapley value   : fair allocation under cooperation (monte-carlo permutations)
Stage 3  surrogate model : neural net mapping latent vector -> value, so new
                           samples can be priced without retraining
Stage 4  redundancy      : multi-metric distance + kernel filtering, aggregated as
                           V(S) = sum_i v_i * (1 - discount * share_i)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import (
    KERNEL_C,
    NC_SURROGATE_EPOCHS,
    NC_SURROGATE_LR,
    REDUNDANCY_DISCOUNT,
    REDUNDANCY_TOPK,
    SHAPLEY_MONTE_CARLO,
)


@dataclass
class ValuationResult:
    values: np.ndarray
    shapley: np.ndarray
    redundancy: np.ndarray
    aggregated: np.ndarray
    selected: np.ndarray
    budget: int
    aggregation_trace: Dict[str, float]


def _dist_matrix(points: np.ndarray) -> Dict[str, np.ndarray]:
    diff = points[:, None, :] - points[None, :, :]
    euclid = np.sqrt((diff ** 2).sum(axis=2) + 1e-12)
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    unit = points / np.maximum(norm, 1e-8)
    cosine_sim = np.clip(unit @ unit.T, -1.0, 1.0)
    cosine = 1.0 - cosine_sim
    manhattan = np.abs(diff).sum(axis=2)
    return {"euclidean": euclid, "cosine": cosine, "manhattan": manhattan}


def normalize(mat: np.ndarray) -> np.ndarray:
    rng = mat.max() - mat.min()
    if rng <= 1e-12:
        return np.zeros_like(mat)
    return (mat - mat.min()) / rng


def multi_metric_distance(points: np.ndarray) -> np.ndarray:
    """Fuse Euclidean, cosine and Manhattan views into a single distance."""
    dists = _dist_matrix(points)
    stack = np.stack([normalize(dists[k]) for k in ("euclidean", "cosine", "manhattan")])
    return stack.mean(axis=0)


def inverse_multiquadric(distance: np.ndarray, c: float = KERNEL_C) -> np.ndarray:
    """Slowly-decaying (heavy-tailed) kernel: captures mid-range redundancy."""
    return 1.0 / np.sqrt(distance ** 2 + c ** 2)


def rbf_kernel(distance: np.ndarray, c: float = KERNEL_C) -> np.ndarray:
    return np.exp(-((distance / max(c, 1e-8)) ** 2))


def laplacian_kernel(distance: np.ndarray, c: float = KERNEL_C) -> np.ndarray:
    return np.exp(-distance / max(c, 1e-8))


KERNELS = {
    "inverse_multiquadric": inverse_multiquadric,
    "rbf": rbf_kernel,
    "laplacian": laplacian_kernel,
}


def redundancy_matrix(points: np.ndarray, c: float = KERNEL_C) -> np.ndarray:
    """Redundancy between neighbours: near samples share information, so value is discounted."""
    dist = multi_metric_distance(points)
    combined = np.zeros_like(dist)
    for name, fn in KERNELS.items():
        combined = combined + fn(dist, c)
    combined = combined / len(KERNELS)
    np.fill_diagonal(combined, 0.0)
    return combined


def leave_one_out_values(points: np.ndarray, metric: Callable[[np.ndarray], float], baseline: float) -> np.ndarray:
    n = len(points)
    values = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        values[i] = baseline - metric(points[mask])
    return values


def shapley_values(
    points: np.ndarray,
    metric: Callable[[np.ndarray], float],
    loo_values: np.ndarray,
    max_players: int = 8,
    samples: int = SHAPLEY_MONTE_CARLO,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Exact Shapley for small coalitions, monte-carlo permutation sampling above."""
    rng = rng or np.random.default_rng(3)
    n = len(points)
    if n == 0:
        return loo_values

    if n <= max_players:
        cache: Dict[tuple, float] = {}

        def utility(idxs: Sequence[int]) -> float:
            key = tuple(sorted(idxs))
            if key not in cache:
                cache[key] = coverage_metric(points[list(key)]) if len(key) else 0.0
            return cache[key]

        shap = np.zeros(n)
        for i in range(n):
            others = [j for j in range(n) if j != i]
            total = 0.0
            for size in range(len(others) + 1):
                weight = (
                    math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                )
                for subset in combinations(others, size):
                    total += weight * (utility(list(subset) + [i]) - utility(list(subset)))
            shap[i] = total
        return shap

    shap = np.zeros(n)
    for _ in range(samples):
        order = rng.permutation(n)
        current: List[int] = []
        previous = 0.0
        for idx in order:
            current.append(int(idx))
            current_utility = coverage_metric(points[current])
            shap[int(idx)] += current_utility - previous
            previous = current_utility
    return shap / samples


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class ValueSurrogate:
    """Two-layer MLP regressing a sample's Shapley value from its latent vector."""

    def __init__(self, input_dim: int, hidden: int = 32, lr: float = NC_SURROGATE_LR, seed: int = 5):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w1 = rng.normal(0, 1 / np.sqrt(input_dim), (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z1 = x @ self.w1 + self.b1
        self.a1 = _relu(self.z1)
        return self.a1 @ self.w2 + self.b2

    def backward(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.forward(x)
        n = len(x)
        dout = 2.0 * (pred - y) / n
        dw2 = self.a1.T @ dout
        db2 = dout.sum(axis=0)
        da1 = dout @ self.w2.T
        dz1 = da1 * _relu_grad(self.z1)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        for attr, grad in (("w2", dw2), ("b2", db2), ("w1", dw1), ("b1", db1)):
            setattr(self, attr, getattr(self, attr) - self.lr * grad)
        return float(((pred - y) ** 2).mean())

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = NC_SURROGATE_EPOCHS) -> List[float]:
        y = y.reshape(-1, 1)
        self._y_lo, self._y_hi = float(y.min()), float(y.max())
        mu, sigma = x.mean(axis=0), x.std(axis=0) + 1e-8
        x = (x - mu) / sigma
        self._mu, self._sigma = mu, sigma
        # Shapley values sit three orders of magnitude below the raw network
        # output, so regressing on them directly leaves a learning rate of 1e-2
        # numerically stalled. Standardising the target puts the loss on a
        # unit-variance scale, which makes the reported loss directly readable
        # as `1 - R^2` on the priced sample.
        self._y_mu, self._y_sigma = float(y.mean()), float(y.std()) + 1e-12
        y = (y - self._y_mu) / self._y_sigma
        losses = []
        for _ in range(epochs):
            losses.append(self.backward(x, y))
        return losses

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = (x - self._mu) / self._sigma
        out = self.forward(x) * self._y_sigma + self._y_mu
        # A ReLU net is piecewise linear and therefore extrapolates without
        # bound outside the hull of its 60 training samples. Clamping to the
        # training target range keeps unpriced samples from being handed a value
        # the Shapley game could never have produced.
        return np.clip(out, self._y_lo, self._y_hi).ravel()


def coverage_metric(points: np.ndarray) -> float:
    """Surrogate utility: negative mean nearest-neighbour distance in latent space.

    Higher coverage means the subset spans the process distribution better, so
    the model generalises further for the same labelling budget.
    """
    if len(points) < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    nnd = dist.min(axis=1)
    return -float(nnd.mean())


def redundancy_share(redundancy: np.ndarray, top_k: int = REDUNDANCY_TOPK) -> np.ndarray:
    """Per-sample redundancy, normalised to [0, 1] across the pool.

    ``redundancy`` is a pairwise kernel value bounded by [0, 1]. A sample's
    redundancy is judged against its ``top_k`` closest neighbours: a sample
    sitting in a dense region duplicates information its neighbours already
    carry, an isolated one does not. Averaging over the local neighbourhood
    instead of summing the whole row keeps the result bounded no matter how
    large the pool is, which is what makes it combinable with a value.
    """
    n = len(redundancy)
    if n < 2:
        return np.zeros(n)
    k = int(np.clip(top_k, 1, n - 1))
    neighbourhood = np.partition(redundancy, -k, axis=1)[:, -k:]
    return normalize(neighbourhood.mean(axis=1))


def aggregate_values(
    values: np.ndarray,
    redundancy: np.ndarray,
    share: Optional[np.ndarray] = None,
    discount: float = REDUNDANCY_DISCOUNT,
) -> float:
    """V(S) = sum_i v_i * (1 - discount * share_i).

    Redundancy enters as a bounded multiplicative discount rather than as the
    absolute ``sum_{i<j} r_ij`` subtraction: the kernel is bounded by [0, 1],
    so summing it over a ~1_500-sample pool reaches ~1_100 while the values
    themselves live at the 1e-2 scale -- subtracting them drowned out the
    ranking entirely. Passing the pool-wide ``share`` in keeps a subset's
    objective exactly consistent with the per-sample column it is built from, so
    ``sum_{i in S} aggregated_i == aggregate_values(values_S, red_SS)``.
    """
    n = len(values)
    if n < 2:
        return float(values.sum())
    if share is None:
        share = redundancy_share(redundancy)
    return float((values * (1.0 - discount * share)).sum())


def _take(share: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
    """Restrict a pool-wide share vector to a candidate subset (None-safe)."""
    return None if share is None else share[idx]


def greedy_select(
    values: np.ndarray,
    redundancy: np.ndarray,
    budget: int,
    share: Optional[np.ndarray] = None,
    discount: float = REDUNDANCY_DISCOUNT,
) -> np.ndarray:
    """Greedy forward selection under a fixed budget, followed by local swap search."""
    n = len(values)
    budget = min(budget, n)
    selected: List[int] = []
    for _ in range(budget):
        best, best_gain = None, -np.inf
        for i in range(n):
            if i in selected:
                continue
            cand = selected + [i]
            idx = np.array(cand)
            gain = aggregate_values(
                values[idx], redundancy[np.ix_(idx, idx)], _take(share, idx), discount
            )
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        selected.append(best)

    improved = True
    while improved:
        improved = False
        idx = np.array(selected)
        current = aggregate_values(
            values[idx], redundancy[np.ix_(idx, idx)], _take(share, idx), discount
        )
        for pos in range(len(selected)):
            for j in range(n):
                if j in selected:
                    continue
                trial = selected.copy()
                trial[pos] = j
                tidx = np.array(trial)
                gain = aggregate_values(
                    values[tidx], redundancy[np.ix_(tidx, tidx)], _take(share, tidx), discount
                )
                if gain > current + 1e-9:
                    selected = trial
                    current = gain
                    improved = True
                    break
            if improved:
                break

    return np.array(sorted(selected))


def run_valuation(
    latent: np.ndarray,
    budget: int,
    rng: Optional[np.random.Generator] = None,
    sample_size: int = 60,
) -> ValuationResult:
    """End-to-end pricing: LOO -> Shapley -> surrogate -> redundancy-discounted selection."""
    rng = rng or np.random.default_rng(9)
    n = len(latent)
    m = min(sample_size, n)
    idx = rng.choice(n, size=m, replace=False)
    subset = latent[idx]

    baseline = coverage_metric(subset)
    loo = leave_one_out_values(subset, coverage_metric, baseline)
    shap = shapley_values(subset, coverage_metric, loo, rng=rng)

    surrogate = ValueSurrogate(input_dim=latent.shape[1])
    losses = surrogate.fit(subset, shap)

    full_values = surrogate.predict(latent)
    red = redundancy_matrix(latent)

    # `share` is each sample's pool-wide redundancy against its REDUNDANCY_TOPK
    # nearest neighbours, min-max scaled to [0, 1]. Pricing it as a bounded
    # multiplicative discount keeps `aggregated` on exactly the same scale as
    # `full_values`, which the previous absolute `sum_i r_ij` subtraction did not:
    # the kernel sums to ~1_100 per row while values live at 1e-2.
    share = redundancy_share(red)
    aggregated = full_values * (1.0 - REDUNDANCY_DISCOUNT * share)

    selected = greedy_select(full_values, red, budget, share=share)

    sidx = selected
    iu = np.triu_indices(len(red), k=1)
    trace = {
        "baseline_utility": round(baseline, 6),
        "surrogate_final_loss": round(losses[-1], 6) if losses else 0.0,
        "mean_sample_value": round(float(full_values.mean()), 6),
        "mean_pairwise_redundancy": round(float(red[iu].mean()), 4),
        "mean_redundancy_share": round(float(share.mean()), 4),
        "aggregated_selected": round(
            aggregate_values(full_values[sidx], red[np.ix_(sidx, sidx)], share[sidx]), 6
        ),
        "aggregated_all": round(aggregate_values(full_values, red, share), 6),
    }

    return ValuationResult(
        values=full_values,
        shapley=shap,
        redundancy=red,
        aggregated=aggregated,
        selected=selected,
        budget=budget,
        aggregation_trace=trace,
    )


def redundancy_report(result: ValuationResult, points: np.ndarray, top_k: int = REDUNDANCY_TOPK) -> pd.DataFrame:
    """Top redundant pairs among the selected subset, with per-metric breakdown."""
    sub = result.redundancy[np.ix_(result.selected, result.selected)]
    iu = np.triu_indices(len(sub), k=1)
    pairs = [(result.selected[i], result.selected[j], sub[i, j]) for i, j in zip(*iu)]
    pairs.sort(key=lambda t: -t[2])

    scale_euclid = float(np.linalg.norm(points, axis=1).mean()) + 1e-8
    scale_manh = float(np.abs(points).sum(axis=1).mean()) + 1e-8

    rows = []
    for a, b, r in pairs[:top_k]:
        da, db = points[a], points[b]
        diff = da - db
        euclid = float(np.linalg.norm(diff))
        cosine = float(1.0 - (da @ db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12))
        manhattan = float(np.abs(diff).sum())
        combined = (euclid / scale_euclid + cosine / 2.0 + manhattan / scale_manh) / 3.0
        rows.append(
            {
                "样本A": f"S{a:05d}",
                "样本B": f"S{b:05d}",
                "综合距离": round(combined, 4),
                "欧氏距离": round(euclid, 4),
                "余弦距离": round(cosine, 4),
                "曼哈顿距离": round(manhattan, 4),
                "冗余度": round(float(r), 4),
                "价值折价": round(float(r) * (result.values[a] + result.values[b]) * 0.5, 4),
            }
        )
    return pd.DataFrame(rows)
