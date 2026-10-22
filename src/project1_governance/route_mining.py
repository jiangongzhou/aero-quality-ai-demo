"""Typical process-route mining with entropy-weighted LCS and PSO-K-means.

Pipeline
    serialization -> entropy weighted LCS similarity -> spectral embedding
    -> kernel density estimation for K -> PSO seeding -> K-means -> 18 routes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import (
    PSO_C1,
    PSO_C2,
    PSO_ITERATIONS,
    PSO_POPULATION,
    PSO_W,
    ROUTE_CLUSTER_K,
    ROUTE_SIM_THRESHOLD,
)


@dataclass
class RouteMiningResult:
    k: int
    routes: pd.DataFrame
    embedding: np.ndarray
    assignment: np.ndarray
    coverage: float
    suggested_k: int = ROUTE_CLUSTER_K
    inertia_trace: List[float] = field(default_factory=list)
    kde_grid: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))


def tokenize(seq: str) -> List[str]:
    return [t.strip() for t in str(seq).split(">") if t.strip()]


def build_entropy_weights(corpus: List[List[str]]) -> Dict[str, float]:
    freq: Dict[str, int] = {}
    for toks in corpus:
        for t in set(toks):
            freq[t] = freq.get(t, 0) + 1
    n = len(corpus)
    weights: Dict[str, float] = {}
    for t, c in freq.items():
        p = c / n
        weights[t] = float(1.0 + (-p * np.log2(p)) / np.log2(n))
    return weights


def lcs_length(a: List[str], b: List[str], weight: Dict[str, float]) -> float:
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + weight.get(a[i - 1], 1.0)
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def similarity_matrix(corpus: List[List[str]], weight: Dict[str, float]) -> np.ndarray:
    n = len(corpus)
    sim = np.eye(n)
    norms = np.array([sum(weight.get(t, 1.0) for t in set(toks)) or 1.0 for toks in corpus])
    for i in range(n):
        for j in range(i + 1, n):
            l = lcs_length(corpus[i], corpus[j], weight)
            s = 2 * l / (norms[i] + norms[j])
            sim[i][j] = s
            sim[j][i] = s
    return sim


def spectral_embedding(sim: np.ndarray, dim: int = 8) -> np.ndarray:
    d = np.diag(sim.sum(axis=1))
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(d), 1e-8)))
    lap = np.eye(len(sim)) - d_inv_sqrt @ sim @ d_inv_sqrt
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)[1: dim + 1]
    return vecs[:, order]


def kde_estimate_k(embedding: np.ndarray, k_max: int = 30, rng: np.random.Generator | None = None) -> int:
    proj = embedding[:, 0] if embedding.shape[1] > 0 else np.zeros(len(embedding))
    grid = np.linspace(proj.min(), proj.max(), 120)
    bandwidth = 0.9 * len(proj) ** (-1 / 5) * (proj.std() or 1e-8)
    density = np.exp(-0.5 * ((grid[:, None] - proj[None, :]) / max(bandwidth, 1e-8)) ** 2).sum(axis=1)
    density /= density.sum() + 1e-12
    peaks = [
        i
        for i in range(1, len(grid) - 1)
        if density[i] > density[i - 1] and density[i] >= density[i + 1]
    ]
    peaks = sorted(peaks, key=lambda i: -density[i])[:k_max]
    return max(6, min(k_max, len(peaks)))


def kmeans_sse(points: np.ndarray, centers: np.ndarray) -> float:
    dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(dist.min(axis=1).sum())


def kmeans(points: np.ndarray, centers: np.ndarray, iters: int = 40) -> Tuple[np.ndarray, np.ndarray, float]:
    for _ in range(iters):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1)
        new_centers = centers.copy()
        for c in range(len(centers)):
            member = points[assign == c]
            if len(member) > 0:
                new_centers[c] = member.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1)
    return assign, centers, kmeans_sse(points, centers)


def pso_seed(points: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n, d = points.shape
    pop = rng.uniform(points.min(axis=0), points.max(axis=0), size=(PSO_POPULATION, k, d))
    vel = rng.normal(scale=0.05, size=(PSO_POPULATION, k, d))
    pbest = pop.copy()
    pbest_score = np.array([kmeans_sse(points, p) for p in pop])
    gbest = pbest[np.argmin(pbest_score)].copy()
    gbest_score = pbest_score.min()

    for _ in range(PSO_ITERATIONS):
        r1 = rng.random((PSO_POPULATION, k, d))
        r2 = rng.random((PSO_POPULATION, k, d))
        vel = PSO_W * vel + PSO_C1 * r1 * (pbest - pop) + PSO_C2 * r2 * (gbest - pop)
        pop = pop + vel
        scores = np.array([kmeans_sse(points, p) for p in pop])
        improved = scores < pbest_score
        pbest[improved] = pop[improved]
        pbest_score[improved] = scores[improved]
        if scores.min() < gbest_score:
            gbest_score = scores.min()
            gbest = pop[np.argmin(scores)].copy()
    return gbest


def _chain_similarity(seq: List[str], chain: List[str], weight: Dict[str, float]) -> float:
    """Entropy-weighted LCS similarity, the same kernel the operator matrix uses."""
    length = lcs_length(seq, chain, weight)
    denom = (sum(weight.get(t, 1.0) for t in set(seq))
             + sum(weight.get(t, 1.0) for t in set(chain)))
    return float(2 * length / max(denom, 1e-9))


def _repair_empty_clusters(points: np.ndarray, centers: np.ndarray,
                           assignment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Move centres that K-means starved onto the records they currently fit worst.

    Re-labelling a point is not enough: the next pass reassigns by nearest centre,
    so a starved centre goes straight back to empty. The centre itself has to
    travel to the poorly-served record, which is what actually carves a genuine
    cluster out of the tail instead of leaving dead prototypes behind.
    """
    k = len(centers)
    for _ in range(k):
        missing = [c for c in range(k) if not (assignment == c).any()]
        if not missing:
            break
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assigned = dist[np.arange(len(points)), assignment]
        for c in missing:
            worst = int(np.argmax(assigned))
            centers[c] = points[worst]
            assigned[worst] = -1.0
        assignment, centers, _ = kmeans(points, centers, iters=25)
    return assignment, centers, kmeans_sse(points, centers)


def mine_routes(process_records: pd.DataFrame, rng: np.random.Generator | None = None) -> RouteMiningResult:
    rng = rng or np.random.default_rng(7)
    seqs = process_records["operation_seq"].astype(str).tolist()
    corpus = [tokenize(s) for s in seqs]

    weight = build_entropy_weights(corpus)
    sim = similarity_matrix(corpus, weight)
    embedding = spectral_embedding(sim, dim=8)

    suggested_k = kde_estimate_k(embedding, k_max=ROUTE_CLUSTER_K, rng=rng)
    k = ROUTE_CLUSTER_K

    centers = pso_seed(embedding, k, rng)
    assignment, centers, sse = kmeans(embedding, centers)
    assignment, centers, sse = _repair_empty_clusters(embedding, centers, assignment)

    rows = []
    sizes = pd.Series(assignment).value_counts()
    for c in range(k):
        members = process_records.iloc[np.where(assignment == c)[0]]
        toks: List[List[str]] = [tokenize(s) for s in members["operation_seq"].tolist()]
        if not toks:
            continue
        length = int(np.median([len(t) for t in toks]))
        aligned: List[str] = []
        for pos in range(length):
            counter: Dict[str, int] = {}
            for t in toks:
                if pos < len(t):
                    counter[t[pos]] = counter.get(t[pos], 0) + 1
            aligned.append(max(counter, key=counter.get) if counter else "-")
        # How faithfully the consensus chain reproduces each member's own sequence.
        # A cluster only "covers" the records it genuinely stands for, so short or
        # idiosyncratic sequences drop out of the headline coverage number.
        sims = [_chain_similarity(t, aligned, weight) for t in toks]
        chain_covered = int(sum(s >= ROUTE_SIM_THRESHOLD for s in sims))
        rows.append(
            {
                "路线编号": f"TR-{c + 1:02d}",
                "代表度": f"{float(np.mean(sims)) * 100:.1f}%",
                "覆盖零件数": chain_covered,
                "代表零件类别": members["category"].mode().iat[0] if "category" in members else "-",
                "典型材料": members["material"].mode().iat[0] if "material" in members else "-",
                "工序数": length,
                "典型工序链": " > ".join(aligned),
                "簇内零件数": int(sizes.get(c, 0)),
                "占比": f"{sizes.get(c, 0) / len(assignment) * 100:.1f}%",
                "建议主轴转速": int(members["spindle"].median()) if "spindle" in members else 0,
                "建议进给速度": round(float(members["feed"].median()), 1) if "feed" in members else 0.0,
            }
        )

    routes = pd.DataFrame(rows).sort_values("覆盖零件数", ascending=False).reset_index(drop=True)
    coverage = float(routes["覆盖零件数"].sum() / len(assignment))

    return RouteMiningResult(
        k=k,
        routes=routes,
        embedding=embedding,
        assignment=assignment,
        coverage=round(coverage, 3),
        inertia_trace=[round(sse, 2)],
        kde_grid=(np.array([]), np.array([])),
        suggested_k=suggested_k,
    )
