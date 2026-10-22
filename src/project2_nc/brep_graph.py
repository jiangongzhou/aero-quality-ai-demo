"""B-rep attributed graph construction from synthetic CAD/CAM data.

Node features (24-d, face level)
    [0:8]   geometry type one-hot : plane / cylinder / cone / sphere / torus /
                                    bspline / fillet / pocket
    [8:16]  geometric measures    : area, perimeter, aspect ratio, curvature,
                                    normal (x, y, z), face depth
    [16:24] process semantics     : tolerance grade, roughness Ra, datum flag,
                                    min tool radius, accessibility, stock
                                    allowance, removal ratio, ops count

Edge features (5-d, adjacency level)
    [0:3] convexity one-hot : convex / concave / tangent
    [3]   dihedral angle (rad)
    [4]   shared edge length

Generation rule: each machining region owns a latent prototype (normal, depth,
roughness, tolerance, geometry family). Faces inside a region scatter around that
prototype, which is what makes the region-label learnable from geometry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import (
    NC_DEPTH_MAX,
    NC_EDGE_FEATURE_DIM,
    NC_EDGE_PER_PART,
    NC_FACE_PER_PART,
    NC_LABEL_NOISE,
    NC_MATERIAL_FACTOR,
    NC_NODE_FEATURE_DIM,
    NC_RA_MAX,
    NC_REGION_PER_PART,
    NC_REMOVAL_MAX,
)

GEOMETRY_TYPES = ["plane", "cylinder", "cone", "sphere", "torus", "bspline", "fillet", "pocket"]
CONVEXITY = ["convex", "concave", "tangent"]
ROUGHNESS_GRADES = np.array([0.4, 0.8, 1.6, 3.2, 6.3, 12.5])
TOLERANCE_GRADES = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])
TOOL_RADII = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.5])


@dataclass
class BRepGraph:
    """A single part represented as an attributed adjacency graph."""

    part_code: str
    category: str
    material: str
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_label: np.ndarray
    region_id: np.ndarray
    region_params: np.ndarray
    node_stats: Optional[np.ndarray] = None

    @property
    def n_faces(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def n_regions(self) -> int:
        return int(len(np.unique(self.region_id[self.region_id >= 0])))


@dataclass
class GraphCorpus:
    graphs: List[BRepGraph]
    stats: Dict[str, float]
    node_feature_names: List[str]
    edge_feature_names: List[str]


def _material_factor(material: str) -> float:
    return float(NC_MATERIAL_FACTOR.get(material, 1.0))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-8)


def _sample_adjacency(
    rng: np.random.Generator,
    region_id: np.ndarray,
    n_faces: int,
    n_edges: int,
    n_regions: int,
    local_ratio: float = 0.45,
    near_ratio: float = 0.40,
):
    """Sample a B-rep adjacency list that respects geometric locality.

    Real B-rep adjacency is local: a face mostly touches faces of the same
    machining region, sometimes a neighbouring region, rarely a distant one.
    Sampling src/dst uniformly at random would leave ~1.5% positive edges and
    the segmentation task would degenerate into a constant predictor.
    """
    all_faces = np.arange(n_faces)
    # an empty region would break the sampler, fall back to any face instead
    members_by_region = [
        m if m.size else all_faces for m in (np.flatnonzero(region_id == r) for r in range(n_regions))
    ]

    draw = rng.random(n_edges)
    local = draw < local_ratio
    near = (draw >= local_ratio) & (draw < local_ratio + near_ratio)

    src = rng.integers(0, n_faces, n_edges)
    dst = rng.integers(0, n_faces, n_edges)

    for mask, is_near in ((local, False), (near, True)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        base_region = region_id[src[idx]]
        if is_near:
            # neighbour region: half of them stay local, half jump to a nearby id
            target_region = np.where(
                rng.random(idx.size) < 0.5,
                base_region,
                np.clip(base_region + rng.integers(1, 3, idx.size), 0, n_regions - 1),
            )
        else:
            target_region = base_region
        picks = np.array(
            [
                members_by_region[int(r)][rng.integers(0, len(members_by_region[int(r)]))]
                for r in target_region
            ]
        )
        dst[idx] = picks

    keep = src != dst
    return src[keep], dst[keep]


def build_part_graph(rng: np.random.Generator, part_code: str, category: str, material: str) -> BRepGraph:
    n_faces = int(rng.integers(int(NC_FACE_PER_PART * 0.6), int(NC_FACE_PER_PART * 1.4)))
    n_regions = int(np.clip(rng.normal(NC_REGION_PER_PART, 12), 3, max(3, n_faces // 2)))

    # balanced assignment: every region owns at least one face, which keeps
    # the prototype-based geometry and the local adjacency sampler well defined
    region_id = np.empty(n_faces, dtype=int)
    region_id[:n_regions] = np.arange(n_regions)
    region_id[n_regions:] = rng.integers(0, n_regions, n_faces - n_regions)
    rng.shuffle(region_id)

    proto_normal = rng.normal(size=(n_regions, 3))
    proto_normal /= np.linalg.norm(proto_normal, axis=1, keepdims=True) + 1e-8
    proto_depth = rng.uniform(1.0, 40.0, n_regions)
    proto_ra = rng.choice(ROUGHNESS_GRADES, n_regions)
    proto_tol = rng.choice(TOLERANCE_GRADES, n_regions).astype(float)
    proto_geom = rng.integers(0, len(GEOMETRY_TYPES), n_regions)
    proto_area = rng.lognormal(3.2, 0.8, n_regions)
    proto_access = rng.uniform(0.2, 1.0, n_regions)

    follow = rng.random(n_faces) < 0.82

    geom_ids = np.where(follow, proto_geom[region_id], rng.integers(0, len(GEOMETRY_TYPES), n_faces))
    area = np.where(follow, proto_area[region_id] * rng.lognormal(0, 0.25, n_faces), rng.lognormal(3.2, 0.8, n_faces))
    perimeter = area ** 0.5 * rng.uniform(3.4, 5.2, n_faces)
    aspect = rng.lognormal(0.0, 0.45, n_faces)
    curvature = np.where(geom_ids <= 0, 0.0, rng.exponential(0.08, n_faces))
    normals = proto_normal[region_id] + rng.normal(scale=0.30, size=(n_faces, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    depth = np.where(follow, proto_depth[region_id] + rng.normal(0, 2.5, n_faces), rng.uniform(0.5, 42.0, n_faces))
    depth = np.clip(depth, 0.3, 45.0)

    tol = np.where(follow, proto_tol[region_id], rng.choice(TOLERANCE_GRADES, n_faces).astype(float))
    ra = np.where(follow, proto_ra[region_id], rng.choice(ROUGHNESS_GRADES, n_faces))
    datum = (rng.random(n_faces) < 0.12).astype(float)
    min_tool = rng.choice(TOOL_RADII, n_faces)

    node_features = np.zeros((n_faces, NC_NODE_FEATURE_DIM))
    node_features[np.arange(n_faces), geom_ids.astype(int)] = 1.0
    node_features[:, 8] = _zscore(area)
    node_features[:, 9] = _zscore(perimeter)
    node_features[:, 10] = _zscore(aspect)
    node_features[:, 11] = _zscore(curvature)
    node_features[:, 12:15] = normals
    node_features[:, 15] = _zscore(depth)
    node_features[:, 16] = _zscore(tol)
    node_features[:, 17] = _zscore(ra)
    node_features[:, 18] = datum
    node_features[:, 19] = _zscore(min_tool)
    node_features[:, 20] = np.clip(np.where(follow, proto_access[region_id], rng.uniform(0.1, 1.0, n_faces)), 0.05, 1.0)
    node_features[:, 21] = _zscore(depth * area)
    node_features[:, 22] = _zscore(1.0 / (1.0 + min_tool))
    node_features[:, 23] = _zscore(np.bincount(region_id, minlength=n_regions)[region_id].astype(float))

    mfg = _material_factor(material)
    region_params = np.zeros((n_regions, 2))
    for r in range(n_regions):
        mask = region_id == r
        if not mask.any():
            continue
        mean_depth = float(depth[mask].mean())
        mean_ra = float(ra[mask].mean())
        spindle = float(np.clip(18000 * mfg / (1 + mean_depth / 12.0), 3500, 20000))
        feed = float(np.clip(2600 * mfg / (1 + mean_ra / 3.0), 260, 3200))
        region_params[r] = [spindle, feed]

    n_edges = min(int(NC_EDGE_PER_PART * 1.1), n_faces * 3)
    src, dst = _sample_adjacency(rng, region_id, n_faces, n_edges, n_regions)
    n_edges = len(src)

    same_region = region_id[src] == region_id[dst]
    u = rng.random(n_edges)

    dihedral = np.where(
        same_region,
        np.where(u < 0.58, np.pi - rng.exponential(0.12, n_edges), rng.uniform(0.2, 1.35, n_edges)),
        rng.uniform(0.15, np.pi, n_edges),
    )
    dihedral = np.clip(dihedral, 0.05, np.pi)

    conv = np.where(
        dihedral > np.pi - 0.18,
        2,
        np.where(rng.random(n_edges) < 0.52, 1, 0),
    )

    edge_index = np.stack([src, dst])
    edge_features = np.zeros((n_edges, NC_EDGE_FEATURE_DIM))
    edge_features[np.arange(n_edges), conv.astype(int)] = 1.0
    edge_features[:, 3] = dihedral / np.pi
    edge_features[:, 4] = rng.uniform(1.0, 160.0, n_edges) / 160.0

    edge_label = same_region.astype(int)
    edge_label = np.where((conv == 2) & (rng.random(n_edges) < NC_LABEL_NOISE), 0, edge_label)

    # per-face statistics carried in absolute process units, so pooling them
    # recovers the region means the cutting-parameter target is built from
    node_stats = np.column_stack([
        depth / NC_DEPTH_MAX,
        ra / NC_RA_MAX,
        np.clip(depth * area, 0.0, NC_REMOVAL_MAX) / NC_REMOVAL_MAX,
    ])

    return BRepGraph(
        part_code=part_code,
        category=category,
        material=material,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_label=edge_label,
        region_id=region_id,
        region_params=region_params,
        node_stats=node_stats,
    )


def build_corpus_graphs(parts: pd.DataFrame, rng: np.random.Generator, limit: int | None = None) -> GraphCorpus:
    frame = parts if limit is None else parts.head(limit)
    graphs = [
        build_part_graph(rng, str(row["part_code"]), str(row["category"]), str(row["material"]))
        for _, row in frame.iterrows()
    ]
    return GraphCorpus(
        graphs=graphs,
        stats=graph_statistics(graphs),
        node_feature_names=node_feature_names(),
        edge_feature_names=edge_feature_names(),
    )


def node_feature_names() -> List[str]:
    return (
        [f"geom::{g}" for g in GEOMETRY_TYPES]
        + ["area(z)", "perimeter(z)", "aspect(z)", "curvature(z)", "nx", "ny", "nz", "depth(z)"]
        + ["tol_grade(z)", "Ra(z)", "is_datum", "min_tool_r(z)", "accessibility", "removal(z)", "tool_penalty(z)", "ops(z)"]
    )


def edge_feature_names() -> List[str]:
    return [f"convexity::{c}" for c in CONVEXITY] + ["dihedral/pi", "shared_len/160"]


def graph_statistics(graphs: List[BRepGraph]) -> Dict[str, float]:
    faces = [g.n_faces for g in graphs]
    edges = [g.n_edges for g in graphs]
    regions = [g.n_regions for g in graphs]
    return {
        "parts": float(len(graphs)),
        "faces_mean": round(float(np.mean(faces)), 1),
        "faces_total": float(sum(faces)),
        "edges_mean": round(float(np.mean(edges)), 1),
        "edges_total": float(sum(edges)),
        "regions_mean": round(float(np.mean(regions)), 1),
        "regions_total": float(sum(regions)),
    }


def union_find_merge(edge_index: np.ndarray, pred: np.ndarray, n_nodes: int, threshold: float = 0.5):
    """Merge faces into machining regions via connected components on predicted 1-edges."""
    parent = np.arange(n_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in range(edge_index.shape[1]):
        if pred[k] < threshold:
            continue
        a, b = int(edge_index[0, k]), int(edge_index[1, k])
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    labels = np.array([find(i) for i in range(n_nodes)])
    _, compressed = np.unique(labels, return_inverse=True)
    return compressed.astype(int), int(len(np.unique(compressed)))
