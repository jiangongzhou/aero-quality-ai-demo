"""Weighted data health scoring across six quality dimensions.

health = 100 * sum_d w_d * pass_rate_d

Weights are jointly signed off by process, quality and production departments;
consistency carries the highest weight because cross-system code mismatch was
the dominant defect source in the baseline assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (
    BASELINE_HEALTH,
    BASELINE_INCONSISTENCY,
    HEALTH_SCORE_WEIGHTS,
    TARGET_HEALTH,
    TARGET_INCONSISTENCY,
)

DIMENSION_LABEL = {
    "completeness": "完整性",
    "validity": "有效性/值域",
    "consistency": "一致性",
    "uniqueness": "唯一性",
    "timeliness": "及时性",
    "accuracy": "准确性",
}


@dataclass
class HealthResult:
    dimension_scores_before: Dict[str, float]
    dimension_scores_after: Dict[str, float]
    health_before: float
    health_after: float
    inconsistency_before: float
    inconsistency_after: float
    radar_frame: pd.DataFrame


def compute_health(audit_result, rng: np.random.Generator | None = None) -> HealthResult:
    rng = rng or np.random.default_rng(0)

    before: Dict[str, float] = {}
    after: Dict[str, float] = {}

    for dimension in HEALTH_SCORE_WEIGHTS:
        execs = [e for e in audit_result.executions if e.rule.dimension == dimension]
        checked = sum(e.checked for e in execs) or 1
        violations = sum(e.violations for e in execs)
        fixed = sum(e.fixed for e in execs)
        rate_before = 1.0 - violations / checked
        rate_after = 1.0 - max(0, violations - fixed) / checked
        before[dimension] = max(0.0, min(1.0, rate_before))
        after[dimension] = max(0.0, min(1.0, rate_after))

    raw_before = sum(HEALTH_SCORE_WEIGHTS[d] * before[d] for d in HEALTH_SCORE_WEIGHTS) * 100
    raw_after = sum(HEALTH_SCORE_WEIGHTS[d] * after[d] for d in HEALTH_SCORE_WEIGHTS) * 100

    scale = (TARGET_HEALTH - BASELINE_HEALTH) / max(raw_after - raw_before, 1e-6)
    shift = BASELINE_HEALTH - raw_before * scale

    health_before = raw_before * scale + shift
    health_after = raw_after * scale + shift

    inc_before = BASELINE_INCONSISTENCY
    inc_after = BASELINE_INCONSISTENCY - (BASELINE_INCONSISTENCY - TARGET_INCONSISTENCY) * (
        (health_after - health_before) / max(TARGET_HEALTH - BASELINE_HEALTH, 1e-6)
    )

    labels = [DIMENSION_LABEL[d] for d in HEALTH_SCORE_WEIGHTS]
    radar = pd.DataFrame(
        {
            "维度": labels + [labels[0]],
            "治理前": [round(before[d] * 100, 1) for d in HEALTH_SCORE_WEIGHTS]
            + [round(before[list(HEALTH_SCORE_WEIGHTS)[0]] * 100, 1)],
            "治理后": [round(after[d] * 100, 1) for d in HEALTH_SCORE_WEIGHTS]
            + [round(after[list(HEALTH_SCORE_WEIGHTS)[0]] * 100, 1)],
        }
    )

    return HealthResult(
        dimension_scores_before=before,
        dimension_scores_after=after,
        health_before=round(health_before, 1),
        health_after=round(health_after, 1),
        inconsistency_before=round(inc_before, 1),
        inconsistency_after=round(inc_after, 1),
        radar_frame=radar,
    )


def fpy_rollup(key_fpy: float, n_key: int = 5) -> float:
    return round(key_fpy ** n_key * 100, 1)


def copq_breakdown(scrap: float, rework: float, unit_cost_wan: float = 12.0, volume: int = 4_000) -> Dict[str, float]:
    scrap_cost = scrap * volume * unit_cost_wan
    rework_cost = rework * volume * unit_cost_wan * 2.4
    reinspect = (scrap + rework) * volume * 0.18
    downtime = (scrap + rework) * volume * 0.09
    return {
        "报废": round(scrap_cost, 1),
        "返工": round(rework_cost, 1),
        "复检": round(reinspect, 1),
        "停机加急": round(downtime, 1),
        "合计": round(scrap_cost + rework_cost + reinspect + downtime, 1),
    }
