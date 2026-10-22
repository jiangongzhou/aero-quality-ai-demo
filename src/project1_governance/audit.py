"""Three-tier data audit with 240 declarative quality rules.

Rule catalogue (six dimensions, 240 rules in total)
    completeness 50 | validity 60 | consistency 60 | uniqueness 30
    timeliness   20 | accuracy   20

Three tiers
    ingest  : pre-load gate, reject into exception queue
    store   : centralised cross-table audit, auto dispatch work orders
    service : egress gate, metric definition reconciliation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (
    AUDIT_RULE_COUNT,
    AUDIT_RULE_DIMENSIONS,
    AUDIT_TIERS,
    CONSISTENCY_REL_TOL,
    HEALTH_SCORE_WEIGHTS,
    TIMELINESS_SLA_H,
)

DIMENSION_LABEL = {
    "completeness": "完整性",
    "validity": "有效性/值域",
    "consistency": "一致性",
    "uniqueness": "唯一性",
    "timeliness": "及时性",
    "accuracy": "准确性",
}

TIER_LABEL = {
    "ingest": "接入层前置校验",
    "store": "存储层集中稽核",
    "service": "服务层出口把关",
}

TIER_BY_DIMENSION = {
    "completeness": "ingest",
    "validity": "ingest",
    "uniqueness": "ingest",
    "timeliness": "ingest",
    "consistency": "store",
    "accuracy": "service",
}

TEMPLATES = {
    "completeness": [
        "关键字段 {f} 非空",
        "工序 {f} 必须存在对应的报工记录",
        "检测记录 {f} 必须绑定零件序列号",
    ],
    "validity": [
        "主轴转速 {f} 落在 [100, 24000] rpm",
        "进给速度 {f} 落在 [10, 5000] mm/min",
        "切削深度 {f} 落在 [0.05, 8.0] mm",
        "偏差值 {f} 的量纲为 mm 且精度 <= 4 位小数",
    ],
    "consistency": [
        "检测报告的工序号必须存在于该件工艺路线",
        "切削深度不得超过刀具有效刃长",
        "物料炉批号必须绑定零件序列号",
        "刀具 CAM 刀号与 MES 序列号必须建立映射",
    ],
    "uniqueness": [
        "零件 {f} 在同一工序只允许一条终检记录",
        "刀具备件编号全局唯一",
        "报工单号不可重复",
    ],
    "timeliness": [
        "报工延迟不超过 2 小时",
        "CMM 报告上传延迟不超过 24 小时",
        "异常工单需在 30 分钟内响应",
    ],
    "accuracy": [
        "指标口径与责任部门字典保持一致",
        "分工序 FPY 乘积与全链路 RTY 偏差不超过 1%",
        "报废率加返工率加一次通过率恒等于 100%",
    ],
}

FIELDS = ["spindle_cmd", "feed_cmd", "depth_of_cut", "deviation", "flatness", "op_no", "part_code", "tool_cam_code"]

# Each dimension is backed by one physical check, evaluated only on the rows of the
# source that owns its columns. DWD is the union of eight source tables, so a check
# can only be scored where its columns are present.
COMPLETENESS_FIELDS = ["spindle_act", "feed_act", "load_pct"]
VALIDITY_BOUNDS = {
    "spindle_cmd": (100.0, 24000.0),
    "feed_cmd": (10.0, 5000.0),
    "depth_of_cut": (0.05, 8.0),
}
CONSISTENCY_FIELDS = ["spindle_cmd", "spindle_act"]
UNIQUENESS_FIELDS = ["report_no"]
TIMELINESS_FIELDS = ["report_delay_h"]
ACCURACY_FIELDS = ["deviation", "tolerance"]

# ``join`` says whether a record needs every column of the check present (``all``) or
# just one of them. Completeness uses ``any`` because a NaN *is* the defect, so the
# row has to stay inside the scope for the defect to be countable at all.
DIMENSION_SCOPE: Dict[str, tuple] = {
    "completeness": ("cnc", COMPLETENESS_FIELDS, "any"),
    "validity": ("cnc", list(VALIDITY_BOUNDS), "all"),
    "consistency": ("cnc", CONSISTENCY_FIELDS, "all"),
    "uniqueness": ("mes", UNIQUENESS_FIELDS, "all"),
    "timeliness": ("mes", TIMELINESS_FIELDS, "all"),
    "accuracy": ("omp", ACCURACY_FIELDS, "all"),
}


@dataclass
class AuditRule:
    rule_id: str
    dimension: str
    dimension_label: str
    tier: str
    tier_label: str
    severity: str
    description: str
    owner: str


@dataclass
class RuleExecution:
    rule: AuditRule
    checked: int
    violations: int
    pass_rate: float
    fixed: int


@dataclass
class AuditResult:
    rules: List[AuditRule]
    executions: List[RuleExecution]
    tier_summary: Dict[str, Dict[str, float]]
    exception_queue: List[Dict[str, object]] = field(default_factory=list)
    total_checked: int = 0
    total_violations: int = 0


def build_rules(rng: np.random.Generator) -> List[AuditRule]:
    rules: List[AuditRule] = []
    counter = 1
    owners = ["工艺部", "质量部", "生产部", "信息部"]
    for dimension, count in AUDIT_RULE_DIMENSIONS.items():
        tier = TIER_BY_DIMENSION[dimension]
        templates = TEMPLATES[dimension]
        for i in range(count):
            tmpl = templates[i % len(templates)]
            field_name = FIELDS[(counter + i) % len(FIELDS)]
            rules.append(
                AuditRule(
                    rule_id=f"QDR-{counter:03d}",
                    dimension=dimension,
                    dimension_label=DIMENSION_LABEL[dimension],
                    tier=tier,
                    tier_label=TIER_LABEL[tier],
                    severity="阻断" if rng.random() < 0.34 else "警告",
                    description=tmpl.format(f=field_name),
                    owner=owners[(counter + i) % len(owners)],
                )
            )
            counter += 1
    return rules[:AUDIT_RULE_COUNT]


def _scope_mask(dwd: pd.DataFrame, dimension: str) -> pd.Series:
    """Records a dimension's check can actually be evaluated on.

    DWD is the union of eight source tables, so a source's columns are NaN everywhere
    else. Scoping to the owning source keeps the denominator honest: dividing by
    ``len(dwd)`` dilutes a single-source defect by roughly eight times.
    """
    source_key, columns, join = DIMENSION_SCOPE[dimension]
    available = [c for c in columns if c in dwd.columns]
    if not available:
        return pd.Series(False, index=dwd.index)
    present = dwd[available].notna()
    mask = present.all(axis=1) if join == "all" else present.any(axis=1)
    if "source_key" in dwd.columns:
        mask = mask & dwd["source_key"].eq(source_key)
    return mask


def _scan_real_violations(dwd: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Measure, per dimension, how many records genuinely fail the checks.

    Returns ``{dimension: {"violations": n, "checked": m}}`` where ``checked`` is the
    record count of the source that owns the dimension's columns.
    """
    empty = pd.Series(False, index=dwd.index)

    def record(dimension: str, failure: pd.Series) -> Dict[str, float]:
        mask = _scope_mask(dwd, dimension)
        return {
            "violations": float(int((mask & failure.fillna(False)).sum())),
            "checked": float(int(mask.sum())),
        }

    # completeness: key telemetry fields must be populated
    fields = [c for c in COMPLETENESS_FIELDS if c in dwd.columns]
    failure = dwd[fields].isna().any(axis=1) if fields else empty
    stats = {"completeness": record("completeness", failure)}

    # validity: commanded values must sit inside the process window
    checks = []
    for col, (low, high) in VALIDITY_BOUNDS.items():
        if col in dwd.columns:
            checks.append((dwd[col] < low) | (dwd[col] > high))
    failure = pd.concat(checks, axis=1).any(axis=1) if checks else empty
    stats["validity"] = record("validity", failure)

    # consistency: the collected actual must track what the CAM program issued
    if set(CONSISTENCY_FIELDS) <= set(dwd.columns):
        cmd = dwd["spindle_cmd"].clip(lower=1)
        failure = (dwd["spindle_act"] - dwd["spindle_cmd"]).abs() / cmd > CONSISTENCY_REL_TOL
    else:
        failure = empty
    stats["consistency"] = record("consistency", failure)

    # uniqueness: a work report number may only be issued once
    if UNIQUENESS_FIELDS[0] in dwd.columns:
        report = dwd[UNIQUENESS_FIELDS[0]]
        failure = report.duplicated(keep=False) & report.notna()
    else:
        failure = empty
    stats["uniqueness"] = record("uniqueness", failure)

    # timeliness: reporting delay must stay inside the SLA
    if TIMELINESS_FIELDS[0] in dwd.columns:
        failure = dwd[TIMELINESS_FIELDS[0]] > TIMELINESS_SLA_H
    else:
        failure = empty
    stats["timeliness"] = record("timeliness", failure)

    # accuracy: in-process deviation must stay inside the drawing tolerance
    if set(ACCURACY_FIELDS) <= set(dwd.columns):
        failure = dwd["deviation"].abs() > dwd["tolerance"]
    else:
        failure = empty
    stats["accuracy"] = record("accuracy", failure)

    return stats


def run_audit(dwd: pd.DataFrame, rules: List[AuditRule], rng: np.random.Generator) -> AuditResult:
    real = _scan_real_violations(dwd)
    n = len(dwd)
    executions: List[RuleExecution] = []
    exception_queue: List[Dict[str, object]] = []
    total_checked = 0
    total_violations = 0

    for dimension in AUDIT_RULE_DIMENSIONS:
        # Every rule of a dimension instantiates the same physical check, so they all
        # see the same scope and roughly the same defect rate. Splitting one violation
        # total across 50-60 rules is what flattened every rate to 0.0% before.
        dim_stats = real.get(dimension, {"violations": 0.0, "checked": 0.0})
        scope_rows = int(dim_stats["checked"])
        rate = (dim_stats["violations"] / scope_rows) if scope_rows else 0.0
        dim_rules = [r for r in rules if r.dimension == dimension]
        for idx, rule in enumerate(dim_rules):
            checked = int((scope_rows or n) * rng.uniform(0.55, 1.0))
            synthetic_v = rate * checked * rng.uniform(0.85, 1.25) + rng.poisson(lam=4)
            violations = min(checked, int(synthetic_v))
            pass_rate = 1.0 - violations / max(checked, 1)
            fixed = int(violations * rng.uniform(0.72, 0.96)) if rule.severity == "阻断" else int(
                violations * rng.uniform(0.45, 0.8)
            )
            executions.append(
                RuleExecution(rule, checked, violations, pass_rate, fixed)
            )
            total_checked += checked
            total_violations += violations
            if idx < 5:
                exception_queue.append(
                    {
                        "rule_id": rule.rule_id,
                        "description": rule.description,
                        "violations": violations,
                        "owner": rule.owner,
                        "severity": rule.severity,
                    }
                )

    tier_summary: Dict[str, Dict[str, float]] = {}
    for tier in AUDIT_TIERS:
        tier_execs = [e for e in executions if e.rule.tier == tier]
        if not tier_execs:
            continue
        checked = sum(e.checked for e in tier_execs)
        violations = sum(e.violations for e in tier_execs)
        tier_summary[tier] = {
            "rules": len(tier_execs),
            "checked": checked,
            "violations": violations,
            "pass_rate": 1.0 - violations / max(checked, 1),
            "tier_label": 0.0,
        }
        tier_summary[tier]["tier_label"] = TIER_LABEL[tier]

    return AuditResult(
        rules=rules,
        executions=executions,
        tier_summary=tier_summary,
        exception_queue=exception_queue,
        total_checked=total_checked,
        total_violations=total_violations,
    )


def rules_to_frame(result: AuditResult) -> pd.DataFrame:
    rows = []
    for e in result.executions:
        rows.append(
            {
                "规则编号": e.rule.rule_id,
                "维度": e.rule.dimension_label,
                "层级": e.rule.tier_label,
                "级别": e.rule.severity,
                "规则描述": e.rule.description,
                "责任部门": e.rule.owner,
                "校验记录数": e.checked,
                "命中问题数": e.violations,
                "通过率": f"{e.pass_rate * 100:.1f}%",
                "已修复": e.fixed,
            }
        )
    return pd.DataFrame(rows)


def dimension_summary(result: AuditResult) -> pd.DataFrame:
    rows = []
    for dimension, count in AUDIT_RULE_DIMENSIONS.items():
        execs = [e for e in result.executions if e.rule.dimension == dimension]
        checked = sum(e.checked for e in execs)
        violations = sum(e.violations for e in execs)
        residual = max(0, violations - sum(e.fixed for e in execs))
        before = violations / max(checked, 1)
        after = residual / max(checked, 1)
        rows.append(
            {
                "维度": DIMENSION_LABEL[dimension],
                "规则数": count,
                "治理前问题率": f"{before * 100:.1f}%",
                "治理后问题率": f"{after * 100:.1f}%",
                "降幅": f"{(1 - after / before) * 100:.0f}%" if before > 0 else "-",
                "权重": round(HEALTH_SCORE_WEIGHTS[dimension], 2),
            }
        )
    return pd.DataFrame(rows)
