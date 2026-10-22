"""Synthetic data generator for aircraft structural part manufacturing.

Everything emitted here is fabricated. Part numbers, machine codes, operator
identifiers and sensor traces are generated from fixed random seeds and carry
no relationship to any real production system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import (
    DATA_SOURCE_CATALOG,
    DEFECT_MODE_COUNT,
    DIRTY_INJECTION,
    KEY_STATION_COUNT,
    KPP_COUNT,
    KQC_COUNT,
    NC_FACE_PER_PART,
    NC_PART_CATEGORIES,
    NC_RECORD_COUNT,
    OPERATION_COUNT,
    get_record_count,
)

MATERIALS = ["TC4", "7050-T7451", "2A12-T4", "300M", "AlSi10Mg"]
MACHINES = [f"MC-{i:03d}" for i in range(1, 13)]
OPERATORS = [f"OP-{i:04d}" for i in range(1, 40)]

OPERATION_POOL = [
    "下料", "粗铣腹板", "时效去应力", "半精铣腹板", "精铣腹板", "粗铣缘条",
    "半精铣缘条", "精铣缘条", "钻铰孔系", "锪窝", "倒角去毛刺", "钳修",
    "超声探伤", "荧光渗透", "三坐标终检", "阳极化", "喷丸强化", "表面处理",
    "清洗", "终包装",
]

DEFECT_MODE_POOL = [
    "平面度超差", "垂直度超差", "尺寸超差", "孔位偏差", "表面粗糙度超差",
    "刀痕振纹", "让刀过切", "毛刺残留", "壁厚超差", "圆角过切",
]

FIVE_M_ONE_E = ["人", "机", "料", "法", "环", "测"]


@dataclass
class SyntheticCorpus:
    """Container holding every synthetic table produced by the generator."""

    sources: Dict[str, pd.DataFrame]
    parts: pd.DataFrame
    process_records: pd.DataFrame
    routes: pd.DataFrame
    defects: pd.DataFrame

    @property
    def tables(self) -> List[str]:
        return list(self.sources.keys())

    def total_records(self) -> int:
        return int(sum(len(df) for df in self.sources.values()))


def _part_codes(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.array([f"AC-{rng.integers(1000, 9999)}-{rng.integers(10, 99)}" for _ in range(n)])


def generate_parts(rng: np.random.Generator, n_parts: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "part_code": _part_codes(rng, n_parts),
            "category": rng.choice(NC_PART_CATEGORIES, n_parts),
            "material": rng.choice(MATERIALS, n_parts),
            "faces": rng.integers(int(NC_FACE_PER_PART * 0.6), int(NC_FACE_PER_PART * 1.4), n_parts),
            "raw_value_wan": np.round(rng.uniform(3.0, 45.0, n_parts), 2),
        }
    )


def _inject_missing(frame: pd.DataFrame, rng: np.random.Generator, columns: List[str], rate: float) -> None:
    for col in columns:
        mask = rng.random(len(frame)) < rate
        frame.loc[mask, col] = np.nan


def _inject_unit_mix(frame: pd.DataFrame, rng: np.random.Generator, col: str, rate: float) -> None:
    mask = rng.random(len(frame)) < rate
    frame.loc[mask, col] = frame.loc[mask, col] * 60.0
    frame.loc[mask, "feed_unit"] = "mm/r"


def _inject_duplicate_keys(frame: pd.DataFrame, rng: np.random.Generator, col: str, rate: float) -> None:
    """Copy an already-issued key onto `rate` of the rows.

    Generated keys are drawn from a space wide enough that they essentially never
    collide by chance, so without this the uniqueness rules would find nothing.
    """
    if col not in frame.columns or len(frame) == 0:
        return
    mask = rng.random(len(frame)) < rate
    hits = int(mask.sum())
    if hits == 0:
        return
    donor = rng.integers(0, len(frame), hits)
    frame.loc[mask, col] = frame[col].to_numpy()[donor]


def _inject_out_of_range(
    frame: pd.DataFrame, rng: np.random.Generator, col: str, rate: float, low: float, high: float
) -> None:
    """Push `rate` of a column outside the window its validity rule allows."""
    if col not in frame.columns or len(frame) == 0:
        return
    mask = rng.random(len(frame)) < rate
    hits = int(mask.sum())
    if hits == 0:
        return
    frame.loc[mask, col] = rng.uniform(low, high, hits).round(2)


def generate_cnc(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="15s"),
            "part_code": parts["part_code"].values[idx],
            "machine": rng.choice(MACHINES, n),
            "operator": rng.choice(OPERATORS, n),
            "op_no": rng.integers(1, OPERATION_COUNT + 1, n),
            "spindle_cmd": rng.uniform(3000, 18000, n).round(0),
            "feed_cmd": rng.uniform(200, 3000, n).round(1),
            "depth_of_cut": rng.uniform(0.2, 6.0, n).round(2),
            "feed_unit": "mm/min",
        }
    )
    frame["spindle_act"] = frame["spindle_cmd"] * rng.normal(1.0, 0.035, n)
    frame["feed_act"] = frame["feed_cmd"] * rng.normal(1.0, 0.048, n)
    frame["load_pct"] = np.clip(rng.normal(45, 12, n), 5, 100).round(1)
    _inject_missing(frame, rng, ["spindle_act", "feed_act", "load_pct"], DIRTY_INJECTION["missing_rate"])
    _inject_unit_mix(frame, rng, "feed_cmd", DIRTY_INJECTION["unit_mix_rate"])
    _inject_out_of_range(
        frame, rng, "depth_of_cut", DIRTY_INJECTION["out_of_range_rate"], 9.0, 14.0
    )
    return frame


def generate_omp(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=n, freq="40s"),
            "part_code": parts["part_code"].values[idx],
            "probe_id": rng.choice(["PRB-A", "PRB-B", "PRB-C"], n),
            "op_no": rng.integers(1, OPERATION_COUNT + 1, n),
            "deviation": rng.normal(0.0, 0.035, n).round(4),
            "tolerance": rng.choice([0.05, 0.08, 0.10, 0.15], n),
        }
    )


def generate_mes(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    start = pd.date_range("2025-01-01", periods=n, freq="3min")
    delay = rng.exponential(1.4, n)
    delayed = rng.random(n) < DIRTY_INJECTION["report_delay_rate"]
    delay[delayed] += rng.uniform(2.0, 26.0, int(delayed.sum()))
    frame = pd.DataFrame(
        {
            "ts": start,
            "part_code": parts["part_code"].values[idx],
            "op_no": rng.integers(1, OPERATION_COUNT + 1, n),
            "setup_min": rng.uniform(8, 95, n).round(1),
            "run_min": rng.uniform(15, 480, n).round(1),
            "report_delay_h": delay.round(2),
            "qty": rng.integers(1, 6, n),
            "report_no": [f"RPT-{i:06d}" for i in range(n)],
        }
    )
    _inject_duplicate_keys(frame, rng, "report_no", DIRTY_INJECTION["code_duplicate_rate"])
    return frame


def generate_tool(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tool_cam_code": [f"T{rng.integers(1, 400):03d}" for _ in range(n)],
            "tool_mes_sn": [f"SER{'-'}%05d" % rng.integers(1, 99999) for _ in range(n)],
            "tool_room_lot": [f"LOT-{rng.integers(1000, 9999)}" for _ in range(n)],
            "diameter": rng.choice([6, 8, 10, 12, 16, 20, 25], n),
            "flute_len": rng.choice([6, 8, 10, 16, 20, 26], n),
            "used_min": rng.uniform(5, 320, n).round(1),
            "rated_min": rng.choice([60, 90, 120, 180, 240], n),
        }
    )


def generate_cmm(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    flat = rng.normal(0.0, 0.06, n)
    frame = pd.DataFrame(
        {
            "ts": pd.Timestamp("2025-01-02") + pd.to_timedelta(rng.integers(0, 86_400 * 180, n), unit="s"),
            "part_code": parts["part_code"].values[idx],
            "op_no": rng.integers(1, OPERATION_COUNT + 2, n),
            "flatness": flat.round(4),
            "perpendicularity": rng.normal(0.0, 0.04, n).round(4),
            "spec_limit": 0.10,
            "source": rng.choice(["CMM-Direct", "PDF-Scan"], n, p=[0.78, 0.22]),
        }
    )
    return frame


def generate_visual(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    templates = [
        "精铣后缘条内侧可见轻微振纹, 判合格",
        "腹板转角处残留毛刺, 已钳修",
        "孔口倒角不均匀, 补锪",
        "表面轻微划伤, 抛光处理",
        "未见异常",
    ]
    return pd.DataFrame(
        {
            "ts": pd.Timestamp("2025-01-02") + pd.to_timedelta(rng.integers(0, 86_400 * 180, n), unit="s"),
            "part_code": parts["part_code"].values[idx],
            "op_no": rng.integers(1, OPERATION_COUNT + 1, n),
            "free_text": rng.choice(templates, n),
            "verdict": rng.choice(["合格", "返修", "报废"], n, p=[0.86, 0.11, 0.03]),
        }
    )


def generate_material(rng: np.random.Generator, n: int, parts: pd.DataFrame) -> pd.DataFrame:
    idx = rng.integers(0, len(parts), n)
    return pd.DataFrame(
        {
            "heat_lot": [f"HL{rng.integers(10000, 99999)}" for _ in range(n)],
            "part_code": parts["part_code"].values[idx],
            "serial_bound": rng.choice([True, False], n, p=[0.74, 0.26]),
            "cert_no": [f"CERT-{rng.integers(100000, 999999)}" for _ in range(n)],
            "ut_result": rng.choice(["OK", "NG"], n, p=[0.97, 0.03]),
        }
    )


def generate_device(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "machine": rng.choice(MACHINES, n),
            "spindle_hours": rng.uniform(200, 9_000, n).round(1),
            "last_calib_days": rng.integers(3, 420, n),
            "maint_ticket": [f"MT-{rng.integers(1000, 9999)}" for _ in range(n)],
            "status": rng.choice(["运行", "待机", "维修"], n, p=[0.82, 0.13, 0.05]),
        }
    )


def generate_process_records(rng: np.random.Generator, n_routes: int, parts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_parts = len(parts)
    part_codes = parts["part_code"].values
    categories = parts["category"].values
    materials = parts["material"].values
    for i in range(n_routes):
        idx = int(rng.integers(0, n_parts))
        category = categories[idx]
        length = rng.integers(8, 21)
        anchors = {
            "腹板类": ["下料", "粗铣腹板", "时效去应力", "半精铣腹板", "精铣腹板", "钳修", "三坐标终检"],
            "框类": ["下料", "粗铣外形", "时效去应力", "半精铣外形", "精铣内形", "钻铰孔系", "钳修", "三坐标终检"],
            "梁类": ["下料", "粗铣上下缘", "时效去应力", "半精铣", "精铣腹板", "锪窝", "三坐标终检"],
            "接头类": ["下料", "粗铣基准", "钻铰孔系", "精镗孔", "倒角去毛刺", "荧光渗透", "三坐标终检"],
            "肋类": ["下料", "粗铣腹板", "半精铣缘条", "精铣缘条", "倒角去毛刺", "三坐标终检"],
            "壁板类": ["下料", "滚弯成形", "淬火", "拉伸校形", "铣切边缘", "超声探伤", "阳极化"],
        }[category]
        extra = rng.choice(OPERATION_POOL, max(0, length - len(anchors)))
        seq = list(anchors) + list(extra)
        synonym_map = {"粗铣腹板": "开粗腹板", "精铣腹板": "精加工腹板"}
        seq = [synonym_map.get(op, op) for op in seq]
        rows.append(
            {
                "route_id": f"R{i:04d}",
                "part_code": part_codes[idx],
                "category": category,
                "material": materials[idx],
                "operation_seq": " > ".join(seq),
                "op_count": len(seq),
                "spindle": round(float(rng.uniform(4000, 16000)), 0),
                "feed": round(float(rng.uniform(300, 2400)), 1),
                "label_confidence": round(float(rng.uniform(0.45, 0.99)), 3),
            }
        )
    return pd.DataFrame(rows)


def generate_defect_cases(rng: np.random.Generator, n_cases: int, parts: pd.DataFrame) -> pd.DataFrame:
    modes = [f"DM-{i:02d}" for i in range(1, DEFECT_MODE_COUNT + 1)]
    cause_names = [
        "进给速度漂移", "刀具磨损超限", "装夹压紧力不足", "主轴热伸长",
        "切削液浓度偏低", "毛坯余量不均", "编程抬刀不足", "测头未校准",
        "上工序基准误差累积", "加工顺序不合理",
    ]
    causes = [f"{m}-{c}" for c in cause_names for m in FIVE_M_ONE_E[:2]]
    rows = []
    for i in range(n_cases):
        chosen = list(rng.choice(causes, size=rng.integers(1, 3), replace=False))
        rows.append(
            {
                "case_id": f"CASE-{i:04d}",
                "part_code": parts["part_code"].values[rng.integers(0, len(parts))],
                "station_no": int(rng.integers(1, KEY_STATION_COUNT + 1)),
                "defect_mode": modes[rng.integers(0, len(modes))],
                "confirmed_causes": chosen,
                "capa_closed": bool(rng.random() < 0.81),
            }
        )
    return pd.DataFrame(rows)


def generate_corpus(full: bool = False, seed: int | None = None) -> SyntheticCorpus:
    """Generate the full synthetic corpus used by both demo projects."""
    from ..config import RANDOM_SEED, seed_everything

    seed_everything(seed or RANDOM_SEED)
    rng = np.random.default_rng(seed or RANDOM_SEED)

    n_total = get_record_count(full)
    n_parts = max(120, n_total // 900)
    parts = generate_parts(rng, n_parts)

    weights = np.array([src["tables"] for src in DATA_SOURCE_CATALOG], dtype=float)
    weights = weights / weights.sum()
    counts = (weights * n_total).astype(int)

    builders = {
        "cnc": generate_cnc,
        "omp": generate_omp,
        "mes": generate_mes,
        "tool": generate_tool,
        "cmm": generate_cmm,
        "vi": generate_visual,
        "mat": generate_material,
        "dev": generate_device,
    }

    sources: Dict[str, pd.DataFrame] = {}
    for src, count in zip(DATA_SOURCE_CATALOG, counts):
        builder = builders[src["key"]]
        frame = builder(rng, count, parts) if src["key"] not in ("tool", "dev") else builder(rng, count)
        frame.attrs["source_name"] = src["name"]
        frame.attrs["channel"] = src["channel"]
        frame.attrs["kind"] = src["kind"]
        sources[src["key"]] = frame

    return SyntheticCorpus(
        sources=sources,
        parts=parts,
        process_records=generate_process_records(rng, NC_RECORD_COUNT, parts),
        routes=pd.DataFrame(),
        defects=generate_defect_cases(rng, 620, parts),
    )
