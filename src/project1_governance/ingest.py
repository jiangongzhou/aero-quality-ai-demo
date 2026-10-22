"""Dual-channel ingestion and lakehouse layering for quality data governance.

Channel design
    stream channel : CNC acquisition, on-machine probing, MES reporting, tool life
    batch  channel : CMM, visual inspection, material lot, device ledger

Lakehouse layers
    ODS -> DWD -> DWS -> ADS, 30 / 30 / 15 / 5 = 80 tables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from ..config import DATA_SOURCE_CATALOG, TABLE_LAYER_COUNTS

SYNONYM_MAP = {
    "粗铣腹板": "粗铣腹板", "开粗腹板": "粗铣腹板", "Rough Mill": "粗铣腹板",
    "精铣腹板": "精铣腹板", "精加工腹板": "精铣腹板", "Finish Mill": "精铣腹板",
    "钻铰孔系": "钻铰孔系", "钻孔铰孔": "钻铰孔系", "Drill-Ream": "钻铰孔系",
    "时效去应力": "时效去应力", "去应力退火": "时效去应力", "Stress Relieve": "时效去应力",
    "三坐标终检": "三坐标终检", "CMM 终检": "三坐标终检", "Final CMM": "三坐标终检",
}

UNIT_CONVERSION = {
    ("mm/r", "mm/min"): "multiply_by_spindle",
    ("mm/min", "mm/min"): "identity",
}


@dataclass
class LayerStats:
    layer: str
    tables: int
    records: int
    purpose: str


@dataclass
class IngestionResult:
    layers: Dict[str, pd.DataFrame]
    layer_stats: List[LayerStats]
    stream_records: int
    batch_records: int
    entity_alignment: Dict[str, int] = field(default_factory=dict)
    unit_normalized: int = 0


def normalize_operations(series: pd.Series) -> pd.Series:
    return series.map(lambda s: SYNONYM_MAP.get(str(s).strip(), str(s).strip()))


def route_signature(seq: str) -> str:
    parts = [SYNONYM_MAP.get(op.strip(), op.strip()) for op in str(seq).split(">")]
    return " > ".join([p for p in parts if p])


def normalize_feed_units(frame: pd.DataFrame, rng) -> pd.DataFrame:
    mixed = frame.get("feed_unit", pd.Series(dtype=str)).eq("mm/r")
    if "feed_cmd" in frame.columns and mixed.any():
        frame.loc[mixed, "feed_cmd"] = (
            frame.loc[mixed, "feed_cmd"] * frame.loc[mixed, "spindle_cmd"].clip(lower=1)
        )
        frame.loc[mixed, "feed_unit"] = "mm/min"
    return frame


def run_ingestion(corpus, rng) -> IngestionResult:
    stream_records = 0
    batch_records = 0
    frames: List[pd.DataFrame] = []

    for src, (key, df) in zip(DATA_SOURCE_CATALOG, corpus.sources.items()):
        df = df.copy()
        df["source_key"] = key
        df["source_name"] = src["name"]
        df["channel"] = src["channel"]
        if src["channel"] == "stream":
            stream_records += len(df)
        else:
            batch_records += len(df)
        frames.append(df)

    ods = pd.concat(frames, ignore_index=True)
    ods["ingest_tier"] = "ingest"

    dwd = ods.copy()
    if "operation_seq" in dwd.columns:
        dwd["operation_seq"] = normalize_operations(dwd["operation_seq"])
    dwd = normalize_feed_units(dwd, rng)
    dwd["part_key"] = dwd["part_code"].astype(str).str.upper().str.strip()
    dwd["op_key"] = dwd.get("op_no", pd.Series(index=dwd.index)).astype(str)
    dwd["ingest_tier"] = "store"
    unit_normalized = int(ods.get("feed_unit", pd.Series(dtype=str)).eq("mm/r").sum())

    dws = (
        dwd.groupby(["part_key", "op_key"], as_index=False)
        .agg(
            spindle_mean=("spindle_cmd", "mean") if "spindle_cmd" in dwd.columns else ("part_key", "size"),
            record_count=("part_key", "size"),
        )
    ) if "spindle_cmd" in dwd.columns else pd.DataFrame({"part_key": dwd["part_key"].unique()})
    dws["ingest_tier"] = "store"

    ads = pd.DataFrame(
        {
            "metric": ["fpy", "rty", "scrap_rate", "rework_rate", "copq_wan"],
            "value": [0.842, 0.423, 0.038, 0.120, 260.0],
            "owner": ["质量部", "质量部", "工艺部", "生产部", "财务部"],
            "refresh": ["daily", "daily", "daily", "daily", "monthly"],
        }
    )
    ads["ingest_tier"] = "service"

    layers = {"ODS": ods, "DWD": dwd, "DWS": dws, "ADS": ads}

    stats = [
        LayerStats("ODS", TABLE_LAYER_COUNTS["ODS"], len(ods), "原样保真, 不做清洗"),
        LayerStats("DWD", TABLE_LAYER_COUNTS["DWD"], len(dwd), "统一编码/单位归一化, 稽核主战场"),
        LayerStats("DWS", TABLE_LAYER_COUNTS["DWS"], len(dws), "按件/工序聚合宽表"),
        LayerStats("ADS", TABLE_LAYER_COUNTS["ADS"], len(ads), "指标口径收敛, 直接喂看板"),
    ]

    return IngestionResult(
        layers=layers,
        layer_stats=stats,
        stream_records=stream_records,
        batch_records=batch_records,
        unit_normalized=unit_normalized,
    )
