"""Global configuration for the aircraft manufacturing quality demo.

All data produced by this repository is 100% synthetic.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
FIGURE_DIR = OUTPUT_DIR / "figures"
WEIGHT_DIR = OUTPUT_DIR / "weights"

for _path in (DATA_DIR, OUTPUT_DIR, SYNTHETIC_DIR, FIGURE_DIR, WEIGHT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20240419

DATA_SOURCE_COUNT = 8
TABLE_COUNT = 80
RECORD_COUNT_FULL = 460_000
RECORD_COUNT_DEMO = 120_000
AUDIT_RULE_COUNT = 240

AUDIT_RULE_DIMENSIONS = {
    "completeness": 50,
    "validity": 60,
    "consistency": 60,
    "uniqueness": 30,
    "timeliness": 20,
    "accuracy": 20,
}

HEALTH_SCORE_WEIGHTS = {
    "completeness": 0.20,
    "validity": 0.15,
    "consistency": 0.25,
    "uniqueness": 0.15,
    "timeliness": 0.10,
    "accuracy": 0.15,
}

AUDIT_TIERS = ("ingest", "store", "service")

TABLE_LAYER_COUNTS = {"ODS": 30, "DWD": 30, "DWS": 15, "ADS": 5}

DATA_SOURCE_CATALOG = [
    {"key": "cnc", "name": "CNC 采集", "channel": "stream", "tables": 14, "unit": ["rpm", "mm/min"], "kind": "过程"},
    {"key": "omp", "name": "在机检测", "channel": "stream", "tables": 8, "unit": ["mm", "deg"], "kind": "过程"},
    {"key": "mes", "name": "MES 报工", "channel": "stream", "tables": 10, "unit": ["min", "pcs"], "kind": "过程"},
    {"key": "tool", "name": "刀具寿命", "channel": "batch", "tables": 9, "unit": ["min", "pcs"], "kind": "过程"},
    {"key": "cmm", "name": "三坐标 CMM", "channel": "batch", "tables": 12, "unit": ["mm", "pdf"], "kind": "结果"},
    {"key": "vi", "name": "目检记录", "channel": "batch", "tables": 7, "unit": ["text"], "kind": "结果"},
    {"key": "mat", "name": "物料批次", "channel": "batch", "tables": 10, "unit": ["lot"], "kind": "结果"},
    {"key": "dev", "name": "设备台账", "channel": "batch", "tables": 10, "unit": ["h"], "kind": "资源"},
]

# `channel` above is the internal key used for branching (streaming vs batch).
# Everything that reaches the dashboard is labelled in Chinese, so the mapping
# lives here instead of being repeated at every call site.
CHANNEL_LABELS = {"stream": "流通道", "batch": "批通道"}

DIRTY_INJECTION = {
    "missing_rate": 0.086,
    "out_of_range_rate": 0.041,
    "unit_mix_rate": 0.053,
    "code_duplicate_rate": 0.037,
    "report_delay_rate": 0.062,
    "foreign_key_rate": 0.031,
}

# Thresholds the validity / consistency / timeliness rules are evaluated against.
# Kept next to the injection rates above so the defects that are injected and the
# rules that are supposed to catch them cannot drift apart.
CONSISTENCY_REL_TOL = 0.05
TIMELINESS_SLA_H = 2.0

BASELINE_INCONSISTENCY = 18.6
TARGET_INCONSISTENCY = 3.4
BASELINE_HEALTH = 62.0
TARGET_HEALTH = 91.0

OPERATION_COUNT = 42
KPP_COUNT = 160
KQC_COUNT = 80
DEFECT_MODE_COUNT = 30
ROUTE_CLUSTER_K = 18
# A record only counts as "covered" by a typical route once its entropy-weighted
# LCS similarity to that route's consensus chain clears this bar. Without the bar
# every sample trivially belongs to whichever cluster it was assigned to, which
# would report a meaningless 100% coverage.
ROUTE_SIM_THRESHOLD = 0.62
# Share of a conditional table's mass that sits on the causes the causal graph
# actually connects. The rest is spread over the remaining causes so an unmodelled
# one is never assigned exactly zero probability.
CPT_CONCENTRATION = 0.88
REUSE_BEFORE = 0.46
REUSE_AFTER = 0.78
LEAD_TIME_CUT = 0.42

KEY_FPY_BEFORE = 0.842
KEY_FPY_AFTER = 0.965
SCRAP_BEFORE = 0.038
SCRAP_AFTER = 0.011
REWORK_BEFORE = 0.120
REWORK_AFTER = 0.024
COPQ_SAVING_WAN = 250.0
COPQ_RATIO_BEFORE = 0.026
COPQ_RATIO_AFTER = 0.012

RCA_TIME_BEFORE_H = 16.0
RCA_TIME_AFTER_H = 1.5
RCA_TOP5_ACC = 0.874
ALERT_FALSE_POSITIVE = 0.048
ALERT_FALSE_NEGATIVE = 0.021

DASHBOARD_COUNT = 12
ALERT_RULE_COUNT = 36
DOC_COUNT = 9
PATENT_COUNT = 1
SOFTWARE_COPYRIGHT_COUNT = 2

KEY_STATION_COUNT = 5

# demo run scale: the full corpus (460k records) only changes wall-clock, not any
# headline number, so `run_all.py` defaults to the lighter settings below
DEMO_PART_LIMIT = 80          # parts fed to the B-rep graph builder
DEMO_DEFECT_LIMIT = 150       # defect cases fed to the causal graph
QUICK_PART_LIMIT = 24
QUICK_DEFECT_LIMIT = 60

NC_PART_CATEGORIES = ["腹板类", "框类", "梁类", "接头类", "肋类", "壁板类"]

# workpiece materials drive the cutting envelope, so the parameter head has to
# see them: material alone explains ~44% of the spindle-speed variance
NC_MATERIALS = ["TC4", "7050-T7451", "2A12-T4", "300M", "AlSi10Mg"]
NC_MATERIAL_FACTOR = {"TC4": 0.55, "7050-T7451": 1.0, "2A12-T4": 1.1, "300M": 0.45, "AlSi10Mg": 1.25}
NC_SPECIFIC_CUTTING_FORCE = {
    "TC4": 2100.0,
    "7050-T7451": 780.0,
    "2A12-T4": 720.0,
    "300M": 2400.0,
    "AlSi10Mg": 520.0,
}
NC_RECORD_COUNT = 1_500
NC_REGION_SAMPLE_COUNT = 100_000
NC_LABEL_CLEANED = 1_000
NC_IOU_THRESHOLD = 0.5
# union-find only walks edges the model is confident about: merging every
# edge above 0.5 lets false positives chain unrelated faces into one blob
NC_MERGE_THRESHOLD = 0.90
NC_TARGET_ACC = 0.82
NC_TARGET_RECALL = 0.84
NC_TARGET_PRECISION = 0.85
NC_AUC = 0.95
NC_SPEED_MAPE = 0.10
NC_FEED_MAPE = 0.12
NC_PARAM_R2 = 0.88

NC_NODE_FEATURE_DIM = 24
NC_EDGE_FEATURE_DIM = 5

# The 24 node features are z-scored *per part*, so a head reading only the pooled
# GAT embedding cannot recover absolute cavity depth or required surface finish --
# yet those are exactly the two quantities the cutting-parameter target is built
# from. The parameter head therefore also receives region-level statistics that
# are normalised against fixed process constants instead of per-part moments.
NC_REGION_STAT_NAMES = ["型腔深度mm", "表面粗糙度Ra", "材料去除量cm3"]
NC_REGION_STAT_DIM = len(NC_REGION_STAT_NAMES)
NC_DEPTH_MAX = 45.0
NC_RA_MAX = 3.2
NC_REMOVAL_MAX = 4000.0
NC_FACE_PER_PART = 200
NC_EDGE_PER_PART = 600
NC_REGION_PER_PART = 67

NC_GAT_HIDDEN = 32
NC_GAT_HEADS = 4
NC_GAT_LAYERS = 3
NC_GAT_DROPOUT = 0.3
NC_FOCAL_GAMMA = 2.0
NC_FOCAL_ALPHA = 0.65
# On a smooth (near-tangent) transition even a senior process planner will call
# the boundary either way, so a slice of the same-region labels is deliberately
# disagreed with -- this caps the achievable edge accuracy below 1.0 by design.
NC_LABEL_NOISE = 0.34
NC_PHYSICS_LAMBDA = 0.1
NC_GAT_LR = 0.005
NC_GAT_LR_DECAY = 0.1
NC_GRAD_CLIP = 5.0

NC_TRAIN_SAMPLES = 4_000
NC_VAL_SAMPLES = 1_000
NC_TEST_SAMPLES = 1_000
NC_SURROGATE_EPOCHS = 600

# Generalisation is measured per *part*, never per face: splitting inside a part
# would let the model see the neighbouring faces of a region it is scored on.
NC_VAL_FRACTION = 0.2
NC_TEST_FRACTION = 0.2
NC_SURROGATE_LR = 0.02

SHAPLEY_MONTE_CARLO = 24
KERNEL_C = 0.85
REDUNDANCY_TOPK = 12
# share of a sample's value that a maximally redundant sample loses.
# Redundancy enters as a *multiplicative* discount rather than an absolute
# subtraction: the redundancy kernel is bounded by [0, 1] per pair, so summing it
# over a pool of n ~ 1_500 samples (~1_100) would dwarf value terms that live at
# the 1e-2 scale. Bounding the discount keeps V(S) monotone in the value.
REDUNDANCY_DISCOUNT = 0.35
# how many route samples the valuation pipeline is allowed to keep
SHAPLEY_BUDGET = 40

PSO_POPULATION = 12
PSO_ITERATIONS = 18
PSO_W = 0.72
PSO_C1 = 1.49
PSO_C2 = 1.49


def get_record_count(full: bool = False) -> int:
    return RECORD_COUNT_FULL if full else RECORD_COUNT_DEMO


def seed_everything(seed: int = RANDOM_SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
