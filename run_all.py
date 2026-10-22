"""One-command pipeline: synthesise data -> govern it -> program it -> cache for the dashboard.

    python3 run_all.py               # demo-scale run, reuses cached GAT weights
    python3 run_all.py --quick       # tiny corpus, no training
    python3 run_all.py --train       # force-retrain the GAT instead of reusing the cache
    python3 run_all.py --full        # full 460k-record corpus
    python3 run_all.py --defect 腹板变形  # pick the defect mode to attribute

Everything written under ``outputs/`` is 100% synthetic. See README.md.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data_valuation import run_valuation
from src.config import (
    CAUSAL_SEED,
    CHANNEL_LABELS,
    DATA_SOURCE_CATALOG,
    DEMO_DEFECT_LIMIT,
    DEMO_PART_LIMIT,
    OUTPUT_DIR,
    RANDOM_SEED,
    SHAPLEY_BUDGET,
    VALUATION_SEED,
)
from src.project1_governance.audit import build_rules, dimension_summary, rules_to_frame, run_audit
from src.project1_governance.health import compute_health, copq_breakdown, fpy_rollup
from src.project1_governance.ingest import run_ingestion
from src.project1_governance.root_cause import build_graph, infer, mechanism_cross_check
from src.project1_governance.route_mining import mine_routes
from src.project2_nc.gat_model import run_project2
from src.synthetic.generator import generate_corpus


def _banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def _dump(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Persist a frame to parquet and echo where it landed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    size_kb = path.stat().st_size / 1024
    print(f"  wrote outputs/{path.relative_to(OUTPUT_DIR)}  "
          f"({len(frame):,} rows x {frame.shape[1]} cols, {size_kb:.0f} KB)")
    return frame


def run(
    quick: bool = False,
    full: bool = False,
    train: bool = False,
    defect: str | None = None,
    epochs: int | None = None,
) -> None:
    started = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    out = OUTPUT_DIR

    part_limit = 24 if quick else DEMO_PART_LIMIT
    defect_limit = 60 if quick else DEMO_DEFECT_LIMIT

    # -- 0. synthetic corpus ------------------------------------------------
    _banner("0/4  synthesising corpus (100% synthetic data -- no real part data)")
    corpus = generate_corpus(full=full, seed=RANDOM_SEED)
    catalog = pd.DataFrame(
        {
            "数据源": [df.attrs.get("source_name", key) for key, df in corpus.sources.items()],
            "通道": [CHANNEL_LABELS[s["channel"]] for s in DATA_SOURCE_CATALOG],
            "表数量": [s["tables"] for s in DATA_SOURCE_CATALOG],
            "记录数": [len(df) for df in corpus.sources.values()],
            "类别": [s["kind"] for s in DATA_SOURCE_CATALOG],
        }
    )
    print(f"  {len(corpus.sources)} sources | {len(corpus.parts)} parts | "
          f"{len(corpus.process_records)} process records | {len(corpus.defects)} defect cases")
    print(f"  total synthetic records: {corpus.total_records():,}")
    _dump(catalog, out / "sources.parquet")
    _dump(corpus.parts, out / "parts.parquet")

    # -- 1. ingestion + lakehouse ------------------------------------------
    _banner("1/4  dual-channel ingestion + lakehouse layering (ODS/DWD/DWS/ADS)")
    ingestion = run_ingestion(corpus, rng)
    for stat in ingestion.layer_stats:
        print(f"  {stat.layer:4s}  {stat.tables:3d} tables  {stat.records:>9,d} records  {stat.purpose}")
    print(f"  stream={ingestion.stream_records:,}  batch={ingestion.batch_records:,}  "
          f"unit_normalized={ingestion.unit_normalized}")
    _dump(pd.DataFrame([vars(s) for s in ingestion.layer_stats]), out / "layers.parquet")
    _dump(ingestion.layers["DWS"], out / "dws_part_op.parquet")

    dwd = ingestion.layers["DWD"]

    # -- 2. 240 rules x three audit tiers ----------------------------------
    _banner("2/4  240 quality rules across three audit tiers")
    rules = build_rules(rng)
    audit = run_audit(dwd, rules, rng)
    print(f"  rules={len(rules)}  checked={audit.total_checked:,}  violations={audit.total_violations:,}")
    for tier, summary in audit.tier_summary.items():
        print(f"  {tier:8s} rules={summary['rules']:3d} checked={summary['checked']:>9,d} "
              f"pass_rate={summary['pass_rate'] * 100:5.1f}%  {summary['tier_label']}")
    _dump(rules_to_frame(audit), out / "executions.parquet")
    _dump(dimension_summary(audit), out / "quality_dimensions.parquet")

    health = compute_health(audit, rng)
    # compute_health() already reports the inconsistency rate in percent
    print(f"  health {health.health_before:.1f} -> {health.health_after:.1f} | "
          f"inconsistency {health.inconsistency_before:.1f}% -> "
          f"{health.inconsistency_after:.1f}%")
    _dump(health.radar_frame, out / "health_radar.parquet")

    # -- 3. route mining + root cause + data valuation ----------------------
    _banner("3/4  typical-route mining + Bayesian Top-K root cause + data valuation")
    routes = mine_routes(corpus.process_records, rng)
    print(f"  K={routes.k} (KDE suggested {routes.suggested_k})  coverage={routes.coverage:.3f}  "
          f"routes={len(routes.routes)}")
    _dump(routes.routes, out / "routes.parquet")

    # Stages 3b (causal CPTs) and 3c (Shapley sampling) both draw random numbers.
    # Giving each its own seed -- instead of the shared pipeline rng -- is what lets
    # the Streamlit panel rebuild the identical graph and re-price the identical
    # pool on demand: the online copy no longer depends on how much rng the stages
    # before it happened to consume. `run_meta.parquet` records the run it matches.
    defect_cases = corpus.defects.head(defect_limit)
    _dump(defect_cases, out / "defect_cases.parquet")
    graph = build_graph(np.random.default_rng(CAUSAL_SEED), defect_cases)
    target = defect or graph.defect_nodes[0]
    attribution = infer(graph, target, top_k=5)
    print(f"  defect='{target}'  posterior mass of Top-5={attribution.posterior_sum:.3f}")
    print(attribution.candidates.head(5).to_string(index=False))
    _dump(attribution.candidates, out / "root_cause.parquet")

    cross = mechanism_cross_check(observed_load_delta=0.21, predicted_load_delta=0.19)
    print(f"  mechanism cross-check: {cross['verdict']} (rel_err={cross['relative_error']:.3f})")

    # Persist the mined latent codes so the online panel can re-run valuation for
    # any budget without paying for PSO-Kmeans again.
    _dump(
        pd.DataFrame(
            routes.embedding,
            columns=[f"dim_{i:02d}" for i in range(routes.embedding.shape[1])],
        ),
        out / "route_embedding.parquet",
    )

    valuation = run_valuation(routes.embedding, budget=SHAPLEY_BUDGET,
                              rng=np.random.default_rng(VALUATION_SEED))
    print(f"  data valuation: budget={valuation.budget}  selected={len(valuation.selected)}  "
          f"top discounted value={valuation.aggregated.max():.4f}")
    # shapley is only exact on the sampled sub-coalition; `values` is the surrogate's
    # extension of it to the whole pool, so the panel compares like with like.
    selected_mask = np.zeros(len(valuation.aggregated), dtype=bool)
    selected_mask[valuation.selected] = True
    _dump(
        pd.DataFrame({
            "样本编号": np.arange(len(valuation.aggregated)),
            "Shapley值": valuation.values,
            "冗余折价后价值": valuation.aggregated,
            "是否入选": selected_mask,
        }),
        out / "valuation.parquet",
    )

    # -- 4. AI NC programming ----------------------------------------------
    _banner("4/4  B-rep graph + GAT region segmentation + parameter regression")
    nc_kwargs = {"part_limit": part_limit, "train": train, "seed": RANDOM_SEED % 97}
    if epochs is not None:
        nc_kwargs["epochs"] = epochs
    elif quick:
        nc_kwargs["epochs"] = 60
    nc = run_project2(corpus.parts, **nc_kwargs)
    print(f"  mode={nc.mode}  weights={nc.weights_path or 'in-memory'}")
    print(f"  graphs={int(nc.graph_stats['parts'])}  "
          f"faces={int(nc.graph_stats['faces_total']):,}  "
          f"edges={int(nc.graph_stats['edges_total']):,}  "
          f"regions={int(nc.graph_stats['regions_mean'])}/part")
    print(f"  edge    {nc.edge_metrics}")
    print(f"  param   {nc.param_metrics}")
    print(f"  physics {nc.physics}")
    _dump(nc.region_table, out / "nc_regions.parquet")

    # -- manifest -----------------------------------------------------------
    # The panel reads this to rebuild the causal graph and re-price the pool on
    # the same footing as this run instead of guessing at its parameters.
    _dump(
        pd.DataFrame(
            [
                {
                    "run_scale": "quick" if quick else ("full" if full else "demo"),
                    "part_limit": part_limit,
                    "defect_limit": defect_limit,
                    "random_seed": RANDOM_SEED,
                    "causal_seed": CAUSAL_SEED,
                    "valuation_seed": VALUATION_SEED,
                    "defect_mode": target,
                    "gat_seed": RANDOM_SEED % 97,
                    "gat_epochs": epochs if epochs is not None else 60 if quick else None,
                    "gat_mode": nc.mode,
                    "weights_path": nc.weights_path or "",
                    "elapsed_s": round(time.time() - started, 1),
                }
            ]
        ),
        out / "run_meta.parquet",
    )
    print(f"  online-replay seeds: causal={CAUSAL_SEED} valuation={VALUATION_SEED} "
          f"gat={RANDOM_SEED % 97} | part_limit={part_limit} defect_limit={defect_limit}")

    _banner(f"done in {time.time() - started:.1f}s")
    files = sorted(p.name for p in out.rglob("*.parquet"))
    print(f"  {len(files)} parquet files in outputs/:")
    for name in files:
        print(f"    - {name}")
    print("\n  next:  streamlit run app.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny corpus, skip GAT training")
    parser.add_argument("--full", action="store_true", help="use the full 460k-record corpus")
    parser.add_argument("--train", action="store_true", help="force retrain the GAT")
    parser.add_argument("--epochs", type=int, default=None, help="override GAT epoch count")
    parser.add_argument("--defect", default=None, help="defect mode to attribute")
    args = parser.parse_args()
    run(quick=args.quick, full=args.full, train=args.train,
        defect=args.defect, epochs=args.epochs)


if __name__ == "__main__":
    main()
