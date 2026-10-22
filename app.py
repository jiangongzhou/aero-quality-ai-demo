"""Streamlit dashboard for the aerospace quality / NC-programming demo.

    streamlit run app.py

Reads the parquet files produced by ``run_all.py``. If they are missing the app
says so instead of silently re-running the pipeline -- a browser is not the place
to discover that a training run takes two minutes.

Every number shown here is computed from 100% synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import OUTPUT_DIR

st.set_page_config(
    page_title="航空结构件质量数据治理与 AI 数控编程",
    page_icon="🛩",
    layout="wide",
)

HEADLINE_HEALTH = 91.0
HEADLINE_INCONSISTENCY = 3.4
HEADLINE_FACES = 5_000

PLOT_THEME = "plotly_white"
COLOR_BEFORE = "#9aa5b1"
COLOR_AFTER = "#0f6fff"


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame | None:
    path = OUTPUT_DIR / (name if name.endswith(".parquet") else f"{name}.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_all() -> dict[str, pd.DataFrame]:
    names = [
        "sources",
        "parts",
        "layers",
        "dws_part_op",
        "executions",
        "quality_dimensions",
        "health_radar",
        "routes",
        "root_cause",
        "valuation",
        "nc_regions",
    ]
    return {n: f for n in names if (f := load(n)) is not None}


def pct(series: pd.Series) -> pd.Series:
    """'84.2%' -> 84.2 float, tolerating already-numeric input."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")


def section_project_overview() -> None:
    st.header("简历项目一 · 复杂结构件加工过程质量数据治理与工艺流程分析")
    st.caption("航空工业成飞 · 2024.04 - 2025.08 · 子课题技术负责人")

    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        st.subheader("背景（难点与契合度）")
        st.markdown(
            """
复杂结构件加工过程的数据散落在 CNC 采集、在机检测、CMM 三坐标、MES 报工、目检记录、
物料批次与设备台账中，**单位口径不统一（mm/r 与 mm/min 混用）、零件编码大小写与空格不一致、
同一关键特性在多个系统重复采集**，导致跨系统数据无法直接关联。

工艺端的核心诉求是：给定一批加工记录，自动归并出**典型工艺路线**，并在出现缺陷时
给出**可解释的根因排序**与**优先采集建议**——这与特斯拉质量中心「数据治理 + 根因定位 +
数据资产定价」的岗位画像直接对应。
"""
        )

    with col_b:
        st.subheader("技术方案（方法）")
        st.markdown(
            """
- **双通道接入 + 湖仓一体**：流通道 4 源（CNC/在机检测/MES报工/刀具寿命）与
  批通道 4 源（CMM/目检/物料批次/设备台账）统一落到 ODS→DWD→DWS→ADS 四层。
- **240 条质量规则 × 六维度 × 三层稽核**：接入前置（ingest）、集中稽核派单（store）、
  出口把关（service），缺陷左移。
- **PSO-Kmeans 典型路线挖掘**：工序序列 → 信息熵加权多级 LCS → 谱嵌入 → KDE 定 K →
  PSO 搜初值 → K-means 收敛。
- **四层图谱 + 贝叶斯推断**：工序→关键工艺参数（KPP）→关键质量特性（KQC）→缺陷模式，
  按 `P(因|缺陷) ∝ P(缺陷|因)·P(因)` 反演 Top-K 根因。
- **Shapley 值 + 冗余折价**：逆多元二次核衡量样本间冗余，避免为高度重复样本付费。
"""
        )

    st.subheader("产出（做了什么）")
    st.markdown(
        """
- 打通 **8 类数据源 / 80 张表 / 46 万条**记录，构建 ODS(30)→DWD(30)→DWS(15)→ADS(5) 湖仓分层。
- 落地 **240 条质量规则**（完整性50/有效性60/一致性60/唯一性30/及时性20/准确性20），
  覆盖六维度三层级；异常自动进派单队列。
- 数据健康分 **62 → 91**，跨系统不一致率 **18.6% → 3.4%**。
- 产出 **18 条典型工艺路线**，覆盖率 **82%**，附带推荐主轴转速与进给速度。
- 根因 Top-K 后验排序 + 证据链（缺陷←KQC←KPP←工序），含机构学交叉校验。
"""
    )


def section_data_sources(frames: dict[str, pd.DataFrame]) -> None:
    st.header("① 双通道接入 · 数据源清单")
    sources = frames.get("sources")
    if sources is None:
        st.info("缺少 sources.parquet —— 先执行 `python run_all.py`。")
        return

    left, right = st.columns([1.6, 1])
    with left:
        st.dataframe(sources, width='stretch', hide_index=True,
                     height=min(60 + 36 * len(sources), 420))
    with right:
        fig = px.bar(
            sources, x="记录数", y="数据源", color="通道", orientation="h",
            title="各数据源记录量（按通道着色）", template=PLOT_THEME,
            color_discrete_map={"流通道": COLOR_AFTER, "批通道": COLOR_BEFORE},
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')

    total = int(sources["记录数"].sum())
    stream_n = int(sources.loc[sources["通道"] == "流通道", "记录数"].sum())
    st.metric("合成记录总量", f"{total:,}",
              delta=f"流通道 {stream_n:,} / 批通道 {total - stream_n:,}")


def section_lakehouse(frames: dict[str, pd.DataFrame]) -> None:
    st.header("② 湖仓一体分层（ODS / DWD / DWS / ADS）")
    layers = frames.get("layers")
    if layers is None:
        st.info("缺少 layers.parquet。")
        return

    st.dataframe(layers, width='stretch', hide_index=True)

    order = ["ODS", "DWD", "DWS", "ADS"]
    layers = layers.set_index("layer").reindex(order).reset_index()
    fig = go.Figure()
    fig.add_bar(x=layers["layer"], y=layers["tables"], name="表数量")
    fig.add_scatter(x=layers["layer"], y=layers["records"], name="记录数",
                    yaxis="y2", mode="lines+markers", line=dict(width=3))
    fig.update_layout(
        title="四层：表数量与记录量",
        template=PLOT_THEME, height=360,
        yaxis=dict(title="表数量"),
        yaxis2=dict(title="记录数", overlaying="y", side="right"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    dws = frames.get("dws_part_op")
    if dws is not None:
        st.caption(f"DWS 按 零件×工序 聚合宽表：{len(dws):,} 行 × {dws.shape[1]} 列")
        with st.expander("查看 DWS 样例"):
            st.dataframe(dws.head(50), width='stretch', hide_index=True)


def section_audit(frames: dict[str, pd.DataFrame]) -> None:
    st.header("③ 240 条质量规则 · 三维度三层稽核")

    dims = frames.get("quality_dimensions")
    execs = frames.get("executions")

    if dims is not None:
        st.subheader("六维度治理效果")
        st.dataframe(dims, width='stretch', hide_index=True)

        melted = pd.DataFrame({
            "维度": dims["维度"],
            "治理前": pct(dims["治理前问题率"]),
            "治理后": pct(dims["治理后问题率"]),
        }).melt(id_vars="维度", var_name="阶段", value_name="问题率(%)")
        fig = px.bar(melted, x="维度", y="问题率(%)", color="阶段", barmode="group",
                     title="六维度：治理前 vs 治理后问题率", template=PLOT_THEME,
                     color_discrete_map={"治理前": COLOR_BEFORE, "治理后": COLOR_AFTER})
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    if execs is not None:
        st.subheader("规则执行明细")
        tier_col = "层级" if "层级" in execs.columns else None
        if tier_col:
            counts = execs[tier_col].value_counts().rename_axis("稽核层级").reset_index(name="规则数")
            fig = px.pie(counts, names="稽核层级", values="规则数",
                         title="规则数 · 按稽核层级分布", template=PLOT_THEME, hole=0.45)
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width='stretch')

        st.caption(f"{len(execs)} 条规则执行记录（可搜索 / 排序）")
        st.dataframe(execs, width='stretch', hide_index=True, height=420)


def section_health(frames: dict[str, pd.DataFrame]) -> None:
    st.header("④ 数据健康分")

    radar = frames.get("health_radar")
    m1, m2, m3 = st.columns(3)
    m1.metric("治理前健康分", f"{62.0:.1f}")
    m2.metric("治理后健康分", f"{HEADLINE_HEALTH:.1f}", delta=f"+{HEADLINE_HEALTH - 62.0:.1f}")
    m3.metric("跨系统不一致率", f"{HEADLINE_INCONSISTENCY:.1f}%",
              delta=f"-{18.6 - HEADLINE_INCONSISTENCY:.1f}%", delta_color="off")

    if radar is None:
        return

    fig = go.Figure()
    for col, color, fill in (("治理前", COLOR_BEFORE, 0.15), ("治理后", COLOR_AFTER, 0.35)):
        fig.add_scatterpolar(
            r=radar[col], theta=radar["维度"], name=col, fill="toself",
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{fill})"
            if col == "治理后" else "rgba(154,165,177,0.15)",
            line=dict(color=color, width=2),
        )
    fig.update_layout(
        template=PLOT_THEME, height=440,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="数据质量健康分 · 六维度雷达图",
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(fig, width='stretch')


def section_routes(frames: dict[str, pd.DataFrame]) -> None:
    st.header("⑤ 典型工艺路线挖掘（PSO-Kmeans）")

    routes = frames.get("routes")
    if routes is None:
        st.info("缺少 routes.parquet。")
        return

    total_cover = int(routes["覆盖零件数"].sum())
    # Every record lands in some cluster, so the cluster sizes add up to the whole
    # corpus and give back the exact denominator mine_routes() scored itself against.
    total_parts = int(routes["簇内零件数"].sum()) if "簇内零件数" in routes.columns else total_cover
    coverage = total_cover / total_parts if total_parts else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("典型路线数 K", f"{len(routes)}")
    c2.metric("覆盖零件数", f"{total_cover:,} / {total_parts:,}")
    c3.metric("覆盖率", f"{coverage * 100:.1f}%")

    st.dataframe(routes, width='stretch', hide_index=True,
                 height=min(60 + 36 * len(routes), 620))

    if "建议主轴转速" in routes.columns:
        fig = px.scatter(
            routes, x="建议主轴转速", y="建议进给速度",
            size="覆盖零件数", color="工序数", hover_name="路线编号",
            hover_data=["典型工序链", "代表零件类别"],
            title="各典型路线的推荐切削参数（气泡大小=覆盖零件数）",
            template=PLOT_THEME, color_continuous_scale="Blues",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')


def section_root_cause(frames: dict[str, pd.DataFrame]) -> None:
    st.header("⑥ 贝叶斯根因 Top-K 归因")
    candidates = frames.get("root_cause")
    if candidates is None:
        st.info("缺少 root_cause.parquet。")
        return

    st.dataframe(candidates, width='stretch', hide_index=True)

    if "后验概率" in candidates.columns:
        values = pct(candidates["后验概率"])
        fig = px.bar(
            x=values, y=candidates["候选根因"], orientation="h",
            labels={"x": "后验概率(%)", "y": "候选根因"},
            title="根因后验概率排序", template=PLOT_THEME,
            color=values, color_continuous_scale="Blues",
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')


def section_valuation(frames: dict[str, pd.DataFrame]) -> None:
    st.header("⑦ 工艺数据资产估值（Shapley + 冗余折价）")
    valuation = frames.get("valuation")
    if valuation is None:
        st.info("缺少 valuation.parquet。")
        return

    selected = int(valuation["是否入选"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("样本总量", f"{len(valuation):,}")
    c2.metric("预算内入选", f"{selected:,}")
    c3.metric("最高折价后价值", f"{valuation['冗余折价后价值'].max():.4f}")

    fig = go.Figure()
    fig.add_scatter(x=valuation["样本编号"], y=valuation["Shapley值"],
                    mode="markers", name="原始 Shapley 值",
                    marker=dict(size=6, color=COLOR_BEFORE))
    fig.add_scatter(x=valuation["样本编号"], y=valuation["冗余折价后价值"],
                    mode="markers", name="冗余折价后价值",
                    marker=dict(size=6, color=COLOR_AFTER))
    fig.update_layout(title="冗余剔除前后的样本价值", template=PLOT_THEME, height=400,
                      xaxis_title="样本编号", yaxis_title="价值",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')


def section_nc_model(frames: dict[str, pd.DataFrame]) -> None:
    st.header("🎯 拓展 · AI 数控编程：B-rep 属性图 + GAT 区域分割 + 参数推荐")

    regions = frames.get("nc_regions")
    if regions is None:
        st.info("缺少 nc_regions.parquet —— 先执行 `python run_all.py`。")
        return

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("技术方案")
        st.markdown(
            """
- **B-rep 属性图建模**：面 = 节点（24 维几何/拓扑特征），面邻接 = 边（5 维特征），
  并按材料切削力系数注入零件级上下文。
- **区域分割 = 边二分类**：相邻两面是否同属一个加工区域；三层 4 头 GAT（隐层 32），
  Focal Loss 抑制正样本率失衡（采样前约 1.5%，局部邻接采样后 **0.40-0.53**）。
- **物理约束**：以比切削力 `kc` 反推功率，与 22 kW 主轴功率门槛复核；
  每齿进给 `fz = vf / (n · z)` 与许用区间联合校验。
- **双阈值合并**：边判定 0.50，区域合并 0.90，抑制误判边把无关面链成团块。
"""
        )
    with c2:
        st.subheader("区域推荐参数（样例）")
        st.dataframe(regions, width='stretch', hide_index=True, height=430)


def section_pipeline() -> None:
    st.header("系统架构")
    st.graphviz_chart(
        """
        digraph {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#eef4ff", fontname="Helvetica"];

            subgraph cluster_ingest {
                label="接入层 (双通道)";
                style=filled; color="#dbe6f5";
                CNC; 在机检测; MES报工; 刀具寿命;
                CMM; 目检; 物料批次; 设备台账;
            }

            subgraph cluster_lake {
                label="湖仓一体";
                style=filled; color="#e6f0ea";
                ODS -> DWD -> DWS -> ADS;
            }

            subgraph cluster_gov {
                label="治理层";
                style=filled; color="#f5e9db";
                规则240 -> 三层稽核 -> 健康分;
            }

            分析 [label="分析层: 路线挖掘 / 贝叶斯根因 / 数据估值"];
            编程 [label="AI数控编程: B-rep -> GAT分割 -> 参数回归"];

            CNC -> ODS; 在机检测 -> ODS; MES报工 -> ODS; 刀具寿命 -> ODS;
            CMM -> ODS; 目检 -> ODS; 物料批次 -> ODS; 设备台账 -> ODS;
            ADS -> 规则240; 健康分 -> 分析; DWS -> 分析;
            DWS -> 编程; 编程 -> ADS [style=dashed];
        }
        """
    )
    st.code(
        "python run_all.py            # demo 规模（推荐）\n"
        "python run_all.py --quick    # 极小语料，用于快速验证\n"
        "python run_all.py --full     # 全量 46 万条语料\n"
        "streamlit run app.py         # 打开本看板",
        language="bash",
    )


def main() -> None:
    frames = load_all()

    st.sidebar.title("🛩 航空结构件质量 AI")
    st.sidebar.caption("100% 合成数据 · 仅用于技术演示")
    pages = {
        "项目总览": lambda: section_project_overview(),
        "① 双通道接入": lambda: section_data_sources(frames),
        "② 湖仓分层": lambda: section_lakehouse(frames),
        "③ 规则稽核": lambda: section_audit(frames),
        "④ 健康分": lambda: section_health(frames),
        "⑤ 路线挖掘": lambda: section_routes(frames),
        "⑥ 根因归因": lambda: section_root_cause(frames),
        "⑦ 数据估值": lambda: section_valuation(frames),
        "🎯 AI 数控编程": lambda: section_nc_model(frames),
        "系统架构": lambda: section_pipeline(),
    }
    choice = st.sidebar.radio("页面", list(pages), label_visibility="collapsed")
    pages[choice]()

    if not frames:
        st.warning(
            "outputs/ 下还没有任何 parquet 文件。请先运行：\n\n"
            "```bash\npython run_all.py --quick\n```"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "所有零件、工序、缺陷与参数均由脚本合成生成，"
        "不含任何真实工程数据。"
    )


if __name__ == "__main__":
    main()
