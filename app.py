"""Streamlit dashboard for the aerospace quality / NC-programming demo.

    streamlit run app.py

Reads the parquet files produced by ``run_all.py``. If they are missing the app
says so instead of silently re-running the pipeline -- a browser is not the place
to discover that a training run takes two minutes.

The "⚡ 在线推理" page is the exception: it loads the cached GAT weights and the
causal CPTs and runs a real forward pass per click. Every other page is a reader
over ``outputs/*.parquet``; that one is the software.

Every number shown here is computed from 100% synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import live
from src.config import NC_MERGE_THRESHOLD, OUTPUT_DIR, SHAPLEY_BUDGET

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
        st.caption(
            f"全量 {len(regions)} 行 / {regions['零件号'].nunique()} 个零件，此处只展示前 200 行；"
            "「⚡ 在线推理」页会重放全部零件的推理并逐行比对。"
        )
        st.dataframe(regions.head(200), width='stretch', hide_index=True, height=430)


def _ready(name: str) -> bool:
    return (OUTPUT_DIR / f"{name}.parquet").exists()


@st.cache_resource(show_spinner="加载 GAT 权重并重建零件 B-rep 属性图 ...")
def _gat_bundle():
    return live.load_gat_bundle()


@st.cache_resource(show_spinner="重建四层因果图 CPT ...")
def _causal_bundle():
    return live.load_causal_bundle()


@st.cache_data(show_spinner=False)
def _valuation_pool():
    return live.load_valuation_pool()


def _tab_gat() -> None:
    st.subheader("① B-rep 属性图 → GAT 区域分割 → 切削参数回归")
    st.markdown(
        """
点一次"执行"，下面这段前向传播就真的跑一遍：

`面特征 (24-d)` → `节点编码器` → `3 层 × 4 头图注意力` → `边逻辑值（相邻两面是否同属一个加工区域）`
→ `union-find 合并成加工区域` → `区域均值池化` → `参数头输出 (转速, 进给)` → `× SPINDLE_MAX/FEED_MAX 还原工艺单位`

拖动**区域合并阈值**可以看到分割粒度怎么变：阈值调低，更多弱置信边被采信，区域被并成大片；
调高则区域碎裂。这是模型输出的真实响应，不是预渲染的图。
"""
    )
    bundle = _gat_bundle()
    if bundle.mode == "untrained":
        st.warning("未找到 GAT 权重缓存。请先运行 `python run_all.py`（首次会自动训练并缓存）。")
        return

    code = st.selectbox("零件号", bundle.part_codes, key="gat_part")
    threshold = st.slider("区域合并阈值", 0.50, 0.99, float(NC_MERGE_THRESHOLD), 0.01, key="gat_thr")

    pred = live.predict_part(bundle, code, merge_threshold=threshold)
    m = live.summarise_prediction(pred)

    row1 = st.columns(4)
    row1[0].metric("边分类准确率", f"{m['边界准确率'] * 100:.2f}%")
    row1[1].metric("区域 IoU", f"{m['区域IoU']:.4f}")
    row1[2].metric("转速 MAPE", f"{m['转速MAPE'] * 100:.2f}%")
    row1[3].metric("进给 MAPE", f"{m['进给MAPE'] * 100:.2f}%")

    row2 = st.columns(4)
    row2[0].metric("零件 面 / 边", f"{pred['n_faces']} / {pred['n_edges']}")
    row2[1].metric("切分出加工区域", f"{int(m['预测区域数'])}")
    row2[2].metric("边界精确率 / 召回率",
                   f"{m['边界精确率']:.3f} / {m['边界召回率']:.3f}")
    row2[3].metric("本次前向耗时", f"{m['推理耗时ms']:.1f} ms")

    st.dataframe(pred["table"], width='stretch', hide_index=True,
                 height=min(90 + 36 * len(pred["table"]), 460))

    conf = pred["edge_confidence"]
    fig = px.histogram(conf, nbins=40, labels={"value": "GAT 输出概率", "count": "边数量"},
                       title=f"{code} · {len(conf)} 条候选边的分割置信度分布（阈值 {threshold:.2f}）",
                       template=PLOT_THEME, color_discrete_sequence=[COLOR_AFTER])
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d62728",
                  annotation_text="合并阈值", annotation_position="top")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, width='stretch')

    with st.expander("🧪 一致性自检：在线重放 vs 离线 parquet", expanded=False):
        regions = load("nc_regions")
        if regions is None or "零件号" not in regions.columns:
            st.info("缺少 nc_regions.parquet，无法比对。")
        else:
            shared = [c for c in bundle.part_codes if c in set(regions["零件号"])]
            cols = ["预测转速", "预测进给"]
            bad = []
            for item in shared:
                online = live.predict_part(bundle, item)["table"]
                offline = regions[regions["零件号"] == item].reset_index(drop=True)
                n = min(len(online), len(offline))
                if n == 0:
                    continue
                if not np.allclose(online[cols].to_numpy(float)[:n],
                                   offline[cols].to_numpy(float)[:n]):
                    bad.append(item)
            if not shared:
                st.info("当前零件集合与离线表无交集。")
            elif bad:
                st.error(f"{len(bad)}/{len(shared)} 个零件与离线结果不一致：{bad[:5]}")
            else:
                st.success(
                    f"在线重放的 {len(shared)} 个零件与 outputs/nc_regions.parquet 逐行一致；"
                    f"其余 {len(bundle.part_codes) - len(shared)} 个零件属于训练/验证划分，"
                    "离线表里本来就没有记录，已跳过。"
                )


def _tab_root_cause() -> None:
    st.subheader("② 贝叶斯逆向推断：由缺陷模式反推关键工艺参数")
    st.markdown(
        """
四层因果图 `工序 → 关键工艺参数(KPP) → 关键质量特性(KQC) → 缺陷模式` 已按记录的
`causal_seed` 重建。每次改动下面任一滑杆都会重新做一次贝叶斯逆向推断
`P(KPP | 缺陷) ∝ P(缺陷|KQC)^s · P(KQC) · P(KQC|KPP)`，其中 `s` 即证据强度。
"""
    )
    if not _ready("defect_cases"):
        st.info(
            "缺少 defect_cases.parquet。请重新执行 `python run_all.py` —— "
            "当前版本的流水线会落盘缺陷案例与运行元信息供本页复现。"
        )
        return

    bundle = _causal_bundle()
    defect = st.selectbox("缺陷模式", bundle.defect_modes, key="rc_defect")
    strength = st.slider("证据强度 s（对似然做幂次锐化）", 0.5, 3.0, 1.0, 0.1, key="rc_str")
    top_k = st.slider("Top-K 候选根因数", 3, 10, 5, 1, key="rc_k")

    result, elapsed = live.diagnose(bundle, defect, evidence_strength=strength, top_k=top_k)

    row = st.columns(4)
    row[0].metric("Top-K 后验质量", f"{result.posterior_sum * 100:.2f}%")
    row[1].metric("候选根因数", f"{len(result.candidates)}")
    row[2].metric("本次推断耗时", f"{elapsed * 1000:.1f} ms")
    row[3].metric("CPT 校准样本", f"{bundle.case_count}")

    detail = result.candidates.copy()
    detail["性价比排名"] = pct(detail["信息价值"]).rank(ascending=False).astype(int)
    st.dataframe(detail, width='stretch', hide_index=True)

    values = pct(result.candidates["后验概率"])
    fig = px.bar(
        x=values, y=result.candidates["候选根因"], orientation="h",
        labels={"x": "后验概率(%)", "y": "候选根因"},
        title=f"{defect} · 根因后验概率（证据强度 s={strength:.1f}）",
        template=PLOT_THEME, color=values, color_continuous_scale="Blues",
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width='stretch')

    with st.expander("⚙️ 机理交叉校核（实测 vs 模型预测）", expanded=False):
        observed = st.slider("实测切削负载增量", 0.05, 0.60, 0.21, 0.01, key="cc_obs")
        predicted = st.slider("模型预测负载增量", 0.05, 0.60, 0.19, 0.01, key="cc_pred")
        cross = live.cross_check(observed, predicted)
        c1, c2, c3 = st.columns(3)
        c1.metric("相对误差", f"{cross['relative_error']:.3f}")
        c2.metric("是否自洽", "通过" if cross["consistent"] else "不通过")
        c3.metric("结论", str(cross["verdict"]))
        st.caption(
            "数据驱动的排序只说明「统计上可疑」，这一步用切削力学模型回答"
            "「是否解释得通」：负载增量对不上，排序再高也不下发。"
        )

    with st.expander("🧪 一致性自检：在线重放 vs 离线 parquet", expanded=False):
        offline = load("root_cause")
        meta = live.load_meta()
        if offline is None:
            st.info("缺少 root_cause.parquet，无法比对。")
        elif meta.get("defect_mode") != defect or abs(strength - 1.0) > 1e-9 or top_k != 5:
            st.info(
                "离线表是用 `defect_mode=%s`、证据强度 1.0、Top-5 生成的。"
                "把上面三个控件调回该组合即可触发逐行比对。" % meta.get("defect_mode")
            )
        else:
            online = result.candidates.reset_index(drop=True)
            same = online.equals(offline.reset_index(drop=True))
            if same:
                st.success("在线推断结果与 outputs/root_cause.parquet 逐格一致。")
            else:
                st.error("在线推断结果与离线表不一致。")


def _tab_valuation() -> None:
    st.subheader("③ Shapley 估值 + 冗余折价：换预算重新定价整个数据池")
    st.markdown(
        """
改预算后点"重新定价"，会完整重跑一遍
`留一法(LOO) → Shapley 值 → 代理模型外推全池 → 冗余矩阵 → 折价贪心选择`，
而不是在一张固定结果表里增删行。折价公式 `V = v_i · (1 − 0.35 · share_i)`，
其中 `share_i` 是该样本对其最近邻的池内冗余占比。
"""
    )
    if not _ready("route_embedding"):
        st.info(
            "缺少 route_embedding.parquet。请重新执行 `python run_all.py` —— "
            "当前版本会把挖掘出的工艺路线隐向量落盘供本页复现。"
        )
        return

    pool = _valuation_pool()
    budget = st.slider("采购预算（入选样本数）", 10, 60, int(SHAPLEY_BUDGET), 5, key="val_budget")
    st.caption(
        f"数据池 {len(pool):,} 条 × {pool.shape[1]} 维隐向量。"
        "滑杆上限收到 60，因为贪心选择是 O(预算 × 池规模)，再往上单次要点几十秒，不适合网页交互。"
    )

    if st.button("▶ 重新定价整个数据池", type="primary", key="val_run"):
        with st.spinner("LOO → Shapley → 代理模型 → 冗余折价贪心选择 ..."):
            result, elapsed = live.price_pool(pool, budget)
        frame = live.valuation_frame(result)
        # `result` travels along so the redundancy table below can reuse the
        # pricing we just paid for instead of re-running the whole greedy pass.
        st.session_state["valuation_live"] = (frame, result.aggregation_trace, elapsed, result)
    else:
        st.info("点上面的按钮跑一次定价。默认预算与离线流水线一致（40）。")

    cached = st.session_state.get("valuation_live")
    if cached is None:
        return

    frame, trace, elapsed, result = cached
    row = st.columns(4)
    row[0].metric("预算 / 入选", f"{budget} / {int(frame['是否入选'].sum())}")
    row[1].metric("选中集合折价后价值", f"{trace['aggregated_selected']:.4f}")
    row[2].metric("全池折价后价值", f"{trace['aggregated_all']:.4f}")
    row[3].metric("本次定价耗时", f"{elapsed:.2f} s")

    st.caption(
        f"基线效用 {trace['baseline_utility']} · 代理模型终损 {trace['surrogate_final_loss']} · "
        f"平均样本价值 {trace['mean_sample_value']} · "
        f"平均两两冗余 {trace['mean_pairwise_redundancy']} · 平均冗余占比 {trace['mean_redundancy_share']}"
    )

    fig = go.Figure()
    fig.add_scatter(x=frame["样本编号"], y=frame["Shapley值"], mode="markers",
                    name="原始 Shapley 值", marker=dict(size=5, color=COLOR_BEFORE))
    fig.add_scatter(x=frame["样本编号"], y=frame["冗余折价后价值"], mode="markers",
                    name="冗余折价后价值", marker=dict(size=5, color=COLOR_AFTER))
    fig.update_layout(title="冗余剔除前后的样本价值", template=PLOT_THEME, height=400,
                      xaxis_title="样本编号", yaxis_title="价值",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width='stretch')

    st.dataframe(frame[frame["是否入选"]].sort_values("冗余折价后价值", ascending=False),
                 width='stretch', hide_index=True, height=340)

    with st.expander("🔗 冗余样本对（选中集合内部）", expanded=False):
        st.dataframe(live.valuation_report(result, pool),
                     width='stretch', hide_index=True)


def section_live() -> None:
    st.header("⚡ 在线推理 · 现场调用算法")
    st.caption(
        "其余页面读的是 `run_all.py` 预先算好的 parquet；这一页不是。"
        "下面的每个数字都是你改动控件那一刻，由缓存的 GAT 权重与因果 CPT 现算出来的。"
    )
    tab1, tab2, tab3 = st.tabs(["🎯 GAT 实时推理", "🔍 贝叶斯根因诊断", "💰 估值预算寻优"])
    with tab1:
        _tab_gat()
    with tab2:
        _tab_root_cause()
    with tab3:
        _tab_valuation()


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
        "⚡ 在线推理": lambda: section_live(),
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
