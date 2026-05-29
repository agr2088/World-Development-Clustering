"""
layers/layer_3_world_map.py
LAYER 3 - WORLD MAP
Full-width choropleth, single-model or 5-model small-multiples toggle.
"""
from html import escape

import streamlit as st
import pandas as pd

from utils.theme import render_section_header, render_gold_divider
from utils.charts import build_choropleth_single, build_choropleth_mini

CLUSTER_COLORS = {
    -1: "#1A1A24",
    0: "#D4A017",
    1: "#7EC8A4",
    2: "#6EB3D4",
    3: "#C97B5E",
}

MODEL_COLS = [
    ("KMeans_Cluster", "KMeans ★"),
    ("GMM_Cluster", "GMM"),
    ("Hierarchical_Cluster", "Agglomerative"),
    ("Spectral_Cluster", "Spectral"),
    ("Birch_Cluster", "BIRCH"),
]


def _is_invalid_model(display_name: str, invalid_names: set[str]) -> bool:
    display_lower = display_name.lower()
    return any(
        display_lower.startswith(name.lower()) or name.lower().startswith(display_lower)
        for name in invalid_names
    )


def render_world_map(df: pd.DataFrame, state: dict, metrics: list) -> None:
    render_section_header(
        "WORLD MAP",
        "Choropleth - Countries Coloured by Cluster Assignment",
    )

    invalid_names = {m["model"] for m in metrics if m.get("invalid", False)}

    left_ctrl, right_ctrl = st.columns([1, 3])
    with left_ctrl:
        view_toggle = st.radio(
            "map_view",
            ["Single Model", "All 5 Models"],
            horizontal=False,
            label_visibility="collapsed",
        )

    if view_toggle == "Single Model":
        active_model = state["active_model"]
        df_map = df.copy()
        if state["active_clusters"]:
            keep_clusters = set(state["active_clusters"])
            df_map.loc[~df_map[active_model].isin(keep_clusters), active_model] = -1
        fig = build_choropleth_single(df_map, active_model, CLUSTER_COLORS)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        row1 = st.columns(3)
        row2 = st.columns(3)
        all_cols = row1 + row2

        for i, (col_name, title) in enumerate(MODEL_COLS):
            if col_name not in df.columns:
                continue
            with all_cols[i]:
                display_title = f"{title} ⚠" if _is_invalid_model(title, invalid_names) else title
                fig = build_choropleth_mini(df, col_name, display_title, CLUSTER_COLORS)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    cluster_names = state.get("cluster_names", {})
    legend_items = []
    for cid in sorted(cluster_names):
        color = CLUSTER_COLORS.get(cid, "#5A5040")
        name = escape(cluster_names.get(cid, f"Cluster {cid}"))
        legend_items.append(
            f'<span class="legend-item" style="color:{color}">■ Cluster {cid} - {name}</span>'
        )
    st.markdown(
        f'<div class="map-legend">{"".join(legend_items)}</div>',
        unsafe_allow_html=True,
    )

    render_gold_divider()
