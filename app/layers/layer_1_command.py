"""
layers/layer_1_command.py
LAYER 1 - COMMAND DECK
No sidebar - all global controls live here.
Returns a state dict consumed by every downstream layer.
"""
import streamlit as st
import pandas as pd


MODEL_MAP: dict[str, str] = {
    "KMeans ★": "KMeans_Cluster",
    "GMM": "GMM_Cluster",
    "Agglomerative": "Hierarchical_Cluster",
    "Spectral": "Spectral_Cluster",
    "BIRCH": "Birch_Cluster",
}

LABEL_MAP: dict[str, str] = {
    "KMeans_Cluster": "KMeans_Cluster_Label",
    "GMM_Cluster": "GMM_Cluster_Label",
    "Hierarchical_Cluster": "Hierarchical_Cluster_Label",
    "Spectral_Cluster": "Spectral_Cluster_Label",
    "Birch_Cluster": "Birch_Cluster_Label",
}


def get_cluster_names(df: pd.DataFrame, cluster_col: str, label_col: str) -> dict[int, str]:
    if cluster_col not in df.columns or label_col not in df.columns:
        return {}

    cluster_names: dict[int, str] = {}
    for cluster_id, labels in df.groupby(cluster_col)[label_col]:
        modes = labels.dropna().mode()
        name = str(modes.iloc[0]) if not modes.empty else f"Cluster {int(cluster_id)}"
        cluster_names[int(cluster_id)] = name
    return cluster_names


def render_command(df: pd.DataFrame) -> dict:
    """
    Renders the sticky command deck with all global controls.

    Returns
    -------
    state : dict
        active_model       : column name, e.g. "KMeans_Cluster"
        active_label_col   : matching label column
        active_clusters    : list[int] - empty means "All"
        selected_features  : list[str]
        country_query      : str - empty means no search
        cluster_names      : dict[int, str]
    """
    st.markdown('<div class="command-deck">', unsafe_allow_html=True)

    col_m, col_c, col_f, col_s = st.columns([2.5, 2, 3, 2])

    with col_m:
        st.markdown('<p class="cmd-label">MODEL</p>', unsafe_allow_html=True)
        active_model_label = st.radio(
            "model",
            list(MODEL_MAP.keys()),
            horizontal=True,
            label_visibility="collapsed",
        )
        active_model = MODEL_MAP[active_model_label]
        active_label_col = LABEL_MAP[active_model]
        cluster_names = get_cluster_names(df, active_model, active_label_col)

    with col_c:
        st.markdown('<p class="cmd-label">CLUSTER FILTER</p>', unsafe_allow_html=True)
        cluster_ids = sorted(int(cid) for cid in df[active_model].dropna().unique())
        cluster_opts = ["All"] + [
            f"{cid} - {cluster_names.get(cid, f'Cluster {cid}')}" for cid in cluster_ids
        ]
        selected_c = st.selectbox(
            "cluster",
            cluster_opts,
            label_visibility="collapsed",
        )
        active_clusters: list[int] = [] if selected_c == "All" else [int(selected_c.split(" - ", 1)[0])]

    with col_f:
        st.markdown('<p class="cmd-label">FEATURES</p>', unsafe_allow_html=True)
        available_features = sorted(
            c for c in df.select_dtypes(include="number").columns
            if not c.endswith("_Cluster") and c != "Label_Agreement"
        )
        default_features = [
            f for f in [
                "GDP",
                "Internet Usage",
                "Infant Mortality Rate",
                "Health Exp % GDP",
                "Life Expectancy Female_trend",
            ]
            if f in available_features
        ]
        selected_features = st.multiselect(
            "features",
            available_features,
            default=default_features,
            max_selections=6,
            label_visibility="collapsed",
        )

    with col_s:
        st.markdown('<p class="cmd-label">COUNTRY SEARCH</p>', unsafe_allow_html=True)
        country_query = st.text_input(
            "country",
            placeholder="e.g. India, Nigeria...",
            label_visibility="collapsed",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "active_model": active_model,
        "active_label_col": active_label_col,
        "active_clusters": active_clusters,
        "selected_features": selected_features,
        "country_query": country_query.strip(),
        "cluster_names": cluster_names,
    }
