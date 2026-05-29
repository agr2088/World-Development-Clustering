"""
utils/loader.py
Cached data loading - no redundant I/O on re-renders.
"""

import os
import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

_here = Path(__file__).resolve()
PROJECT_ROOT = next(
    p for p in [_here, *_here.parents]
    if (p / "config").is_dir() and (p / "src").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CLUSTERED_DATA, REPORTS_DIR, PROCESSED_DIR


@st.cache_data(show_spinner=False)
def load_clustered_data() -> pd.DataFrame:
    df = pd.read_csv(CLUSTERED_DATA)

    cluster_cols = [c for c in df.columns if c.endswith("_Cluster")]
    for col in cluster_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    label_cols = [c for c in df.columns if c.endswith("_Label") and c != "Majority_Label"]

    def _recompute_label_consensus() -> None:
        if label_cols:
            modes = df[label_cols].mode(axis=1)
            df["Majority_Label"] = (
                modes.iloc[:, 0].fillna("Unknown") if not modes.empty else "Unknown"
            )
            df["Label_Agreement"] = (
                df[label_cols].eq(df["Majority_Label"], axis=0).sum(axis=1) / len(label_cols)
            ).astype(float)
        else:
            df["Majority_Label"] = "Unknown"
            df["Label_Agreement"] = 1.0

    if (
        "Label_Agreement" not in df.columns
        or df["Label_Agreement"].isna().all()
        or "Majority_Label" not in df.columns
        or df["Majority_Label"].isna().all()
    ):
        _recompute_label_consensus()
    else:
        df["Majority_Label"] = df["Majority_Label"].fillna("Unknown")
        df["Label_Agreement"] = pd.to_numeric(df["Label_Agreement"], errors="coerce")
        max_val = df["Label_Agreement"].max()
        if max_val > 1.0:
            df["Label_Agreement"] = df["Label_Agreement"] / max_val
        df["Label_Agreement"] = df["Label_Agreement"].fillna(1.0)

    return df


@st.cache_data(show_spinner=False)
def load_pca_data() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "pca_data.csv")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> list:
    path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_cluster_profiles(model_col: str) -> pd.DataFrame:
    """
    Load cluster_profiles_{model_col}.csv.
    Returns a DataFrame indexed by cluster id (int).
    The first column is assumed to be the cluster ID column.
    """
    fname = f"cluster_profiles_{model_col}.csv"
    path = os.path.join(REPORTS_DIR, fname)
    df = pd.read_csv(path)

    first_col = df.columns[0]
    df = df.set_index(first_col)
    df.index = df.index.astype(int)
    return df
