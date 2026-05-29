"""
layers/layer_0_hero.py
LAYER 0 - HERO BANNER
Animated gold title, tagline, model pills, stat badges.
"""
import streamlit as st
import pandas as pd


def render_hero(df: pd.DataFrame):
    country_count = len(df)
    feature_count = len([
        c for c in df.select_dtypes(include="number").columns
        if not c.endswith("_Cluster") and c != "Label_Agreement"
    ])

    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-crown">✦</div>
          <h1 class="hero-title">WORLD DEVELOPMENT INTELLIGENCE</h1>
          <p class="hero-tagline">Unsupervised Intelligence · {country_count} Nations · 5 Clustering Models</p>
          <div class="hero-pills">
            <span class="pill">KMeans ★</span>
            <span class="pill">GMM</span>
            <span class="pill">Agglomerative</span>
            <span class="pill">Spectral</span>
            <span class="pill">BIRCH</span>
          </div>
          <div class="hero-badges">
            <span class="badge">k = 4 Clusters</span>
            <span class="badge">{country_count} Countries</span>
            <span class="badge">{feature_count} Features</span>
            <span class="badge">4 Trend Signals</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
