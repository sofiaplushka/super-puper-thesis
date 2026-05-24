from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

from thesis_pipeline.text_normalization import truncate_text


def add_hover_text(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["text_preview"] = result["text"].map(lambda x: truncate_text(x, 300))
    return result


def save_umap2d(df: pd.DataFrame, path: str | Path) -> None:
    data = add_hover_text(df)
    fig = px.scatter(
        data,
        x="umap2_x",
        y="umap2_y",
        color=data["cluster_leiden"].astype(str),
        hover_data=[
            "id",
            "year",
            "month",
            "tags_raw",
            "macro_tags",
            "cluster_leiden",
            "text_preview",
        ],
        title="UMAP 2D visualization colored by Leiden cluster",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.75})
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def save_umap3d(df: pd.DataFrame, path: str | Path) -> None:
    data = add_hover_text(df)
    fig = px.scatter_3d(
        data,
        x="umap3_x",
        y="umap3_y",
        z="umap3_z",
        color=data["cluster_leiden"].astype(str),
        hover_data=[
            "id",
            "year",
            "month",
            "tags_raw",
            "macro_tags",
            "cluster_leiden",
            "text_preview",
        ],
        title="UMAP 3D visualization colored by Leiden cluster",
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.75})
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def save_pca3d(df: pd.DataFrame, coords, path: str | Path) -> None:
    data = add_hover_text(df)
    data["pca_x"] = coords[:, 0]
    data["pca_y"] = coords[:, 1]
    data["pca_z"] = coords[:, 2]
    fig = px.scatter_3d(
        data,
        x="pca_x",
        y="pca_y",
        z="pca_z",
        color=data["cluster_leiden"].astype(str),
        hover_data=[
            "id",
            "year",
            "month",
            "tags_raw",
            "macro_tags",
            "cluster_leiden",
            "text_preview",
        ],
        title="PCA 3D baseline colored by Leiden cluster",
    )
    fig.update_traces(marker={"size": 4, "opacity": 0.75})
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
