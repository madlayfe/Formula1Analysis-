from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import rolling_median

PLOTLY_TEMPLATE = "simple_white"
PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]

def fig_laptime_scatter(df: pd.DataFrame, color_by: str="Driver") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig

    if color_by == "Driver":
        for i, (drv, sub) in enumerate(df.groupby("Driver")):
            hover = f"Lap %{{x}} · %{{y:.3f}} s<br>%{{customdata[0]}} — %{{customdata[1]}}<extra>{drv}</extra>"
            fig.add_trace(go.Scatter(
                x=sub["LapNumber"], y=sub["LapTime_s"], mode="markers", name=drv,
                marker=dict(size=6, color=PALETTE[i % len(PALETTE)]),
                customdata=np.stack([sub["Compound"], sub["Team"]], axis=-1),
                hovertemplate=hover
            ))
    else:
        for i, (cpd, sub) in enumerate(df.groupby(df["Compound"].str.upper())):
            hover = f"Lap %{{x}} · %{{y:.3f}} s<br>{cpd} — %{{customdata[1]}}<extra>%{{customdata[0]}}</extra>"
            fig.add_trace(go.Scatter(
                x=sub["LapNumber"], y=sub["LapTime_s"], mode="markers", name=str(cpd),
                marker=dict(size=6, color=PALETTE[i % len(PALETTE)]),
                customdata=np.stack([sub["Driver"], sub["Team"]], axis=-1),
                hovertemplate=hover
            ))

    fig.update_layout(template=PLOTLY_TEMPLATE, hovermode="closest",
                      xaxis_title="Lap", yaxis_title="Lap time (s)", legend_title=None,
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_driver_trend(df: pd.DataFrame, max_drivers:int=6, roll_window:int=5) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig

    order = (df.groupby("Driver")["LapTime_s"].median().sort_values().index.tolist())[:max_drivers]
    for i, drv in enumerate(order):
        sub = df[df["Driver"] == drv].sort_values("LapNumber")
        sub = sub[sub["LapTime_s"].notna()]
        if sub.empty: continue
        y = rolling_median(sub["LapTime_s"], window=roll_window)
        fig.add_trace(go.Scatter(
            x=sub["LapNumber"], y=y, mode="lines", name=drv,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
            hovertemplate=f"{drv} · Lap %{{x}} · %{{y:.3f}} s<extra></extra>"
        ))
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Lap",
                      yaxis_title=f"Rolling median (w={roll_window}) — lap time (s)",
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_position_progress(df: pd.DataFrame, drivers: List[str]) -> go.Figure:
    fig = go.Figure()
    if df.empty or not drivers:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig
    for i, drv in enumerate(drivers):
        sub = df[df["Driver"] == drv].sort_values("LapNumber")
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["LapNumber"], y=sub["Position"], mode="lines+markers", name=drv,
            line=dict(width=2, color=PALETTE[i % len(PALETTE)]),
            hovertemplate=f"{drv} · Lap %{{x}} · Pos %{{y}}<extra></extra>"
        ))
    fig.update_layout(template=PLOTLY_TEMPLATE, yaxis_title="Position (1 = lead)", xaxis_title="Lap",
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_pit_map(stops: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if stops.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig
    sizes = stops["Stationary_s"].fillna(stops["Stationary_s"].median() if stops["Stationary_s"].notna().any() else 2.5)
    fig.add_trace(go.Scatter(
        x=stops["LapNumber"], y=stops["Driver"], mode="markers",
        marker=dict(size=np.clip(sizes, 2, 12), color="#636EFA", opacity=0.7),
        hovertemplate="Lap %{x}<br>%{y}<br>Stationary: %{marker.size:.1f}s<extra></extra>"
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Pit stop lap", yaxis_title="Driver",
                      margin=dict(l=10,r=10,t=10,b=10))
    return fig

def fig_tyre_donut(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig
    counts = (df.groupby(df["Compound"].str.upper())["LapNumber"].count()
              .reset_index(name="Laps").sort_values("Laps", ascending=False))
    fig = go.Figure(go.Pie(labels=counts["Compound"], values=counts["Laps"], hole=0.6))
    fig.update_traces(textinfo="percent+label", hovertemplate=" %{label}: %{value} laps (%{percent})<extra></extra>")
    fig.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def fig_team_avg_pace(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig
    v = df[df["ValidLap"]]
    t = (v.groupby("Team")["LapTime_s"].mean().sort_values().reset_index())
    fig.add_trace(go.Bar(x=t["Team"], y=t["LapTime_s"], marker_color="#1f77b4"))
    fig.update_layout(template=PLOTLY_TEMPLATE, xaxis_title="Team", yaxis_title="Avg lap time (s, valid laps)",
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

def fig_multi_race_box(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=PLOTLY_TEMPLATE); return fig
    order = (df.groupby("Driver")["LapTime_s"].median().sort_values().index.tolist())
    for i, drv in enumerate(order):
        sub = df[df["Driver"] == drv]
        fig.add_trace(go.Box(y=sub["LapTime_s"], name=drv, boxmean=True,
                             marker_color=PALETTE[i % len(PALETTE)],
                             hovertemplate=f"Median ~ %{{y:.3f}} s<extra>{drv}</extra>"))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Lap-time distribution across selected races",
                      yaxis_title="Lap time (s)", xaxis_title="Driver",
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig
