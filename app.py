from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import enable_cache, get_schedule, load_multi
from utils import (
    filter_laps, fastest_per_driver, stint_summary, pit_from_laps,
    summarize_race
)
from plots import (
    fig_laptime_scatter, fig_driver_trend, fig_position_progress,
    fig_pit_map, fig_tyre_donut, fig_team_avg_pace, fig_multi_race_box
)

# We’ll use FastF1 directly for telemetry (no changes needed elsewhere)
import fastf1

# ───────────────────────────
# Page & cache setup + subtle styling
# ───────────────────────────
st.set_page_config(page_title="F1 Race Dashboard · FastF1", page_icon="🏁", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1240px;}
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] { padding: 10px 14px; background: #f7f7f9; border-radius: 12px; }
      .stTabs [aria-selected="true"] { background: white; box-shadow: 0 1px 6px rgba(0,0,0,0.07); }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏁 F1 Race Dashboard (FastF1)")
st.caption("Race facts, pace & strategy visuals, telemetry comparisons, and exportable tables. Select season/session/races and click Load.")

enable_cache()  # FastF1 on-disk cache

# ───────────────────────────
# Controls
# ───────────────────────────
topA, topB, topC = st.columns([1, 1, 2])
with topA:
    year = st.selectbox("Season", list(range(2025, 2017, -1)), index=0)
with topB:
    session_code = st.selectbox("Session", ["R","Q","SQ","SS","FP1","FP2","FP3"], index=0)
with topC:
    schedule = get_schedule(year)
    labels = [f"Rnd {int(r)} — {n}" for r, n in zip(schedule["RoundNumber"], schedule["EventName"])]
    name_map = dict(zip(labels, schedule["EventName"]))
    selection = st.multiselect("Races (1–4)", labels, default=labels[:1], max_selections=4)
    selected_events = [name_map[x] for x in selection]

run = st.button("Load / Refresh", type="primary")
if not run:
    st.info("Pick season, session, and at least one race, then click **Load / Refresh**.")
    st.stop()

# ───────────────────────────
# Load sessions
# ───────────────────────────
with st.spinner("Loading timing, results & telemetry metadata (first load caches data)…"):
    data = load_multi(year, selected_events, session_code)

laps_all      = data["laps"]
stints_all    = data["stints"]
pitc_all      = data["pit_counts"]
pits_all      = data["pit_stops"]
best_all      = data["best_per_driver"]
results_all   = data["results"]

if laps_all.empty:
    st.warning("No laps for your selection. Try a different session or race.")
    st.stop()

# Focus race ordering consistent with official schedule listing
events_loaded = sorted(
    laps_all["EventName"].unique().tolist(),
    key=lambda n: schedule["EventName"].tolist().index(n)
)
focus_event = st.selectbox("Focus race", events_loaded, index=0)

# Global filters (apply in Pace/Strategy/Tables; Telemetry loads its own fastest lap)
fc1, fc2, fc3, fc4 = st.columns(4)
valid_only = fc1.checkbox("Valid laps only", True)
excl_inout = fc2.checkbox("Exclude in/out laps", True)
compound   = fc3.selectbox("Compound filter", ["ALL","SOFT","MEDIUM","HARD","INTERMEDIATE","WET"], 0)
color_by   = fc4.selectbox("Color scatter by", ["Driver", "Compound"], 0)

# Filtered focus-laps
laps_focus = filter_laps(
    laps_all,
    event=focus_event,
    valid_only=valid_only,
    exclude_inout=excl_inout,
    compound=compound
)

# ───────────────────────────
# Tabs
# ───────────────────────────
tabs = st.tabs(["🏁 Summary", "📈 Pace", "🧠 Strategy", "📊 Telemetry", "📋 Tables", "🧭 Season"])
tab_summary, tab_pace, tab_strategy, tab_telemetry, tab_tables, tab_season = tabs

# ───────────────────────────
# SUMMARY TAB
# ───────────────────────────
with tab_summary:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader(f"Race summary — {focus_event} ({year})")
        summary = summarize_race(results_all, laps_focus, focus_event)
        c1, c2, c3 = st.columns(3)
        c1.metric("Winner", summary.get("Winner") or "—")
        c2.metric("Official Fastest Lap", summary.get("PurpleLap") or "—")
        c3.metric("Session", session_code)
    with right:
        # Robust podium/Top 10 table: auto-detect columns and rename for display
        if not results_all.empty:
            res_focus = results_all[results_all["EventName"] == focus_event] if "EventName" in results_all.columns else results_all
            if not res_focus.empty and "Position" in res_focus.columns:
                cols = set(res_focus.columns)
                driver_col = next((c for c in ["Driver", "FullName", "DriverAbbr", "Abbreviation", "DriverNumber"] if c in cols), None)
                team_col   = "Team" if "Team" in cols else ("TeamName" if "TeamName" in cols else None)
                grid_col   = next((c for c in ["GridPosition", "GridPos"] if c in cols), None)
                df_show = res_focus.sort_values("Position", na_position="last").head(10).copy()
                if driver_col is None:
                    df_show["Driver"] = "N/A"; driver_col = "Driver"
                out_cols = ["Position", driver_col] + [c for c in [team_col, grid_col] if c]
                rename   = {driver_col: "Driver"}
                if team_col: rename[team_col] = "Team"
                if grid_col: rename[grid_col] = "Grid"
                st.dataframe(df_show[out_cols].rename(columns=rename), use_container_width=True, height=240)
            else:
                st.info("No classified results available for this session.")
    st.markdown("---")
    # A small “key insights” block
    st.markdown(
        "_**Tip**: Use the tabs above. **Pace** shows lap-time scatter and rolling trends. **Strategy** has pit and tyre views. "
        "**Telemetry** lets you compare two drivers’ fastest laps by distance (Speed / Throttle / Brake / Gear / DRS)._"
    )

# ───────────────────────────
# PACE TAB
# ───────────────────────────
with tab_pace:
    r1c1, r1c2 = st.columns([1.6, 1])
    r1c1.plotly_chart(fig_laptime_scatter(laps_focus, color_by=color_by), use_container_width=True)
    r1c1.caption("Each dot = a lap. Lower is faster. Use color to reveal driver pace or tyre effects. Filters remove noise.")
    r1c2.plotly_chart(fig_driver_trend(laps_focus, max_drivers=6, roll_window=5), use_container_width=True)
    r1c2.caption("Rolling-median lap times (w=5) for the fastest drivers. Smooths noise; shows true pace & degradation.")

# ───────────────────────────
# STRATEGY TAB
# ───────────────────────────
with tab_strategy:
    r2c1, r2c2 = st.columns([1.6, 1])
    dlist = sorted(laps_focus["Driver"].dropna().unique().tolist())
    pick = r2c2.multiselect("Drivers (Race Progress)", dlist, default=dlist[:4])
    r2c1.plotly_chart(fig_position_progress(laps_focus, drivers=pick), use_container_width=True)
    r2c1.caption("Position vs lap. Downward moves = overtakes/undercuts; flat lines = stable track position.")

    pc_focus, stops_focus = pit_from_laps(laps_all[laps_all["EventName"] == focus_event])
    r2c2.plotly_chart(fig_pit_map(stops_focus), use_container_width=True)
    r2c2.caption("Pit stops per driver. Bubble size ≈ stationary time (if available). Clusters show strategy windows.")

    st.markdown("---")
    r3c1, r3c2 = st.columns([1, 1])
    r3c1.plotly_chart(fig_tyre_donut(laps_focus), use_container_width=True)
    r3c1.caption("Share of laps by compound. Hints at dominant strategies and stint lengths.")
    r3c2.plotly_chart(fig_team_avg_pace(laps_focus), use_container_width=True)
    r3c2.caption("Average valid lap time by team (lower is faster). Quick read on relative race pace.")

# ───────────────────────────
# TELEMETRY TAB (new)
# ───────────────────────────
with tab_telemetry:
    st.subheader("Fastest-lap comparison (Telemetry)")
    # Driver pickers from the focus race
    drivers_in_focus = sorted(laps_focus["Driver"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([1, 1, 2])
    drv_a = c1.selectbox("Driver A", drivers_in_focus, index=0 if drivers_in_focus else None) if drivers_in_focus else ""
    drv_b = c2.selectbox("Driver B", drivers_in_focus, index=1 if len(drivers_in_focus) > 1 else 0) if drivers_in_focus else ""
    align_mode = c3.radio("Align distance to pit-exit?", ["No (raw fastest lap)", "Yes (start at 0 m)"], index=1, horizontal=True)

    # Load raw session for telemetry (not cached by us; FastF1 uses its own cache)
    if drv_a and drv_b and drv_a != drv_b:
        try:
            with st.spinner("Loading fastest laps & telemetry…"):
                ses = fastf1.get_session(year, focus_event, session_code)
                ses.load()

                la = ses.laps.pick_driver(drv_a).pick_fastest()
                lb = ses.laps.pick_driver(drv_b).pick_fastest()
                if la is None or lb is None:
                    st.warning("Telemetry not available for one or both drivers.")
                else:
                    ta = la.get_telemetry().add_distance()
                    tb = lb.get_telemetry().add_distance()

                    # Distance alignment (normalize both to start at 0 m)
                    if align_mode.startswith("Yes"):
                        if "Distance" in ta.columns:
                            ta["Distance"] = ta["Distance"] - float(ta["Distance"].iloc[0])
                        if "Distance" in tb.columns:
                            tb["Distance"] = tb["Distance"] - float(tb["Distance"].iloc[0])

                    # Build a stacked plot: Speed, Throttle, Brake, Gear, DRS(when present)
                    fig = make_subplots(
                        rows=5, cols=1, shared_xaxes=True,
                        subplot_titles=("Speed (km/h)", "Throttle (%)", "Brake (%)", "Gear", "DRS"),
                        vertical_spacing=0.06
                    )

                    def add_pair(row, ycol, name, dash="dash"):
                        fig.add_trace(go.Scatter(
                            x=ta["Distance"], y=ta[ycol], name=f"{drv_a} {name}", line=dict(width=2)
                        ), row=row, col=1)
                        fig.add_trace(go.Scatter(
                            x=tb["Distance"], y=tb[ycol], name=f"{drv_b} {name}", line=dict(width=2, dash=dash)
                        ), row=row, col=1)

                    # Speed / Throttle / Brake / Gear
                    if "Speed" in ta.columns and "Speed" in tb.columns:       add_pair(1, "Speed", "Speed")
                    if "Throttle" in ta.columns and "Throttle" in tb.columns: add_pair(2, "Throttle", "Throttle")
                    if "Brake" in ta.columns and "Brake" in tb.columns:       add_pair(3, "Brake", "Brake")
                    if "nGear" in ta.columns and "nGear" in tb.columns:       add_pair(4, "nGear", "Gear")

                    # DRS can be missing depending on session/season
                    if "DRS" in ta.columns and "DRS" in tb.columns:
                        add_pair(5, "DRS", "DRS")
                    else:
                        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="DRS N/A", showlegend=False), row=5, col=1)

                    fig.update_layout(
                        template="simple_white", height=940, margin=dict(l=10, r=10, t=40, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig.update_xaxes(title_text="Distance along lap (m)", row=5, col=1)

                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(
                        "Two drivers’ fastest laps over distance. Solid = Driver A, Dashed = Driver B. "
                        "Look for where one carries more speed, applies throttle earlier, brakes later, or uses different gears/DRS."
                    )
        except Exception as e:
            st.error(f"Telemetry load error: {e}")
    else:
        st.info("Pick two different drivers from this race to compare their fastest laps.")

# ───────────────────────────
# TABLES TAB
# ───────────────────────────
with tab_tables:
    st.subheader("Details & Exports (focus race)")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Fastest lap per driver**")
        best_focus = fastest_per_driver(laps_focus)
        st.dataframe(best_focus, use_container_width=True, height=280)
        if not best_focus.empty:
            st.download_button(
                "Download fastest laps (CSV)",
                data=best_focus.to_csv(index=False).encode("utf-8"),
                file_name=f"{year}_{focus_event}_{session_code}_fastest_per_driver.csv",
                mime="text/csv"
            )
    with cB:
        st.markdown("**Stint summary**")
        ss_focus = stint_summary(laps_focus) if len(laps_focus) else pd.DataFrame()
        st.dataframe(ss_focus, use_container_width=True, height=280)
        if not ss_focus.empty:
            st.download_button(
                "Download stints (CSV)",
                data=ss_focus.to_csv(index=False).encode("utf-8"),
                file_name=f"{year}_{focus_event}_{session_code}_stints.csv",
                mime="text/csv"
            )

    st.markdown("**All laps (filtered)**")
    st.dataframe(laps_focus, use_container_width=True, height=320)
    if not laps_focus.empty:
        st.download_button(
            "Download laps (CSV)",
            data=laps_focus.to_csv(index=False).encode("utf-8"),
            file_name=f"{year}_{focus_event}_{session_code}_laps.csv",
            mime="text/csv"
        )

# ───────────────────────────
# SEASON TAB
# ───────────────────────────
with tab_season:
    st.subheader("Season Overview (selected races)")
    if len(events_loaded) < 2:
        st.info("Select multiple races to compare season-wide patterns.")
    else:
        df = filter_laps(laps_all, valid_only=valid_only, exclude_inout=excl_inout, compound=compound)
        if df.empty:
            st.info("No laps after filters across selected races.")
        else:
            st.plotly_chart(fig_multi_race_box(df), use_container_width=True)
            st.caption("Boxplots aggregate lap-time distributions across the races you loaded. Lower medians & tighter boxes = pace & consistency.")
