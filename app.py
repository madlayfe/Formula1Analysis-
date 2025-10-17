# app.py — Formula 1 Gradient Dashboard (FastF1 + Streamlit + Plotly)
# Run:  python3 -m streamlit run app.py

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
from fastf1 import plotting as f1plot

BUILD = "F1-GRADIENT-ONEFILE-FIX-2025-10-17"

# ─────────────────────────────────────────────────────────
# PAGE + THEME
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="F1 Gradient Dashboard", page_icon="🏁", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

with st.sidebar:
    st.markdown("### 🎨 Theme")
    st.session_state.theme = st.radio(
        "Mode", ["Light", "Dark"],
        index=(1 if st.session_state.theme == "Dark" else 0),
        horizontal=True
    )

TPL = "plotly_white" if st.session_state.theme == "Light" else "plotly_dark"
is_light = st.session_state.theme == "Light"
card_bg = "rgba(255,255,255,0.78)" if is_light else "rgba(28,28,30,0.85)"
card_border = "rgba(0,0,0,0.06)" if is_light else "rgba(255,255,255,0.06)"
text_muted = "#444" if is_light else "#bbb"

# ─────────────────────────────────────────────────────────
# GLOBAL CSS (gradient look + soft cards)
# ─────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
    background:
      radial-gradient(1200px 600px at 0% 0%, rgba(255,75,75,0.22), transparent 40%),
      radial-gradient(900px 600px at 100% 0%, rgba(80,130,255,0.15), transparent 45%),
      linear-gradient(160deg, #0e0f12 0%, #121418 60%, #0e1013 100%);
  }}
  .block-container {{ max-width: 1320px; padding-top: .6rem; }}

  .panel {{
    background:
      radial-gradient(900px 400px at 10% -20%, rgba(255,70,70,0.55), transparent 60%),
      linear-gradient(135deg, #0f1115 0%, #171a20 55%, #101317 100%);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 18px 20px;
    color: #fff; box-shadow: 0 6px 22px rgba(0,0,0,0.28);
    margin-bottom: 16px;
  }}
  .panel h2 {{ margin: 0 0 6px 0; font-size: 1.15rem; letter-spacing: .2px; }}
  .panel-sub {{ opacity: .88; font-size: .94rem; margin-bottom: 10px; }}

  .card {{
    background: {card_bg};
    border: 1px solid {card_border};
    backdrop-filter: blur(8px);
    border-radius: 16px; padding: 14px 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.10);
  }}
  .card h3 {{ margin: 0 0 8px 0; font-size: 1.06rem; }}
  .muted {{ color: {text_muted}; font-size:.92rem; }}
  .metric {{ display:flex; flex-direction:column; gap:4px; font-size:1.08rem; }}

  .grid-3 {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .grid-2 {{ display:grid; grid-template-columns: 1.6fr 1fr; gap: 14px; }}
  .grid-2-even {{ display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
  @media (max-width: 1000px) {{
    .grid-3, .grid-2, .grid-2-even {{ grid-template-columns: 1fr; }}
  }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{ padding: 10px 14px; border-radius: 12px; }}
  .stTabs [aria-selected="true"] {{
    box-shadow: 0 1px 10px rgba(0,0,0,0.10);
  }}

  .stDataFrame {{ border-radius: 12px !important; overflow: hidden; }}
  .wm {{ color:#8a8a8a; text-align:right; font-size:12px; margin-top:8px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FASTF1 DISK CACHE
# ─────────────────────────────────────────────────────────
fastf1.Cache.enable_cache("f1cache", ignore_version=True)

# ─────────────────────────────────────────────────────────
# HELPERS (schedule, loading, transforms)
# ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_schedule(year: int) -> pd.DataFrame:
    sch = fastf1.get_event_schedule(year, include_testing=False)
    if "EventName" not in sch.columns and "Event" in sch.columns:
        sch = sch.rename(columns={"Event": "EventName"})
    return sch[["EventName", "RoundNumber"]].copy()

def _safe_secs(td) -> float | None:
    try:
        return float(td.total_seconds())
    except Exception:
        return None

def normalize_laps(laps: pd.DataFrame, event_name: str) -> pd.DataFrame:
    if laps is None or laps.empty:
        return pd.DataFrame()
    df = laps.copy()
    df["EventName"] = event_name
    # Lap time in seconds
    if "LapTime" in df.columns:
        df["LapTime_s"] = df["LapTime"].apply(_safe_secs)
    else:
        df["LapTime_s"] = np.nan
    # Valid lap heuristic
    valid = pd.Series(True, index=df.index)
    if "IsAccurate" in df.columns:
        valid &= df["IsAccurate"].fillna(False)
    if "Deleted" in df.columns:
        valid &= ~df["Deleted"].fillna(False)
    df["ValidLap"] = valid.fillna(False)
    # Standard columns
    for c in ["Driver", "Team", "Compound", "Stint", "LapNumber", "Position"]:
        if c not in df.columns:
            df[c] = np.nan
    # In/Out lap flags
    df["IsOutLap"] = df.get("PitOutTime").notna() if "PitOutTime" in df.columns else False
    df["IsInLap"]  = df.get("PitInTime").notna()  if "PitInTime" in df.columns else False
    return df

@st.cache_data(show_spinner=False)
def load_sessions(year: int, event_names: list[str], session_code: str) -> dict:
    """Returns dict with: laps (all), results (best-effort), per_session list."""
    all_laps = []
    results_rows = []
    per_session = []

    for ev in event_names:
        try:
            ses = fastf1.get_session(year, ev, session_code)
            ses.load(laps=True, telemetry=False, weather=True, messages=False)
        except Exception as e:
            st.warning(f"Failed to load {year} {ev} {session_code}: {e}")
            continue

        laps = ses.laps
        df = normalize_laps(laps, ev)
        if not df.empty:
            all_laps.append(df)
            per_session.append((ev, df))

        res = getattr(ses, "results", None)
        if res is not None and not res.empty:
            row = res.copy()
            cols = row.columns
            drv_col = next((c for c in ["Driver","FullName","Abbreviation",
                                        "DriverAbbreviation","DriverNumber"] if c in cols), None)
            row["Driver"] = row[drv_col] if drv_col else row.get("DriverNumber", np.nan)
            if "TeamName" in row.columns and "Team" not in row.columns:
                row = row.rename(columns={"TeamName": "Team"})
            row["EventName"] = ev
            results_rows.append(row)

    laps_all = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
    results_all = pd.concat(results_rows, ignore_index=True) if results_rows else pd.DataFrame()
    return {"laps": laps_all, "results": results_all, "per_session": per_session}

def filter_laps(df: pd.DataFrame, event: str | None = None,
                valid_only: bool = True, exclude_inout: bool = True,
                compound: str = "ALL") -> pd.DataFrame:
    if df.empty:
        return df
    q = df.copy()
    if event:
        q = q[q["EventName"] == event]
    if valid_only:
        q = q[q["ValidLap"]]
    if exclude_inout:
        q = q[~(q["IsOutLap"] | q["IsInLap"])]
    if compound and compound != "ALL" and "Compound" in q.columns:
        q = q[q["Compound"].astype(str).str.upper() == compound.upper()]
    return q.sort_values(["Driver", "LapNumber"])

def fastest_per_driver(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    best = (df.dropna(subset=["LapTime_s"])
              .sort_values("LapTime_s")
              .groupby("Driver", as_index=False)
              .first()[["Driver","Team","LapNumber","LapTime_s","Compound","Stint"]])
    best = best.sort_values("LapTime_s")
    best["LapTime_s"] = best["LapTime_s"].round(3)
    return best

def stint_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Stint" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["Driver","Stint","Compound"], dropna=False)
    out = g.agg(
        n_laps=("LapNumber","count"),
        avg_lap_s=("LapTime_s","mean"),
        best_lap_s=("LapTime_s","min")
    ).reset_index()
    out["avg_lap_s"] = out["avg_lap_s"].round(3)
    out["best_lap_s"] = out["best_lap_s"].round(3)
    return out.sort_values(["Driver","Stint"])

def summarize_race(results_all: pd.DataFrame, laps_focus: pd.DataFrame, focus_event: str) -> dict:
    out = {"Winner": None, "PurpleLap": None}
    if not results_all.empty and "Position" in results_all.columns:
        r = results_all[results_all["EventName"] == focus_event]
        if not r.empty:
            r1 = r.sort_values("Position").head(1)
            name = r1["Driver"].iloc[0] if "Driver" in r1.columns else None
            team = r1["Team"].iloc[0] if "Team" in r1.columns else None
            out["Winner"] = f"{name} ({team})" if name is not None else None
    if not laps_focus.empty:
        b = laps_focus.dropna(subset=["LapTime_s"]).sort_values("LapTime_s").head(1)
        if not b.empty:
            out["PurpleLap"] = f"{b['Driver'].iloc[0]} · {b['LapTime_s'].iloc[0]:.3f}s"
    return out

# NEW: robust pit-stop extraction + plotting (no NaN sizes)
def get_pit_stops_for_event(year: int, event: str, session_code: str) -> pd.DataFrame:
    """Derive pit stops from laps with PitInTime; estimate stationary time if possible, clean NaNs/negatives."""
    try:
        ses = fastf1.get_session(year, event, session_code)
        ses.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = ses.laps
        stops = laps.loc[laps["PitInTime"].notna(), ["Driver","LapNumber","PitInTime","PitOutTime"]].copy()
        if stops.empty:
            return pd.DataFrame()
        if "PitOutTime" in stops.columns:
            dt = (stops["PitOutTime"] - stops["PitInTime"]).dt.total_seconds()
            stops["Stationary_s"] = dt
        else:
            stops["Stationary_s"] = np.nan
        # Clean values
        stops.loc[~np.isfinite(stops["Stationary_s"]), "Stationary_s"] = np.nan
        stops.loc[stops["Stationary_s"] < 0, "Stationary_s"] = np.nan
        return stops.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────
PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

def _team_color(team: str | None, fallback: str) -> str:
    if not team:
        return fallback
    try:
        return f1plot.team_color(team)
    except Exception:
        return fallback

def fig_laptime_scatter(df: pd.DataFrame, color_by: str="Driver", template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=template); return fig
    if color_by == "Driver":
        for i, (drv, sub) in enumerate(df.groupby("Driver")):
            col = _team_color(sub["Team"].iloc[0] if "Team" in sub else None, PALETTE[i % len(PALETTE)])
            fig.add_trace(go.Scatter(
                x=sub["LapNumber"], y=sub["LapTime_s"], mode="markers", name=str(drv),
                marker=dict(size=6, color=col, opacity=0.92),
                customdata=np.stack([sub.get("Compound","").astype(str), sub.get("Team","").astype(str)], axis=-1),
                hovertemplate="Lap %{x} · %{y:.3f}s<br>%{customdata[0]} — %{customdata[1]}<extra>%{fullData.name}</extra>"
            ))
    else:
        for i, (cpd, sub) in enumerate(df.groupby(df["Compound"].astype(str).str.upper())):
            fig.add_trace(go.Scatter(
                x=sub["LapNumber"], y=sub["LapTime_s"], mode="markers", name=str(cpd),
                marker=dict(size=6, color=PALETTE[i % len(PALETTE)], opacity=0.92),
                customdata=np.stack([sub.get("Driver","").astype(str), sub.get("Team","").astype(str)], axis=-1),
                hovertemplate="Lap %{x} · %{y:.3f}s<br>%{fullData.name} — %{customdata[1]}<extra>%{customdata[0]}</extra>"
            ))
    fig.update_yaxes(autorange="reversed", title="Lap time (s)")
    fig.update_xaxes(title="Lap")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig

def fig_driver_trend(df: pd.DataFrame, max_drivers=6, roll_window=5, template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=template); return fig
    order = (df.groupby("Driver")["LapTime_s"].median().sort_values().index.tolist())[:max_drivers]
    for i, drv in enumerate(order):
        sub = df[df["Driver"] == drv].sort_values("LapNumber")
        if sub.empty: continue
        y = sub["LapTime_s"].rolling(window=roll_window, center=True, min_periods=max(1, roll_window//2)).median()
        col = _team_color(sub["Team"].iloc[0] if "Team" in sub else None, PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatter(x=sub["LapNumber"], y=y, mode="lines", name=str(drv),
                                 line=dict(width=2, color=col),
                                 hovertemplate="%{fullData.name} · Lap %{x} · %{y:.3f}s<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=f"Rolling median (w={roll_window}) — lap time (s)")
    fig.update_xaxes(title="Lap")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig

def fig_position_progress(df: pd.DataFrame, drivers: list[str], template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty or not drivers:
        fig.update_layout(template=template); return fig
    for i, drv in enumerate(drivers):
        sub = df[df["Driver"] == drv].sort_values("LapNumber")
        if sub.empty: continue
        col = _team_color(sub["Team"].iloc[0] if "Team" in sub else None, PALETTE[i % len(PALETTE)])
        fig.add_trace(go.Scatter(x=sub["LapNumber"], y=sub["Position"], mode="lines+markers", name=str(drv),
                                 line=dict(width=2, color=col), marker=dict(size=5),
                                 hovertemplate="%{fullData.name} · Lap %{x} · Pos %{y}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Position (1 = lead)")
    fig.update_xaxes(title="Lap")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig

def fig_pit_map(stops: pd.DataFrame, template="plotly_white") -> go.Figure:
    """Robust pit map — uses default sizes when stationary time is missing; friendly hover."""
    fig = go.Figure()
    if stops.empty:
        fig.update_layout(template=template)
        return fig

    sizes = stops["Stationary_s"].astype(float)
    sizes = sizes.where(np.isfinite(sizes), np.nan).fillna(6.0).clip(2.0, 14.0)

    stationary_label = stops["Stationary_s"].apply(
        lambda x: f"{x:.1f}s" if pd.notna(x) and np.isfinite(x) and x >= 0 else "N/A"
    )

    fig.add_trace(go.Scatter(
        x=stops["LapNumber"], y=stops["Driver"], mode="markers",
        marker=dict(size=sizes.tolist()),
        customdata=np.stack([stationary_label], axis=-1),
        hovertemplate="Lap %{x}<br>%{y}<br>Stationary: %{customdata[0]}<extra></extra>",
        name="Pit stop"
    ))
    fig.update_xaxes(title="Pit stop lap")
    fig.update_yaxes(title="Driver")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_tyre_donut(df: pd.DataFrame, template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=template); return fig
    counts = (df.groupby(df["Compound"].astype(str).str.upper())["LapNumber"].count()
                .reset_index(name="Laps").sort_values("Laps", ascending=False))
    fig.add_trace(go.Pie(labels=counts["Compound"], values=counts["Laps"], hole=0.6,
                         hovertemplate="%{label}: %{value} laps (%{percent})<extra></extra>"))
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def fig_team_avg_pace(df: pd.DataFrame, template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=template); return fig
    v = df[df["ValidLap"]]
    if v.empty:
        fig.update_layout(template=template); return fig
    t = v.groupby("Team")["LapTime_s"].mean().sort_values().reset_index()
    colors = [_team_color(team, PALETTE[i % len(PALETTE)]) for i, team in enumerate(t["Team"])]
    fig.add_trace(go.Bar(x=t["Team"], y=t["LapTime_s"], marker_color=colors))
    fig.update_yaxes(autorange="reversed", title="Avg lap time (s, valid laps)")
    fig.update_xaxes(title="Team")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_multi_race_box(df: pd.DataFrame, template="plotly_white") -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(template=template); return fig
    order = (df.groupby("Driver")["LapTime_s"].median().sort_values().index.tolist())
    for drv in order:
        sub = df[df["Driver"] == drv]
        fig.add_trace(go.Box(y=sub["LapTime_s"], name=str(drv), boxmean=True,
                             hovertemplate="Median ~ %{y:.3f}s<extra>%{fullData.name}</extra>"))
    fig.update_yaxes(autorange="reversed", title="Lap time (s)")
    fig.update_xaxes(title="Driver")
    fig.update_layout(template=template, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ─────────────────────────────────────────────────────────
# SESSION STATE DATA CACHE (prevents reload on driver compare)
# ─────────────────────────────────────────────────────────
def get_ss(name, default=None):
    if name not in st.session_state:
        st.session_state[name] = default
    return st.session_state[name]

if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}  # key: (year, tuple(events), session_code) -> dict
if "loaded_key" not in st.session_state:
    st.session_state.loaded_key = None

def params_key(year: int, events: list[str], session_code: str) -> tuple:
    return (year, tuple(events), session_code)

# ─────────────────────────────────────────────────────────
# GRADIENT CONTROL PANEL (form + resilient defaults)
# ─────────────────────────────────────────────────────────
default_year = get_ss("year", 2025)
_ = get_schedule(default_year)  # warm cache

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<h2>🏁 Load Data</h2><div class="panel-sub">Pick season, session & races. Data only reloads when you press the button.</div>', unsafe_allow_html=True)

with st.form("load_form", clear_on_submit=False):
    colA, colB = st.columns(2)

    with colA:
        years = list(range(2025, 2017, -1))
        year_idx = years.index(default_year) if default_year in years else 0
        year = st.selectbox("Season", years, index=year_idx, key="year")

        sess_opts = ["R","Q","SQ","SS","FP1","FP2","FP3"]
        session_code = st.selectbox("Session", sess_opts, index=sess_opts.index(get_ss("session_code","R")), key="session_code")

    with colB:
        schedule = get_schedule(year)
        labels = [f"Rnd {int(r)} — {n}" for r, n in zip(schedule["RoundNumber"], schedule["EventName"])]
        name_map = dict(zip(labels, schedule["EventName"]))
        saved = get_ss("race_labels", labels[:1]) or labels[:1]
        default_sel = [d for d in saved if d in labels] or labels[:1]
        selection = st.multiselect("Races (1–4)", options=labels, default=default_sel, max_selections=4, key="race_labels")

    submit = st.form_submit_button("🔄 Load / Refresh", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

selected_events = [name_map[x] for x in (st.session_state.get("race_labels") or []) if x in name_map]
if not selected_events and labels:
    st.session_state["race_labels"] = [labels[0]]
    selected_events = [name_map[labels[0]]]

key = params_key(year, selected_events, session_code)
should_load = submit or (st.session_state.loaded_key != key) or (key not in st.session_state.data_cache)

if should_load:
    with st.spinner("Loading timing, results & telemetry metadata…"):
        data = load_sessions(year, selected_events, session_code)
    st.session_state.data_cache[key] = data
    st.session_state.loaded_key = key

data = st.session_state.data_cache[key]
laps_all, results_all = data["laps"], data["results"]
if laps_all.empty:
    st.warning("No laps for your selection. Try a different session or race.")
    st.stop()

# ─────────────────────────────────────────────────────────
# FOCUS + FILTERS
# ─────────────────────────────────────────────────────────
events_loaded = sorted(
    laps_all["EventName"].unique().tolist(),
    key=lambda n: get_schedule(year)["EventName"].tolist().index(n)
)
focus_default = get_ss("focus_event", events_loaded[0])
focus_index = events_loaded.index(focus_default) if focus_default in events_loaded else 0
focus_event = st.selectbox("Focus race", events_loaded, index=focus_index, key="focus_event")

f1c, f2c, f3c, f4c = st.columns(4)
valid_only = f1c.toggle("Valid laps only", get_ss("valid_only", True), key="valid_only")
excl_inout = f2c.toggle("Exclude in/out laps", get_ss("excl_inout", True), key="excl_inout")
compound   = f3c.selectbox("Compound", ["ALL","SOFT","MEDIUM","HARD","INTERMEDIATE","WET"],
                           index=["ALL","SOFT","MEDIUM","HARD","INTERMEDIATE","WET"].index(get_ss("compound","ALL")), key="compound")
color_by   = f4c.selectbox("Scatter color by", ["Driver","Compound"],
                           index=["Driver","Compound"].index(get_ss("color_by","Driver")), key="color_by")

laps_focus = filter_laps(laps_all, event=focus_event, valid_only=valid_only, exclude_inout=excl_inout, compound=compound)

TEAM_COLORS = {}
try:
    f1plot.setup_mpl()
    for t in laps_focus["Team"].dropna().unique():
        try:
            TEAM_COLORS[t] = f1plot.team_color(t)
        except Exception:
            TEAM_COLORS[t] = None
except Exception:
    pass

st.markdown("---")

# ─────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────
tabs = st.tabs(["📌 Summary", "📈 Pace", "🧠 Strategy", "📊 Telemetry", "📋 Tables", "🧭 Season"])
tab_summary, tab_pace, tab_strategy, tab_telemetry, tab_tables, tab_season = tabs

# SUMMARY
with tab_summary:
    summary = summarize_race(results_all, laps_focus, focus_event)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="metric">🏆 <b>Winner</b>'
                f'<span class="muted">{summary.get("Winner") or "—"}</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="metric">🟣 <b>Fastest Lap</b>'
                f'<span class="muted">{summary.get("PurpleLap") or "—"}</span></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><div class="metric">🗓️ <b>Session</b>'
                f'<span class="muted">{session_code}</span></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not results_all.empty and "Position" in results_all.columns:
        res_focus = results_all[results_all["EventName"] == focus_event] if "EventName" in results_all.columns else results_all
        if not res_focus.empty:
            cols = set(res_focus.columns)
            driver_col = next((c for c in ["Driver","FullName","Abbreviation","DriverAbbreviation","DriverNumber"] if c in cols), None)
            team_col   = "Team" if "Team" in cols else ("TeamName" if "TeamName" in cols else None)
            grid_col   = next((c for c in ["GridPosition","GridPos"] if c in cols), None)
            show = res_focus.sort_values("Position", na_position="last").head(10).copy()
            if not driver_col:
                show["Driver"] = "N/A"; driver_col = "Driver"
            out_cols = ["Position", driver_col] + [c for c in [team_col, grid_col] if c]
            rename = {driver_col:"Driver"}
            if team_col: rename[team_col] = "Team"
            if grid_col: rename[grid_col] = "Grid"
            st.markdown('<div class="card"><h3>Podium / Top 10</h3>', unsafe_allow_html=True)
            st.dataframe(show[out_cols].rename(columns=rename), use_container_width=True, height=260)
            st.markdown('</div>', unsafe_allow_html=True)

# PACE
with tab_pace:
    c1, c2 = st.columns([1.6, 1])
    c1.plotly_chart(fig_laptime_scatter(laps_focus, color_by=color_by, template=TPL), use_container_width=True)
    c1.caption("Each dot = a lap. Lower is faster. Color by driver or compound to reveal patterns.")
    c2.plotly_chart(fig_driver_trend(laps_focus, max_drivers=6, roll_window=5, template=TPL), use_container_width=True)
    c2.caption("Rolling-median lap times (w=5) for the fastest drivers; shows degradation/clean air effects.")

# STRATEGY
with tab_strategy:
    r1, r2 = st.columns([1.6, 1])
    drivers_in_focus = sorted(laps_focus["Driver"].dropna().unique().tolist())
    default_pick = [d for d in get_ss("pos_drivers", drivers_in_focus[:4]) if d in drivers_in_focus]
    pick = r2.multiselect("Drivers (Race Progress)", drivers_in_focus, default=default_pick, key="pos_drivers")
    r1.plotly_chart(fig_position_progress(laps_focus, drivers=pick, template=TPL), use_container_width=True)

    stops = get_pit_stops_for_event(year, focus_event, session_code)
    r2.plotly_chart(fig_pit_map(stops, template=TPL), use_container_width=True)

    st.markdown("---")
    a, b = st.columns(2)
    a.plotly_chart(fig_tyre_donut(laps_focus, template=TPL), use_container_width=True)
    b.plotly_chart(fig_team_avg_pace(laps_focus, template=TPL), use_container_width=True)

# TELEMETRY (no reload when switching drivers)
with tab_telemetry:
    st.subheader("Fastest-lap comparison (Speed · Throttle · Brake · Gear · DRS)")
    drivers = sorted(laps_focus["Driver"].dropna().unique().tolist())
    idx_a = drivers.index(get_ss("drv_a", drivers[0])) if drivers else 0
    idx_b = drivers.index(get_ss("drv_b", (drivers[1] if len(drivers)>1 else drivers[0]))) if drivers else 0
    colA, colB, colC = st.columns([1,1,2])
    drv_a = colA.selectbox("Driver A", drivers, index=idx_a if idx_a < len(drivers) else 0, key="drv_a") if drivers else ""
    drv_b = colB.selectbox("Driver B", drivers, index=idx_b if idx_b < len(drivers) else 0, key="drv_b") if drivers else ""
    align = colC.radio("Align distance to 0 m?", ["Yes","No"], index=(0 if get_ss("align","Yes")=="Yes" else 1), key="align", horizontal=True)

    if drv_a and drv_b and drv_a != drv_b:
        try:
            with st.spinner("Loading fastest laps & telemetry…"):
                ses = fastf1.get_session(year, focus_event, session_code); ses.load()
                la = ses.laps.pick_driver(drv_a).pick_fastest()
                lb = ses.laps.pick_driver(drv_b).pick_fastest()
                if la is None or lb is None:
                    st.warning("Telemetry not available for one or both drivers.")
                else:
                    ta = la.get_telemetry().add_distance()
                    tb = lb.get_telemetry().add_distance()
                    if align == "Yes":
                        if "Distance" in ta: ta["Distance"] -= float(ta["Distance"].iloc[0])
                        if "Distance" in tb: tb["Distance"] -= float(tb["Distance"].iloc[0])

                    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                                        subplot_titles=("Speed (km/h)","Throttle (%)","Brake (%)","Gear","DRS"),
                                        vertical_spacing=0.06)
                    def add_pair(row, ycol, name, dash="dash"):
                        fig.add_trace(go.Scatter(x=ta["Distance"], y=ta[ycol], name=f"{drv_a} {name}", line=dict(width=2)), row=row, col=1)
                        fig.add_trace(go.Scatter(x=tb["Distance"], y=tb[ycol], name=f"{drv_b} {name}", line=dict(width=2, dash=dash)), row=row, col=1)
                    if "Speed" in ta and "Speed" in tb:      add_pair(1, "Speed", "Speed")
                    if "Throttle" in ta and "Throttle" in tb:add_pair(2, "Throttle", "Throttle")
                    if "Brake" in ta and "Brake" in tb:      add_pair(3, "Brake", "Brake")
                    if "nGear" in ta and "nGear" in tb:      add_pair(4, "nGear", "Gear")
                    if "DRS" in ta and "DRS" in tb:          add_pair(5, "DRS", "DRS")
                    else:
                        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="DRS N/A", showlegend=False), row=5, col=1)

                    fig.update_layout(template=TPL, height=940, margin=dict(l=10,r=10,t=40,b=10),
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                      hovermode="x unified")
                    fig.update_xaxes(title_text="Distance along lap (m)", row=5, col=1)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Solid = Driver A · Dashed = Driver B. Compare speed traces, throttle pickup, braking points, gear usage, and DRS.")
        except Exception as e:
            st.error(f"Telemetry load error: {e}")
    else:
        st.info("Pick two different drivers to compare their fastest laps.")

# TABLES
with tab_tables:
    tA, tB = st.columns(2)
    with tA:
        st.markdown('<div class="card"><h3>Fastest lap per driver</h3>', unsafe_allow_html=True)
        best = fastest_per_driver(laps_focus)
        st.dataframe(best, use_container_width=True, height=280)
        if not best.empty:
            st.download_button("Download fastest laps (CSV)",
                               data=best.to_csv(index=False).encode("utf-8"),
                               file_name=f"{year}_{focus_event}_{session_code}_fastest_per_driver.csv",
                               mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)
    with tB:
        st.markdown('<div class="card"><h3>Stint summary</h3>', unsafe_allow_html=True)
        ss = stint_summary(laps_focus) if len(laps_focus) else pd.DataFrame()
        st.dataframe(ss, use_container_width=True, height=280)
        if not ss.empty:
            st.download_button("Download stints (CSV)",
                               data=ss.to_csv(index=False).encode("utf-8"),
                               file_name=f"{year}_{focus_event}_{session_code}_stints.csv",
                               mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><h3>All laps (filtered)</h3>', unsafe_allow_html=True)
    st.dataframe(laps_focus, use_container_width=True, height=320)
    if not laps_focus.empty:
        st.download_button("Download laps (CSV)",
                           data=laps_focus.to_csv(index=False).encode("utf-8"),
                           file_name=f"{year}_{focus_event}_{session_code}_laps.csv",
                           mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# SEASON
with tab_season:
    st.markdown('<div class="card"><h3>Season Overview</h3>', unsafe_allow_html=True)
    if len(events_loaded) < 2:
        st.info("Select multiple races to compare season-wide patterns.")
    else:
        df = filter_laps(laps_all, valid_only=valid_only, exclude_inout=excl_inout, compound=compound)
        if df.empty:
            st.info("No laps after filters across selected races.")
        else:
            st.plotly_chart(fig_multi_race_box(df, template=TPL), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<div class="wm">Build {BUILD}</div>', unsafe_allow_html=True)
