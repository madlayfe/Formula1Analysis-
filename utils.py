from __future__ import annotations
from typing import Tuple, Dict, Optional
import pandas as pd
import numpy as np

# -------- basic helpers --------
def td_to_sec(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(s) else pd.Series([np.nan]*len(s), index=s.index)

def clean_laps_df(laps) -> pd.DataFrame:
    df = laps.reset_index(drop=True).copy()
    need = ["Driver","DriverNumber","Team","Stint","Compound","TyreLife","LapNumber",
            "LapTime","Sector1Time","Sector2Time","Sector3Time","PitInTime","PitOutTime",
            "TrackStatus","IsAccurate","IsPersonalBest","Position"]
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA

    df["LapTime_s"] = td_to_sec(df["LapTime"])
    df["S1_s"]      = td_to_sec(df["Sector1Time"])
    df["S2_s"]      = td_to_sec(df["Sector2Time"])
    df["S3_s"]      = td_to_sec(df["Sector3Time"])

    df["ValidLap"]  = df["LapTime_s"].notna() & (df["LapTime_s"] > 0)
    df["IsInLap"]   = df["PitInTime"].notna()
    df["IsOutLap"]  = df["PitOutTime"].notna()

    for c in ["Driver","Team","Compound"]:
        try: df[c] = df[c].astype(str)
        except: pass
    return df

def stint_blocks(laps_df: pd.DataFrame) -> pd.DataFrame:
    if laps_df.empty:
        return pd.DataFrame(columns=["Driver","Stint","start_lap","end_lap","compound","length"])
    g = (laps_df.groupby(["Driver","Stint"], dropna=False)
         .agg(start_lap=("LapNumber","min"), end_lap=("LapNumber","max"),
              compound=("Compound", lambda s: s.mode().iat[0] if not s.isna().all() else "UNK"))
         .reset_index())
    g["length"] = (g["end_lap"] - g["start_lap"] + 1).astype(int)
    return g

def stint_summary(laps_df: pd.DataFrame) -> pd.DataFrame:
    v = laps_df[laps_df["ValidLap"]]
    out = (v.groupby(["Driver","Stint","Compound"], as_index=False)
           .agg(n_laps=("LapNumber","count"),
                avg_lap_s=("LapTime_s","mean"),
                best_lap_s=("LapTime_s","min"),
                tyre_life_max=("TyreLife","max")))
    out["avg_lap_s"]  = out["avg_lap_s"].round(3)
    out["best_lap_s"] = out["best_lap_s"].round(3)
    return out.sort_values(["Driver","Stint"]).reset_index(drop=True)

def pit_from_laps(laps_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = (laps_df.groupby("Driver")["PitInTime"].apply(lambda s: int(s.notna().sum()))
              .rename("PitStops").reset_index())
    stops = (laps_df[laps_df["PitInTime"].notna()]
             [["Driver","LapNumber","PitInTime","PitOutTime","Compound","Stint"]]
             .sort_values(["Driver","LapNumber"]).reset_index(drop=True))
    if not stops.empty:
        try:
            t_in  = pd.to_datetime(stops["PitInTime"])
            t_out = pd.to_datetime(stops["PitOutTime"])
            stops["Stationary_s"] = (t_out - t_in).dt.total_seconds().round(2)
        except Exception:
            stops["Stationary_s"] = np.nan
    else:
        stops["Stationary_s"] = np.nan
    return counts, stops

def fastest_per_driver(laps_df: pd.DataFrame) -> pd.DataFrame:
    v = laps_df[laps_df["ValidLap"]]
    if v.empty: return pd.DataFrame()
    best = (v.sort_values(["Driver","LapTime_s"])
              .groupby("Driver", as_index=False)
              .first()[["Driver","Team","LapNumber","LapTime_s","Compound","Stint"]])
    return best.sort_values("LapTime_s").reset_index(drop=True)

def rolling_median(series: pd.Series, window:int=5) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=max(1, window//2)).median()

def filter_laps(laps_all: pd.DataFrame, event: Optional[str]=None, valid_only=True,
                exclude_inout=True, compound="ALL") -> pd.DataFrame:
    df = laps_all.copy()
    if event:
        df = df[df["EventName"] == event]
    if valid_only:
        df = df[df["LapTime_s"].notna() & (df["LapTime_s"] > 0)]
    if exclude_inout:
        df = df[~(df["IsInLap"] | df["IsOutLap"])]
    if compound != "ALL":
        df = df[df["Compound"].str.upper() == compound]
    return df

# -------- results normalization + summary --------
def normalize_results_df(res: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize FastF1 session.results to a consistent schema across seasons/sessions.
    Ensures we have: Position, Driver, Team, GridPosition, FastestLapTime (may be NaT).
    """
    if res is None or (isinstance(res, pd.DataFrame) and res.empty):
        return pd.DataFrame()

    df = res.copy()

    # Team name normalization
    if "Team" not in df.columns and "TeamName" in df.columns:
        df = df.rename(columns={"TeamName": "Team"})

    # Driver name normalization
    if "Driver" not in df.columns:
        if "FullName" in df.columns:
            df["Driver"] = df["FullName"]
        elif "DriverAbbr" in df.columns:
            df["Driver"] = df["DriverAbbr"]
        elif "Abbreviation" in df.columns:
            df["Driver"] = df["Abbreviation"]
        elif "DriverNumber" in df.columns:
            df["Driver"] = df["DriverNumber"].astype(str)
        else:
            df["Driver"] = "N/A"

    # Grid position normalization
    if "GridPosition" not in df.columns and "GridPos" in df.columns:
        df = df.rename(columns={"GridPos": "GridPosition"})
    if "GridPosition" not in df.columns:
        df["GridPosition"] = pd.NA

    # Fastest lap time may be missing; ensure column exists
    if "FastestLapTime" not in df.columns:
        df["FastestLapTime"] = pd.NaT

    return df

def summarize_race(results_all: pd.DataFrame, laps_focus: pd.DataFrame, event_name: str) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"Winner": None, "Podium": None, "PurpleLap": None}

    # Select results for the current event if present
    res = pd.DataFrame()
    if isinstance(results_all, pd.DataFrame) and not results_all.empty:
        res = results_all[results_all["EventName"] == event_name] if "EventName" in results_all.columns else results_all

    # Winner & podium
    if not res.empty and "Position" in res.columns:
        podium = res.sort_values("Position").head(3)
        if len(podium):
            out["Winner"] = f"{podium.iloc[0].get('Driver', 'N/A')} ({podium.iloc[0].get('Team','')})"
            out["Podium"] = ", ".join([f"{int(row['Position'])}. {row.get('Driver','N/A')}" for _, row in podium.iterrows()])

    # Official fastest lap if available; else fallback to laps
    if not res.empty and "FastestLapTime" in res.columns and res["FastestLapTime"].notna().any():
        tmp = res[res["FastestLapTime"].notna()].copy()
        try:
            secs = pd.to_timedelta(tmp["FastestLapTime"]).dt.total_seconds()
        except Exception:
            secs = pd.Series([np.nan]*len(tmp), index=tmp.index)
        tmp["FLT_s"] = secs
        row = tmp.sort_values("FLT_s").head(1)
        if not row.empty and pd.notna(row.iloc[0]["FLT_s"]):
            drv = row.iloc[0].get("Driver", "N/A"); team = row.iloc[0].get("Team","")
            out["PurpleLap"] = f"{drv} — {row.iloc[0]['FLT_s']:.3f}s ({team})"
    else:
        v = laps_focus[laps_focus["ValidLap"]] if isinstance(laps_focus, pd.DataFrame) and not laps_focus.empty else pd.DataFrame()
        if not v.empty:
            r = v.sort_values("LapTime_s").head(1).iloc[0]
            out["PurpleLap"] = f"{r['Driver']} — {r['LapTime_s']:.3f}s ({r['Team']})"

    return out
