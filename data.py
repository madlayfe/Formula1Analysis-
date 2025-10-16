from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import fastf1

from utils import (
    clean_laps_df, stint_blocks, pit_from_laps, fastest_per_driver,
    normalize_results_df
)

CACHE_DIR = Path("./f1cache")

def enable_cache() -> None:
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

def get_schedule(year: int) -> pd.DataFrame:
    sch = fastf1.get_event_schedule(year, include_testing=False).reset_index(drop=True)
    cols = ["RoundNumber","EventName","EventDate","Country","Location"]
    sch = sch[cols]
    sch["RoundNumber"] = sch["RoundNumber"].astype(int)
    return sch

def _load_session(year: int, event: str|int, session_code: str) -> Dict[str, pd.DataFrame]:
    ses = fastf1.get_session(year, event, session_code)
    ses.load()

    laps = clean_laps_df(ses.laps)
    results = getattr(ses, "results", None)
    res_df = normalize_results_df(results if isinstance(results, pd.DataFrame) else pd.DataFrame())

    meta = {"EventName": ses.event["EventName"], "Year": int(ses.event["EventDate"].year)}

    pit_counts, pit_stops = pit_from_laps(laps)
    return {
        "laps": laps.assign(EventName=meta["EventName"], Year=meta["Year"]),
        "stints": stint_blocks(laps).assign(EventName=meta["EventName"], Year=meta["Year"]),
        "pit_counts": pit_counts.assign(EventName=meta["EventName"], Year=meta["Year"]),
        "pit_stops": pit_stops.assign(EventName=meta["EventName"], Year=meta["Year"]),
        "best_per_driver": fastest_per_driver(laps).assign(EventName=meta["EventName"], Year=meta["Year"]),
        "results": res_df.assign(EventName=meta["EventName"], Year=meta["Year"]) if not res_df.empty else pd.DataFrame()
    }

def load_multi(year: int, events: List[str], session_code: str) -> Dict[str, pd.DataFrame]:
    frames = {"laps":[], "stints":[], "pit_counts":[], "pit_stops":[], "best_per_driver":[], "results":[]}
    for ev in events:
        d = _load_session(year, ev, session_code)
        for k in frames:
            if not d[k].empty:
                frames[k].append(d[k])
    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in frames.items()}
