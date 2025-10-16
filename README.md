🤝 Credits

Timing & telemetry: FastF1

UI: Streamlit + Plotly


# 🏁 Formula 1 Analysis Dashboard (FastF1 + Streamlit)

- Pick a **season**, **session** (Race, Quali, FP…), and **one or more races**.
- Get a **Race Summary** (winner, podium, official fastest lap).
- Explore **Pace** (lap-time scatter, rolling trends) and **Strategy** (position vs lap, pit stops, tyres, team pace).
- Dive into **Telemetry** with a **fastest-lap comparison** between two drivers (Speed / Throttle / Brake / Gear / DRS).
- Export **CSV** tables (fastest per driver, stint summary, filtered laps).


## ✨ Features

- **Tabbed UI**: Summary · Pace · Strategy · Telemetry · Tables · Season
- **Race Summary**: Winner, podium (top 10), official purple lap (fallback to fastest valid lap if missing)
- **Pace**
  - Lap-time scatter (color by **Driver/Compound**; filter invalid + in/out laps)
  - Rolling median trend (w=5) for quickest drivers
- **Strategy**
  - Position vs Lap (overtakes/undercuts visible)
  - Pit Stop Map (bubble size ~ stationary time when available)
  - Tyre Usage Donut
  - Team Average Pace (lower is faster)
- **Telemetry**
  - Compare **two drivers’ fastest laps** aligned by **distance**
  - Speed, Throttle, Brake, Gear, DRS time series
- **Season Overview**
  - Boxplots for lap-time distribution across your selected races
- **Exports**
  - CSV: fastest per driver, stints, filtered laps
- **Caching**
  - FastF1 on-disk cache for significantly faster reloads

---

## 🧰 Tech Stack

- **[FastF1]** (timing, telemetry & session data)
- **Streamlit** (UI)
- **Plotly** (interactive charts)
- **Pandas / NumPy** (data wrangling)

---

## 📦 Requirements

- **Python 3.10+** (tested up to 3.13)
- Internet access (FastF1 fetches timing/telemetry once, then uses local cache)
- OS: macOS, Linux, or Windows

---

## 🚀 Quick Start

> **macOS/Linux** (zsh/bash)

```bash
# clone your repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# install deps
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# if your shell complains about brackets, run:
# python3 -m pip install "fastf1[plotting]" streamlit plotly pandas numpy

# run
python3 -m streamlit run app.py
# Formula1Analysis-


