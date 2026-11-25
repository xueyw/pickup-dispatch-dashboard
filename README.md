# Small-Batch Pickup: Dispatch Planning, Daily Report & Return Alerts

English | [Français](README.fr.md) | [中文](README.zh-CN.md)

This is a simulation project built on the public Cainiao [LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) dataset (Jilin pickup data, about 262,000 orders). It models a pickup operation serving small-batch merchants under these constraints:

- Vehicles are planned the night before. **Each vehicle makes a single trip per day**, and the plan can't change much during the day.
- An ad-hoc extra vehicle takes more than 1 hour to arrive.
- **Every vehicle must be back at the depot by 18:00** to unload.

> Vehicle capacity, parcel volume and travel speed are not in the original data. They are simulated, and each script's header documents the assumptions.

## Modules

| Module | File | What it does |
|---|---|---|
| Data cleaning | `src/clean.py` | Three-level flags: dirty `dq_*` (excluded), suspicious `sus_*` (kept), business anomaly `biz_*` (analysed). Schema validation with pandera. |
| Data warehouse | `sql/`, `src/load_db.py` | PostgreSQL raw → stg → mart layers. SQL and Python cleaning results are cross-checked rule by rule. |
| Route optimisation | `src/capacity_sensitivity.py` | OR-Tools CVRPTW: vehicle capacity, time windows and the 18:00 return deadline. |
| Map dashboard | `app.py` | Streamlit + folium. Draws each vehicle's route and highlights vehicles with less than 60 min of return slack. |
| Buffer simulation | `src/buffer_sim.py` | Plans the night before from the forecast, then executes on the day's actual orders. Compares spare vehicles against relaxing morning time windows. |
| Merchant segmentation | `src/segment.py` | Segments AOIs by visit frequency and time-window profile. Estimates how many vehicles are saved by negotiating later morning windows with the top merchants. |
| Demand forecast | `src/forecast.py` | Quantile GBM (P50/P80) with holiday features. Detects days where the data collapses. |
| Return alerts | `src/alert.py` | Replays checkpoints from 13:00 to 17:00. Scores alert rules on precision and on whether they fire by 16:00, which still leaves time to add a vehicle. |
| BI report | Power BI | Connects directly to the PostgreSQL mart layer. |

## Tech Stack

| Purpose | Libraries / tools |
|---|---|
| Data processing | pandas, NumPy, PyArrow (Parquet storage) |
| Data validation | pandera (schema and value-range checks) |
| Database | PostgreSQL 16, psycopg (bulk COPY load), SQLAlchemy |
| Route optimisation | Google OR-Tools (capacitated VRP with time windows) |
| Forecasting | scikit-learn (HistGradientBoostingRegressor, quantile loss) |
| Visualisation | Streamlit, folium + streamlit-folium (maps), Plotly, Power BI |
| Other | tabulate (Markdown reports), concurrent.futures (parallel simulation) |

## Pipeline

```
Raw CSV (262k rows)
  │  clean.py: parse fields → deduplicate → flag with 15 rules → pandera validation
  ▼
Cleaned data (Parquet, all rows kept, split by is_clean)
  ├─ capacity_sensitivity.py  minimum fleet size for each vehicle capacity
  ├─ buffer_sim.py            night-before plan vs. same-day execution (multiprocess, 8 days × 2 regions × 6 scenarios)
  ├─ segment.py               merchant segmentation + time-window negotiation experiment
  ├─ forecast.py              region-day panel → anomaly detection → features → P50/P80 forecast
  └─ alert.py                 checkpoint replay → alert rule evaluation
  ▼
load_db.py: raw (as-is) → stg (SQL cleaning) → mart (daily report, courier-day, DQ summary)
  ▼
Power BI daily report / Streamlit map dashboard
```

## Problems Solved

**Data cleaning**
- **Timestamps without a year.** The raw format is `MM-DD HH:MM:SS`. I added the year from the `ds` field and checked that each pickup time matches its date.
- **A distance rule that wrongly flagged 9% of rows.** The first rule (more than 100 km from the city centre) flagged a whole rural county as dirty data. I changed it to "more than 30 km from the order's own region median", which now catches only 2 genuinely bad records.
- **Three-level handling.** Missing GPS (16%) and GPS drift can't simply be deleted without losing many valid orders. Rules are split into dirty (excluded), suspicious (kept and flagged) and business anomaly (analysed), and each rule reports its hit count and share.
- **SQL and Python results cross-checked.** The SQL and Python pipelines' hit counts are compared for all 13 rules. They all match, so the database and the analysis use the same definitions.
- **Data-collection problems detected.** Early May is the dataset's ramp-up period. From 20 to 25 June, 11 urban regions dropped to almost zero while the county stayed normal, which points to missing data rather than a real stop in business. Any region-day below 30% of its trailing 28-day median is flagged automatically and left out of training and evaluation.

**Modelling and analysis**
- **Stops dropped because of cross-day time windows.** Some orders have time windows that fall on the next day. Clipping them at 18:00 made them impossible to schedule, so these orders are now treated as flexible (any time that day).
- **Finding what really drives fleet size.** My first guess was a lack of time buffer, but vehicles were returning only half full. Further analysis showed that 73% of planned stops must be picked up between 09:00 and 11:00: the morning windows decide how many vehicles are needed. I redesigned the simulation as "spare vehicles × morning-window relaxation".
- **A first segmentation with almost every merchant labelled "mixed".** One AOI holds several merchants, so order-share rules don't separate them. Segmenting instead by "share of days with a must-be-morning order" gave clear groups.
- **Too many peak alerts.** Flagging on P80 raised 574 alerts out of 875 region-days, almost every day. Flagging on P50 cut this to 199, with 33% precision and 40% recall.
- **Alert timeliness redefined.** A 17:00 alert used to count as "in time", even though by then it is too late to add a vehicle. Only alerts raised by 16:00 now count, and an "early alert" mode is evaluated separately.
- **Stable plans.** When routes are re-optimised during the day, moving a stop to another vehicle is penalised, which keeps the "plan can't change much" constraint.

## Key Findings

- **The night-before plan is the main lever.** In the large urban region, planning 6.75 vehicles leaves the work unfinished on 38% of days. Adding one spare vehicle brings that to 0%.
- **Morning windows (09:00–11:00) set the fleet size.** Negotiating a 1-hour later window with the top 20 merchants by morning volume saves about one vehicle (1.13 in the urban region, 1 in the county).
- **Forecast.** Overall WAPE is on par with the 28-day-median baseline (12.8% vs. 12.6%). The gain is on peak days: underestimation drops from 20.3% to 13.0%, and P80 coverage is 80.5%.
- **Same-day alerts are a safety net.** The simplest rule, "2 or more stops still pending at 16:00", catches 67% of late returns while there is still time to add a vehicle. It beats the ETA-based rules.
- **Data issues must be handled first.** The early-May ramp-up and the late-June urban outage both have to be detected and removed, or both the forecast and the baseline are distorted.

Detailed results are in `reports/` (in Chinese).

## Getting Started

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw && curl -L -o data/raw/pickup_jl.csv \
  https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/pickup/pickup_jl.csv

.venv/bin/python src/clean.py
.venv/bin/python src/capacity_sensitivity.py
.venv/bin/python src/buffer_sim.py
.venv/bin/python src/segment.py
.venv/bin/python src/forecast.py
.venv/bin/python src/alert.py
.venv/bin/python src/load_db.py      # requires local PostgreSQL with a database named lade
.venv/bin/streamlit run app.py
```

## Limitations

- LaDe couriers also handle deliveries, so the raw data has a high baseline late rate.
- The simulation assumes all of the day's orders are known in the morning.
- Each region is tested on only 8 days.
