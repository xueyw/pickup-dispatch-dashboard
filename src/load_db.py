"""
把原始数据和各分析模块的结果导入 PostgreSQL（raw → stg → mart 三层）

前提：PostgreSQL 已启动，存在数据库 lade，当前 WSL 用户有权限
    sudo service postgresql start
用法：
    .venv/bin/python src/load_db.py

流程：
    1. sql/01_raw.sql   建 schema 与原始表
    2. COPY             把 data/raw/pickup_jl.csv 原样导入 raw.pickup
    3. sql/02_stg.sql   SQL 清洗与打标记 → stg.pickup
    4. Python 产出入库  片区-日面板、清洗规则说明、商户分层、预测、预警、排车模拟
    5. sql/03_mart.sql  日报表、揽收员-日表、数据质量汇总（含 SQL 与 Python 结果交叉核对）
    6. 校验并打印结果
"""
import time

import pandas as pd
import psycopg
from sqlalchemy import create_engine

from capacity_sensitivity import ROOT

DB = "lade"
SQL_DIR = ROOT / "sql"
REPORTS = ROOT / "reports"

# Python 产出（中文列名）→ 数据库表（英文列名，方便写 SQL；取值里的中文保留，看板上直接显示）
CSV_TABLES = {
    "stg.dq_rule_meta": ("pickup_jl_dq_rules.csv", {
        "类型": "kind", "规则": "rule", "说明": "description", "命中行数": "py_hits", "占比%": "py_pct"}),
    "mart.aoi_segment": ("aoi_segments.csv", {
        "订单数": "orders", "活跃天数": "active_days", "活跃日均件": "avg_units_active_day",
        "件数波动CV": "units_cv", "上午订单占比": "morning_order_share",
        "傍晚订单占比": "evening_order_share", "全天窗口占比": "allday_share",
        "提前揽收率": "early_pickup_rate", "上午日均件": "morning_units_per_day",
        "傍晚日均件": "evening_units_per_day", "活跃率": "active_rate",
        "上午必到天数占比": "morning_required_day_share", "傍晚必到天数占比": "evening_required_day_share",
        "频率层": "freq_segment", "时间窗层": "tw_segment"}),
    "mart.alert_log": ("alert_log_test.csv", {
        "日期": "ds", "片区": "region_id", "揽收员": "courier_id", "检查点": "checkpoint",
        "已揽收单": "picked_orders", "未揽收点": "pending_stops", "预计回仓": "projected_return",
        "实际回仓": "actual_return", "实际是否超时": "actual_late"}),
    "mart.capacity_sensitivity": ("capacity_sensitivity.csv", {
        "片区": "region", "日期类型": "day_type", "日期": "day", "订单数": "orders", "揽收点": "stops",
        "总标准件": "total_units", "实际揽收员数": "actual_couriers", "容量": "capacity",
        "求解状态": "status", "需要车辆": "vehicles", "总里程km": "total_km",
        "平均装载率%": "avg_load_rate_pct", "最晚回仓": "latest_return",
        "17点后回仓车数": "vehicles_return_after_17", "未能揽收点": "dropped_stops"}),
    "mart.buffer_tradeoff": ("buffer_tradeoff_detail.csv", {
        "片区": "region_id", "日期": "ds", "机动车": "spare_vehicles",
        "上午时间窗放宽(分钟)": "morning_relax_min", "计划车辆": "planned_vehicles",
        "实际动用机动车": "spare_used", "计划揽收点": "planned_stops", "实际揽收点": "actual_stops",
        "计划外揽收点": "unplanned_stops", "实际标准件": "actual_units", "未完成点": "unfinished_stops",
        "未完成标准件": "unfinished_units", "改派点": "reassigned_stops",
        "17点后回仓车": "vehicles_return_after_17", "最晚回仓分钟": "latest_return_min"}),
}
FORECAST_COLS = ["region_id", "ds", "units", "stops", "base_units", "base_stops", "holiday",
                 "after_holiday", "units_pred_p50", "units_pred_p80", "stops_pred_p50", "stops_pred_p80"]
PANEL_COLS = ["region_id", "ds", "units", "stops", "orders", "ramp_up", "anomaly", "usable",
              "base_units", "base_stops"]


def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def run_sql(conn, name):
    step(f"执行 {name}")
    conn.execute((SQL_DIR / name).read_text(encoding="utf-8"))
    conn.commit()


def main():
    engine = create_engine(f"postgresql+psycopg:///{DB}")
    with psycopg.connect(dbname=DB) as conn:
        run_sql(conn, "01_raw.sql")

        step("COPY 原始 CSV → raw.pickup")
        with conn.cursor() as cur, open(ROOT / "data/raw/pickup_jl.csv", "rb") as f:
            with cur.copy("COPY raw.pickup FROM STDIN WITH (FORMAT csv, HEADER true)") as cp:
                while chunk := f.read(1 << 20):
                    cp.write(chunk)
        conn.commit()

        run_sql(conn, "02_stg.sql")

        step("导入 Python 分析结果")
        panel = pd.read_parquet(ROOT / "data/clean/region_day_panel.parquet")[PANEL_COLS]
        panel.to_sql("region_day", engine, schema="stg", if_exists="replace", index=False)
        for table, (fname, cols) in CSV_TABLES.items():
            schema, name = table.split(".")
            pd.read_csv(REPORTS / fname).rename(columns=cols) \
              .to_sql(name, engine, schema=schema, if_exists="replace", index=False)
        fc = pd.read_csv(REPORTS / "forecast_test_predictions.csv", parse_dates=["ds"])[FORECAST_COLS]
        fc.to_sql("forecast_test", engine, schema="mart", if_exists="replace", index=False)

        run_sql(conn, "03_mart.sql")

        step("校验")
        tables = conn.execute("""
            SELECT table_schema || '.' || table_name
            FROM information_schema.tables
            WHERE table_schema IN ('raw', 'stg', 'mart')
            ORDER BY table_schema, table_name""").fetchall()
        for (t,) in tables:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<32} {n:>9,}")

    dq = pd.read_sql("SELECT kind, rule, sql_hits, py_hits, pct, sql_python_match FROM mart.dq_summary",
                     engine)
    print("\n数据质量：SQL 与 Python 清洗结果交叉核对")
    print(dq.to_string(index=False))
    if not dq["sql_python_match"].all():
        print("⚠ 存在不一致的规则，请检查")


if __name__ == "__main__":
    main()
