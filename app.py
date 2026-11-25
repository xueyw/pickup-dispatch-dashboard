"""
排车模拟地图页面

运行：
    .venv/bin/streamlit run app.py
"""
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent / "src"))
from capacity_sensitivity import (SHIFT_START_H, add_simulated_fields,  # noqa: E402
                                  build_stops, depot_of, solve)

st.set_page_config(page_title="揽收排车模拟", layout="wide")

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46a3a3",
          "#f032e6", "#9a6324", "#800000", "#808000", "#000075", "#2f4f4f"]
TILES = {
    "高德地图": ("https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1"
                 "&style=8&x={x}&y={y}&z={z}", "高德地图"),
    "OpenStreetMap": ("OpenStreetMap", None),
}


def fmt(minutes: float) -> str:
    """距 08:00 的分钟数 → HH:MM"""
    m = int(minutes)
    return f"{SHIFT_START_H + m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------- 数据与计算（带缓存）
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_parquet(Path(__file__).parent / "data/clean/pickup_jl_clean.parquet")
    return add_simulated_fields(df[df["is_clean"]].copy())


@st.cache_data(show_spinner=False)
def run_plan(region: int, day: str, capacity: int):
    df = load_data()
    dr = df[df["region_id"] == region]
    day_ts = pd.Timestamp(day)
    orders = dr[dr["ds"] == day_ts]
    stops = build_stops(orders, day_ts, capacity)
    depot = depot_of(dr)
    return orders, stops, depot, solve(stops, depot, capacity)


# ---------------------------------------------------------------- 侧边栏
df = load_data()
st.sidebar.header("排车参数")
regions = df.groupby("region_id").size().sort_values(ascending=False)
region = st.sidebar.selectbox(
    "片区", regions.index, format_func=lambda r: f"片区 {r}（共 {regions[r]:,} 单）")

daily = df[df["region_id"] == region].groupby("ds").size()
day_mode = st.sidebar.radio("日期", ["高峰日", "常规日", "自选"], horizontal=True)
if day_mode == "高峰日":
    day = daily.idxmax()
elif day_mode == "常规日":
    day = (daily - daily.median()).abs().idxmin()
else:
    day = pd.Timestamp(st.sidebar.date_input(
        "选择日期", daily.index.max().date(),
        min_value=daily.index.min().date(), max_value=daily.index.max().date()))
    if day not in daily.index:
        st.warning("该片区这一天没有揽收订单，请换一天")
        st.stop()

capacity = st.sidebar.slider("车辆容量（标准件）", 20, 200, 120, step=10)
tile_name = st.sidebar.selectbox("底图", list(TILES))
st.sidebar.caption("仓库位置、包裹体积、车辆容量为模拟参数，详见 src/capacity_sensitivity.py")

# ---------------------------------------------------------------- 求解
st.title("揽收排车模拟")
st.caption(f"片区 {region} · {day:%Y-%m-%d}（{day_mode}）· 容量 {capacity} 标准件 · "
           "08:00 出发，18:00 前必须回仓")

with st.spinner("OR-Tools 正在排车，约 8 秒……"):
    orders, stops, depot, res = run_plan(int(region), f"{day:%Y-%m-%d}", capacity)

if res.get("status") != "OK":
    st.error("求解失败：在当前约束下找不到可行方案")
    st.stop()

routes = res["_routes"]
c = st.columns(6)
c[0].metric("订单数", f"{len(orders):,}")
c[1].metric("揽收点", orders["aoi_id"].nunique())
c[2].metric("需要车辆", res["vehicles"])
c[3].metric("总里程", f"{res['total_km']} km")
c[4].metric("最晚回仓", res["latest_return"])
c[5].metric("17点后回仓车数", res["routes_return_after_17"],
            delta="有风险" if res["routes_return_after_17"] else None, delta_color="inverse")
if res["dropped_nodes"]:
    st.warning(f"有 {res['dropped_nodes']} 个揽收点无法在约束内完成")

# ---------------------------------------------------------------- 地图
tiles, attr = TILES[tile_name]
m = folium.Map(location=[stops["lat"].mean(), stops["lng"].mean()], zoom_start=13,
               tiles=tiles, attr=attr)
folium.Marker(depot, tooltip="仓库", icon=folium.Icon(color="black", icon="home")).add_to(m)

for i, r in enumerate(routes):
    color = COLORS[i % len(COLORS)]
    late = r["return_min"] > 540
    group = folium.FeatureGroup(name=f"车 {i + 1}{' ⚠' if late else ''}")
    path = [depot] + [(stops.iloc[k]["lat"], stops.iloc[k]["lng"]) for k, _ in r["seq"]] + [depot]
    folium.PolyLine(path, color=color, weight=4 if late else 3, opacity=0.8,
                    dash_array="8" if late else None,
                    tooltip=f"车 {i + 1}：{r['stops']} 个点，{r['load']} 件，"
                            f"回仓 {fmt(r['return_min'])}").add_to(group)
    for order_no, (k, arrive) in enumerate(r["seq"], start=1):
        s = stops.iloc[k]
        folium.CircleMarker(
            (s["lat"], s["lng"]), radius=4 + min(s["units"], 20) / 2,
            color=color, fill=True, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>车 {i + 1} · 第 {order_no} 站</b><br>AOI {s['aoi_id']}<br>"
                f"{s['packages']} 单 / {s['units']} 标准件<br>"
                f"时间窗 {fmt(s['tw_s'])}–{fmt(s['tw_e'])}<br>预计到达 {fmt(arrive)}",
                max_width=220),
        ).add_to(group)
    group.add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

st_folium(m, height=560, use_container_width=True, returned_objects=[])
st.caption("线路为直线示意，实际行驶时间按直线距离 × 1.3 绕路系数 ÷ 20 km/h 估算；虚线粗线 = 17:00 后回仓的车")

# ---------------------------------------------------------------- 路线明细
st.subheader("每辆车的安排")
table = pd.DataFrame([{
    "车辆": f"车 {i + 1}",
    "揽收点数": r["stops"],
    "装载（标准件）": r["load"],
    "装载率": f"{r['load'] / capacity:.0%}",
    "出发": fmt(r["start_min"]),
    "回仓": fmt(r["return_min"]),
    "距 18:00 余量（分钟）": 600 - r["return_min"],
    "里程 km": round(r["km"], 1),
} for i, r in enumerate(routes)])
st.dataframe(
    table.style.apply(lambda row: ["background-color: #ffe0e0"] * len(row)
                      if row["距 18:00 余量（分钟）"] < 60 else [""] * len(row), axis=1),
    hide_index=True, use_container_width=True)
st.caption("红色行：回仓余量不足 60 分钟。临时加车需要 1 小时以上，这些车一旦出现延误就来不及补救。")
