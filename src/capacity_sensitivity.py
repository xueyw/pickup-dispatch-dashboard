"""
车辆容量敏感性分析：容量变化时，需要几辆车、能否 18:00 前回仓

用法：
    .venv/bin/python src/capacity_sensitivity.py

输入：data/clean/pickup_jl_clean.parquet（先运行 src/clean.py）
输出：reports/capacity_sensitivity.csv、reports/capacity_sensitivity.md

—— 模拟层说明（LaDe 中没有这些字段，以下全部为模拟假设，写进 README）——
  仓库位置   每个片区所有揽收点坐标的中位数
  包裹体积   小件 70% = 1 标准件，中件 25% = 2，大件 5% = 5（随机种子固定，可复现）
  车辆容量   以"标准件"计；参考值 40 ≈ 揽收员单日揽收量 P95（41 单）
  行驶时间   直线距离 × 1.3 绕路系数 ÷ 20 km/h
  服务时间   每个点 5 分钟 + 每件 1 分钟
  班次       08:00 从仓库出发，18:00 前必须回仓（硬约束）
  排车口径   用当天实际发生的订单（相当于"完美预测"），每车一天只跑一趟
"""
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parent.parent

SEED = 42
VOLUME_CLASSES = {"小件": (0.70, 1), "中件": (0.25, 2), "大件": (0.05, 5)}
CAPACITIES = [30, 40, 60, 120]
REGIONS = {109: "市区-大片区", 90: "市区-中片区", 132: "外县-稀疏片区"}
DETOUR, SPEED_KMH = 1.3, 20
SERVICE_BASE_MIN, SERVICE_PER_UNIT_MIN = 5, 1
SHIFT_START_H, SHIFT_END_H = 8, 18
HORIZON = (SHIFT_END_H - SHIFT_START_H) * 60          # 600 分钟
VEHICLE_FIXED_COST = 100_000                          # 每多用一辆车 ≈ 多跑 100km，优先少用车
DROP_PENALTY = 10_000_000                             # 放弃一个点的惩罚，保证尽量全部揽收
TIME_LIMIT_S = 8


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- 模拟层
def add_simulated_fields(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    names = list(VOLUME_CLASSES)
    probs = [VOLUME_CLASSES[n][0] for n in names]
    df["volume_class"] = rng.choice(names, size=len(df), p=probs)
    df["units"] = df["volume_class"].map({n: v[1] for n, v in VOLUME_CLASSES.items()})
    return df


def depot_of(df_region: pd.DataFrame) -> tuple[float, float]:
    return df_region["lat"].median(), df_region["lng"].median()


def minutes_in_shift(ts: pd.Series, day: pd.Timestamp) -> pd.Series:
    """时间点 → 距当天 08:00 的分钟数，并截断到 [0, 600]"""
    m = (ts - (day + pd.Timedelta(hours=SHIFT_START_H))).dt.total_seconds() / 60
    return m.clip(0, HORIZON)


def build_stops(orders: pd.DataFrame, day: pd.Timestamp, capacity: int) -> pd.DataFrame:
    """同一 AOI 的订单合并为一个揽收点；时间窗取最早截止的那个；超过车容量的点拆成多次"""
    orders = orders.assign(tw_s=minutes_in_shift(orders["time_window_start"], day),
                           tw_e=minutes_in_shift(orders["time_window_end"], day))
    # 时间窗在 18:00 之后（多为次日窗口、被提前揽收的订单）：视为当天任意时段可揽
    # 否则会被截断成"必须 18:00 整到达"，导致无解
    flexible = orders["time_window_start"] >= day + pd.Timedelta(hours=SHIFT_END_H)
    orders.loc[flexible, ["tw_s", "tw_e"]] = [0, HORIZON]
    stops = orders.groupby("aoi_id").agg(
        lat=("lat", "median"), lng=("lng", "median"),
        packages=("order_id", "size"), units=("units", "sum"),
        tw_s=("tw_s", "min"), tw_e=("tw_e", "min")).reset_index()
    stops["tw_e"] = np.maximum(stops["tw_e"], stops["tw_s"])

    rows = []
    for r in stops.itertuples(index=False):
        n_split = ceil(r.units / capacity)
        for k in range(n_split):
            u = min(capacity, r.units - k * capacity)
            rows.append({**r._asdict(), "units": u, "split": k})
    out = pd.DataFrame(rows)
    out["service_min"] = SERVICE_BASE_MIN + SERVICE_PER_UNIT_MIN * out["units"]
    return out


# ---------------------------------------------------------------- 求解 CVRPTW
def solve(stops: pd.DataFrame, depot: tuple[float, float], capacity: int) -> dict:
    lat = np.r_[depot[0], stops["lat"].to_numpy()]
    lng = np.r_[depot[1], stops["lng"].to_numpy()]
    n = len(lat)
    dist_km = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :]) * DETOUR
    dist_m = (dist_km * 1000).round().astype(int)
    travel = np.ceil(dist_km / SPEED_KMH * 60).astype(int)
    service = np.r_[0, stops["service_min"].to_numpy()].astype(int)
    demand = np.r_[0, stops["units"].to_numpy()].astype(int)
    windows = [(0, HORIZON)] + list(zip(stops["tw_s"].astype(int), stops["tw_e"].astype(int)))

    n_vehicles = min(n - 1, max(5, 3 * ceil(demand.sum() / capacity)))
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(i, j):
        return int(dist_m[manager.IndexToNode(i), manager.IndexToNode(j)])

    def time_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(service[a] + travel[a, b])

    def demand_cb(i):
        return int(demand[manager.IndexToNode(i)])

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cb))
    routing.SetFixedCostOfAllVehicles(VEHICLE_FIXED_COST)

    t_idx = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(t_idx, HORIZON, HORIZON, False, "Time")   # 允许等待；总时长 ≤ 600 分钟
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(*windows[node])
        routing.AddDisjunction([manager.NodeToIndex(node)], DROP_PENALTY)
    for v in range(n_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(0, HORIZON)
        time_dim.CumulVar(routing.End(v)).SetRange(0, HORIZON)      # 18:00 前回仓
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(v)))

    d_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(d_idx, 0, [capacity] * n_vehicles, True, "Load")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = TIME_LIMIT_S
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return {"status": "无解"}

    routes = []
    for v in range(n_vehicles):
        if not routing.IsVehicleUsed(sol, v):
            continue
        idx, km, load, nodes = routing.Start(v), 0.0, 0, 0
        seq = []   # 依次经过的揽收点：(stops 表行号, 到达分钟)
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            a, b = manager.IndexToNode(idx), manager.IndexToNode(nxt)
            km += dist_km[a, b]
            load += demand[b]
            nodes += b != 0
            if b != 0:
                seq.append((b - 1, sol.Value(time_dim.CumulVar(nxt))))
            idx = nxt
        routes.append({"km": km, "load": load, "stops": nodes, "seq": seq,
                       "start_min": sol.Value(time_dim.CumulVar(routing.Start(v))),
                       "return_min": sol.Value(time_dim.CumulVar(routing.End(v)))})
    served = sum(r["stops"] for r in routes)
    r = pd.DataFrame(routes)
    return {
        "status": "OK",
        "vehicles": len(routes),
        "total_km": round(r["km"].sum(), 1),
        "avg_load_rate%": round(r["load"].mean() / capacity * 100, 1),
        "latest_return": f"{SHIFT_START_H + r['return_min'].max() // 60:02d}:{r['return_min'].max() % 60:02d}",
        "routes_return_after_17": int((r["return_min"] > HORIZON - 60).sum()),
        "dropped_nodes": (n - 1) - served,
        "_routes": routes,   # 明细路线，供地图页面使用；汇总报表里会去掉
    }


# ---------------------------------------------------------------- 主流程
def pick_days(df_region: pd.DataFrame) -> dict[str, pd.Timestamp]:
    daily = df_region.groupby("ds").size()
    median_day = (daily - daily.median()).abs().idxmin()
    return {"常规日": median_day, "高峰日": daily.idxmax()}


def main():
    df = pd.read_parquet(ROOT / "data/clean/pickup_jl_clean.parquet")
    df = add_simulated_fields(df[df["is_clean"]].copy())

    results = []
    for region, label in REGIONS.items():
        dr = df[df["region_id"] == region]
        depot = depot_of(dr)
        for day_type, day in pick_days(dr).items():
            orders = dr[dr["ds"] == day]
            for cap in CAPACITIES:
                stops = build_stops(orders, day, cap)
                res = solve(stops, depot, cap)
                res.pop("_routes", None)
                row = {"片区": f"{region} {label}", "日期类型": day_type,
                       "日期": f"{day:%m-%d}", "订单数": len(orders),
                       "揽收点": orders["aoi_id"].nunique(),
                       "总标准件": int(orders["units"].sum()),
                       "实际揽收员数": orders["courier_id"].nunique(),
                       "容量": cap, **res}
                results.append(row)
                print(row)

    out = pd.DataFrame(results).rename(columns={
        "vehicles": "需要车辆", "total_km": "总里程km", "avg_load_rate%": "平均装载率%",
        "latest_return": "最晚回仓", "routes_return_after_17": "17点后回仓车数",
        "dropped_nodes": "未能揽收点", "status": "求解状态"})
    (ROOT / "reports").mkdir(exist_ok=True)
    out.to_csv(ROOT / "reports/capacity_sensitivity.csv", index=False, encoding="utf-8-sig")
    md = ["# 车辆容量敏感性分析", "", "模拟假设见 src/capacity_sensitivity.py 文件头。", "",
          out.to_markdown(index=False), ""]
    (ROOT / "reports/capacity_sensitivity.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
