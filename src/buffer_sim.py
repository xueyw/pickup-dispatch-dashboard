"""
缓冲策略模拟：前一晚按预测排车、车辆固定；当天按实际订单执行，只允许小调整

用法：
    .venv/bin/python src/buffer_sim.py

输出：
    reports/buffer_tradeoff_detail.csv   每个片区 × 日期 × 缓冲组合的明细
    reports/buffer_tradeoff.csv / .md    按缓冲组合汇总

—— 模拟流程 ——
  前一晚（计划）
    1. 预测：用前 28 天历史，活跃天数占比 ≥ 50% 的揽收点进入计划，
       件数取其活跃日的中位数，时间窗取历史中位数
    2. 排车：每车容量 120 标准件、一天一趟、18:00 前回仓，目标是用最少的车
    3. 结果：确定车辆数 K，以及每个揽收点分给哪辆车 —— 这两样当天不能改
  当天（执行）
    4. 用当天实际订单，K 辆计划车 + 预留的机动车，18:00 必须回仓（硬约束）
    5. 允许：同车内调整顺序（1 级）、少量改派给其他车（2 级，每次改派有惩罚）、
       计划外的新揽收点插入任意车、动用机动车
    6. 仍然完成不了的点 → 需要临时加车或与商户改约（统计为"未完成"）
  场景维度：机动车数 × 是否与商户协商放宽上午时间窗

—— 局限（写进 README）——
  * 假设当天订单在出发前全部已知；实际中部分订单是白天陆续产生的，结果偏乐观
  * 揽收点、体积、容量、速度等模拟假设同 src/capacity_sensitivity.py
"""
import itertools
from concurrent.futures import ProcessPoolExecutor
from math import ceil

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from capacity_sensitivity import (DETOUR, HORIZON, ROOT, SERVICE_BASE_MIN, SERVICE_PER_UNIT_MIN,
                                  SHIFT_END_H, SHIFT_START_H, SPEED_KMH, add_simulated_fields,
                                  build_stops, depot_of, haversine_km)

CAPACITY = 120                      # 车辆物理容量（标准件，面包车；每车每天只跑一趟）
HISTORY_DAYS = 28
ACTIVE_THRESHOLD = 0.5              # 活跃天数占比达到该值才进入计划
SPARE_VEHICLES = [0, 1, 2]          # 每个片区额外预留的机动车数
MORNING_RELAX = [0, 60]             # 与商户协商：11:00 前截止的时间窗延长多少分钟
MORNING_DEADLINE = 180              # 11:00（距 08:00 的分钟数）
REGIONS = {109: "市区-大片区", 132: "外县-稀疏片区"}

# 第一轮（v1）用的是 40 件容量 × 时间缓冲 × 装载比例，结果存档在 reports/buffer_tradeoff_cap40_v1*。
# 发现：一天一趟的车装 40 件中午就回仓，时间缓冲完全无效；换成 120 件后，
# 真正决定车辆数的是 09:00–11:00 时间窗过于集中（计划点中约 73%），所以第二轮改为模拟
# "机动车"和"放宽上午时间窗"这两个杠杆。
N_DAYS = 8                          # 每个片区抽取的测试日数
SEED = 7
TIME_LIMIT_S = 4
WORKERS = 8

VEHICLE_FIXED_COST = 100_000        # 每多用一辆车 ≈ 多跑 100km
DROP_PENALTY = 10_000_000           # 放弃一个点
REASSIGN_PENALTY = 5_000            # 当天把点改派给非计划车辆 ≈ 多跑 5km
LATE_PENALTY_PER_MIN = 5_000        # 计划阶段：超过"18:00 − 缓冲"每分钟的惩罚


# ---------------------------------------------------------------- 通用求解器
def solve_vrp(nodes: pd.DataFrame, depot, capacity: int, n_vehicles: int,
              soft_end: int = HORIZON, fixed_cost: int = 0, planned_vehicle=None):
    """
    nodes: 需含 lat, lng, units, service_min, tw_s, tw_e（距 08:00 的分钟）
    soft_end: 回仓软上限（分钟）；硬上限始终是 18:00
    planned_vehicle: 每个点的计划车辆编号（-1 表示计划外），用于改派惩罚
    """
    lat = np.r_[depot[0], nodes["lat"].to_numpy()]
    lng = np.r_[depot[1], nodes["lng"].to_numpy()]
    n = len(lat)
    dist_km = haversine_km(lat[:, None], lng[:, None], lat[None, :], lng[None, :]) * DETOUR
    dist_m = (dist_km * 1000).round().astype(int)
    travel = np.ceil(dist_km / SPEED_KMH * 60).astype(int)
    service = np.r_[0, nodes["service_min"].to_numpy()].astype(int)
    demand = np.r_[0, nodes["units"].to_numpy()].astype(int)
    tw = np.c_[np.r_[0, nodes["tw_s"].to_numpy()], np.r_[HORIZON, nodes["tw_e"].to_numpy()]]
    planned = None if planned_vehicle is None else np.r_[-1, planned_vehicle]

    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    if planned is None:
        cb = routing.RegisterTransitCallback(
            lambda i, j: int(dist_m[manager.IndexToNode(i), manager.IndexToNode(j)]))
        routing.SetArcCostEvaluatorOfAllVehicles(cb)
    else:
        # 每辆车单独的成本：去"计划给别的车"的点要加改派惩罚
        for v in range(n_vehicles):
            def cost(i, j, v=v):
                b = manager.IndexToNode(j)
                extra = REASSIGN_PENALTY if planned[b] not in (-1, v) else 0
                return int(dist_m[manager.IndexToNode(i), b] + extra)
            routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(cost), v)
    if fixed_cost:
        routing.SetFixedCostOfAllVehicles(fixed_cost)

    t_idx = routing.RegisterTransitCallback(
        lambda i, j: int(service[manager.IndexToNode(i)]
                         + travel[manager.IndexToNode(i), manager.IndexToNode(j)]))
    routing.AddDimension(t_idx, HORIZON, HORIZON, False, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, n):
        idx = manager.NodeToIndex(node)
        time_dim.CumulVar(idx).SetRange(int(tw[node, 0]), int(tw[node, 1]))
        routing.AddDisjunction([idx], DROP_PENALTY)
    for v in range(n_vehicles):
        end = routing.End(v)
        time_dim.CumulVar(end).SetRange(0, HORIZON)
        if soft_end < HORIZON:
            time_dim.SetCumulVarSoftUpperBound(end, soft_end, LATE_PENALTY_PER_MIN)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(end))

    d_idx = routing.RegisterUnaryTransitCallback(lambda i: int(demand[manager.IndexToNode(i)]))
    routing.AddDimensionWithVehicleCapacity(d_idx, 0, [capacity] * n_vehicles, True, "Load")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = TIME_LIMIT_S
    sol = routing.SolveWithParameters(params)
    if sol is None:
        return None

    assign = np.full(n - 1, -1)          # 每个点最终由哪辆车完成（-1 = 未完成）
    routes = []
    for v in range(n_vehicles):
        if not routing.IsVehicleUsed(sol, v):
            continue
        idx, km, load = routing.Start(v), 0.0, 0
        while not routing.IsEnd(idx):
            nxt = sol.Value(routing.NextVar(idx))
            a, b = manager.IndexToNode(idx), manager.IndexToNode(nxt)
            km += dist_km[a, b]
            load += demand[b]
            if b:
                assign[b - 1] = v
            idx = nxt
        routes.append({"vehicle": v, "km": km, "load": load,
                       "return_min": sol.Value(time_dim.CumulVar(routing.End(v)))})
    return {"routes": routes, "assign": assign}


# ---------------------------------------------------------------- 预测（前一晚）
def split_by_capacity(stops: pd.DataFrame, capacity: int) -> pd.DataFrame:
    rows = []
    for r in stops.itertuples(index=False):
        for k in range(ceil(r.units / capacity)):
            rows.append({**r._asdict(), "units": min(capacity, r.units - k * capacity)})
    out = pd.DataFrame(rows)
    out["service_min"] = SERVICE_BASE_MIN + SERVICE_PER_UNIT_MIN * out["units"]
    return out


def forecast_stops(hist: pd.DataFrame, plan_capacity: int) -> pd.DataFrame:
    base = hist["ds"] + pd.Timedelta(hours=SHIFT_START_H)
    tw_s = ((hist["time_window_start"] - base).dt.total_seconds() / 60).clip(0, HORIZON)
    tw_e = ((hist["time_window_end"] - base).dt.total_seconds() / 60).clip(0, HORIZON)
    flexible = hist["time_window_start"] >= hist["ds"] + pd.Timedelta(hours=SHIFT_END_H)
    tw_s[flexible], tw_e[flexible] = 0, HORIZON

    daily = hist.assign(tw_s=tw_s, tw_e=tw_e).groupby(["aoi_id", "ds"]).agg(
        units=("units", "sum"), lat=("lat", "median"), lng=("lng", "median"),
        tw_s=("tw_s", "min"), tw_e=("tw_e", "min"))
    n_days = hist["ds"].nunique()
    per = daily.groupby("aoi_id").agg(
        active=("units", "size"), units=("units", "median"), lat=("lat", "median"),
        lng=("lng", "median"), tw_s=("tw_s", "median"), tw_e=("tw_e", "median")).reset_index()
    per = per[per["active"] / n_days >= ACTIVE_THRESHOLD].copy()
    per["units"] = np.ceil(per["units"]).astype(int)
    per["tw_e"] = np.maximum(per["tw_e"], per["tw_s"])
    return split_by_capacity(per.drop(columns="active"), plan_capacity)


# ---------------------------------------------------------------- 单个场景
_DF = None


def _init_worker():
    global _DF
    df = pd.read_parquet(ROOT / "data/clean/pickup_jl_clean.parquet")
    _DF = add_simulated_fields(df[df["is_clean"]].copy())


def relax_morning(nodes: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """协商后的时间窗：11:00 前截止的点，截止时间延后 minutes 分钟"""
    if not minutes:
        return nodes
    nodes = nodes.copy()
    morning = nodes["tw_e"] <= MORNING_DEADLINE
    nodes.loc[morning, "tw_e"] = np.minimum(nodes.loc[morning, "tw_e"] + minutes, HORIZON)
    return nodes


def run_scenario(args):
    region, day, spare, relax = args
    dr = _DF[_DF["region_id"] == region]
    depot = depot_of(dr)
    hist = dr[(dr["ds"] < day) & (dr["ds"] >= day - pd.Timedelta(days=HISTORY_DAYS))]
    actual = dr[dr["ds"] == day]

    # —— 前一晚：按预测排车
    plan_nodes = relax_morning(forecast_stops(hist, CAPACITY), relax)
    plan = solve_vrp(plan_nodes, depot, CAPACITY, min(len(plan_nodes), 30),
                     fixed_cost=VEHICLE_FIXED_COST)
    used = [r["vehicle"] for r in plan["routes"]]
    remap = {v: k for k, v in enumerate(used)}           # 车辆重新编号为 0..K-1
    K = len(used)
    aoi_vehicle = {}
    for aoi, v in zip(plan_nodes["aoi_id"], plan["assign"]):
        if v >= 0:
            aoi_vehicle.setdefault(aoi, remap[v])

    # —— 当天：K 辆计划车 + spare 辆机动车，按实际订单执行
    exec_nodes = relax_morning(build_stops(actual, day, CAPACITY), relax)
    planned_v = exec_nodes["aoi_id"].map(aoi_vehicle).fillna(-1).astype(int).to_numpy()
    ex = solve_vrp(exec_nodes, depot, CAPACITY, K + spare, planned_vehicle=planned_v)

    done = ex["assign"] >= 0
    returns = np.array([r["return_min"] for r in ex["routes"]]) if ex["routes"] else np.array([0])
    return {
        "片区": region, "日期": f"{day:%Y-%m-%d}",
        "机动车": spare, "上午时间窗放宽(分钟)": relax,
        "计划车辆": K,
        "实际动用机动车": sum(r["vehicle"] >= K for r in ex["routes"]),
        "计划揽收点": plan_nodes["aoi_id"].nunique(),
        "实际揽收点": actual["aoi_id"].nunique(),
        "计划外揽收点": int(exec_nodes.loc[planned_v == -1, "aoi_id"].nunique()),
        "实际标准件": int(actual["units"].sum()),
        "未完成点": int((~done).sum()),
        "未完成标准件": int(exec_nodes.loc[~done, "units"].sum()),
        "改派点": int(((planned_v >= 0) & done & (ex["assign"] != planned_v)).sum()),
        "17点后回仓车": int((returns > HORIZON - 60).sum()),
        "最晚回仓分钟": int(returns.max()),
    }


def pick_days(df: pd.DataFrame, region: int) -> list[pd.Timestamp]:
    daily = df[df["region_id"] == region].groupby("ds").size()
    eligible = daily[daily.index >= daily.index.min() + pd.Timedelta(days=HISTORY_DAYS)]
    rng = np.random.default_rng(SEED + region)
    days = set(rng.choice(eligible.index, N_DAYS - 1, replace=False))
    days.add(eligible.idxmax())                          # 一定包含高峰日
    return sorted(pd.Timestamp(d) for d in days)


def main():
    _init_worker()
    tasks = [(region, day, spare, relax)
             for region in REGIONS
             for day in pick_days(_DF, region)
             for spare, relax in itertools.product(SPARE_VEHICLES, MORNING_RELAX)]
    print(f"共 {len(tasks)} 个场景，{WORKERS} 进程并行")
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as pool:
        detail = pd.DataFrame(list(pool.map(run_scenario, tasks)))
    detail.to_csv(ROOT / "reports/buffer_tradeoff_detail.csv", index=False, encoding="utf-8-sig")

    detail["总车辆"] = detail["计划车辆"] + detail["机动车"]
    summary = detail.groupby(["片区", "上午时间窗放宽(分钟)", "机动车"]).agg(
        平均计划车辆=("计划车辆", "mean"),
        平均总车辆=("总车辆", "mean"),
        机动车动用率=("实际动用机动车", lambda s: (s > 0).mean()),
        出现未完成的天数占比=("未完成点", lambda s: (s > 0).mean()),
        平均未完成标准件=("未完成标准件", "mean"),
        平均改派点=("改派点", "mean"),
        平均17点后回仓车=("17点后回仓车", "mean"),
    ).round(2).reset_index()
    summary["片区"] = summary["片区"].map(lambda r: f"{r} {REGIONS[r]}")
    summary.to_csv(ROOT / "reports/buffer_tradeoff.csv", index=False, encoding="utf-8-sig")
    md = ["# 缓冲策略模拟：用车数 vs 当天未完成风险", "",
          f"每个片区抽取 {N_DAYS} 天（含高峰日）；车辆物理容量 {CAPACITY} 标准件。"
          "模拟流程与假设见 src/buffer_sim.py 文件头。", "",
          summary.to_markdown(index=False), ""]
    (ROOT / "reports/buffer_tradeoff.md").write_text("\n".join(md), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
