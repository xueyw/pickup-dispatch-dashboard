"""
商户（揽收点）分层分析

用法：
    .venv/bin/python src/segment.py

输出：
    reports/aoi_segments.csv     每个揽收点的指标与分层结果
    reports/segment_report.md    分层报告 + "只和前 N 家谈"的省车实验

说明：LaDe 没有商户 ID，用 aoi_id（揽收点/区域）代替"商户"，一个 AOI 里可能有多个商户。

分层维度：
  1. 下单频率   高频（≥50% 的天有单）/ 中频（20%~50%）/ 低频（<20%）
  2. 时间窗类型 上午必到（≥60% 的活跃日有 11:00 前截止的订单）/ 傍晚必到（≥30% 的活跃日全部订单 17:00 后才开始）/
                全天灵活（≥50% 订单为全天窗口）/ 灵活
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import buffer_sim as bs
from capacity_sensitivity import HORIZON, ROOT, SHIFT_END_H, SHIFT_START_H, add_simulated_fields

MORNING_END = 180            # 11:00
EVENING_START = 540          # 17:00
TOP_N = [0, 5, 10, 20, 40]   # 省车实验：只和"上午贡献最大"的前 N 个揽收点协商
RELAX_MIN = 60


# ---------------------------------------------------------------- 指标
def order_windows(df: pd.DataFrame) -> pd.DataFrame:
    """订单级时间窗（距当天 08:00 的分钟），规则与排车模块一致"""
    base = df["ds"] + pd.Timedelta(hours=SHIFT_START_H)
    tw_s = ((df["time_window_start"] - base).dt.total_seconds() / 60).clip(0, HORIZON)
    tw_e = ((df["time_window_end"] - base).dt.total_seconds() / 60).clip(0, HORIZON)
    flexible = df["time_window_start"] >= df["ds"] + pd.Timedelta(hours=SHIFT_END_H)
    tw_s[flexible], tw_e[flexible] = 0, HORIZON
    return df.assign(
        is_morning=tw_e <= MORNING_END,
        is_evening=tw_s >= EVENING_START,
        is_allday=df["tw_type"] == "全天窗口",
    )


def aoi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    n_days = df["ds"].nunique()
    daily = df.groupby(["aoi_id", "ds"])["units"].sum()
    g = df.groupby("aoi_id")
    m = pd.DataFrame({
        "region_id": g["region_id"].agg(lambda s: s.mode().iloc[0]),
        "lat": g["lat"].median(), "lng": g["lng"].median(),
        "订单数": g.size(),
        "活跃天数": daily.groupby("aoi_id").size(),
        "活跃日均件": daily.groupby("aoi_id").mean(),
        "件数波动CV": daily.groupby("aoi_id").std() / daily.groupby("aoi_id").mean(),
        "上午订单占比": g["is_morning"].mean(),
        "傍晚订单占比": g["is_evening"].mean(),
        "全天窗口占比": g["is_allday"].mean(),
        "提前揽收率": g["biz_early"].mean(),
        "上午日均件": df[df["is_morning"]].groupby("aoi_id")["units"].sum() / n_days,
        "傍晚日均件": df[df["is_evening"]].groupby("aoi_id")["units"].sum() / n_days,
    }).fillna({"上午日均件": 0, "傍晚日均件": 0, "件数波动CV": 0})
    m["活跃率"] = m["活跃天数"] / n_days

    # 排车时一个揽收点当天只要有一单 11:00 前截止，整个点就必须上午去（取最早截止）；
    # 只有当天全部订单都在 17:00 后才开始，才会被排到傍晚。所以按"天"统计更贴近排车。
    day = df.groupby(["aoi_id", "ds"]).agg(any_morning=("is_morning", "any"),
                                           all_evening=("is_evening", "all"))
    m["上午必到天数占比"] = day.groupby("aoi_id")["any_morning"].mean()
    m["傍晚必到天数占比"] = day.groupby("aoi_id")["all_evening"].mean()

    m["频率层"] = np.select([m["活跃率"] >= 0.5, m["活跃率"] >= 0.2], ["高频", "中频"], "低频")
    m["时间窗层"] = np.select(
        [m["上午必到天数占比"] >= 0.6, m["傍晚必到天数占比"] >= 0.3, m["全天窗口占比"] >= 0.5],
        ["上午必到", "傍晚必到", "全天灵活"], "灵活")
    return m.reset_index()


# ---------------------------------------------------------------- 省车实验
_DF = None


def _init():
    global _DF
    bs._init_worker()
    _DF = bs._DF


def plan_vehicles(args):
    """前一晚计划：只放宽 top_aois 的上午时间窗，返回计划车辆数"""
    region, day, n, top_aois = args
    dr = _DF[_DF["region_id"] == region]
    hist = dr[(dr["ds"] < day) & (dr["ds"] >= day - pd.Timedelta(days=bs.HISTORY_DAYS))]
    nodes = bs.forecast_stops(hist, bs.CAPACITY)
    sel = nodes["aoi_id"].isin(top_aois) & (nodes["tw_e"] <= MORNING_END)
    nodes.loc[sel, "tw_e"] = np.minimum(nodes.loc[sel, "tw_e"] + RELAX_MIN, HORIZON)
    plan = bs.solve_vrp(nodes, bs.depot_of(dr), bs.CAPACITY, min(len(nodes), 30),
                        fixed_cost=bs.VEHICLE_FIXED_COST)
    return {"片区": region, "日期": day, "协商揽收点数": n, "计划车辆": len(plan["routes"])}


def negotiation_experiment(seg: pd.DataFrame) -> pd.DataFrame:
    tasks = []
    for region in bs.REGIONS:
        ranked = (seg[(seg["region_id"] == region) & (seg["上午日均件"] > 0)]
                  .sort_values("上午日均件", ascending=False)["aoi_id"].tolist())
        for day in bs.pick_days(_DF, region):
            for n in TOP_N + [len(ranked)]:
                tasks.append((region, day, n, set(ranked[:n])))
    with ProcessPoolExecutor(bs.WORKERS, initializer=_init) as pool:
        res = pd.DataFrame(list(pool.map(plan_vehicles, tasks)))
    out = res.groupby(["片区", "协商揽收点数"])["计划车辆"].mean().round(2).reset_index()
    out["片区"] = out["片区"].map(lambda r: f"{r} {bs.REGIONS[r]}")
    return out


# ---------------------------------------------------------------- 报告
def pct(x):
    return f"{x:.1%}"


def build_report(seg: pd.DataFrame, exp: pd.DataFrame) -> str:
    total_units = seg["订单数"].sum()
    cross = seg.groupby(["频率层", "时间窗层"]).agg(
        揽收点数=("aoi_id", "size"), 订单数=("订单数", "sum"),
        上午日均件=("上午日均件", "sum"), 傍晚日均件=("傍晚日均件", "sum")).reset_index()
    cross["订单占比"] = (cross["订单数"] / total_units).map(pct)
    cross["上午件占比"] = (cross["上午日均件"] / seg["上午日均件"].sum()).map(pct)
    cross["傍晚件占比"] = (cross["傍晚日均件"] / seg["傍晚日均件"].sum()).map(pct)
    cross = cross.sort_values("订单数", ascending=False)[
        ["频率层", "时间窗层", "揽收点数", "订单占比", "上午件占比", "傍晚件占比"]]

    freq = seg.groupby("频率层").agg(揽收点数=("aoi_id", "size"), 订单数=("订单数", "sum"))
    freq["点数占比"] = (freq["揽收点数"] / len(seg)).map(pct)
    freq["订单占比"] = (freq["订单数"] / total_units).map(pct)

    # 上午峰值的集中度（帕累托）
    mor = seg.sort_values("上午日均件", ascending=False)["上午日均件"]
    cum = mor.cumsum() / mor.sum()
    pareto = [f"- 前 {k} 个揽收点贡献了 {cum.iloc[k - 1]:.0%} 的上午件量"
              for k in (10, 30, 100) if k <= len(cum)]
    n80 = int((cum < 0.8).sum()) + 1

    cols = ["aoi_id", "region_id", "频率层", "活跃率", "上午日均件", "上午必到天数占比", "提前揽收率"]
    top_morning = seg.sort_values("上午日均件", ascending=False).head(15)[cols]
    cols_e = ["aoi_id", "region_id", "频率层", "活跃率", "傍晚日均件", "傍晚订单占比"]
    top_evening = seg.sort_values("傍晚日均件", ascending=False).head(10)[cols_e]

    fmt = {"活跃率": pct, "上午必到天数占比": pct, "傍晚订单占比": pct, "提前揽收率": pct,
           "上午日均件": "{:.1f}".format, "傍晚日均件": "{:.1f}".format}

    def show(d):
        d = d.copy()
        for c, f in fmt.items():
            if c in d:
                d[c] = d[c].map(f)
        return d.to_markdown(index=False)

    base = exp[exp["协商揽收点数"] == 0].set_index("片区")["计划车辆"]
    exp = exp.assign(比不协商少用车=lambda d: (d["片区"].map(base) - d["计划车辆"]).round(2))

    return "\n".join([
        "# 商户（揽收点）分层分析", "",
        f"共 {len(seg)} 个揽收点（AOI），{total_units:,} 单。LaDe 无商户 ID，以 AOI 代替商户。", "",
        "## 1. 下单频率", "", freq.reset_index().to_markdown(index=False), "",
        "## 2. 频率 × 时间窗 交叉分层", "", cross.to_markdown(index=False), "",
        "## 3. 上午高峰的集中度", "", *pareto,
        f"- 只需 {n80} 个揽收点（占 {n80 / len(seg):.0%}）就贡献了 80% 的上午件量", "",
        "### 上午件量最大的 15 个揽收点（优先协商对象）", "", show(top_morning), "",
        "## 4. 傍晚窗口揽收点（回仓风险来源）", "", show(top_evening), "",
        f"## 5. 省车实验：只和上午件量最大的前 N 个揽收点协商（截止时间延后 {RELAX_MIN} 分钟）", "",
        "每个片区 8 个测试日的前一晚计划车辆平均值；最后一行 = 片区内所有上午揽收点都协商。", "",
        exp.to_markdown(index=False), "",
    ])


def main():
    _init()
    df = order_windows(_DF)
    seg = aoi_metrics(df)
    seg.to_csv(ROOT / "reports/aoi_segments.csv", index=False, encoding="utf-8-sig")
    print("分层完成，开始省车实验……")
    exp = negotiation_experiment(seg)
    md = build_report(seg, exp)
    (ROOT / "reports/segment_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
