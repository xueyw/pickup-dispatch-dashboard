"""
回仓超时预警：规则设计 + 历史回放检验

用法：
    .venv/bin/python src/alert.py

输出：
    reports/alert_report.md        预警规则评估报告
    reports/alert_log_test.csv     测试期每条预警的明细（相当于推送记录）

—— 回放方法 ——
  * 把 LaDe 中每个"揽收员 × 天"当作一辆车，仓库 = 所在片区中心
  * 在检查点 13:00 / 14:00 / 15:00 / 16:00 / 17:00，只使用当时已知的信息：
      已揽收的订单、已接单但未揽收的订单（接单时间 ≤ 检查点）。之后才产生的订单当时看不到
  * 真实结果：最后一单揽收时间 + 返回仓库的行驶时间 > 18:00 → 记为"回仓超时"
  * 规则参数在训练期（05-16 ~ 08-31）调优，在测试期（09-01 ~ 10-31）评估
—— 速度标定 ——
  用揽收员相邻两个揽收点之间的实际间隔，按距离分箱取中位数后拟合：
      间隔(分钟) = 每点固定耗时 a + 每公里耗时 b × 距离
  实际间隔里也包含了揽收员顺路派件等工作，所以标定出的是"真实节奏"，不是车速
—— 三条规则 ——
  A 简单规则：16:00 时未揽收点数 ≥ N
  B 时间推算：从当前位置按"最近邻"顺序跑完已知未揽收点再回仓，预计回仓时间 > 18:00 − m
  C 时间推算 + 预计新增：在 B 的基础上，加上该片区历史上检查点之后平均还会新增的揽收点
"""
import numpy as np
import pandas as pd

from capacity_sensitivity import ROOT, SHIFT_START_H, haversine_km
from forecast import TRAIN_END, VALID_FROM

CHECKPOINTS = {300: "13:00", 360: "14:00", 420: "15:00", 480: "16:00", 540: "17:00"}
DEADLINE = 600              # 18:00（距 08:00 的分钟）
ADD_VEHICLE_LEAD = 60       # 临时加车需要的提前量：17:00 之前发出的预警才来得及加车
MIN_ORDERS = 5              # 每天至少 5 单的揽收员-日才纳入（太少的不像一趟车）


# ---------------------------------------------------------------- 数据准备
def load() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(ROOT / "data/clean/pickup_jl_clean.parquet")
    df = df[df["is_clean"]].copy()
    panel = pd.read_parquet(ROOT / "data/clean/region_day_panel.parquet")
    usable = panel.loc[panel["usable"], ["region_id", "ds"]]

    base = df["ds"] + pd.Timedelta(hours=SHIFT_START_H)
    df["acc_m"] = (df["accept_time"] - base).dt.total_seconds() / 60
    df["pick_m"] = (df["pickup_time"] - base).dt.total_seconds() / 60
    df = df[df["ds"] >= VALID_FROM].merge(usable, on=["region_id", "ds"])

    # 揽收员-日：取当天主要片区；过滤单量太少的
    cd = df.groupby(["courier_id", "ds"]).agg(
        n=("order_id", "size"), region=("region_id", lambda s: s.mode().iloc[0]))
    cd = cd[cd["n"] >= MIN_ORDERS]
    df = df.merge(cd[["region"]].reset_index(), on=["courier_id", "ds"])
    depots = df.groupby("region_id")[["lat", "lng"]].median()
    return df, {r: tuple(v) for r, v in depots.iterrows()}


# ---------------------------------------------------------------- 速度标定
def calibrate(df: pd.DataFrame) -> tuple[float, float, float, pd.DataFrame]:
    d = df[df["ds"] <= TRAIN_END].sort_values(["courier_id", "ds", "pick_m"])
    # 连续在同一个 AOI 的揽收合并为一次停靠
    new_stop = ((d["aoi_id"] != d["aoi_id"].shift()) | (d["courier_id"] != d["courier_id"].shift())
                | (d["ds"] != d["ds"].shift()))
    d = d.assign(stop_id=new_stop.cumsum())
    stops = d.groupby("stop_id").agg(courier_id=("courier_id", "first"), ds=("ds", "first"),
                                     t=("pick_m", "min"), lat=("lat", "median"), lng=("lng", "median"))
    prev = stops.shift()
    same = (stops["courier_id"] == prev["courier_id"]) & (stops["ds"] == prev["ds"])
    hop = pd.DataFrame({
        "gap": (stops["t"] - prev["t"])[same],
        "km": haversine_km(prev["lat"], prev["lng"], stops["lat"], stops["lng"])[same],
    })
    hop = hop[(hop["gap"] > 0) & (hop["gap"] <= 120) & (hop["km"] <= 10)]
    hop["bin"] = pd.cut(hop["km"], [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 10])
    binned = hop.groupby("bin", observed=True).agg(km=("km", "median"), gap=("gap", "median"),
                                                   n=("gap", "size"))
    b, a = np.polyfit(binned["km"], binned["gap"], 1, w=np.sqrt(binned["n"]))
    return float(a), float(b), float(hop["km"].median()), binned


# ---------------------------------------------------------------- 回放
def nn_tour_minutes(pos, pts: np.ndarray, depot, a, b) -> float:
    """从 pos 出发，按最近邻顺序访问 pts（N×2 的 lat,lng），最后回仓，返回总分钟数"""
    total, cur, left = 0.0, np.array(pos), pts.copy()
    while len(left):
        d = haversine_km(cur[0], cur[1], left[:, 0], left[:, 1])
        k = int(np.argmin(d))
        total += a + b * d[k]
        cur = left[k]
        left = np.delete(left, k, axis=0)
    return total + b * float(haversine_km(cur[0], cur[1], depot[0], depot[1]))


def replay(df: pd.DataFrame, depots: dict, a: float, b: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, days = [], []
    for (courier, ds), g in df.groupby(["courier_id", "ds"], sort=False):
        region = g["region"].iloc[0]
        depot = depots[region]
        last = g.loc[g["pick_m"].idxmax()]
        actual_return = last["pick_m"] + b * float(
            haversine_km(last["lat"], last["lng"], depot[0], depot[1]))
        days.append({"courier_id": courier, "ds": ds, "region": region, "orders": len(g),
                     "actual_return": actual_return, "late": actual_return > DEADLINE})
        for T in CHECKPOINTS:
            done = g[g["pick_m"] <= T]
            todo = g[(g["acc_m"] <= T) & (g["pick_m"] > T)]
            new_after = g.loc[g["acc_m"] > T, "aoi_id"].nunique()
            if len(done):
                lp = done.loc[done["pick_m"].idxmax()]
                pos, t0 = (lp["lat"], lp["lng"]), T
            else:
                pos, t0 = depot, max(T, 0)
            pts = todo.groupby("aoi_id")[["lat", "lng"]].median().to_numpy()
            if len(pts):
                proj = t0 + nn_tour_minutes(pos, pts, depot, a, b)
            elif len(done):      # 已知的单都揽完了：按最后一单之后直接回仓估算
                proj = lp["pick_m"] + b * float(haversine_km(pos[0], pos[1], depot[0], depot[1]))
            else:
                proj = 0.0
            rows.append({"courier_id": courier, "ds": ds, "region": region, "T": T,
                         "done": len(done), "todo_stops": len(pts), "new_after": new_after,
                         "proj_known": proj})
    return pd.DataFrame(rows), pd.DataFrame(days)


# ---------------------------------------------------------------- 规则与评估
def first_alert(cp: pd.DataFrame, cond: pd.Series) -> pd.Series:
    """每个揽收员-日第一次触发预警的检查点（未触发为 NaN）"""
    return cp[cond].groupby(["courier_id", "ds"])["T"].min()


def score(days: pd.DataFrame, first: pd.Series) -> dict:
    d = days.set_index(["courier_id", "ds"])
    d["alert_T"] = first.reindex(d.index)
    alerted = d["alert_T"].notna()
    tp = (alerted & d["late"]).sum()
    precision = tp / max(alerted.sum(), 1)
    recall = tp / max(d["late"].sum(), 1)
    # 加车要 1 小时以上：17:00 的预警只剩 60 分钟，来不及；16:00 及以前的才算来得及
    in_time = (d["alert_T"] < DEADLINE - ADD_VEHICLE_LEAD) & d["late"]
    return {
        "预警次数": int(alerted.sum()),
        "每天平均预警": round(alerted.sum() / d.index.get_level_values("ds").nunique(), 1),
        "命中率": precision, "召回率": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-9),
        "16点前预警到的超时占比（来得及加车）": in_time.sum() / max(d["late"].sum(), 1),
    }


def rules(cp: pd.DataFrame, future_mean: pd.Series, a, b, hop_km) -> dict:
    """返回 {规则名: (参数列表, 参数→触发条件 的函数)}"""
    exp_future = cp.set_index(["region", "T"]).index.map(future_mean).fillna(0).to_numpy()
    proj_future = cp["proj_known"] + exp_future * (a + b * hop_km)
    return {
        "A 简单规则（16:00 未揽收点 ≥ N）":
            (range(1, 16), lambda N: (cp["T"] == 480) & (cp["todo_stops"] >= N)),
        "B 时间推算（预计回仓 > 18:00 − m）":
            (range(-60, 121, 10), lambda m: cp["proj_known"] > DEADLINE - m),
        "C 时间推算 + 预计新增":
            (range(-60, 121, 10), lambda m: proj_future > DEADLINE - m),
    }, proj_future


def main():
    df, depots = load()
    a, b, hop_km, binned = calibrate(df)
    print(f"标定：每点 {a:.1f} 分钟 + 每公里 {b:.1f} 分钟（等效 {60 / b:.1f} km/h）")
    cp, days = replay(df, depots, a, b)
    train_mask = cp["ds"] <= TRAIN_END
    future_mean = cp[train_mask].groupby(["region", "T"])["new_after"].mean()

    rule_defs, proj_future = rules(cp, future_mean, a, b, hop_km)
    cp["proj_future"] = proj_future
    tr_days, te_days = days[days["ds"] <= TRAIN_END], days[days["ds"] > TRAIN_END]

    # 两种模式：全部检查点（含 17:00）/ 早预警（只用 16:00 及以前的检查点，来得及加车）
    early = cp["T"] < DEADLINE - ADD_VEHICLE_LEAD
    modes = {"全部检查点": pd.Series(True, index=cp.index), "早预警（≤16:00）": early}
    results, chosen = [], {}
    for mode, allowed in modes.items():
        for name, (grid, make) in rule_defs.items():
            def train_f1(p):
                c = make(p) & allowed
                return score(tr_days, first_alert(cp[train_mask], c[train_mask]))["F1"]
            best = max(grid, key=train_f1)
            cond = make(best) & allowed
            chosen[(mode, name)] = (best, cond)
            s = score(te_days, first_alert(cp[~train_mask], cond[~train_mask]))
            results.append({"模式": mode, "规则": name, "调优后参数": best, **s})
    res = pd.DataFrame(results)
    for c in ["命中率", "召回率", "F1", "16点前预警到的超时占比（来得及加车）"]:
        res[c] = res[c].map(lambda x: f"{x:.0%}")

    # 规则 C（早预警模式）在各检查点的单独表现
    best_name = "C 时间推算 + 预计新增"
    m_best, cond_best = chosen[("早预警（≤16:00）", best_name)]
    per_T = []
    for T, label in CHECKPOINTS.items():
        sub = cp[~train_mask & (cp["T"] == T)].merge(te_days[["courier_id", "ds", "late"]])
        fire = sub["proj_future"] > DEADLINE - m_best
        tp = (fire & sub["late"]).sum()
        per_T.append({"检查点": label, "触发": int(fire.sum()),
                      "命中率": f"{tp / max(fire.sum(), 1):.0%}",
                      "召回率": f"{tp / max(sub['late'].sum(), 1):.0%}",
                      "平均已知未揽收点": round(sub["todo_stops"].mean(), 1),
                      "平均之后还会新增的点": round(sub["new_after"].mean(), 1)})

    # 预警推送明细（测试期，规则 C 早预警，首次触发）
    log = cp[~train_mask & cond_best].sort_values("T").groupby(["courier_id", "ds"]).head(1)
    log = log.merge(te_days[["courier_id", "ds", "late", "actual_return"]])
    hhmm = lambda m: f"{SHIFT_START_H + int(m) // 60:02d}:{int(m) % 60:02d}"
    log = log.assign(检查点=log["T"].map(CHECKPOINTS), 预计回仓=log["proj_future"].map(hhmm),
                     实际回仓=log["actual_return"].map(hhmm))
    log[["ds", "region", "courier_id", "检查点", "done", "todo_stops", "预计回仓", "实际回仓", "late"]] \
        .rename(columns={"ds": "日期", "region": "片区", "courier_id": "揽收员", "done": "已揽收单",
                         "todo_stops": "未揽收点", "late": "实际是否超时"}) \
        .to_csv(ROOT / "reports/alert_log_test.csv", index=False, encoding="utf-8-sig")
    ex = log[log["late"]].iloc[0]
    message = (f"【回仓预警】{ex['ds']:%m-%d} {ex['检查点']} 片区 {ex['region']} 揽收员 {ex['courier_id']}："
               f"已揽 {ex['done']} 单，还剩 {ex['todo_stops']} 个点，预计 {ex['预计回仓']} 回仓，"
               f"距 18:00 余量不足 {m_best} 分钟。建议：现在联系机动车，或调剂附近车辆接手部分点位。")

    late_rate = days["late"].mean()
    md = [
        "# 回仓超时预警：规则设计与历史回放", "",
        "## 1. 速度标定（来自真实揽收间隔）", "",
        f"间隔(分钟) = **{a:.1f}** + **{b:.1f}** × 距离(km)，即每个点约 {a:.0f} 分钟、"
        f"路上等效 {60 / b:.1f} km/h；相邻揽收点距离中位数 {hop_km:.2f} km。", "",
        binned.reset_index().assign(bin=lambda x: x["bin"].astype(str)).round(2)
              .rename(columns={"bin": "距离区间km", "km": "距离中位数", "gap": "间隔中位数(分钟)", "n": "样本数"})
              .to_markdown(index=False), "",
        "## 2. 回放样本", "",
        f"- 揽收员-日 {len(days):,} 个（每天 ≥ {MIN_ORDERS} 单；已剔除爬坡期与停摆日）",
        f"- 其中实际回仓超时 {days['late'].sum():,} 个，占 {late_rate:.1%}",
        f"- 训练期 {len(tr_days):,} 个，测试期 {len(te_days):,} 个", "",
        "## 3. 三条规则在测试期的表现", "",
        res.to_markdown(index=False), "",
        f"## 4. 规则 C（早预警参数 m = {m_best}）在各检查点的表现", "",
        pd.DataFrame(per_T).to_markdown(index=False), "",
        "## 5. 推送示例", "", f"> {message}", "",
    ]
    (ROOT / "reports/alert_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
