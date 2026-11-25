"""
片区日揽收量预测（前一晚预测第二天）

用法：
    .venv/bin/python src/forecast.py

输出：
    data/clean/region_day_panel.parquet   片区 × 日期面板（含异常标记、特征），后续日报/看板复用
    reports/forecast_test_predictions.csv 测试期逐日预测
    reports/forecast_report.md            预测评估报告

预测对象：每个片区每天的 ① 标准件数 ② 揽收点数（一天一趟的车，点数比件数更决定用车）

—— 数据可用性（建模前必须处理）——
  * 2022-05-01 ~ 05-15：数据集采集爬坡期，片区和揽收员覆盖不全 → 整段剔除
  * 片区-日停摆：当天量 < 近 28 天中位数的 30%（如 06-20 ~ 06-25 市区 11 个片区同时接近 0，
    外县正常）→ 标记为异常日，不参与训练和评估，也不作为后续日期的特征
—— 时间划分 ——
  训练：截至 2022-08-31；测试：2022-09-01 ~ 10-31（含中秋、国庆，模型只在训练期见过端午）
—— 特征（全部是前一晚可知的信息）——
  星期、节假日（假期中 / 节后第几天 / 节前第几天 / 调休上班日）、
  昨天、上周同日、近 7 天均值，均除以"近 28 天中位数"做归一化，使 15 个片区可以合并训练
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from capacity_sensitivity import ROOT, add_simulated_fields

VALID_FROM = pd.Timestamp("2022-05-16")
TRAIN_END = pd.Timestamp("2022-08-31")
ANOMALY_RATIO = 0.3
PEAK_RATIO = 1.15          # 实际量 ≥ 基准 × 1.15 视为高峰日
PEAK_FLAG = 1.10           # 预测 P50 ≥ 基准 × 1.10 时发出"高峰预警"
TARGETS = {"units": "标准件", "stops": "揽收点"}

# 2022 年法定节假日（数据期内）与调休上班日
HOLIDAYS = {"端午": ("2022-06-03", "2022-06-05"),
            "中秋": ("2022-09-10", "2022-09-12"),
            "国庆": ("2022-10-01", "2022-10-07")}
ADJUSTED_WORKDAYS = ["2022-10-08", "2022-10-09"]


# ---------------------------------------------------------------- 面板与异常
def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(df["ds"].min(), df["ds"].max())
    regions = sorted(df["region_id"].unique())
    idx = pd.MultiIndex.from_product([regions, days], names=["region_id", "ds"])
    p = (df.groupby(["region_id", "ds"])
           .agg(units=("units", "sum"), stops=("aoi_id", "nunique"), orders=("order_id", "size"))
           .reindex(idx, fill_value=0).reset_index())

    p["ramp_up"] = p["ds"] < VALID_FROM
    # 基准：过去 28 天（不含当天）正常日的中位数
    p["base_units"] = np.nan
    p["base_stops"] = np.nan
    p["anomaly"] = False
    for r, g in p.groupby("region_id"):
        g = g.sort_values("ds").copy()
        units = g["units"].where(~g["ramp_up"]).astype(float)
        stops = g["stops"].where(~g["ramp_up"]).astype(float)
        anomaly = np.zeros(len(g), dtype=bool)
        # 逐日向前滚动，已判为异常的日子不进入后续基准
        u_hist, s_hist = [], []
        b_u, b_s = [], []
        for i in range(len(g)):
            bu = np.median(u_hist[-28:]) if len(u_hist) >= 7 else np.nan
            bs = np.median(s_hist[-28:]) if len(s_hist) >= 7 else np.nan
            b_u.append(bu)
            b_s.append(bs)
            if np.isnan(units.iloc[i]):
                continue
            if not np.isnan(bu) and units.iloc[i] < ANOMALY_RATIO * bu:
                anomaly[i] = True
                continue
            u_hist.append(units.iloc[i])
            s_hist.append(stops.iloc[i])
        p.loc[g.index, "base_units"] = b_u
        p.loc[g.index, "base_stops"] = b_s
        p.loc[g.index, "anomaly"] = anomaly
    p["usable"] = ~p["ramp_up"] & ~p["anomaly"] & p["base_units"].notna()
    return p


# ---------------------------------------------------------------- 特征
def calendar_features(ds: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame(index=ds.index)
    f["dow"] = ds.dt.day_of_week
    f["holiday"] = 0
    f["after_holiday"] = 0     # 节后第几天（1~3），否则 0
    f["before_holiday"] = 0    # 节前第几天（1~2），否则 0
    for start, end in HOLIDAYS.values():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        f.loc[ds.between(s, e), "holiday"] = 1
        after = (ds - e).dt.days
        f.loc[after.between(1, 3), "after_holiday"] = after
        before = (s - ds).dt.days
        f.loc[before.between(1, 2), "before_holiday"] = before
    f["adjusted_workday"] = ds.isin(pd.to_datetime(ADJUSTED_WORKDAYS)).astype(int)
    return f


def add_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.sort_values(["region_id", "ds"]).copy()
    p = pd.concat([p, calendar_features(p["ds"])], axis=1)
    for t in TARGETS:
        clean = p[t].where(p["usable"])          # 异常日不作为特征
        ratio = clean / p[f"base_{t}"]
        g = ratio.groupby(p["region_id"])
        p[f"{t}_lag1"] = g.shift(1)
        p[f"{t}_lag7"] = g.shift(7)
        p[f"{t}_ma7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
        p[f"{t}_ratio"] = p[t] / p[f"base_{t}"]
    return p


def feature_cols(t: str) -> list[str]:
    return ["dow", "holiday", "after_holiday", "before_holiday", "adjusted_workday",
            f"{t}_lag1", f"{t}_lag7", f"{t}_ma7"]


# ---------------------------------------------------------------- 建模与评估
def wape(actual, pred):
    return np.abs(actual - pred).sum() / actual.sum()


def fit_predict(train, test, t):
    X_tr, y_tr = train[feature_cols(t)], train[f"{t}_ratio"]
    X_te = test[feature_cols(t)]
    out = {}
    for name, q in [("GBM_P50", 0.5), ("GBM_P80", 0.8)]:
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300,
                                          learning_rate=0.05, max_depth=4, min_samples_leaf=20,
                                          categorical_features=[0], random_state=0)
        m.fit(X_tr, y_tr)
        out[name] = m.predict(X_te) * test[f"base_{t}"]
    return out


def evaluate(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = p[p["usable"]].dropna(subset=[f"{t}_lag7" for t in TARGETS])
    train = data[data["ds"] <= TRAIN_END]
    test = data[data["ds"] > TRAIN_END].copy()

    rows, peak_rows = [], []
    for t, label in TARGETS.items():
        preds = {
            "基线：近28天中位数（现有做法）": test[f"base_{t}"],
            "基线：上周同日": test[f"{t}_lag7"] * test[f"base_{t}"],
            "基线：近7天均值": test[f"{t}_ma7"] * test[f"base_{t}"],
            **fit_predict(train, test, t),
        }
        test[f"{t}_pred_p50"] = preds["GBM_P50"]
        test[f"{t}_pred_p80"] = preds["GBM_P80"]
        actual = test[t]
        is_peak = test[f"{t}_ratio"] >= PEAK_RATIO
        for name, pred in preds.items():
            if name == "GBM_P80":
                continue
            rows.append({"预测对象": label, "方法": name,
                         "WAPE（全部测试日）": f"{wape(actual, pred):.1%}",
                         "WAPE（高峰日）": f"{wape(actual[is_peak], pred[is_peak]):.1%}",
                         "高峰日平均低估": f"{((actual - pred) / actual)[is_peak].mean():.1%}"})
        cover = (actual <= preds["GBM_P80"]).mean()
        flag = preds["GBM_P50"] / test[f"base_{t}"] >= PEAK_FLAG
        tp = (flag & is_peak).sum()
        peak_rows.append({"预测对象": label, "测试片区-日": len(test), "其中高峰日": int(is_peak.sum()),
                          "P80 覆盖率（应≈80%）": f"{cover:.1%}",
                          "高峰预警次数": int(flag.sum()),
                          "预警命中率（预警中真是高峰）": f"{tp / max(flag.sum(), 1):.0%}",
                          "高峰召回率（高峰被预警到）": f"{tp / max(is_peak.sum(), 1):.0%}"})

    # 节假日前后：全片区合计的实际 vs 预测
    hol = test[test["holiday"].eq(1) | test["after_holiday"].gt(0) | test["before_holiday"].gt(0)
               | test["adjusted_workday"].eq(1)]
    hol_tbl = hol.groupby("ds").agg(
        实际揽收点=("stops", "sum"), 中位数基线=("base_stops", "sum"),
        GBM_P50=("stops_pred_p50", "sum"), GBM_P80=("stops_pred_p80", "sum")).round(0)
    hol_tbl.index = hol_tbl.index.strftime("%m-%d")
    return pd.DataFrame(rows), pd.DataFrame(peak_rows), hol_tbl, test


def main():
    df = pd.read_parquet(ROOT / "data/clean/pickup_jl_clean.parquet")
    df = add_simulated_fields(df[df["is_clean"]].copy())
    panel = add_features(build_panel(df))
    panel.to_parquet(ROOT / "data/clean/region_day_panel.parquet", index=False)

    metrics, peaks, hol_tbl, test = evaluate(panel)
    test.to_csv(ROOT / "reports/forecast_test_predictions.csv", index=False, encoding="utf-8-sig")

    anom = panel[panel["anomaly"]]
    anom_by_day = anom.groupby("ds")["region_id"].apply(lambda s: ", ".join(map(str, s)))
    md = [
        "# 片区日揽收量预测", "",
        "## 1. 数据可用性检查", "",
        f"- 采集爬坡期 {panel['ds'].min():%m-%d} ~ {VALID_FROM - pd.Timedelta(days=1):%m-%d}：整段剔除",
        f"- 停摆/缺失的片区-日（当天量 < 近 28 天中位数的 {ANOMALY_RATIO:.0%}）：共 {len(anom)} 个",
        "",
        "| 日期 | 异常片区 |", "|---|---|",
        *[f"| {d:%m-%d} | {r} |" for d, r in anom_by_day.items()], "",
        "## 2. 预测准确度（测试期 09-01 ~ 10-31，15 个片区）", "",
        "WAPE = 总绝对误差 / 总实际量，越低越好。", "",
        metrics.to_markdown(index=False), "",
        "## 3. 高峰识别与 P80 区间", "",
        f"高峰日：实际 ≥ 基准 × {PEAK_RATIO}；高峰预警：预测 P50 ≥ 基准 × {PEAK_FLAG}。", "",
        peaks.to_markdown(index=False), "",
        "## 4. 节假日前后（全部片区合计的揽收点数）", "",
        hol_tbl.to_markdown(), "",
    ]
    (ROOT / "reports/forecast_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
