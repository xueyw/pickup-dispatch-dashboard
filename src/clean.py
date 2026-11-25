"""
LaDe-P 揽收数据清洗脚本

用法：
    .venv/bin/python src/clean.py data/raw/pickup_jl.csv

输出：
    data/clean/<文件名>_clean.parquet   清洗后数据（保留全部行，带标记字段）
    reports/<文件名>_dq_rules.csv       每条规则的命中统计
    reports/<文件名>_dq_report.md       数据质量报告

清洗原则：
    1. 原始数据不动，清洗结果另存
    2. 只打标记、不删行；分析时用 is_clean 过滤
    3. 区分三类标记：
       dq_*   脏数据      -> 分析时剔除
       sus_*  可疑数据    -> 保留，但在特定分析里谨慎使用
       biz_*  业务异常    -> 真实发生的业务现象，是分析和预警的对象，必须保留
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

YEAR = "2022"            # 原始数据时间不含年份，数据集说明为 2022-05 ~ 2022-10
FAR_KM = 30              # 距所在片区中心超过该距离视为坐标异常
GPS_DRIFT_KM = 2         # 揽收 GPS 与商户坐标相距超过该距离视为漂移
NIGHT_HOUR = 6           # 早于该小时的揽收视为可疑
RETURN_HOUR = 18         # 仓库要求 18 点前回仓

TIME_COLS = ["accept_time", "time_window_start", "time_window_end",
             "pickup_time", "pickup_gps_time", "accept_gps_time"]


def haversine_km(lat1, lng1, lat2, lng2):
    """两点间球面距离（公里），支持向量运算"""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- 1. 读取
def load(path: Path) -> pd.DataFrame:
    # ds 是 "0821" 这种带前导 0 的字符串，必须按字符串读，否则会变成 821
    return pd.read_csv(path, dtype={"ds": str})


# ---------------------------------------------------------------- 2. 格式修正
def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in TIME_COLS:
        # 补年份；无法解析的变成 NaT，后面用 dq_parse_fail 统计
        df[col] = pd.to_datetime(YEAR + "-" + df[col], format="%Y-%m-%d %H:%M:%S",
                                 errors="coerce")
    df["ds"] = pd.to_datetime(YEAR + df["ds"], format="%Y%m%d", errors="coerce")
    return df


# ---------------------------------------------------------------- 3. 打标记
def flag(df: pd.DataFrame) -> pd.DataFrame:
    # ---- 脏数据 dq_*：逻辑上不可能成立，分析时剔除
    df["dq_parse_fail"] = df[["accept_time", "pickup_time", "ds"]].isna().any(axis=1)
    df["dq_duplicate"] = df.duplicated("order_id", keep="first")
    df["dq_time_reverse"] = df["pickup_time"] < df["accept_time"]
    df["dq_tw_invalid"] = df["time_window_end"] < df["time_window_start"]
    df["dq_accept_after_tw"] = df["accept_time"] > df["time_window_end"]
    df["dq_date_mismatch"] = df["pickup_time"].dt.normalize() != df["ds"]
    # 注意：最初用"距全城中心 > 100km"，结果误判了 9.28% 的数据——
    # 片区 132 是整个外县（65 名揽收员），本身就离市区 105km。
    # 改为"距所在片区中心"判断，才能抓到真正的坐标漂移。
    region_lat = df.groupby("region_id")["lat"].transform("median")
    region_lng = df.groupby("region_id")["lng"].transform("median")
    df["dist_to_region_km"] = haversine_km(df["lat"], df["lng"], region_lat, region_lng)
    df["dq_far_away"] = df["dist_to_region_km"] > FAR_KM

    # ---- 可疑数据 sus_*：可能是真实情况，也可能是采集问题，保留但标出
    df["sus_gps_missing"] = df["pickup_gps_lat"].isna()
    df["gps_offset_km"] = haversine_km(df["lat"], df["lng"],
                                       df["pickup_gps_lat"], df["pickup_gps_lng"])
    df["sus_gps_drift"] = df["gps_offset_km"] > GPS_DRIFT_KM   # GPS 缺失时为 False
    df["sus_night"] = df["pickup_time"].dt.hour < NIGHT_HOUR
    # 揽收员单日揽收量用 IQR 判断离群（按揽收员-日期聚合后回填到每行）
    stops = df.groupby(["courier_id", "ds"])["order_id"].transform("count")
    daily = df.groupby(["courier_id", "ds"]).size()
    q1, q3 = daily.quantile([0.25, 0.75])
    df["courier_day_stops"] = stops
    df["sus_courier_day_outlier"] = stops > q3 + 3 * (q3 - q1)

    # ---- 业务异常 biz_*：真实业务现象，是分析与预警的对象
    df["biz_late"] = df["pickup_time"] > df["time_window_end"]
    df["biz_early"] = df["pickup_time"] < df["time_window_start"]
    df["biz_cross_day"] = df["pickup_time"].dt.normalize() > df["accept_time"].dt.normalize()
    df["biz_after_return"] = df["pickup_time"].dt.hour >= RETURN_HOUR

    # ---- 衍生字段
    tw_hours = (df["time_window_end"] - df["time_window_start"]).dt.total_seconds() / 3600
    df["tw_hours"] = tw_hours
    df["tw_type"] = np.select([tw_hours.round(1) == 2, tw_hours >= 9],
                              ["2小时窗口", "全天窗口"], "其他")
    df["lead_minutes"] = (df["pickup_time"] - df["accept_time"]).dt.total_seconds() / 60
    df["pickup_hour"] = df["pickup_time"].dt.hour

    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    df["is_clean"] = ~df[dq_cols].any(axis=1)
    return df


# ---------------------------------------------------------------- 4. 校验
SCHEMA = pa.DataFrameSchema({
    "order_id": pa.Column(int, unique=True),
    "courier_id": pa.Column(int),
    "aoi_id": pa.Column(int),
    "lat": pa.Column(float, pa.Check.in_range(40, 47)),       # 吉林省大致范围
    "lng": pa.Column(float, pa.Check.in_range(121, 132)),
    "accept_time": pa.Column("datetime64[ns]", nullable=False),
    "pickup_time": pa.Column("datetime64[ns]", nullable=False),
    "lead_minutes": pa.Column(float, pa.Check.ge(0)),
}, strict=False)


# ---------------------------------------------------------------- 5. 报告
RULE_DESC = {
    "dq_parse_fail": "关键时间字段无法解析",
    "dq_duplicate": "order_id 重复",
    "dq_time_reverse": "揽收时间早于接单时间",
    "dq_tw_invalid": "时间窗结束早于开始",
    "dq_accept_after_tw": "接单时间晚于时间窗结束",
    "dq_date_mismatch": "揽收日期与 ds 不一致",
    "dq_far_away": f"距所在片区中心 > {FAR_KM} km",
    "sus_gps_missing": "揽收 GPS 缺失",
    "sus_gps_drift": f"揽收 GPS 与商户坐标相距 > {GPS_DRIFT_KM} km",
    "sus_night": f"{NIGHT_HOUR} 点前揽收",
    "sus_courier_day_outlier": "揽收员单日揽收量极端（> Q3 + 3×IQR）",
    "biz_late": "晚于时间窗揽收（超时）",
    "biz_early": "早于时间窗开始揽收（提前）",
    "biz_cross_day": "跨天揽收（接单当天未揽）",
    "biz_after_return": f"{RETURN_HOUR} 点后揽收（影响回仓）",
}
KIND = {"dq": "脏数据（剔除）", "sus": "可疑（保留标记）", "biz": "业务异常（保留分析）"}


def build_report(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, str]:
    n = len(df)
    rules = pd.DataFrame([
        {"类型": KIND[c.split("_")[0]], "规则": c, "说明": RULE_DESC[c],
         "命中行数": int(df[c].sum()), "占比%": round(df[c].mean() * 100, 2)}
        for c in RULE_DESC
    ])
    clean = df[df["is_clean"]]
    tw = clean["tw_type"].value_counts()
    lines = [
        f"# 数据质量报告：{name}",
        "",
        "## 概览",
        "",
        f"- 原始行数：{n:,}",
        f"- 可用行数（剔除脏数据后）：{len(clean):,}（{len(clean) / n:.2%}）",
        f"- 时间范围：{df['ds'].min():%Y-%m-%d} ~ {df['ds'].max():%Y-%m-%d}，共 {df['ds'].nunique()} 天",
        f"- 揽收员 {df['courier_id'].nunique()} 人，AOI（揽收点）{df['aoi_id'].nunique()} 个，"
        f"片区 {df['region_id'].nunique()} 个",
        "",
        "## 规则命中统计",
        "",
        rules.to_markdown(index=False),
        "",
        "## 关键发现",
        "",
        f"- 时间窗类型：" + "，".join(f"{k} {v:,} 单（{v / len(clean):.1%}）" for k, v in tw.items()),
        f"- 接单到揽收的时长中位数 {clean['lead_minutes'].median():.0f} 分钟，"
        f"P90 {clean['lead_minutes'].quantile(0.9):.0f} 分钟",
        f"- 提前揽收占 {clean['biz_early'].mean():.1%}，超时揽收占 {clean['biz_late'].mean():.2%}",
        f"- {RETURN_HOUR} 点后仍在揽收的订单占 {clean['biz_after_return'].mean():.2%}，"
        f"涉及揽收员-日 {clean.loc[clean['biz_after_return'], ['courier_id', 'ds']].drop_duplicates().shape[0]:,} 个",
        f"- 揽收员单日揽收量中位数 {clean.groupby(['courier_id', 'ds']).size().median():.0f} 单",
        "",
    ]
    return rules, "\n".join(lines)


def main(path_str: str):
    path = Path(path_str)
    name = path.stem
    root = Path(__file__).resolve().parent.parent

    df = load(path)
    print(f"读取 {len(df):,} 行")
    df = flag(fix_types(df))

    SCHEMA.validate(df[df["is_clean"]], lazy=True)
    print("pandera 校验通过")

    rules, md = build_report(df, name)
    (root / "data/clean").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / f"data/clean/{name}_clean.parquet", index=False)
    rules.to_csv(root / f"reports/{name}_dq_rules.csv", index=False, encoding="utf-8-sig")
    (root / f"reports/{name}_dq_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/pickup_jl.csv")
