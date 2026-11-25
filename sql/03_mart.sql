-- =====================================================================
-- 03 分析层 mart：日报、看板直接读取的表
-- 依赖：stg.pickup（02_stg.sql）、stg.region_day（Python 预测模块产出，由 load_db.py 导入）
-- =====================================================================

-- ---------- 片区日报 ----------
DROP TABLE IF EXISTS mart.daily_region_kpi CASCADE;
CREATE TABLE mart.daily_region_kpi AS
SELECT p.ds,
       p.region_id,
       COUNT(*)                                          AS orders,
       COUNT(DISTINCT p.aoi_id)                          AS stops,
       COUNT(DISTINCT p.courier_id)                      AS couriers,
       ROUND(AVG((NOT p.biz_late)::int), 4)              AS on_time_rate,
       ROUND(AVG(p.biz_early::int), 4)                   AS early_rate,
       ROUND(AVG(p.biz_after_return::int), 4)            AS after_18_rate,
       ROUND(AVG(p.sus_gps_missing::int), 4)             AS gps_missing_rate,
       rd.base_stops                                     AS baseline_stops,   -- 近 28 天中位数
       ROUND((COUNT(DISTINCT p.aoi_id) / NULLIF(rd.base_stops, 0))::numeric, 3) AS stops_vs_baseline,  -- ROUND(x, n) 只接受 numeric
       COALESCE(rd.anomaly, false)                       AS is_anomaly,
       COALESCE(rd.ramp_up, false)                       AS is_ramp_up
FROM stg.pickup p
LEFT JOIN stg.region_day rd ON rd.region_id = p.region_id AND rd.ds = p.ds
WHERE p.is_clean
GROUP BY p.ds, p.region_id, rd.base_stops, rd.anomaly, rd.ramp_up;

-- ---------- 全市日报（看板首页） ----------
CREATE OR REPLACE VIEW mart.v_daily_overview AS
SELECT ds,
       SUM(orders)                                             AS orders,
       SUM(stops)                                              AS stops,
       SUM(couriers)                                           AS couriers,
       ROUND(SUM(orders * on_time_rate) / SUM(orders), 4)      AS on_time_rate,
       ROUND(SUM(orders * after_18_rate) / SUM(orders), 4)     AS after_18_rate,
       COUNT(*) FILTER (WHERE is_anomaly)                      AS anomaly_regions,
       ROUND(AVG(SUM(orders)) OVER (ORDER BY ds ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)) AS orders_ma7
FROM mart.daily_region_kpi
WHERE NOT is_ramp_up
GROUP BY ds;

-- ---------- 揽收员-日（回仓风险） ----------
DROP TABLE IF EXISTS mart.courier_day CASCADE;
CREATE TABLE mart.courier_day AS
SELECT courier_id,
       ds,
       MODE() WITHIN GROUP (ORDER BY region_id)   AS region_id,
       COUNT(*)                                   AS orders,
       COUNT(DISTINCT aoi_id)                     AS stops,
       MIN(pickup_ts)                             AS first_pickup,
       MAX(pickup_ts)                             AS last_pickup,
       MAX(pickup_ts)::time > '17:00'             AS last_pickup_after_17
FROM stg.pickup
WHERE is_clean
GROUP BY courier_id, ds;

-- ---------- 数据质量汇总：SQL 重新计算 + 与 Python 结果交叉核对 ----------
DROP TABLE IF EXISTS mart.dq_summary CASCADE;
CREATE TABLE mart.dq_summary AS
WITH sql_hits AS (
    SELECT rule, hits FROM (
        SELECT
            SUM(dq_duplicate::int)       AS dq_duplicate,
            SUM(dq_time_reverse::int)    AS dq_time_reverse,
            SUM(dq_tw_invalid::int)      AS dq_tw_invalid,
            SUM(dq_accept_after_tw::int) AS dq_accept_after_tw,
            SUM(dq_date_mismatch::int)   AS dq_date_mismatch,
            SUM(dq_far_away::int)        AS dq_far_away,
            SUM(sus_gps_missing::int)    AS sus_gps_missing,
            SUM(sus_gps_drift::int)      AS sus_gps_drift,
            SUM(sus_night::int)          AS sus_night,
            SUM(biz_late::int)           AS biz_late,
            SUM(biz_early::int)          AS biz_early,
            SUM(biz_cross_day::int)      AS biz_cross_day,
            SUM(biz_after_return::int)   AS biz_after_return
        FROM stg.pickup
    ) s
    CROSS JOIN LATERAL (VALUES
        ('dq_duplicate', s.dq_duplicate), ('dq_time_reverse', s.dq_time_reverse),
        ('dq_tw_invalid', s.dq_tw_invalid), ('dq_accept_after_tw', s.dq_accept_after_tw),
        ('dq_date_mismatch', s.dq_date_mismatch), ('dq_far_away', s.dq_far_away),
        ('sus_gps_missing', s.sus_gps_missing), ('sus_gps_drift', s.sus_gps_drift),
        ('sus_night', s.sus_night), ('biz_late', s.biz_late), ('biz_early', s.biz_early),
        ('biz_cross_day', s.biz_cross_day), ('biz_after_return', s.biz_after_return)
    ) v(rule, hits)
)
SELECT m.kind, m.rule, m.description,
       h.hits                                                  AS sql_hits,
       m.py_hits,
       ROUND(h.hits * 100.0 / (SELECT COUNT(*) FROM stg.pickup), 2) AS pct,
       h.hits = m.py_hits                                      AS sql_python_match
FROM stg.dq_rule_meta m
JOIN sql_hits h USING (rule);
