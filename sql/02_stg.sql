-- =====================================================================
-- 02 清洗层 stg：类型转换 + 补年份 + 三级标记（与 src/clean.py 规则一致）
--   dq_*  脏数据（分析时剔除）  sus_* 可疑（保留）  biz_* 业务异常（保留分析）
-- =====================================================================

-- 两点球面距离（公里）；任一坐标为空时返回空
CREATE OR REPLACE FUNCTION stg.haversine_km(lat1 float8, lng1 float8, lat2 float8, lng2 float8)
RETURNS float8 LANGUAGE sql IMMUTABLE AS $$
  SELECT 2 * 6371 * asin(sqrt(
           power(sin(radians(lat2 - lat1) / 2), 2) +
           cos(radians(lat1)) * cos(radians(lat2)) * power(sin(radians(lng2 - lng1) / 2), 2)))
$$;

DROP TABLE IF EXISTS stg.pickup CASCADE;
CREATE TABLE stg.pickup AS
WITH typed AS (
    SELECT
        order_id::bigint                                   AS order_id,
        region_id::int                                     AS region_id,
        city,
        courier_id::bigint                                 AS courier_id,
        aoi_id::int                                        AS aoi_id,
        aoi_type::int                                      AS aoi_type,
        lng::float8                                        AS lng,
        lat::float8                                        AS lat,
        ('2022-' || accept_time)::timestamp                AS accept_ts,
        ('2022-' || time_window_start)::timestamp          AS tw_start,
        ('2022-' || time_window_end)::timestamp            AS tw_end,
        ('2022-' || pickup_time)::timestamp                AS pickup_ts,
        NULLIF(pickup_gps_lng, '')::float8                 AS pickup_gps_lng,
        NULLIF(pickup_gps_lat, '')::float8                 AS pickup_gps_lat,
        to_date('2022' || ds, 'YYYYMMDD')                  AS ds
    FROM raw.pickup
),
region_center AS (           -- 片区中心：用于判断坐标是否漂移到片区外
    SELECT region_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY lat) AS c_lat,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY lng) AS c_lng
    FROM typed GROUP BY region_id
),
flagged AS (
    SELECT t.*,
        -- 脏数据
        ROW_NUMBER() OVER (PARTITION BY t.order_id ORDER BY t.pickup_ts) > 1   AS dq_duplicate,
        t.pickup_ts < t.accept_ts                                              AS dq_time_reverse,
        t.tw_end < t.tw_start                                                  AS dq_tw_invalid,
        t.accept_ts > t.tw_end                                                 AS dq_accept_after_tw,
        t.pickup_ts::date <> t.ds                                              AS dq_date_mismatch,
        stg.haversine_km(t.lat, t.lng, c.c_lat, c.c_lng) > 30                  AS dq_far_away,
        -- 可疑
        t.pickup_gps_lat IS NULL                                               AS sus_gps_missing,
        COALESCE(stg.haversine_km(t.lat, t.lng, t.pickup_gps_lat, t.pickup_gps_lng) > 2, false)
                                                                               AS sus_gps_drift,
        EXTRACT(hour FROM t.pickup_ts) < 6                                     AS sus_night,
        -- 业务异常
        t.pickup_ts > t.tw_end                                                 AS biz_late,
        t.pickup_ts < t.tw_start                                               AS biz_early,
        t.pickup_ts::date > t.accept_ts::date                                  AS biz_cross_day,
        EXTRACT(hour FROM t.pickup_ts) >= 18                                   AS biz_after_return
    FROM typed t JOIN region_center c USING (region_id)
)
SELECT *,
       NOT (dq_duplicate OR dq_time_reverse OR dq_tw_invalid
            OR dq_accept_after_tw OR dq_date_mismatch OR dq_far_away) AS is_clean
FROM flagged;

CREATE INDEX ON stg.pickup (ds, region_id);
CREATE INDEX ON stg.pickup (courier_id, ds);
CREATE INDEX ON stg.pickup (aoi_id);
ANALYZE stg.pickup;
