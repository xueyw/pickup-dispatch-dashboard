-- =====================================================================
-- 01 原始层 raw：CSV 原样导入，全部字段 text，不做任何修改
-- 数据导入由 src/load_db.py 用 COPY 完成
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS mart;

DROP TABLE IF EXISTS raw.pickup CASCADE;
CREATE TABLE raw.pickup (
    order_id          text,
    region_id         text,
    city              text,
    courier_id        text,
    accept_time       text,   -- 格式 "MM-DD HH:MM:SS"，没有年份
    time_window_start text,
    time_window_end   text,
    lng               text,
    lat               text,
    aoi_id            text,
    aoi_type          text,
    pickup_time       text,
    pickup_gps_time   text,
    pickup_gps_lng    text,
    pickup_gps_lat    text,
    accept_gps_time   text,
    accept_gps_lng    text,
    accept_gps_lat    text,
    ds                text    -- "0821" = 8 月 21 日
);
