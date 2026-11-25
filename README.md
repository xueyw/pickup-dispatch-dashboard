# 小批量揽收：排车 + 日报 + 预警

基于菜鸟公开数据集 [LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe)（吉林揽收数据，26 万条订单）做的模拟项目。场景是面向小批量商户的揽收业务，有以下约束：

- 车辆在前一晚排定，**每车每天只跑一趟**，当天基本不能大改
- 临时加车要 1 小时以上才能到
- 所有车辆 **18:00 前必须回仓**卸货

> 车辆容量、包裹体积、车速等字段在原数据中没有，是模拟出来的，具体假设写在各脚本开头的注释里。

## 模块

| 模块 | 文件 | 内容 |
|---|---|---|
| 数据清洗 | `src/clean.py` | 三级标记：脏数据 `dq_*`（剔除）/ 可疑 `sus_*`（保留）/ 业务异常 `biz_*`（用于分析）；用 pandera 做 schema 校验 |
| 数据仓库 | `sql/`、`src/load_db.py` | PostgreSQL 分 raw → stg → mart 三层，SQL 和 Python 的清洗结果逐条规则交叉核对 |
| 排车求解 | `src/capacity_sensitivity.py` | OR-Tools CVRPTW（容量 + 时间窗 + 18:00 回仓） |
| 地图看板 | `app.py` | Streamlit + folium，按车辆画路线，标出回仓余量不足 60 分钟的车 |
| 缓冲仿真 | `src/buffer_sim.py` | 前一晚按预测排车，再用当天实际订单执行，对比"机动车 × 上午时间窗放宽"的效果 |
| 商户分层 | `src/segment.py` | 按频率层和时间窗层划分 AOI，模拟"和头部商户协商放宽上午时间窗"能省多少车 |
| 需求预测 | `src/forecast.py` | 分位数 GBM 预测 P50/P80，含节假日特征，并识别数据停摆的异常日 |
| 回仓预警 | `src/alert.py` | 回放 13:00–17:00 各检查点，评估预警规则的命中率，以及能否在 16:00 前报出来（留出加车时间） |
| BI 报表 | Power BI | 直连 PostgreSQL 的 mart 层 |

## 主要结论

- **前一晚的计划是主要手段**：市区大片区按计划用 6.75 辆车，有 38% 的天完不成；加 1 辆机动车后完不成的天数降到 0。
- **上午 9–11 点的时间窗决定了要派几辆车**：和上午件量前 20 的商户协商把时间窗放宽 1 小时，可以省 1 辆左右（市区少 1.13 辆，外县少 1 辆）。
- **预测**：WAPE 和 28 天中位数基线差不多（12.8% 对 12.6%），提升主要在高峰日：高峰日低估从 20.3% 降到 13.0%，P80 覆盖率 80.5%。
- **当天预警是兜底手段**：最简单的规则"16:00 还有 ≥2 个点没揽"，能在来得及加车时报出 67% 的超时，比基于 ETA 的规则效果更好。
- **数据问题**：5/1–5/15 是采集爬坡期，6/20–6/25 市区片区停摆，这两段都要先识别出来剔除，否则预测和基线都会失真。

详细结果见 `reports/`。

## 运行

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw && curl -L -o data/raw/pickup_jl.csv \
  https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/pickup/pickup_jl.csv

.venv/bin/python src/clean.py
.venv/bin/python src/capacity_sensitivity.py
.venv/bin/python src/buffer_sim.py
.venv/bin/python src/segment.py
.venv/bin/python src/forecast.py
.venv/bin/python src/alert.py
.venv/bin/python src/load_db.py      # 需要本地 PostgreSQL 和数据库 lade
.venv/bin/streamlit run app.py
```

## 局限

- LaDe 的揽收员同时也做派送，所以原始数据里迟到的基础比例偏高
- 仿真假设当天所有订单在早上就已知
- 每个片区只测试了 8 天
