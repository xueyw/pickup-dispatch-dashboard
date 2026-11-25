# 小批量揽收：排车 + 日报 + 预警

[English](README.md) | [Français](README.fr.md) | 中文

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

## 技术栈

| 用途 | 库 / 工具 |
|---|---|
| 数据处理 | pandas、NumPy、PyArrow（Parquet 存储） |
| 数据质量校验 | pandera（schema 与取值范围校验） |
| 数据库 | PostgreSQL 16、psycopg（COPY 批量导入）、SQLAlchemy |
| 路径优化 | Google OR-Tools（带容量和时间窗的车辆路径问题 CVRPTW） |
| 需求预测 | scikit-learn（HistGradientBoostingRegressor 分位数回归） |
| 可视化 | Streamlit、folium + streamlit-folium（地图）、Plotly、Power BI |
| 其他 | tabulate（生成 Markdown 报告）、concurrent.futures（多进程并行仿真） |

## 数据流程

```
原始 CSV（26 万行）
  │  clean.py：字段解析 → 去重 → 15 条清洗规则打标记 → pandera 校验
  ▼
清洗后数据（Parquet，全部行都保留，用 is_clean 区分）
  ├─ capacity_sensitivity.py  不同车辆容量下最少需要几辆车
  ├─ buffer_sim.py            前一晚计划 vs 当天实际执行（多进程，8 天 × 2 片区 × 6 种方案）
  ├─ segment.py               商户分层 + 协商放宽时间窗的实验
  ├─ forecast.py              片区-日面板 → 异常日识别 → 特征 → P50/P80 预测
  └─ alert.py                 按检查点回放当天进度 → 预警规则评估
  ▼
load_db.py：raw（原样导入）→ stg（SQL 清洗）→ mart（日报、揽收员-日、DQ 汇总）
  ▼
Power BI 日报 / Streamlit 地图看板
```

## 解决的问题

**数据清洗**
- **时间字段没有年份**：原始数据格式是 `MM-DD HH:MM:SS`，结合 `ds` 字段补上年份后解析，并检查揽收时间和日期是否一致。
- **"距离过远"规则误杀了 9% 的数据**：最初按到市中心超过 100km 判定，结果把一整个外县片区都判成了脏数据。改成"到本片区中位点超过 30km"之后，只命中 2 条真正异常的记录。
- **规则分三级处理**：GPS 缺失（16%）、漂移这类问题不能直接删，否则会丢掉大量有效订单。所以分成脏数据（剔除）、可疑（保留并标记）、业务异常（拿来做分析），每条规则都有命中数和占比报告。
- **SQL 和 Python 两套清洗结果交叉核对**：13 条规则的命中数逐条比对，结果全部一致，保证数据库里的数据和分析结果是同一口径。
- **识别数据本身的问题**：5 月上旬是采集爬坡期；6/20–6/25 市区 11 个片区同时接近 0 单，但外县正常，说明是数据缺失而不是业务真的停了。按"低于近 28 天中位数的 30%"自动标记为异常日，不参与训练和评估。

**建模与分析**
- **跨天的时间窗导致点位被丢**：有些订单的时间窗在第二天，被截断到 18:00 后变成了无解。识别出这类订单后，把它们设为当天任意时间可揽。
- **找到决定用车数的真正因素**：一开始以为是时间缓冲不够，但车只装一半就回仓了。进一步分析发现，73% 的计划点要求在 9–11 点揽收，是上午的时间窗决定了要派几辆车，于是把仿真方案改成"机动车 × 上午时间窗放宽"。
- **商户分层第一版几乎全是"混合"**：因为一个 AOI 里有多个商户，订单比例很难分开。改成按"有上午必到订单的天数占比"来分层，结果就清楚了。
- **高峰预警太多**：用 P80 判断高峰时，875 次里有 574 次报警，几乎天天报。改用 P50 判断后降到 199 次，命中率 33%，召回率 40%。
- **预警评估口径**：原来"17:00 预警"也算作及时，但这时已经来不及加车了。改成只有 16:00 前报出的才算及时，并单独评估"早预警"模式。
- **前一晚计划的稳定性**：当天重新规划路线时，给改派到其他车的点位加惩罚，保证"计划好之后不大改"这个约束成立。

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
