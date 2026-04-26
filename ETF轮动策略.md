# ETF轮动策略

已按你的要求完成：新增了一个完全独立的 ETF 轮动策略实现，没有改动原有策略主流程。

**交付结果**
1. 独立策略核心实现  
[ momentum_rotation.py ](src/quant_etf/momentum_rotation.py)  
实现了：
- 动量因子：年化斜率 × R²（基于过去 N 日对 log(close) 线性回归）
- 周度调仓（每周最后一个交易日）
- 截面 Top4 等权（每只 25%）
- 复用现有回测与报告框架（数据仓库、回测引擎、报告导出）

2. 独立运行入口（不影响原入口）  
[ run_momentum_rotation_backtest.py ](main/run_momentum_rotation_backtest.py)  
支持参数：
- symbols
- lookback-days
- top-n
- start-date / end-date
- initial-capital
- output-dir
- data-dir

3. 单元测试新增并通过  
[ test_momentum_rotation.py ](tests/test_momentum_rotation.py)  
包含：
- 因子正确性测试（干净上升趋势分数高于噪声上升）
- 周度 Top4 等权选股测试（每期 4 只、权重和 1、每只 0.25）

4. README 使用说明更新  
[ README.md ](README.md)  
新增了独立策略运行章节和示例命令。

**实跑结果（已完成一次回测）**
输出目录：  data/reports_momentum_rotation 

关键指标：
- 年化收益：4.93%
- 最大回撤：-30.51%
- 夏普比率：0.2365
- 调仓次数：136

报告文件已生成，例如：
- [ metrics.json ](data/reports_momentum_rotation/metrics.json)
- [ report.html ](data/reports_momentum_rotation/report.html)
- [ weekly_signals.csv ](data/reports_momentum_rotation/weekly_signals.csv)

**运行命令**
    python run_momentum_rotation_backtest.py \
      --config-dir configs \
      --output-dir data/reports_momentum_rotation \
      --lookback-days 90 \
      --top-n 4

可选自定义资产池：
    python run_momentum_rotation_backtest.py \
      --symbols 513100.SH,518880.SH,159915.SZ,159928.SZ,515000.SH,512010.SH,516160.SH,510300.SH,510500.SH