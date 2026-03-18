# QuantETF

A-share ETF rotation trading system.

Default remote provider: `easyquotation`.

Note: `easyquotation` exposes live market snapshots and ETF metadata. Historical backtests still depend on the local CSV cache for older trading days.

Daily snapshot cache example:
`/Users/Paul/python_env/my_ml_env/bin/python main/cache_data.py --today`
