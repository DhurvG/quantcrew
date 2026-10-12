# Backtest comparison

Stocks: INFY.NS. Period: 2025-07-01 to 2025-12-31. Horizon: medium.

| Configuration | Calls | Strategy return % | Buy & hold % | Strategy Sharpe | B&H Sharpe | Strategy max DD % | B&H max DD % | Time in market % | Hit rate | Calls changed by debate |
|---|---|---|---|---|---|---|---|---|---|---|
| crew | 6 | 9.0 | 5.42 | 1.45 | 0.71 | 0.0 | -1.91 | 16.7 | 0.75 | 2 |
| technical | 6 | 10.89 | 5.42 | 1.76 | 0.71 | 0.0 | -1.91 | 33.3 | 1.0 | 0 |

Strategy: hold the stock for the horizon on BUY, otherwise cash; equal-weight across stocks. No transaction costs. Fundamental and Sentiment agents are excluded from backtests because no point-in-time data exists for them.
