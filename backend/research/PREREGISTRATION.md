# Pre-registration: GARCH(1,1)-FHS vs the current method

Written and committed **before any evaluation data was downloaded**. Nothing below may
change after results are seen (Rule 1). Data was not fetched until this file was in git.

## Question
Does the candidate GARCH(1,1) filtered-historical-simulation forecaster (`garch_fhs.py`,
frozen as delivered, SHA-256 `31a665de14c07cf3876ad8f19aa072944aa63beed84216e86da00c02ab6394ab`)
give better BTC close forecasts 1, 15, 30 and 60 minutes ahead than the project's
current method?
- **Primary score:** CRPS on log prices.
- **Secondary score:** absolute error of the median.

## Evaluation period (disjoint from tuning)
- **Forecast origins:** every 60 minutes from **2026-08-01 00:00 UTC** to
  **2026-09-15 00:00 UTC**, which is about 1,081 origins.
- **The current method was chosen and tuned on 2026-10-14 → 2026-11-13** (replay
  scoring). The evaluation period ends a month before that window and does not overlap it.
- **GARCH warm-up:** 7 days (10,080 minutes) before the first origin. The `prices` series
  runs from 2026-07-25 00:01 to 2026-09-15 01:00 UTC on the close-time index.
- **History for the current method's time-of-day profile:** up to 30 days before each
  origin, so candles are fetched from 2026-07-02 00:00 UTC. It only ever sees data at or
  before the origin.

## Data
- **Source:** Coinbase Exchange public `/products/{id}/candles`, 1-minute granularity.
- **Products:** BTC-USD (target), plus ETH-USD, SOL-USD, XRP-USD and LTC-USD (the
  current method's factor basket).
- **Timestamp convention:** Coinbase `time` is the bucket **open** time. The target
  series is re-indexed by **close time** = open + 1 min, so the value at index t is the
  close known at t. The current method only sees candles with open time ≤ origin − 1 min.
- **Shared series:** every model is scored against `prepare_minute_closes` of that BTC
  series (forward-fill ≤ 5 min; the longest contiguous segment is kept if longer gaps
  exist, which is reported).

## Exact call
```python
rolling_backtest(
    prices,
    eval_horizons=(1, 15, 30, 60),
    min_train=10_080, rolling_window=10_080, step=60,
    n_simulations=2_000, drift_shrinkage=0.0, drift_uncertainty=True,
    external_models={"current": current_method},
    n_bootstrap=2_000, confidence=0.99, seed=42,
)
```

## The current method (`research/current_method_adapter.py`)
This is production's `CONE_MODEL = "ensemble"` at commit `ac3ba79`: an equal-weight
quantile average of three members.
- **`factor`:** PCA-CAPM with range-informed GJR-GARCH-X, time-of-day adjustment and FHS.
- **`gjr`:** arch AR(1)-GJR-GARCH skew-t on BTC.
- **`ewma`:** RiskMetrics EWMA.

How it's run in the backtest:
- **Fit window:** each member is fitted on the 72 h before the origin.
- **Seasonality profile:** re-estimated from the 30 days before each origin.
- **Refit schedule:** every origin, i.e. every 60 minutes. Production refits every minute,
  so each origin's forecast is exactly what production would have shown at that minute.
- **Output:** 199 evenly spaced quantile levels (0.5% to 99.5%) per horizon, as log
  prices, anchored on the shared series' close at the origin.
- **Seeds:** taken from the backtest's seeded RNG.
- **Inputs:** the method uses the factor coins and BTC high/low up to the origin. That's
  part of the method, and still no look-ahead.

## Reporting
The report follows the brief's template:
- CRPS and median-AE per model.
- `improvement_vs_*` with 99% moving-block-bootstrap CIs and verdicts.
- `skill_vs_current`.
- 95% coverage and misses (descriptive only).
- Fit failures, flagged if above about 3%.
- Median α+β, and the share of fits at the bound.

Seasonality and microstructure are listed as known limitations of the GARCH-FHS
candidate. They're not addressed in this run.
