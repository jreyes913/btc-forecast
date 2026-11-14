# GARCH(1,1)-FHS vs the current method: held-out backtest report

**Verdict: GARCH-FHS does not beat the current method.**
- **CRPS (primary):** it is **significantly worse at all four horizons**, though only by
  about 1%. Every 99% CI excludes zero.
- **Median-AE (secondary):** **inconclusive** everywhere.
- **Against the two built-in baselines** (i.i.d. bootstrap and Gaussian random walk):
  GARCH-FHS is significantly better on CRPS at every horizon, by about 4%.

The method, settings and evaluation period were frozen in
[`../PREREGISTRATION.md`](../PREREGISTRATION.md), committed before any data was downloaded.
Nothing was changed after seeing results.

## Data
| | |
|---|---|
| Source | Coinbase Exchange public API, 1-minute candles. Target BTC-USD; the current method also uses ETH, SOL, XRP and LTC as inputs. |
| Timestamp convention | Coinbase `time` is the candle **open** time (verified: a 00:00–00:04 request returns candles stamped 00:00 … 00:04). Re-indexed to close time = open + 1 min, so the value at t is the close known at t. |
| Price series (close time) | 2026-07-25 00:01 → 2026-09-15 01:00 UTC: **74,940 minutes** |
| Missing minutes before fill | **0** in this range (BTC had 1 no-trade minute in the whole download, outside it) |
| Segment kept | the full range (no gaps over 5 minutes, no warnings) |
| Forecast origins | **1,081**, hourly, 2026-08-01 00:00 → 2026-09-15 00:00 UTC |
| Tuning window of the current method | 2026-10-14 → 11-13, disjoint from this period |

## Settings
```python
rolling_backtest(prices, eval_horizons=(1, 15, 30, 60),
    min_train=10_080, rolling_window=10_080, step=60,
    n_simulations=2_000, drift_shrinkage=0.0, drift_uncertainty=True,
    external_models={"current": current_method},
    n_bootstrap=2_000, confidence=0.99, seed=42)
```

**The current method:** production `CONE_MODEL = "ensemble"` at commit `ac3ba79`. It is an
equal-weight quantile average of three members:
- `factor`: PCA-CAPM with range-informed GJR-GARCH-X, time-of-day adjustment and FHS.
- `gjr`: AR(1)-GJR-GARCH skew-t.
- `ewma`.

How it was run:
- **Refit:** at every origin (hourly) on the 72 h before it. The seasonal profile was
  re-estimated each time from the prior 30 days. Production refits every minute, so each
  origin's forecast is exactly what production would have shown at that minute.
- **Output:** 199 quantile levels (0.5%–99.5%) per horizon.
- **No look-ahead:** it only sees candles closed by the origin, which is tested in
  `tests/test_research.py` by corrupting all later data.

Runtime: 393 s.

## Results
Values are in **basis points of log price** (×10⁻⁴). Improvement = competitor − GARCH, so
**positive means GARCH is better**, with a 99% moving-block-bootstrap CI.

### CRPS (primary) and median absolute error (secondary)
| Horizon | GARCH CRPS | Current CRPS | iid CRPS | Gaussian CRPS | GARCH med-AE | Current med-AE | iid med-AE | Gaussian med-AE |
|---|---|---|---|---|---|---|---|---|
| h1 | 2.654 | **2.627** | 2.760 | 2.756 | 3.591 | 3.588 | 3.595 | 3.600 |
| h15 | 9.117 | **9.041** | 9.473 | 9.520 | 12.325 | 12.296 | 12.301 | 12.299 |
| h30 | 12.897 | **12.775** | 13.397 | 13.459 | 17.393 | 17.375 | 17.345 | 17.392 |
| h60 | 17.407 | **17.279** | 18.220 | 18.293 | 23.307 | 23.353 | 23.330 | 23.315 |

### GARCH vs the current method
| Horizon | CRPS improvement [99% CI] | Verdict | `skill_vs_current` | Med-AE improvement [99% CI] | Med-AE verdict |
|---|---|---|---|---|---|
| h1 | −0.026 [−0.046, −0.009] | **garch_worse** | −1.00% | −0.003 [−0.015, +0.010] | inconclusive |
| h15 | −0.076 [−0.125, −0.035] | **garch_worse** | −0.84% | −0.029 [−0.071, +0.007] | inconclusive |
| h30 | −0.122 [−0.189, −0.055] | **garch_worse** | −0.96% | −0.018 [−0.083, +0.059] | inconclusive |
| h60 | −0.128 [−0.272, −0.005] | **garch_worse** | −0.74% | +0.046 [−0.056, +0.158] | inconclusive |

### GARCH vs the built-in baselines (CRPS)
| Horizon | vs iid [99% CI] | Skill | Verdict | vs Gaussian [99% CI] | Skill | Verdict |
|---|---|---|---|---|---|---|
| h1 | +0.107 [+0.045, +0.184] | +3.9% | garch_better | +0.102 [+0.038, +0.171] | +3.7% | garch_better |
| h15 | +0.356 [+0.195, +0.540] | +3.8% | garch_better | +0.402 [+0.222, +0.578] | +4.2% | garch_better |
| h30 | +0.500 [+0.148, +0.882] | +3.7% | garch_better | +0.562 [+0.205, +0.900] | +4.2% | garch_better |
| h60 | +0.813 [+0.534, +1.106] | +4.5% | garch_better | +0.886 [+0.611, +1.174] | +4.8% | garch_better |

On median-AE, every comparison with the baselines is inconclusive. All models use zero
drift, so their medians are nearly identical.

### Coverage (descriptive only)
Origins and horizons overlap, so these counts are not independent samples.

| Horizon | GARCH 68 / 95 / 99 | Current 68 / 95 / 99 | iid 68 / 95 / 99 | Gaussian 68 / 95 / 99 |
|---|---|---|---|---|
| h1 | 55.7 / 88.5 / 97.3 | 57.4 / 87.6 / 94.3 | 59.1 / 92.1 / 97.9 | 73.8 / 92.9 / 96.9 |
| h15 | 66.8 / 92.0 / 97.7 | 67.2 / 91.5 / 95.9 | 76.3 / 94.6 / 98.6 | 78.9 / 95.0 / 98.0 |
| h30 | 67.2 / 93.3 / 98.1 | 67.6 / 92.2 / 96.4 | 78.3 / 95.1 / 97.7 | 79.8 / 95.2 / 97.3 |
| h60 | 71.0 / 94.4 / 98.1 | 70.9 / 92.6 / 96.9 | 80.6 / 95.2 / 98.0 | 81.5 / 95.3 / 97.8 |

Misses of the 95% band, out of 1,081 per horizon:

| Model | h1 | h15 | h30 | h60 |
|---|---|---|---|---|
| GARCH | 124 | 86 | 72 | 60 |
| Current | 134 | 92 | 84 | 80 |
| iid | 85 | 58 | 53 | 52 |
| Gaussian | 77 | 54 | 52 | 51 |

What the coverage shows:
- **Too narrow at 1 minute:** both GARCH and the current method have bands that are too
  narrow at h1, likely from microstructure and intraday seasonality.
- **Tails:** the current method's 99% band is the most under-covering (94–97%), while
  GARCH's tails are closer to nominal.
- **Too wide in the middle:** the baselines' 68% bands are far too wide (76–82%). Their
  constant volatility is usually too high (see below).

## Diagnostics
- **GARCH fit failures:** **0 / 1,081**. No origin used carried-over parameters.
- **Median α+β:** **0.99900**. **Share of fits at the bound (> 0.998): 100%.** Every fit
  is pinned at the stationarity constraint. This is the minute-frequency limitation the
  brief anticipated.
- **The current method's own fits:** its GJR member (via `arch`) raised 11 optimizer
  convergence warnings, about 1% of origins. Those forecasts were used as-is, which is
  what production would have done.

### The section-3 check (`garch_longrun_diagnostic.json`)
This refits the frozen `fit_garch` on the same windows at every 24th origin (46 fits). It's
read-only and changes no result.

| Quantity | Median | p10 | p90 |
|---|---|---|---|
| Long-run variance ω/(1−α−β) ÷ sample variance | 4.88 | 2.46 | 12.67 |
| σ²_{T+1} ÷ sample variance | 0.46 | 0.15 | 1.07 |
| Mean GARCH variance over the next 60 min ÷ sample variance | 0.65 | 0.29 | 1.23 |
| Volatility half-life | 692 min | | |

With persistence pinned at 0.999, the half-life is about 11.5 h. Over a 60-minute horizon,
GARCH barely mean-reverts, so its long-run variance (about 5× the sample level, an
artifact of the bound) never matters. Its edge over the i.i.d. and Gaussian baselines is a
**level effect**: at a typical origin, the current conditional variance is about half the
7-day sample variance, which the constant-volatility baselines can't see. That is real
volatility clustering at work, but it is "the current level" rather than "better
dynamics toward a long-run mean".

## Conclusions by horizon
| Horizon | vs current: CRPS | vs current: median-AE |
|---|---|---|
| h1 | **worse** (−1.0%) | inconclusive |
| h15 | **worse** (−0.8%) | inconclusive |
| h30 | **worse** (−1.0%) | inconclusive |
| h60 | **worse** (−0.7%; upper CI −0.005) | inconclusive |

**Does the edge fade with horizon, as the theory predicts?**
- **Against the current method** there's no edge to fade. The deficit is roughly constant
  in relative terms (−0.7% to −1.0%).
- **Against the baselines** the edge **doesn't fade either.** Skill is flat to rising
  (3.7–4.8%), because persistence sits at the bound and volatility doesn't mean-revert
  within an hour. The "fade" prediction assumes meaningful mean reversion inside the
  horizon, and these fits don't have it.

**Recommendation:** keep the current method. GARCH-FHS is a sound, well-behaved
volatility model, clearly better than constant-volatility baselines. But the current
ensemble, which adds time-of-day seasonality, intraminute range information, the
crypto-market factor and model averaging, is consistently a little better on the primary
score.

## Known limitations of the candidate (not addressed in this run)
- No intraday or weekly seasonality, so GARCH(1,1) absorbs it as persistence. Consistent
  with this, 100% of fits sit at the bound.
- No microstructure or bid-ask-bounce modeling, which likely explains why every model's
  h1 bands are too narrow.
- No exogenous inputs (volume, order book).

Any follow-ups on these should be judged on a **later** held-out period, not this one.
