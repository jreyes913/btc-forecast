# BTC-USD Intraday Econometric Forecasting Dashboard

A full-stack dashboard that ingests 1-minute OHLCV candles from Coinbase Exchange for
`BTC-USD` and a basket of crypto majors (`ETH-USD`, `SOL-USD`, `XRP-USD`, `LTC-USD`). It keeps
30 days of history in DuckDB and fits a **PCA factor CAPM + range-informed GJR-GARCH-X** model
with time-of-day adjustment. The result is a forecast cone for the next hour built by
filtered historical simulation. Every forecast is scored once its deadline passes. Everything
is shown in a React + TradingView Lightweight Charts UI served by a single FastAPI process.

---

## The model

All returns are 1-minute log returns in basis points.

1. **Data.** Prices are put on a complete 1-minute grid and carried forward for at most 5
   minutes, so an untraded minute counts as a zero return; longer holes are dropped. Each
   candle's high and low give a **Parkinson range variance**, (ln H/L)² / 4 ln 2, which
   shows how much the price moved *within* the minute.
2. **Time-of-day adjustment.** Volatility follows a daily rhythm: the US open and data
   releases are busier, nights and weekends calmer. A per-asset profile s(t), in 15-minute
   buckets plus a weekend multiplier, is estimated from the full stored history. Returns are
   divided by s(t) before fitting, and forecasts are multiplied back by s(t) for the minutes
   they cover. The adjustment switches on once 7 days of history exist.
3. **Market factor (PCA).** PC1 of the four standardized basket returns, with its sign fixed
   so the ETH loading is positive.
4. **CAPM.** OLS regression of BTC on the factor: \(r_{\text{BTC},t} = \alpha + \beta\,\text{PC1}_t + \varepsilon_t\).
5. **Volatility: GJR-GARCH-X.** Both the factor and the idiosyncratic residual get

   σ²ₜ = ω + (α + γ·1[εₜ₋₁<0])·ε²ₜ₋₁ + β·σ²ₜ₋₁ + **δ·RVₜ₋₁**

   where RV is the range variance. If the range adds nothing, δ is estimated near 0. The
   residual also has an AR(1) mean. Fitting is by Gaussian quasi-maximum likelihood
   (`backend/src/services/garch.py`, checked against `arch` in the tests).
6. **Forecast cone: filtered historical simulation (FHS).** Instead of drawing shocks from
   a textbook distribution, each simulated minute resamples a *real past minute's*
   standardized shocks, for the factor and the residual **jointly**. The cone inherits
   BTC's real shape: many tiny moves, occasional jumps, and moves shared across coins.
7. **Zero drift.** α and the AR constant are estimated and reported but not simulated.
   A few days of drift is noise, not a forecast.

The dashed and dotted lines are the 16/84th and 2.5/97.5th percentiles of 1,000 simulated
paths. A strike on the lower dashed line at the deadline's point on the cone means about an
84% model probability of finishing above it.

---

## Forecast scoring

Every forecast is logged (`forecast_log` table) and scored once its target minute closes, so
the sample **grows over time**.

- **Live:** the production model and two baselines are logged every 5 minutes, at horizons of
  1, 15, 30 and 60 minutes.
- **Replay:** a background thread re-runs every model at past 30-minute origins using
  only data available then. With 30 days stored, that's about 1,100 historical forecasts per
  model, available within the first hour or so of running.
- **Compared:**
  - `factor`: the production model.
  - `factor@24h` / `factor@7d`: the same model fitted on other window lengths, replay only.
  - `gjr`: AR(1)-GJR-GARCH skew-t on BTC alone, via `arch`.
  - `ewma`: RiskMetrics EWMA.
  - `ensemble`: an equal-weight average of `factor`, `gjr` and `ewma`. It averages their
    quantiles at each probability level, which keeps the bands sharp. It's built from the
    logged member forecasts at scoring time, so it gets scored on all history, including
    forecasts logged before it existed.
  - `ensemble_bma` / `ensemble_loss`: the same quantile averaging, but with **adaptive
    weights**.
    - **How the weights update:** when each forecast resolves, every member's weight is
      updated by how well it predicted the outcome. The Bayesian version uses its predictive
      density there (dynamic Bayesian model averaging). The loss-based version uses its
      pinball loss, scaled by the forecast SD known in advance so the rule stays proper.
    - **Forgetting factor** (`ENSEMBLE_FORGETTING`, default 0.99): older evidence fades, so
      weights can shift between market regimes instead of locking onto one model.
    - **No look-ahead:** an outcome only affects forecasts made after it resolved.
    - **Caveat:** learned weights often fail to beat equal weights when members are close
      (the "forecast combination puzzle"), so these are challengers. Scoring decides.
- **Metrics:**
  - Band coverage.
  - QLIKE (lower is better).
  - **Brier score at Kalshi-style $100 strikes.** All models are scored on the same strikes.
  - A **calibration chart**: forecast vs observed frequency.
  - **Diebold-Mariano tests** against the factor model, with Newey-West errors for overlapping
    horizons.
- **Fit window:** the dashboard names the fit window with the lowest 30-minute QLIKE.
  Change `FIT_WINDOW` in `settings.py` to adopt it.
- **Cone model:** the dashboard cone comes from the factor model by default. If the ensemble
  scores better (lower QLIKE and Brier, DM p < 0.05), set `CONE_MODEL = "ensemble"` in
  `settings.py`. Scoring always logs the members separately, so the comparison stays fair
  whichever model draws the cone.

---

## Architecture

| Piece | File | Role |
|---|---|---|
| Ingestion | `backend/src/services/candles.py` | Every 15 s, fetches only the missing closed bars, newest first, then backfills history towards 30 days a chunk at a time. Upserts into DuckDB and prunes beyond 30 days (about 20 MB). `LivePoller` keeps the forming bar in memory for the chart only. |
| Seasonality | `backend/src/services/seasonality.py` | Time-of-day volatility profile. |
| Volatility | `backend/src/services/garch.py` | GJR-GARCH-X estimator and joint FHS. |
| Model | `backend/src/services/quant.py` | The pipeline above. |
| Cache | `backend/src/services/forecast_cache.py` | One fit per new closed bar per horizon. The profile is refreshed hourly. |
| Scoring | `backend/src/services/{scoring,evaluation,baselines}.py` | Forecast log, resolver, metrics, replay. |
| API | `backend/app.py` | `GET /candles?product=&since=`, `GET /forecast?horizon=`, `GET /live`, `GET /backtest?source=all\|live\|replay`. Also serves `frontend/dist` at `/`. |
| UI | `frontend/src/` | Candles with the cone and the live bar, plus a badge when price has moved since the cone's origin. Volatility vs its typical level for the time of day. Diagnostics, and the scoring table with a calibration chart. |

---

## Setup & running

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Frontend (production build, served by FastAPI)
cd ../frontend
npm ci
npm run build

# Run
cd ../backend
fastapi run app.py
```

Open **http://localhost:8000**.

- **Within about a minute:** the forecast appears. The last 72 hours download first.
- **Next few minutes:** the remaining 30 days backfill in the background.
- **After that:** replay scoring starts and fills in over roughly the next hour.

> **Upgrading from an earlier version:** delete `backend/data/candles.duckdb` (and `.wal`)
> first. The tables changed.

### Frontend development

Run `fastapi run app.py` in `backend/`, then run `npm run dev` in `frontend/`. Vite proxies the
API routes to port 8000.

### Tests, lint, scoring summary

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q && ruff check .
python -m scripts.backtest --synthetic   # offline: simulate 10 days, replay, score
python -m scripts.backtest               # summary from the stored DB (stop the app first)

cd ../frontend && npm run typecheck
```

CI (`.github/workflows/ci.yml`) runs all of these on every push.

### Configuration

Set in `backend/src/settings.py`:
- products
- `STORE_WINDOW` and `FIT_WINDOW`
- the forward-fill limit
- the number of simulations
- logging cadence and horizons
