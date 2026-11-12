# BTC-USD Intraday Econometric Forecasting Dashboard

A full-stack dashboard that ingests 1-minute OHLCV candles from Coinbase Exchange for
`BTC-USD` and a basket of crypto majors (`ETH-USD`, `SOL-USD`, `XRP-USD`, `LTC-USD`). It keeps
a rolling 72-hour window in DuckDB and fits a two-stage **PCA factor CAPM + GARCH** model. The
result is a Monte Carlo forecast cone for the next hour, shown in a React + TradingView
Lightweight Charts UI served by a single FastAPI process.

---

## The model

All returns are 1-minute log returns in basis points.

1. **Market factor (PCA).** The four basket returns are standardized and their first principal
   component, PC1, is the "crypto market" factor. Its sign is fixed so the ETH loading is
   positive.
2. **CAPM.** OLS regression of BTC on the factor:
   \(r_{\text{BTC},t} = \alpha + \beta\,\text{PC1}_t + \varepsilon_t\).
3. **Volatility.**
   - The factor gets a zero-mean GARCH(1,1) with Student-t innovations.
   - The idiosyncratic residual \(\varepsilon_t\) gets an AR(1)-GJR-GARCH(1,1) with skewed-t
     innovations.
4. **Forecast cone.**
   - Simulate 1,000 seeded paths of both components. They are independent because OLS
     residuals are orthogonal to the regressor.
   - Sum the paths into cumulative BTC returns and read off empirical quantiles: the median,
     the 68% band (p16–p84) and the 95% band (p2.5–p97.5).
   - The bands therefore include factor risk \(\beta^2\sigma_f^2\) as well as idiosyncratic risk.
   - The drift is essentially zero at this horizon, so the width of the cone is the forecast.

**Data hygiene.**
- Coinbase omits minutes with no trades. Prices are put on a complete 1-minute grid and carried
  forward for at most 5 minutes, so an untraded minute counts as a zero return. Longer holes are
  dropped rather than treated as a single 1-minute return.
- The bar for the current, still-forming minute is never stored.

**Backtest.**
- A rolling-origin evaluation refits the model on a trailing 48-hour window every 30 minutes.
  It scores 1, 15 and 60-minute forecasts on band coverage and QLIKE.
- The model is compared against an EWMA baseline and a GJR-GARCH fitted directly to BTC
  returns.
- Results appear in the dashboard once about 49 hours of data have accumulated.

---

## Architecture

| Piece | File | Role |
|---|---|---|
| Ingestion | `backend/src/services/candles.py` | Each minute, fetches only the bars missing for each product, so outages and restarts backfill themselves. Upserts into DuckDB and prunes rows older than 72 h. The table stays at about 22k rows (1–2 MB). |
| Model | `backend/src/services/quant.py` | Stateless, testable functions for each stage above. |
| Cache | `backend/src/services/forecast_cache.py` | One fit per new closed bar per horizon. Requests never trigger extra refits. |
| Backtest | `backend/src/services/backtest.py`, `backend/scripts/backtest.py` | Rolling out-of-sample evaluation. Refreshed hourly in the app, or run from the CLI. |
| API | `backend/app.py` | `GET /candles?product=&since=`, `GET /forecast?horizon=`, `GET /backtest`. Also serves `frontend/dist` at `/`. |
| UI | `frontend/src/` | Candles with the forecast cone overlaid, total vs idiosyncratic conditional volatility, and a diagnostics and backtest panel. |

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

Open **http://localhost:8000**. On first start, the worker downloads 72 hours of history for
five products, which takes a minute or two. The forecast appears once at least 500 aligned
returns are available.

### Frontend development

Run `fastapi run app.py` in `backend/`, then run `npm run dev` in `frontend/`. Vite proxies the
API routes to port 8000.

### Tests, lint, backtest

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q && ruff check .
python -m scripts.backtest              # on the stored candles
python -m scripts.backtest --synthetic  # offline demo

cd ../frontend && npm run typecheck
```

CI (`.github/workflows/ci.yml`) runs all of these on every push.

### Configuration

Products, the window length, the forward-fill limit and the number of simulations are set in
`backend/src/settings.py`.
