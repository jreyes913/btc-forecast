# BTC-USD Intraday Econometric Forecasting Dashboard

A full-stack quantitative trading and forecasting dashboard that ingests real-time 1-minute OHLCV candles from Coinbase Exchange (`BTC-USD`, `ETH-USD`, `SPX-USD`), stores a rolling 72-hour window in DuckDB, fits a two-stage **PCA-CAPM + AR(1)-GJR-GARCH(1,1) Skew-t** model, and serves an interactive React + TradingView Lightweight Charts UI from a single FastAPI application.

---

## Architecture

1. **Data Ingestion (`backend/src/services/candles.py`)**
   - Polls Coinbase Exchange public `/candles` API at 1-minute granularity (`granularity=60`).
   - Maintains a rolling 72-hour window (4,320 bars per product) across `BTC-USD`, `ETH-USD`, and `SPX-USD` in **DuckDB** (`backend/data/candles.duckdb`).
2. **Quantitative Engine (`backend/src/services/quant.py`)**
   - **Stage 1 (PCA Market Factor & CAPM):** Standardizes `ETH-USD` and `SPX-USD` log returns, extracts the first Principal Component (`PC1`), and fits OLS CAPM (\(r_{\text{BTC}, t} = \alpha + \beta \cdot \text{PC1}_t + \varepsilon_t\)).
   - **Stage 2 (Conditional Mean & Volatility):** Models idiosyncratic CAPM residuals (\(\varepsilon_t\)) using an **AR(1) + GJR-GARCH(1,1)** specification with **Skewed Student's t** innovations (`arch`), simulating multi-step price trajectories and ± 1 SD confidence cones.
3. **API & Static Server (`backend/app.py`)**
   - Exposes `GET /candles` and `GET /forecast?horizon=60` endpoints.
   - Serves the compiled React production bundle (`frontend/dist`) at `/` so the entire stack runs as one process.
4. **Frontend Dashboard (`frontend/src/`)**
   - **Top Left:** Interactive `BTC-USD` 1-minute candlestick chart with timeframe presets (`1H`, `6H`, `24H`, `All 72H`), zoom controls, and Central Time (`America/Chicago`) formatting.
   - **Top Right:** Historical GJR-GARCH conditional volatility (\(\sigma_t\) in bps).
   - **Bottom Panel:** 60-minute forward BTC price forecast with ± 1 SD volatility bands alongside live CAPM (α, β), PCA loadings, and GJR-GARCH parameters.

---

## Project Structure

```text
btc-forecast/
├── backend/
│   ├── app.py                  # FastAPI application, endpoints, & static file server
│   ├── requirements.txt        # Python dependencies
│   └── src/
│       ├── settings.py         # Coinbase endpoints, product list, & DuckDB path
│       ├── models/
│       │   └── api.py          # Pydantic response schemas
│       └── services/
│           ├── candles.py      # Background Coinbase polling worker & DuckDB writer
│           └── quant.py        # PCA-CAPM + AR(1)-GJR-GARCH(1,1) Skew-t model
└── frontend/
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    └── src/
        ├── App.tsx             # Main dashboard grid & polling state
        └── components/
            ├── BtcChart.tsx        # Candlestick chart with zoom/rehome toolbar
            ├── VolatilityChart.tsx # Conditional volatility area chart
            └── ForecastPanel.tsx   # Forecast cone chart & econometric metrics
```

---

## Setup & Running

### 1. Build the React Frontend
```bash
cd frontend
npm install
npx vite build
```

### 2. Set Up & Start the FastAPI Backend
```bash
cd ../backend
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi[standard] duckdb pandas requests arch scikit-learn statsmodels
fastapi run app.py
```

### 3. Open the Dashboard
Navigate to **`http://localhost:8000`** in your browser. *(On initial startup, allow ~10–15 seconds for the background worker to download the initial 72-hour candle history across all three products.)*
