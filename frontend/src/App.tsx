import { useCallback, useRef, useState } from "react";
import {
  getJson,
  type BacktestResponse,
  type BacktestSource,
  type Candle,
  type ForecastResponse,
  type LiveResponse,
} from "./api";
import { BtcChart } from "./components/BtcChart";
import { MetricsPanel } from "./components/MetricsPanel";
import { VolatilityChart } from "./components/VolatilityChart";
import { usePolling } from "./hooks/usePolling";
import { colors } from "./theme";

const WINDOW_MS = 72 * 60 * 60 * 1000;
// Re-request a few recent bars each poll so the backend's late bar revisions reach the chart.
const OVERLAP_MS = 3 * 60 * 1000;

function mergeCandles(prev: Candle[], incoming: Candle[]): Candle[] {
  if (incoming.length === 0) return prev;
  const byTime = new Map(prev.map((c) => [c.time, c]));
  for (const c of incoming) byTime.set(c.time, c);
  const merged = [...byTime.values()].sort((a, b) => Date.parse(a.time) - Date.parse(b.time));
  const cutoff = Date.parse(merged[merged.length - 1].time) - WINDOW_MS;
  return merged.filter((c) => Date.parse(c.time) >= cutoff);
}

function App() {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [backtests, setBacktests] = useState<Partial<Record<BacktestSource, BacktestResponse>>>({});
  const [live, setLive] = useState<LiveResponse | null>(null);
  const lastCandleRef = useRef<string | null>(null);

  const candleStatus = usePolling(
    useCallback(async (signal: AbortSignal) => {
      const last = lastCandleRef.current;
      const since = last ? `?since=${encodeURIComponent(new Date(Date.parse(last) - OVERLAP_MS).toISOString())}` : "";
      const rows = await getJson<Candle[]>(`/candles${since}`, signal);
      if (rows.length) lastCandleRef.current = rows[rows.length - 1].time;
      setCandles((prev) => mergeCandles(prev, rows));
    }, []),
  );

  const forecastStatus = usePolling(
    useCallback(async (signal: AbortSignal) => {
      setForecast(await getJson<ForecastResponse>("/forecast", signal));
    }, []),
  );

  usePolling(
    useCallback(async (signal: AbortSignal) => {
      setLive(await getJson<LiveResponse>("/live", signal));
    }, []),
    10_000,
    10_000,
  );

  const backtestStatus = usePolling(
    useCallback(async (signal: AbortSignal) => {
      const sources: BacktestSource[] = ["all", "live", "replay"];
      const results = await Promise.all(
        sources.map((s) => getJson<BacktestResponse>(`/backtest?source=${s}`, signal)),
      );
      setBacktests(Object.fromEntries(sources.map((s, i) => [s, results[i]])));
    }, []),
    2 * 60_000,
    30_000,
  );

  return (
    <main
      style={{
        backgroundColor: colors.page,
        minHeight: "100vh",
        padding: "1rem",
        boxSizing: "border-box",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {forecastStatus && (
        <div style={{ color: colors.warn, marginBottom: "0.75rem", fontSize: "0.9rem" }}>
          Forecast: {forecastStatus}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 520px), 1fr))",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        <BtcChart
          candles={candles}
          forecast={forecast?.forecast ?? []}
          live={live}
          status={candleStatus}
        />
        <VolatilityChart data={forecast?.historical_volatility ?? []} />
      </div>

      <MetricsPanel
        metrics={forecast?.metrics ?? null}
        diagnostics={forecast?.diagnostics ?? null}
        backtests={backtests}
        backtestStatus={backtestStatus}
      />
    </main>
  );
}

export default App;
