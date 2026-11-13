import { useCallback, useRef, useState } from "react";
import {
  getJson,
  type BacktestResponse,
  type Candle,
  type ForecastResponse,
  type LadderResponse,
  type LiveResponse,
} from "./api";
import { BtcChart } from "./components/BtcChart";
import { Insights } from "./components/Insights";
import { MetricsPanel } from "./components/MetricsPanel";
import { ReliabilityCard } from "./components/ReliabilityCard";
import { StrikeLadder } from "./components/StrikeLadder";
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
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [live, setLive] = useState<LiveResponse | null>(null);
  const [ladder, setLadder] = useState<LadderResponse | null>(null);
  const [which, setWhich] = useState<"next" | "following">("next");
  const whichRef = useRef(which);
  whichRef.current = which;
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

  const ladderTask = useCallback(async (signal: AbortSignal) => {
    const res = await getJson<LadderResponse>(`/ladder?which=${whichRef.current}`, signal);
    if (res.which === whichRef.current) setLadder(res);
  }, []);
  const ladderStatus = usePolling(ladderTask, 15_000, 5_000);

  const chooseWhich = (w: "next" | "following") => {
    setWhich(w);
    whichRef.current = w;
    setLadder(null);
    ladderTask(new AbortController().signal).catch(() => undefined);
  };

  const backtestStatus = usePolling(
    useCallback(async (signal: AbortSignal) => {
      setBacktest(await getJson<BacktestResponse>("/backtest?source=all", signal));
    }, []),
    2 * 60_000,
    30_000,
  );

  const twoUp = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 520px), 1fr))",
    gap: "1rem",
    marginBottom: "1rem",
  } as const;

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

      <div style={{ ...twoUp, alignItems: "start" }}>
        <StrikeLadder ladder={ladder} status={ladderStatus} which={which} onWhich={chooseWhich} />
        <Insights forecast={forecast} />
      </div>

      <div style={twoUp}>
        <BtcChart
          candles={candles}
          forecast={forecast?.forecast ?? []}
          live={live}
          status={candleStatus}
        />
        <VolatilityChart data={forecast?.historical_volatility ?? []} />
      </div>

      <div style={{ marginBottom: "1rem" }}>
        <ReliabilityCard
          backtest={backtest}
          status={backtestStatus}
          model={forecast?.diagnostics.cone_model ?? "ensemble"}
        />
      </div>

      <MetricsPanel metrics={forecast?.metrics ?? null} diagnostics={forecast?.diagnostics ?? null} />
    </main>
  );
}

export default App;
