import { useEffect, useState } from "react";
import { BtcChart } from "./components/BtcChart";
import { VolatilityChart, VolatilityPoint } from "./components/VolatilityChart";
import { ForecastPanel, ForecastPoint, QuantMetrics } from "./components/ForecastPanel";

function App() {
  const [histVol, setHistVol] = useState<VolatilityPoint[]>([]);
  const [forecast, setForecast] = useState<ForecastPoint[]>([]);
  const [metrics, setMetrics] = useState<QuantMetrics | null>(null);
  const [forecastStatus, setForecastStatus] = useState<string>("");

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const res = await fetch("/forecast?horizon=60");
        if (res.status === 503) {
          setForecastStatus("Waiting for BTC, ETH, and SPX data to finish loading...");
          setTimeout(fetchForecast, 5000);
          return;
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        setHistVol(data.historical_volatility ?? []);
        setForecast(data.forecast ?? []);
        setMetrics(data.metrics ?? null);
        setForecastStatus("");
      } catch (err) {
        console.error("Forecast fetch failed:", err);
        setForecastStatus("Unable to fetch forecast metrics.");
      }
    };

    fetchForecast();
    const interval = setInterval(fetchForecast, 60_000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main
      style={{
        backgroundColor: "#020617",
        minHeight: "100vh",
        padding: "1rem",
        boxSizing: "border-box",
        fontFamily: "system-ui, -apple-system, sans-serif",
      }}
    >
      {forecastStatus && (
        <div style={{ color: "#facc15", marginBottom: "0.75rem", fontSize: "0.9rem" }}>
          {forecastStatus}
        </div>
      )}

      {/* Top Row: Candlestick (Left) & Historical Volatility (Right) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(520px, 1fr))",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        <BtcChart />
        <VolatilityChart data={histVol} />
      </div>

      {/* Bottom Row: Forecast & CAPM / GJR-GARCH Metrics */}
      <ForecastPanel forecast={forecast} metrics={metrics} />
    </main>
  );
}

export default App;
