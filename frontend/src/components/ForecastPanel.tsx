import React, { useEffect, useRef } from "react";
import {
  createChart,
  LineSeries,
  LineStyle,
  ColorType,
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
} from "lightweight-charts";

export interface ForecastPoint {
  time: string;
  expected_price: number;
  upper_1sd: number;
  lower_1sd: number;
}

export interface QuantMetrics {
  alpha: number;
  beta: number;
  pca_explained_variance_ratio: number;
  pca_loadings: { eth: number; sp500: number };
  residual_std_error: number;
  garch_params: Record<string, number>;
}

interface Props {
  forecast: ForecastPoint[];
  metrics: QuantMetrics | null;
}

export const ForecastPanel: React.FC<Props> = ({ forecast, metrics }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const expectedRef = useRef<ISeriesApi<"Line"> | null>(null);
  const upperRef = useRef<ISeriesApi<"Line"> | null>(null);
  const lowerRef = useRef<ISeriesApi<"Line"> | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f172a" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      width: containerRef.current.clientWidth,
      height: 280,
      localization: {
        timeFormatter: (timestamp: number) =>
          new Intl.DateTimeFormat("en-US", {
            timeZone: "America/Chicago",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
          }).format(new Date(timestamp * 1000)),
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (timestamp: number) =>
          new Intl.DateTimeFormat("en-US", {
            timeZone: "America/Chicago",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          }).format(new Date(timestamp * 1000)),
      },
    });

    expectedRef.current = chart.addSeries(LineSeries, {
      color: "#38bdf8",
      lineWidth: 2,
      title: "Expected",
    });

    upperRef.current = chart.addSeries(LineSeries, {
      color: "#22c55e",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: "+1 SD",
    });

    lowerRef.current = chart.addSeries(LineSeries, {
      color: "#ef4444",
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      title: "-1 SD",
    });

    chartRef.current = chart;

    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (!expectedRef.current || !upperRef.current || !lowerRef.current || forecast.length === 0) return;

    expectedRef.current.setData(
      forecast.map((d) => ({
        time: Math.floor(new Date(d.time).getTime() / 1000) as UTCTimestamp,
        value: d.expected_price,
      }))
    );

    upperRef.current.setData(
      forecast.map((d) => ({
        time: Math.floor(new Date(d.time).getTime() / 1000) as UTCTimestamp,
        value: d.upper_1sd,
      }))
    );

    lowerRef.current.setData(
      forecast.map((d) => ({
        time: Math.floor(new Date(d.time).getTime() / 1000) as UTCTimestamp,
        value: d.lower_1sd,
      }))
    );

    chartRef.current?.timeScale().fitContent();
  }, [forecast]);

  const cardStyle: React.CSSProperties = {
    backgroundColor: "#1e293b",
    borderRadius: "8px",
    padding: "0.75rem 1rem",
    border: "1px solid #334155",
  };

  return (
    <div
      style={{
        backgroundColor: "#0f172a",
        borderRadius: "10px",
        padding: "1rem",
        border: "1px solid #1e293b",
        display: "grid",
        gridTemplateColumns: "2fr 1fr",
        gap: "1rem",
      }}
    >
      {/* Left: Price Forecast Trajectory + +/- 1 SD Bands */}
      <div>
        <h3 style={{ margin: "0 0 0.75rem 0", color: "#f8fafc", fontSize: "1rem" }}>
          BTC Price Forecast &amp; ±1 SD Skew-t Volatility Cone (Next {forecast.length}m)
        </h3>
        <div ref={containerRef} style={{ width: "100%", borderRadius: "6px", overflow: "hidden" }} />
      </div>

      {/* Right: CAPM (Alpha / Beta) & GJR-GARCH Metrics */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
        <h3 style={{ margin: 0, color: "#f8fafc", fontSize: "1rem" }}>
          CAPM &amp; AR(1)-GJR-GARCH Metrics
        </h3>

        {metrics ? (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.65rem" }}>
              <div style={cardStyle}>
                <div style={{ color: "#94a3b8", fontSize: "0.75rem" }}>CAPM Alpha (α)</div>
                <div style={{ color: "#38bdf8", fontSize: "1.15rem", fontWeight: 700 }}>
                  {metrics.alpha.toFixed(5)}%
                </div>
              </div>
              <div style={cardStyle}>
                <div style={{ color: "#94a3b8", fontSize: "0.75rem" }}>CAPM Beta (β to PC1)</div>
                <div style={{ color: "#22c55e", fontSize: "1.15rem", fontWeight: 700 }}>
                  {metrics.beta.toFixed(4)}
                </div>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.65rem" }}>
              <div style={cardStyle}>
                <div style={{ color: "#94a3b8", fontSize: "0.75rem" }}>PC1 Variance Explained</div>
                <div style={{ color: "#f8fafc", fontSize: "1rem", fontWeight: 600 }}>
                  {(metrics.pca_explained_variance_ratio * 100).toFixed(1)}%
                </div>
              </div>
              <div style={cardStyle}>
                <div style={{ color: "#94a3b8", fontSize: "0.75rem" }}>Residual Std Error</div>
                <div style={{ color: "#facc15", fontSize: "1rem", fontWeight: 600 }}>
                  {(metrics.residual_std_error * 10_000).toFixed(2)} bps
                </div>
              </div>
            </div>

            <div style={cardStyle}>
              <div style={{ color: "#94a3b8", fontSize: "0.75rem", marginBottom: "0.35rem" }}>
                PCA Loadings &amp; GJR-GARCH(1,1) Skew-t Params
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "0.25rem 0.75rem",
                  fontSize: "0.8rem",
                  color: "#cbd5e1",
                }}
              >
                <div>ETH Loading: <strong>{metrics.pca_loadings.eth.toFixed(3)}</strong></div>
                <div>SPX Loading: <strong>{metrics.pca_loadings.sp500.toFixed(3)}</strong></div>
                {Object.entries(metrics.garch_params).map(([key, val]) => (
                  <div key={key}>
                    {key}: <strong>{val.toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div style={{ color: "#94a3b8", fontSize: "0.9rem" }}>
            Waiting for econometric model fit...
          </div>
        )}
      </div>
    </div>
  );
};
