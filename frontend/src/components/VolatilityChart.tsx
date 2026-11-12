import React, { useEffect, useRef } from "react";
import {
  createChart,
  AreaSeries,
  ColorType,
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
} from "lightweight-charts";

export interface VolatilityPoint {
  time: string;
  volatility: number;
}

interface Props {
  data: VolatilityPoint[];
}

export const VolatilityChart: React.FC<Props> = ({ data }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);
  const hasFittedRef = useRef<boolean>(false);
  const totalBarsRef = useRef<number>(0);

  const handleRehome = () => {
    const totalBars = totalBarsRef.current;
    if (!chartRef.current || totalBars === 0) return;

    const padding = Math.ceil(totalBars * 0.05);
    chartRef.current.timeScale().setVisibleLogicalRange({
      from: -padding,
      to: totalBars + padding,
    });
    chartRef.current.priceScale("right").applyOptions({ autoScale: true });
  };

  // Zoom In / Out around the center of the current viewport
  const handleZoom = (zoomIn: boolean) => {
    const timeScale = chartRef.current?.timeScale();
    if (!timeScale) return;

    const range = timeScale.getVisibleLogicalRange();
    if (!range) return;

    const span = range.to - range.from;
    const center = (range.from + range.to) / 2;
    const scaleFactor = zoomIn ? 0.7 : 1.4;
    const newSpan = Math.max(10, span * scaleFactor);

    timeScale.setVisibleLogicalRange({
      from: center - newSpan / 2,
      to: center + newSpan / 2,
    });
  };

  // Zoom to the latest N minutes (e.g., 60 for 1H, 360 for 6H)
  const handleTimeframe = (minutes: number) => {
    const totalBars = totalBarsRef.current;
    if (!chartRef.current || totalBars === 0) return;

    const padding = Math.ceil(minutes * 0.05);
    chartRef.current.timeScale().setVisibleLogicalRange({
      from: Math.max(0, totalBars - minutes),
      to: totalBars + padding,
    });
    chartRef.current.priceScale("right").applyOptions({ autoScale: true });
  };

  // 1. Initialize Chart once on mount
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
      height: 360,
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
        priceFormatter: (val: number) => `${val.toFixed(2)} bps`,
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        minBarSpacing: 0.05,
        tickMarkFormatter: (timestamp: number) =>
          new Intl.DateTimeFormat("en-US", {
            timeZone: "America/Chicago",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          }).format(new Date(timestamp * 1000)),
      },
    });

    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: "#f59e0b",
      topColor: "rgba(245, 158, 11, 0.35)",
      bottomColor: "rgba(245, 158, 11, 0.02)",
      lineWidth: 2,
    });

    chartRef.current = chart;
    seriesRef.current = areaSeries;

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

  // 2. Update Chart Series whenever incoming data changes
  useEffect(() => {
    if (!seriesRef.current || data.length === 0) return;

    const formatted = data.map((d) => ({
      time: Math.floor(new Date(d.time).getTime() / 1000) as UTCTimestamp,
      value: d.volatility * 10_000,
    }));

    seriesRef.current.setData(formatted);
    totalBarsRef.current = formatted.length;

    if (!hasFittedRef.current) {
      handleRehome();
      hasFittedRef.current = true;
    }
  }, [data]);

  const btnStyle: React.CSSProperties = {
    backgroundColor: "#1e293b",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: "6px",
    padding: "0.35rem 0.75rem",
    cursor: "pointer",
    fontSize: "0.85rem",
    fontWeight: 500,
  };

  return (
    <div style={{ backgroundColor: "#0f172a", borderRadius: "10px", padding: "1rem", border: "1px solid #1e293b" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "0.75rem",
          flexWrap: "wrap",
          gap: "0.5rem",
        }}
      >
        <h3 style={{ margin: 0, color: "#f8fafc", fontSize: "1rem" }}>
          Historical Conditional Volatility (GJR-GARCH σₜ)
        </h3>

        <div style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}>
          <button style={btnStyle} onClick={() => handleTimeframe(60)}>1H</button>
          <button style={btnStyle} onClick={() => handleTimeframe(360)}>6H</button>
          <button style={btnStyle} onClick={() => handleTimeframe(1440)}>24H</button>
          <button style={btnStyle} onClick={handleRehome}>All (72H)</button>
          <span style={{ color: "#475569", margin: "0 0.25rem" }}>|</span>
          <button style={btnStyle} onClick={() => handleZoom(true)} title="Zoom In">+</button>
          <button style={btnStyle} onClick={() => handleZoom(false)} title="Zoom Out">−</button>
          <button
            style={{ ...btnStyle, backgroundColor: "#0284c7", borderColor: "#0369a1" }}
            onClick={handleRehome}
            title="Reset View"
          >
            Rehome
          </button>
        </div>
      </div>

      {/* Chart Canvas Container */}
      <div ref={containerRef} style={{ width: "100%", borderRadius: "6px", overflow: "hidden" }} />
    </div>
  );
};
