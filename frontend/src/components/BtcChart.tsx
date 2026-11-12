import React, { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  ColorType,
  IChartApi,
  ISeriesApi,
  UTCTimestamp,
} from "lightweight-charts";

interface ApiCandle {
  time: string;
  low: number;
  high: number;
  open: number;
  close: number;
  volume: number;
}

export const BtcChart: React.FC = () => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  const totalBarsRef = useRef<number>(0);
  const hasInitialFitRef = useRef<boolean>(false);

  const [status, setStatus] = useState<string>("Loading chart data...");

  // Rehome: Reset zoom & pan to show all 72h data + 5% padding
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

  // 1. Initialize Chart & Candlestick Series
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0f172a" },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      width: chartContainerRef.current.clientWidth,
      height: 360,
      localization: {
        timeFormatter: (timestamp: number) => {
          return new Intl.DateTimeFormat("en-US", {
            timeZone: "America/Chicago",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
          }).format(new Date(timestamp * 1000));
        },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        minBarSpacing: 0.05,
        tickMarkFormatter: (timestamp: number) => {
          return new Intl.DateTimeFormat("en-US", {
            timeZone: "America/Chicago",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          }).format(new Date(timestamp * 1000));
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    chartRef.current = chart;
    candleSeriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
    };
  }, []);

  // 2. Fetch Data & Poll Every 60s
  useEffect(() => {
    const fetchCandles = async () => {
      try {
        const res = await fetch("/candles");

        if (res.status === 503) {
          setStatus("Backend is downloading initial 72h data... retrying in 5s");
          setTimeout(fetchCandles, 5000);
          return;
        }

        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);

        const rawData: ApiCandle[] = await res.json();

        const candleData = rawData.map((d) => ({
          time: Math.floor(new Date(d.time).getTime() / 1000) as UTCTimestamp,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));

        candleSeriesRef.current?.setData(candleData);
        totalBarsRef.current = candleData.length;

        if (!hasInitialFitRef.current) {
          handleRehome();
          hasInitialFitRef.current = true;
        }

        setStatus("");
      } catch (err) {
        console.error("Failed to fetch candles:", err);
        setStatus("Unable to connect to backend API.");
      }
    };

    fetchCandles();
    const interval = setInterval(fetchCandles, 60_000);
    return () => clearInterval(interval);
  }, []);

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
        <h2 style={{ margin: 0, color: "#f8fafc" }}>BTC-USD (1m Candles)</h2>

        {/* Zoom & Rehome Controls */}
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

      {status && (
        <div style={{ color: "#facc15", marginBottom: "0.5rem" }}>{status}</div>
      )}

      <div
        ref={chartContainerRef}
        style={{ width: "100%", borderRadius: "8px", overflow: "hidden" }}
      />
    </div>
  );
};
