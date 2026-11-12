import { useEffect, useRef } from "react";
import { CandlestickSeries, LineSeries, LineStyle, type IChartApi } from "lightweight-charts";
import type { Candle, ForecastPoint } from "../api";
import { toTimestamp } from "../charts/format";
import { useChart } from "../charts/useChart";
import { colors } from "../theme";
import { ChartPanel } from "./ChartPanel";
import { ChartToolbar } from "./ChartToolbar";

type Band = "p025" | "p16" | "p50" | "p84" | "p975";

const BANDS: { key: Band; color: string; style: LineStyle; width: 1 | 2 }[] = [
  { key: "p975", color: "rgba(56, 189, 248, 0.45)", style: LineStyle.Dotted, width: 1 },
  { key: "p84", color: colors.accent, style: LineStyle.Dashed, width: 1 },
  { key: "p50", color: colors.accent, style: LineStyle.Solid, width: 2 },
  { key: "p16", color: colors.accent, style: LineStyle.Dashed, width: 1 },
  { key: "p025", color: "rgba(56, 189, 248, 0.45)", style: LineStyle.Dotted, width: 1 },
];

function setupSeries(chart: IChartApi) {
  const candles = chart.addSeries(CandlestickSeries, {
    upColor: colors.up,
    downColor: colors.down,
    borderVisible: false,
    wickUpColor: colors.up,
    wickDownColor: colors.down,
  });
  const bands = Object.fromEntries(
    BANDS.map(({ key, color, style, width }) => [
      key,
      chart.addSeries(LineSeries, {
        color,
        lineStyle: style,
        lineWidth: width,
        priceLineVisible: false,
        lastValueVisible: key === "p50",
        crosshairMarkerVisible: false,
      }),
    ]),
  ) as Record<Band, ReturnType<IChartApi["addSeries"]>>;
  return { candles, bands };
}

interface Props {
  candles: Candle[];
  forecast: ForecastPoint[];
  status: string;
}

export function BtcChart({ candles, forecast, status }: Props) {
  const { containerRef, handle } = useChart(setupSeries);
  const hasFitRef = useRef(false);

  useEffect(() => {
    if (!handle || candles.length === 0) return;
    handle.series.candles.setData(
      candles.map((c) => ({
        time: toTimestamp(c.time),
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
      })),
    );
    if (!hasFitRef.current) {
      handle.chart.timeScale().fitContent();
      hasFitRef.current = true;
    }
  }, [handle, candles]);

  useEffect(() => {
    if (!handle) return;
    for (const { key } of BANDS) {
      handle.series.bands[key].setData(
        forecast.map((p) => ({ time: toTimestamp(p.time), value: p[key] })),
      );
    }
  }, [handle, forecast]);

  const lastTime = candles.length ? toTimestamp(candles[candles.length - 1].time) : null;
  const forecastEnd = forecast.length ? toTimestamp(forecast[forecast.length - 1].time) : null;

  return (
    <ChartPanel
      title="BTC-USD · 1m candles with forecast cone"
      toolbar={<ChartToolbar chart={handle?.chart ?? null} lastTime={lastTime} rightEdge={forecastEnd} />}
      status={status}
      containerRef={containerRef}
      height={420}
      legend={
        <>
          Cone: median (solid), 68% band (dashed), 95% band (dotted), from Monte Carlo paths
          of factor + idiosyncratic GARCH shocks. Drift is ≈ 0 at this horizon; the width is
          the forecast.
        </>
      }
    />
  );
}
