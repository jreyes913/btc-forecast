import { useEffect, useRef } from "react";
import { CandlestickSeries, LineSeries, LineStyle, type IChartApi } from "lightweight-charts";
import type { Candle, ForecastPoint, LiveResponse } from "../api";
import { formatCrosshairTime, toTimestamp } from "../charts/format";
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
  live: LiveResponse | null;
  status: string;
}

const LIVE_UP = "rgba(34, 197, 94, 0.45)";
const LIVE_DOWN = "rgba(239, 68, 68, 0.45)";

/** "Cone from $X at 16:43 · live $Y (+Z bps)": makes a stale cone origin obvious. */
function OriginBadge({ live }: { live: LiveResponse | null }) {
  if (!live?.cone_origin_price || !live.cone_origin_time) return null;
  const origin = live.cone_origin_price;
  const now = live.bar?.close ?? null;
  const drift = now === null ? null : Math.log(now / origin) * 1e4;
  const big = drift !== null && Math.abs(drift) >= 10;
  return (
    <div style={{ fontSize: "0.8rem", color: colors.textMuted, marginBottom: "0.4rem" }}>
      Cone from <b style={{ color: colors.text }}>${origin.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b>{" "}
      at {formatCrosshairTime(toTimestamp(live.cone_origin_time))}
      {now !== null && (
        <>
          {" · live "}
          <b style={{ color: colors.text }}>${now.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b>{" "}
          <span style={{ color: big ? colors.warn : colors.textMuted }}>
            ({drift! >= 0 ? "+" : ""}
            {drift!.toFixed(1)} bps{big ? ": price has moved since the cone's origin; shift strikes by this much" : ""})
          </span>
        </>
      )}
    </div>
  );
}

export function BtcChart({ candles, forecast, live, status }: Props) {
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
    // The forming bar (display only, muted) goes after the last closed bar.
    const bar = live?.bar;
    if (bar && toTimestamp(bar.time) > toTimestamp(candles[candles.length - 1].time)) {
      const color = bar.close >= bar.open ? LIVE_UP : LIVE_DOWN;
      handle.series.candles.update({
        time: toTimestamp(bar.time),
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
        color,
        wickColor: color,
      });
    }
    if (!hasFitRef.current) {
      handle.chart.timeScale().fitContent();
      hasFitRef.current = true;
    }
  }, [handle, candles, live]);

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
      header={<OriginBadge live={live} />}
      containerRef={containerRef}
      height={420}
      legend={
        <>
          Cone: middle (solid), 68% range (dashed), 95% range (dotted) for the next two hours.
          It shows how far BTC is likely to move, not which way. The strike table above turns
          it into chances for each Kalshi strike.
        </>
      }
    />
  );
}
