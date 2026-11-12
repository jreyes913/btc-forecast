import { useEffect, useRef } from "react";
import { AreaSeries, LineSeries, type IChartApi } from "lightweight-charts";
import type { VolatilityPoint } from "../api";
import { toTimestamp } from "../charts/format";
import { useChart } from "../charts/useChart";
import { colors } from "../theme";
import { ChartPanel } from "./ChartPanel";
import { ChartToolbar } from "./ChartToolbar";

function setupSeries(chart: IChartApi) {
  chart.applyOptions({ localization: { priceFormatter: (v: number) => `${v.toFixed(2)} bps` } });
  return {
    total: chart.addSeries(AreaSeries, {
      lineColor: colors.vol,
      topColor: "rgba(245, 158, 11, 0.35)",
      bottomColor: "rgba(245, 158, 11, 0.02)",
      lineWidth: 2,
    }),
    idio: chart.addSeries(LineSeries, { color: colors.volIdio, lineWidth: 1 }),
  };
}

export function VolatilityChart({ data }: { data: VolatilityPoint[] }) {
  const { containerRef, handle } = useChart(setupSeries);
  const hasFitRef = useRef(false);

  useEffect(() => {
    if (!handle || data.length === 0) return;
    handle.series.total.setData(data.map((d) => ({ time: toTimestamp(d.time), value: d.total_bps })));
    handle.series.idio.setData(
      data.map((d) => ({ time: toTimestamp(d.time), value: d.idiosyncratic_bps })),
    );
    if (!hasFitRef.current) {
      handle.chart.timeScale().fitContent();
      hasFitRef.current = true;
    }
  }, [handle, data]);

  const lastTime = data.length ? toTimestamp(data[data.length - 1].time) : null;

  return (
    <ChartPanel
      title="Conditional volatility σₜ (bps / minute)"
      toolbar={<ChartToolbar chart={handle?.chart ?? null} lastTime={lastTime} />}
      containerRef={containerRef}
      height={420}
      legend={
        <>
          <span style={{ color: colors.vol }}>■</span> total √(β²σ²_f + σ²_ε) ·{" "}
          <span style={{ color: colors.volIdio }}>■</span> idiosyncratic σ_ε
        </>
      }
    />
  );
}
