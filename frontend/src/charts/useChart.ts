import { useEffect, useRef, useState } from "react";
import { ColorType, createChart, type IChartApi } from "lightweight-charts";
import { colors } from "../theme";
import { formatCrosshairTime, formatTick } from "./format";

export interface ChartHandle<S> {
  chart: IChartApi;
  series: S;
}

/**
 * Creates a themed chart in `containerRef` and calls `setup` once to add its series.
 * `autoSize` keeps the chart sized to its container, so no resize listeners are needed.
 */
export function useChart<S>(setup: (chart: IChartApi) => S) {
  const containerRef = useRef<HTMLDivElement>(null);
  const setupRef = useRef(setup);
  const [handle, setHandle] = useState<ChartHandle<S> | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: colors.panel },
        textColor: colors.textBody,
      },
      grid: { vertLines: { color: colors.grid }, horzLines: { color: colors.grid } },
      localization: { timeFormatter: formatCrosshairTime },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
        minBarSpacing: 0.05,
        tickMarkFormatter: formatTick,
      },
    });
    setHandle({ chart, series: setupRef.current(chart) });

    return () => {
      setHandle(null);
      chart.remove();
    };
  }, []);

  return { containerRef, handle };
}
