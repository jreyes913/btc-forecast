import type { IChartApi, UTCTimestamp } from "lightweight-charts";
import { buttonStyle, colors } from "../theme";

interface Props {
  chart: IChartApi | null;
  /** Time of the last *observed* bar; presets show the N minutes leading up to it. */
  lastTime: UTCTimestamp | null;
  /** Right edge of the view (e.g. the end of the forecast). Defaults to `lastTime`. */
  rightEdge?: UTCTimestamp | null;
}

const PRESETS: [string, number][] = [
  ["1H", 60],
  ["6H", 360],
  ["24H", 1440],
];

export function ChartToolbar({ chart, lastTime, rightEdge }: Props) {
  // Time-based (not bar-count-based) ranges, so missing minutes don't skew the presets.
  const showLast = (minutes: number) => {
    if (!chart || lastTime === null) return;
    const to = (rightEdge ?? lastTime) as number;
    chart.timeScale().setVisibleRange({
      from: (lastTime - minutes * 60) as UTCTimestamp,
      to: (to + Math.ceil(minutes * 0.05) * 60) as UTCTimestamp,
    });
    chart.priceScale("right").applyOptions({ autoScale: true });
  };

  const rehome = () => {
    if (!chart) return;
    chart.timeScale().fitContent();
    chart.priceScale("right").applyOptions({ autoScale: true });
  };

  const zoom = (factor: number) => {
    const ts = chart?.timeScale();
    const range = ts?.getVisibleLogicalRange();
    if (!ts || !range) return;
    const center = (range.from + range.to) / 2;
    const half = Math.max(10, (range.to - range.from) * factor) / 2;
    ts.setVisibleLogicalRange({ from: center - half, to: center + half });
  };

  return (
    <div style={{ display: "flex", gap: "0.4rem", alignItems: "center", flexWrap: "wrap" }}>
      {PRESETS.map(([label, minutes]) => (
        <button key={label} style={buttonStyle} onClick={() => showLast(minutes)}>
          {label}
        </button>
      ))}
      <span style={{ color: "#475569", margin: "0 0.25rem" }}>|</span>
      <button style={buttonStyle} onClick={() => zoom(0.7)} title="Zoom in">+</button>
      <button style={buttonStyle} onClick={() => zoom(1.4)} title="Zoom out">−</button>
      <button
        style={{ ...buttonStyle, backgroundColor: colors.button, borderColor: colors.buttonBorder }}
        onClick={rehome}
        title="Show everything"
      >
        All
      </button>
    </div>
  );
}
