import type { Calibration } from "../api";
import { colors } from "../theme";

interface Props {
  series: { label: string; color: string; hollow?: boolean; data: Calibration | undefined }[];
}

const SIZE = 220;
const PAD_L = 44;
const PAD = 28;

/**
 * Reliability diagram: forecast probability (x) vs how often it happened (y). Points
 * on the diagonal are perfectly calibrated; point area scales with the sample count.
 */
export function CalibrationChart({ series }: Props) {
  const inner = SIZE - PAD - PAD_L;
  const x = (p: number) => PAD_L + p * inner;
  const y = (p: number) => SIZE - PAD - p * (SIZE - 2 * PAD);
  const maxN = Math.max(1, ...series.flatMap((s) => s.data?.bins.map((b) => b.n) ?? []));
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div>
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        style={{ width: "100%", maxWidth: 300, display: "block" }}
        role="img"
        aria-label="Calibration chart: forecast probability versus observed frequency"
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} y1={y(0)} x2={x(t)} y2={y(1)} stroke={colors.grid} />
            <line x1={x(0)} y1={y(t)} x2={x(1)} y2={y(t)} stroke={colors.grid} />
            <text x={x(t)} y={SIZE - PAD + 14} fill={colors.textMuted} fontSize="9" textAnchor="middle">
              {Math.round(t * 100)}%
            </text>
            <text x={PAD_L - 5} y={y(t) + 3} fill={colors.textMuted} fontSize="9" textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={colors.textMuted} strokeDasharray="4 3" />
        {series.map((s) => (
          <g key={s.label}>
            <polyline
              points={(s.data?.bins ?? []).map((b) => `${x(b.p_mean)},${y(b.obs_freq)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeOpacity={0.6}
            />
            {(s.data?.bins ?? []).map((b, i) => (
              <circle
                key={i}
                cx={x(b.p_mean)}
                cy={y(b.obs_freq)}
                r={2 + 4 * Math.sqrt(b.n / maxN)}
                fill={s.hollow ? "none" : s.color}
                stroke={s.color}
              >
                <title>
                  {s.label}: forecast {(b.p_mean * 100).toFixed(0)}% → happened{" "}
                  {(b.obs_freq * 100).toFixed(0)}% (n={b.n})
                </title>
              </circle>
            ))}
          </g>
        ))}
        <text x={SIZE / 2} y={SIZE - 4} fill={colors.textMuted} fontSize="9" textAnchor="middle">
          Forecast P(BTC ≥ strike)
        </text>
        <text
          x={10}
          y={SIZE / 2}
          fill={colors.textMuted}
          fontSize="9"
          textAnchor="middle"
          transform={`rotate(-90 10 ${SIZE / 2})`}
        >
          Observed frequency
        </text>
      </svg>
      <div style={{ display: "flex", gap: "0.75rem", fontSize: "0.75rem", color: colors.textMuted }}>
        {series.map((s) => (
          <span key={s.label}>
            <span style={{ color: s.color }}>{s.hollow ? "○" : "●"}</span> {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
