import { useState, type CSSProperties } from "react";
import type { BacktestResponse, BacktestRow, Calibration } from "../api";
import { buttonStyle, colors, headingStyle, panelStyle } from "../theme";
import { CalibrationChart } from "./CalibrationChart";

function coverageVerdict(actual: number, nominal: number): { text: string; color: string } {
  const diff = actual - nominal;
  if (Math.abs(diff) <= 0.03) return { text: "on target", color: colors.up };
  if (diff > 0) return { text: Math.abs(diff) <= 0.07 ? "slightly cautious" : "too cautious", color: colors.warn };
  return { text: Math.abs(diff) <= 0.07 ? "slightly overconfident" : "overconfident", color: colors.down };
}

/** The calibration bin that covers the probabilities used for ~1-sigma strikes. */
function binNear(cal: Calibration | undefined, p: number) {
  if (!cal || !cal.bins.length) return null;
  return cal.bins.reduce((best, b) => (Math.abs(b.p_mean - p) < Math.abs(best.p_mean - p) ? b : best));
}

interface Props {
  backtest: BacktestResponse | null;
  status: string;
  model: string;
}

export function ReliabilityCard({ backtest, status, model }: Props) {
  const [horizon, setHorizon] = useState(30);
  const rows = (backtest?.rows ?? []).filter((r) => r.model === model);
  const horizons = rows.map((r) => r.horizon).filter((h) => h > 1).sort((a, b) => a - b);
  const row: BacktestRow | undefined = rows.find((r) => r.horizon === horizon);
  const cal = backtest?.calibration.find((c) => c.model === model && c.horizon === horizon);
  const near84 = binNear(cal, 0.84);
  const toggle = (active: boolean): CSSProperties => ({
    ...buttonStyle,
    padding: "0.2rem 0.6rem",
    ...(active ? { backgroundColor: colors.button, borderColor: colors.buttonBorder } : {}),
  });
  const p = (x: number) => `${Math.round(x * 100)}%`;

  return (
    <section style={{ ...panelStyle, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
        <h2 style={headingStyle}>How reliable are these probabilities?</h2>
        <div style={{ display: "flex", gap: "0.3rem" }}>
          {horizons.map((h) => (
            <button key={h} style={toggle(h === horizon)} onClick={() => setHorizon(h)}>
              {h} min out
            </button>
          ))}
        </div>
      </div>

      {!row ? (
        <div style={{ color: colors.textMuted, fontSize: "0.85rem" }}>
          {status || "Not enough scored forecasts yet. The first ones resolve within an hour."}
        </div>
      ) : (
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
          <ul
            style={{
              flex: "1 1 320px",
              margin: 0,
              paddingLeft: "1.1rem",
              color: colors.textBody,
              fontSize: "0.88rem",
              lineHeight: 1.6,
            }}
          >
            <li>
              Checked against <b style={{ color: colors.text }}>{row.n.toLocaleString()}</b> past
              forecasts made {horizon} minutes before the outcome.
            </li>
            {(() => {
              const v = coverageVerdict(row.cov68, 0.68);
              return (
                <li>
                  The <b>68% range</b> contained the final price{" "}
                  <b style={{ color: colors.text }}>{p(row.cov68)}</b> of the time (target 68%):{" "}
                  <span style={{ color: v.color }}>{v.text}</span>.
                </li>
              );
            })()}
            {(() => {
              const v = coverageVerdict(row.cov95, 0.95);
              return (
                <li>
                  The <b>95% range</b> contained it <b style={{ color: colors.text }}>{p(row.cov95)}</b> of
                  the time (target 95%): <span style={{ color: v.color }}>{v.text}</span>.
                </li>
              );
            })()}
            {near84 && (
              <li>
                When it gave about a <b>{p(near84.p_mean)}</b> chance of BTC finishing above a strike,
                it actually did <b style={{ color: colors.text }}>{p(near84.obs_freq)}</b> of the time (
                {near84.n.toLocaleString()} cases).{" "}
                {Math.abs(near84.obs_freq - near84.p_mean) <= 0.03
                  ? "Close enough to take the numbers at face value."
                  : near84.obs_freq > near84.p_mean
                    ? "The model has been a little too cautious here."
                    : "The model has been a little too confident here, so widen your margin."}
              </li>
            )}
            <li style={{ color: colors.textMuted }}>
              "Cautious" means the real odds have been slightly better than shown. That's the safe
              direction for your buy-at-or-below prices.
            </li>
          </ul>
          <div style={{ flex: "0 1 300px", minWidth: 0 }}>
            <CalibrationChart series={[{ label: "Forecast vs reality", color: colors.accent, data: cal }]} />
          </div>
        </div>
      )}
    </section>
  );
}
