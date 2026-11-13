import { useState, type CSSProperties } from "react";
import type { BacktestResponse, BacktestRow, BacktestSource } from "../api";
import { buttonStyle, cardStyle, colors } from "../theme";
import { CalibrationChart } from "./CalibrationChart";

const SOURCE_LABELS: Record<BacktestSource, string> = {
  all: "All",
  live: "Live",
  replay: "Replay",
};

function modelLabel(model: string): string {
  if (model === "factor") return "Factor model";
  if (model.startsWith("factor@")) return `Factor (${model.slice(7)} fit)`;
  if (model === "gjr") return "GJR-GARCH only";
  if (model === "ewma") return "EWMA";
  if (model === "ensemble") return "Ensemble (avg of 3)";
  return model;
}

function coverageColor(actual: number, nominal: number) {
  return Math.abs(actual - nominal) <= 0.05 ? colors.up : colors.warn;
}

function DmCell({ row }: { row: BacktestRow }) {
  if (row.model === "factor") return <>—</>;
  if (row.dm_p_value === null || row.dm_mean_diff === null) return <>n/a</>;
  const factorBetter = row.dm_mean_diff > 0;
  const significant = row.dm_p_value < 0.05;
  return (
    <span
      style={{ color: significant ? (factorBetter ? colors.up : colors.down) : colors.textBody }}
      title={
        factorBetter
          ? "Positive: the factor model's QLIKE loss is lower"
          : "Negative: this model's QLIKE loss is lower"
      }
    >
      {row.dm_p_value < 0.001 ? "<0.001" : row.dm_p_value.toFixed(3)} {factorBetter ? "▲" : "▼"}
    </span>
  );
}

interface Props {
  backtests: Partial<Record<BacktestSource, BacktestResponse>>;
  status: string;
}

export function BacktestPanel({ backtests, status }: Props) {
  const [source, setSource] = useState<BacktestSource>("all");
  const [horizon, setHorizon] = useState(30);
  const bt = backtests[source];
  const anyData = Object.values(backtests).some((b) => b && b.rows.length);

  const rows = [...(bt?.rows ?? [])].sort(
    (a, b) => a.horizon - b.horizon || a.model.localeCompare(b.model),
  );
  const horizons = [...new Set(rows.map((r) => r.horizon))];
  const cal = (model: string) => bt?.calibration.find((c) => c.model === model && c.horizon === horizon);
  const cell: CSSProperties = { padding: "0.25rem 0.5rem", textAlign: "right", whiteSpace: "nowrap" };
  const toggle = (active: boolean): CSSProperties => ({
    ...buttonStyle,
    padding: "0.2rem 0.6rem",
    ...(active ? { backgroundColor: colors.button, borderColor: colors.buttonBorder } : {}),
  });
  const progress = backtests.all?.replay_progress;

  return (
    <div style={{ ...cardStyle, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
        <div style={{ color: colors.text, fontWeight: 600 }}>Forecast scoring (accumulating)</div>
        <div style={{ display: "flex", gap: "0.3rem" }}>
          {(Object.keys(SOURCE_LABELS) as BacktestSource[]).map((s) => (
            <button key={s} style={toggle(s === source)} onClick={() => setSource(s)}>
              {SOURCE_LABELS[s]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ color: colors.textMuted, fontSize: "0.75rem" }}>
        Every forecast is scored once its deadline passes. <b>Replay</b> re-runs the models at
        past times using only data available then; <b>Live</b> is real time. Coverage should sit
        near 68% / 95%; lower QLIKE and Brier are better. DM p &lt; 0.05 means the difference vs
        the factor model is unlikely to be luck (▲ factor model better, ▼ worse).
        {progress && progress.total > 0 && (
          <>
            {" "}
            Replay progress: {progress.done.toLocaleString()} / {progress.total.toLocaleString()} past
            origins.
          </>
        )}
      </div>

      {!anyData ? (
        <div style={{ color: colors.textMuted, fontSize: "0.85rem" }}>
          {status || "No scored forecasts yet. The first ones resolve within an hour."}
        </div>
      ) : (
        <>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", fontSize: "0.8rem", color: colors.textBody, width: "100%" }}>
              <thead>
                <tr style={{ color: colors.textMuted }}>
                  <th style={{ ...cell, textAlign: "left" }}>Model</th>
                  <th style={cell}>h (min)</th>
                  <th style={cell}>n</th>
                  <th style={cell}>68% cov.</th>
                  <th style={cell}>95% cov.</th>
                  <th style={cell}>QLIKE</th>
                  <th style={cell}>Brier</th>
                  <th style={cell}>DM p</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={`${r.model}-${r.horizon}`} style={r.model === "factor" ? { color: colors.text } : undefined}>
                    <td style={{ ...cell, textAlign: "left" }}>{modelLabel(r.model)}</td>
                    <td style={cell}>{r.horizon}</td>
                    <td style={cell}>{r.n.toLocaleString()}</td>
                    <td style={{ ...cell, color: coverageColor(r.cov68, 0.68) }}>{(r.cov68 * 100).toFixed(0)}%</td>
                    <td style={{ ...cell, color: coverageColor(r.cov95, 0.95) }}>{(r.cov95 * 100).toFixed(0)}%</td>
                    <td style={cell}>{r.qlike.toFixed(3)}</td>
                    <td style={cell}>{r.brier === null ? "—" : r.brier.toFixed(4)}</td>
                    <td style={cell}>
                      <DmCell row={r} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {bt?.recommended_fit_window && (
            <div style={{ fontSize: "0.8rem", color: colors.textBody }}>
              Best fit window so far (lowest 30-minute QLIKE in replay):{" "}
              <b style={{ color: colors.accent }}>{bt.recommended_fit_window}</b>. Change{" "}
              <code>FIT_WINDOW</code> in <code>settings.py</code> to adopt it.
            </div>
          )}

          {horizons.length > 0 && (
            <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "flex-start" }}>
              <div style={{ flex: "0 1 300px", minWidth: 0 }}>
                <div style={{ display: "flex", gap: "0.3rem", marginBottom: "0.4rem" }}>
                  {horizons.map((h) => (
                    <button key={h} style={toggle(h === horizon)} onClick={() => setHorizon(h)}>
                      {h}m
                    </button>
                  ))}
                </div>
                <CalibrationChart
                  series={[
                    { label: "Factor model", color: colors.accent, data: cal("factor") },
                    { label: "GJR-GARCH only", color: colors.vol, hollow: true, data: cal("gjr") },
                    { label: "Ensemble", color: colors.up, hollow: true, data: cal("ensemble") },
                  ]}
                />
              </div>
              <div style={{ flex: "1 1 220px", fontSize: "0.75rem", color: colors.textMuted }}>
                Calibration at Kalshi-style $100 strikes. When the model says 84%, the event should
                happen about 84% of the time. Points <b>below</b> the diagonal mean the model is
                overconfident at that probability; points <b>above</b> mean it is too cautious.
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
