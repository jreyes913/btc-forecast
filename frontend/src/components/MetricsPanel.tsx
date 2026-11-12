import type { CSSProperties, ReactNode } from "react";
import type { BacktestResponse, BacktestRow, Diagnostics, Metrics } from "../api";
import { cardStyle, colors, headingStyle, panelStyle } from "../theme";

const grid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
  gap: "0.65rem",
};

function Stat({ label, value, color = colors.text }: { label: string; value: ReactNode; color?: string }) {
  return (
    <div style={cardStyle}>
      <div style={{ color: colors.textMuted, fontSize: "0.75rem" }}>{label}</div>
      <div style={{ color, fontSize: "1.1rem", fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function Params({ title, params }: { title: string; params: Record<string, number> }) {
  return (
    <div style={{ ...cardStyle, fontSize: "0.8rem", color: colors.textBody }}>
      <div style={{ color: colors.textMuted, fontSize: "0.75rem", marginBottom: "0.35rem" }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: "0.2rem 0.75rem" }}>
        {Object.entries(params).map(([k, v]) => (
          <div key={k}>
            {k}: <strong>{v.toFixed(4)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

const MODEL_LABELS: Record<BacktestRow["model"], string> = {
  factor: "Factor model",
  gjr: "GJR-GARCH only",
  ewma: "EWMA",
};

function coverageColor(actual: number, nominal: number) {
  return Math.abs(actual - nominal) <= 0.07 ? colors.up : colors.warn;
}

function BacktestTable({ backtest }: { backtest: BacktestResponse }) {
  const rows = [...backtest.rows].sort((a, b) => a.horizon - b.horizon || a.model.localeCompare(b.model));
  const cell: CSSProperties = { padding: "0.25rem 0.5rem", textAlign: "right" };
  return (
    <div style={{ ...cardStyle, overflowX: "auto" }}>
      <div style={{ color: colors.textMuted, fontSize: "0.75rem", marginBottom: "0.35rem" }}>
        Rolling out-of-sample backtest (48h fit window). Coverage should be near 68% / 95%;
        lower QLIKE is better.
      </div>
      <table style={{ borderCollapse: "collapse", fontSize: "0.8rem", color: colors.textBody, width: "100%" }}>
        <thead>
          <tr style={{ color: colors.textMuted }}>
            <th style={{ ...cell, textAlign: "left" }}>Model</th>
            <th style={cell}>h (min)</th>
            <th style={cell}>n</th>
            <th style={cell}>68% cov.</th>
            <th style={cell}>95% cov.</th>
            <th style={cell}>QLIKE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.model}-${r.horizon}`}>
              <td style={{ ...cell, textAlign: "left" }}>{MODEL_LABELS[r.model]}</td>
              <td style={cell}>{r.horizon}</td>
              <td style={cell}>{r.n}</td>
              <td style={{ ...cell, color: coverageColor(r.cov68, 0.68) }}>{(r.cov68 * 100).toFixed(0)}%</td>
              <td style={{ ...cell, color: coverageColor(r.cov95, 0.95) }}>{(r.cov95 * 100).toFixed(0)}%</td>
              <td style={cell}>{r.qlike.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface Props {
  metrics: Metrics | null;
  diagnostics: Diagnostics | null;
  backtest: BacktestResponse | null;
  backtestStatus: string;
}

export function MetricsPanel({ metrics, diagnostics, backtest, backtestStatus }: Props) {
  return (
    <section style={{ ...panelStyle, display: "flex", flexDirection: "column", gap: "0.65rem" }}>
      <h2 style={headingStyle}>Factor CAPM &amp; GARCH diagnostics</h2>

      {!metrics || !diagnostics ? (
        <div style={{ color: colors.textMuted, fontSize: "0.9rem" }}>Waiting for model fit…</div>
      ) : (
        <>
          {(!diagnostics.resid_converged || !diagnostics.factor_converged) && (
            <div style={{ color: colors.warn, fontSize: "0.85rem" }}>
              ⚠ A GARCH fit did not converge on the latest data; treat the cone with caution.
            </div>
          )}

          <div style={grid}>
            <Stat label="Beta to PC1" value={metrics.beta.toFixed(3)} color={colors.up} />
            <Stat label="CAPM R²" value={`${(metrics.capm_r2 * 100).toFixed(1)}%`} />
            <Stat
              label="Factor share of 1h forecast variance"
              value={`${(metrics.factor_variance_share * 100).toFixed(0)}%`}
            />
            <Stat label="Alpha (bps / min)" value={metrics.alpha_bps.toFixed(3)} color={colors.accent} />
            <Stat label="Residual SD (bps / min)" value={metrics.resid_std_bps.toFixed(2)} color={colors.warn} />
            <Stat
              label="PC1 variance explained"
              value={`${(metrics.pca_explained_variance_ratio * 100).toFixed(1)}%`}
            />
          </div>

          <Params title="PC1 loadings" params={metrics.pca_loadings} />
          <Params title="Residual AR(1)-GJR-GARCH(1,1) skew-t" params={metrics.resid_garch_params} />
          <Params title="Factor GARCH(1,1) Student-t" params={metrics.factor_garch_params} />

          <div style={{ color: colors.textMuted, fontSize: "0.75rem" }}>
            Fitted on {diagnostics.n_obs.toLocaleString()} aligned one-minute returns.
          </div>
        </>
      )}

      {backtest ? (
        <BacktestTable backtest={backtest} />
      ) : (
        <div style={{ color: colors.textMuted, fontSize: "0.8rem" }}>
          Backtest: {backtestStatus || "loading…"}
        </div>
      )}
    </section>
  );
}
