import type { CSSProperties, ReactNode } from "react";
import type { Diagnostics, Metrics } from "../api";
import { cardStyle, colors, panelStyle } from "../theme";

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

const PARAM_HINTS: Record<string, string> = {
  delta: "weight on the high/low range (0 = range adds nothing)",
  persistence: "how long volatility shocks last (closer to 1 = longer)",
  gamma: "extra volatility after drops",
  phi: "AR(1): carry-over of last minute's move",
};

function Params({ title, params }: { title: string; params: Record<string, number> }) {
  return (
    <div style={{ ...cardStyle, fontSize: "0.8rem", color: colors.textBody }}>
      <div style={{ color: colors.textMuted, fontSize: "0.75rem", marginBottom: "0.35rem" }}>{title}</div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
          gap: "0.2rem 0.75rem",
        }}
      >
        {Object.entries(params).map(([k, v]) => (
          <div key={k} title={PARAM_HINTS[k]}>
            {k}: <strong>{v.toFixed(4)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Props {
  metrics: Metrics | null;
  diagnostics: Diagnostics | null;
}

/** Raw model numbers, collapsed by default: the plain-English cards cover day-to-day use. */
export function MetricsPanel({ metrics, diagnostics }: Props) {
  return (
    <details style={{ ...panelStyle }}>
      <summary style={{ color: colors.textBody, cursor: "pointer", fontSize: "0.9rem" }}>
        Model details (for the curious): factor CAPM and GARCH-X parameters
      </summary>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem", marginTop: "0.65rem" }}>

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
            <Stat
              label="Alpha (bps / min, not used in cone)"
              value={metrics.alpha_bps.toFixed(3)}
              color={colors.textMuted}
            />
            <Stat label="Residual SD (bps / min)" value={metrics.resid_std_bps.toFixed(2)} color={colors.warn} />
            <Stat
              label="PC1 variance explained"
              value={`${(metrics.pca_explained_variance_ratio * 100).toFixed(1)}%`}
            />
          </div>

          <Params title="PC1 loadings" params={metrics.pca_loadings} />
          <Params title="Residual AR(1)-GJR-GARCH-X(1,1), range-informed" params={metrics.resid_garch_params} />
          <Params title="Factor GJR-GARCH-X(1,1), range-informed" params={metrics.factor_garch_params} />

          <div style={{ color: colors.textMuted, fontSize: "0.75rem" }}>
            Fitted on {diagnostics.n_obs.toLocaleString()} one-minute returns (last{" "}
            {Math.round(diagnostics.fit_window_hours)}h). Bands come from filtered historical
            simulation (resampled real shocks).{" "}
            <b style={{ color: colors.textBody }}>
              Cone drawn by:{" "}
              {diagnostics.cone_model === "ensemble"
                ? "the ensemble (average of factor, GJR-only and EWMA)"
                : "the factor model"}
              .
            </b>{" "}
            {diagnostics.seasonality_active
              ? "Time-of-day adjustment: on."
              : "Time-of-day adjustment: off until 7 days of history are stored."}
          </div>
        </>
      )}
      </div>
    </details>
  );
}
