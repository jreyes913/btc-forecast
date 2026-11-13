import type { ReactNode } from "react";
import type { ForecastResponse } from "../api";
import { cardStyle, colors, headingStyle, panelStyle } from "../theme";

function Card({ title, tone, children }: { title: string; tone?: string; children: ReactNode }) {
  return (
    <div style={{ ...cardStyle, borderLeft: `3px solid ${tone ?? colors.border}` }}>
      <div style={{ color: colors.textMuted, fontSize: "0.75rem", marginBottom: "0.2rem" }}>{title}</div>
      <div style={{ color: colors.textBody, fontSize: "0.88rem", lineHeight: 1.4 }}>{children}</div>
    </div>
  );
}

function volatilityCard(f: ForecastResponse) {
  const last = f.historical_volatility[f.historical_volatility.length - 1];
  if (!last || last.typical_bps <= 0) return null;
  const ratio = last.total_bps / last.typical_bps;
  const label = ratio > 1.25 ? "elevated" : ratio < 0.8 ? "calm" : "normal";
  const tone = label === "elevated" ? colors.warn : label === "calm" ? colors.up : colors.accent;
  const advice =
    label === "elevated"
      ? "Expect wider swings than usual. Bands are wider, and far strikes are more likely than they look."
      : label === "calm"
        ? "Expect smaller swings than usual. Bands are tighter."
        : "Swings are about what's typical for this time of day.";
  return (
    <Card title="Volatility right now" tone={tone}>
      <b style={{ color: colors.text }}>{ratio.toFixed(1)}× normal</b> for this time of day
      ({label}). {advice}
    </Card>
  );
}

export function Insights({ forecast }: { forecast: ForecastResponse | null }) {
  return (
    <section style={{ ...panelStyle, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
      <h2 style={headingStyle}>What the model is saying</h2>
      {!forecast ? (
        <div style={{ color: colors.textMuted, fontSize: "0.9rem" }}>Waiting for the first forecast…</div>
      ) : (
        <>
          {volatilityCard(forecast)}

          <Card title="Where the risk comes from" tone={colors.accent}>
            About <b style={{ color: colors.text }}>{Math.round(forecast.metrics.factor_variance_share * 100)}%</b>{" "}
            of BTC's next-hour risk is the whole crypto market moving together. The rest is
            BTC-specific. A high share means a big move in ETH or SOL will likely drag BTC with it.
          </Card>

          <Card title="Direction" tone={colors.textMuted}>
            <b style={{ color: colors.text }}>No directional call.</b> The model forecasts how far
            BTC is likely to move, not which way. The middle of the cone stays at today's price.
          </Card>

          {(() => {
            const d = forecast.diagnostics;
            const ok = d.resid_converged && d.factor_converged;
            return (
              <Card title="Model health" tone={ok ? colors.up : colors.warn}>
                {ok ? (
                  <>
                    <b style={{ color: colors.text }}>Fit OK</b> on the last {Math.round(d.fit_window_hours)}h
                    of data.
                  </>
                ) : (
                  <>
                    <b style={{ color: colors.warn }}>Fit warning:</b> a model didn't fully converge on the
                    latest data, so treat the probabilities with extra caution.
                  </>
                )}{" "}
                {d.seasonality_active
                  ? "Adjusting for time of day."
                  : "Time-of-day adjustment is off until 7 days of history are stored."}{" "}
                Probabilities come from{" "}
                {d.cone_model === "ensemble" ? "an average of 3 models" : "the factor model"}.
              </Card>
            );
          })()}
        </>
      )}
    </section>
  );
}
