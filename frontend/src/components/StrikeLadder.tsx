import type { CSSProperties } from "react";
import type { LadderResponse, LadderRow } from "../api";
import { buttonStyle, colors, headingStyle, panelStyle } from "../theme";

const etTime = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
});

const usd = (v: number) => `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

function pct(row: LadderRow, side: "above" | "below"): string {
  const p = side === "above" ? row.p_above : row.p_below;
  // Outside the model's 1-99% range the exact figure isn't meaningful.
  if (row.beyond_grid === "high") return side === "above" ? "<1%" : ">99%";
  if (row.beyond_grid === "low") return side === "above" ? ">99%" : "<1%";
  return `${(p * 100).toFixed(p > 0.02 && p < 0.98 ? 0 : 1)}%`;
}

/** Beyond the model's 1-99% range the true chance is more extreme, so the cap is a floor. */
const cents = (c: number | null, beyond: boolean) =>
  c === null ? "—" : `${Math.floor(c)}¢${beyond ? "+" : ""}`;

const BAND_STYLE: Record<string, { bg: string; fg: string }> = {
  "95% low": { bg: "rgba(56, 189, 248, 0.10)", fg: colors.accent },
  "68% low": { bg: "rgba(56, 189, 248, 0.18)", fg: colors.accent },
  median: { bg: "rgba(148, 163, 184, 0.18)", fg: colors.text },
  "68% high": { bg: "rgba(56, 189, 248, 0.18)", fg: colors.accent },
  "95% high": { bg: "rgba(56, 189, 248, 0.10)", fg: colors.accent },
};

interface Props {
  ladder: LadderResponse | null;
  status: string;
  which: "next" | "following";
  onWhich: (w: "next" | "following") => void;
}

export function StrikeLadder({ ladder, status, which, onWhich }: Props) {
  const toggle = (active: boolean): CSSProperties => ({
    ...buttonStyle,
    padding: "0.2rem 0.6rem",
    ...(active ? { backgroundColor: colors.button, borderColor: colors.buttonBorder } : {}),
  });
  const cell: CSSProperties = { padding: "0.3rem 0.6rem", textAlign: "right", whiteSpace: "nowrap" };

  return (
    <section style={{ ...panelStyle, display: "flex", flexDirection: "column", gap: "0.6rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "0.5rem" }}>
        <h2 style={headingStyle}>Strike probabilities</h2>
        <div style={{ display: "flex", gap: "0.3rem" }}>
          <button style={toggle(which === "next")} onClick={() => onWhich("next")}>
            Next hour
          </button>
          <button style={toggle(which === "following")} onClick={() => onWhich("following")}>
            Hour after
          </button>
        </div>
      </div>

      {!ladder ? (
        <div style={{ color: colors.warn, fontSize: "0.9rem" }}>{status || "Loading…"}</div>
      ) : (
        <>
          <div style={{ color: colors.textBody, fontSize: "0.9rem" }}>
            Deadline <b style={{ color: colors.text }}>{etTime.format(new Date(ladder.deadline))} ET</b>
            {" · "}
            {ladder.minutes_left} min left{" · "}
            from {ladder.reference_is_live ? "live price" : "last closed bar"}{" "}
            <b style={{ color: colors.text }}>{usd(ladder.reference_price)}</b>
          </div>

          {!ladder.available ? (
            <div style={{ color: colors.warn, fontSize: "0.85rem" }}>
              That deadline is further out than the forecast covers right now. It becomes
              available once the live price updates.
            </div>
          ) : (
            <>
              {ladder.ranges && (
                <div style={{ color: colors.textBody, fontSize: "0.9rem" }}>
                  <b style={{ color: colors.accent }}>68%</b> chance BTC finishes between{" "}
                  <b style={{ color: colors.text }}>
                    {usd(ladder.ranges.p68[0])} and {usd(ladder.ranges.p68[1])}
                  </b>
                  {" · "}
                  <b style={{ color: colors.accent }}>95%</b>:{" "}
                  {usd(ladder.ranges.p95[0])} – {usd(ladder.ranges.p95[1])}
                </div>
              )}

              <div style={{ overflowX: "auto" }}>
                <table
                  style={{ borderCollapse: "collapse", fontSize: "0.85rem", color: colors.textBody, width: "100%" }}
                >
                  <thead>
                    <tr style={{ color: colors.textMuted, fontSize: "0.75rem" }}>
                      <th style={{ ...cell, textAlign: "left" }}>Strike</th>
                      <th style={cell}>Chance ≥ strike</th>
                      <th style={cell}>Chance below</th>
                      <th style={cell} title={`Model chance minus ${ladder.edge_points} points`}>
                        Buy Yes at ≤
                      </th>
                      <th style={cell} title={`Model chance minus ${ladder.edge_points} points`}>
                        Buy No at ≤
                      </th>
                      <th style={{ ...cell, textAlign: "left" }}>Band</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...ladder.rows].reverse().map((r) => {
                      const band = r.bands[0];
                      const style = band ? BAND_STYLE[band] : undefined;
                      return (
                        <tr
                          key={r.strike}
                          style={{
                            backgroundColor: style?.bg,
                            borderTop: `1px solid ${colors.grid}`,
                            outline: r.is_current ? `1px solid ${colors.warn}` : undefined,
                          }}
                        >
                          <td style={{ ...cell, textAlign: "left", color: colors.text, fontWeight: 600 }}>
                            {usd(r.strike)}
                            {r.is_current && (
                              <span style={{ color: colors.warn, fontWeight: 400 }}> ◂ now</span>
                            )}
                          </td>
                          <td style={cell}>{pct(r, "above")}</td>
                          <td style={cell}>{pct(r, "below")}</td>
                          <td style={{ ...cell, color: r.max_yes_cents === null ? colors.textMuted : colors.up }}>
                            {cents(r.max_yes_cents, r.beyond_grid === "low")}
                          </td>
                          <td style={{ ...cell, color: r.max_no_cents === null ? colors.textMuted : colors.down }}>
                            {cents(r.max_no_cents, r.beyond_grid === "high")}
                          </td>
                          <td style={{ ...cell, textAlign: "left", color: style?.fg ?? colors.textMuted }}>
                            {r.bands.join(" · ")}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div style={{ color: colors.textMuted, fontSize: "0.75rem" }}>
                <b>Buy at ≤</b> is the most a contract is worth to you: the model's chance minus{" "}
                {ladder.edge_points} points, your safety margin. Only buy if Kalshi's price is at or
                below it, after fees. "—" means no price leaves that margin. Shaded rows mark the
                cone's band lines, from the 95% low through the median to the 95% high.
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
