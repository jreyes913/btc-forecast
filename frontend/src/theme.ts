import type { CSSProperties } from "react";

export const colors = {
  page: "#020617",
  panel: "#0f172a",
  card: "#1e293b",
  border: "#334155",
  grid: "#1e293b",
  text: "#f8fafc",
  textMuted: "#94a3b8",
  textBody: "#cbd5e1",
  up: "#22c55e",
  down: "#ef4444",
  accent: "#38bdf8",
  warn: "#facc15",
  vol: "#f59e0b",
  volIdio: "#a78bfa",
  button: "#0284c7",
  buttonBorder: "#0369a1",
};

export const panelStyle: CSSProperties = {
  backgroundColor: colors.panel,
  borderRadius: "10px",
  padding: "1rem",
  border: `1px solid ${colors.grid}`,
  minWidth: 0,
};

export const cardStyle: CSSProperties = {
  backgroundColor: colors.card,
  borderRadius: "8px",
  padding: "0.75rem 1rem",
  border: `1px solid ${colors.border}`,
};

export const buttonStyle: CSSProperties = {
  backgroundColor: colors.card,
  color: "#e2e8f0",
  border: `1px solid ${colors.border}`,
  borderRadius: "6px",
  padding: "0.35rem 0.75rem",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontWeight: 500,
};

export const headingStyle: CSSProperties = { margin: 0, color: colors.text, fontSize: "1rem" };
