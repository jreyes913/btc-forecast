import type { UTCTimestamp } from "lightweight-charts";

const TIME_ZONE = "America/Chicago";

const crosshairFormat = new Intl.DateTimeFormat("en-US", {
  timeZone: TIME_ZONE,
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: true,
});

const tickFormat = new Intl.DateTimeFormat("en-US", {
  timeZone: TIME_ZONE,
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});

export const formatCrosshairTime = (ts: number) => crosshairFormat.format(new Date(ts * 1000));
export const formatTick = (ts: number) => tickFormat.format(new Date(ts * 1000));

export const toTimestamp = (iso: string) =>
  Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
