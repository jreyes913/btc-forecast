export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ForecastPoint {
  time: string;
  mean: number;
  p025: number;
  p16: number;
  p50: number;
  p84: number;
  p975: number;
}

export interface VolatilityPoint {
  time: string;
  total_bps: number;
  idiosyncratic_bps: number;
  typical_bps: number;
}

export interface Metrics {
  alpha_bps: number;
  beta: number;
  capm_r2: number;
  pca_explained_variance_ratio: number;
  pca_loadings: Record<string, number>;
  resid_std_bps: number;
  factor_variance_share: number;
  resid_garch_params: Record<string, number>;
  factor_garch_params: Record<string, number>;
}

export interface Diagnostics {
  n_obs: number;
  last_bar_time: string;
  last_close: number;
  resid_converged: boolean;
  factor_converged: boolean;
  seasonality_active: boolean;
  fit_window_hours: number;
  cone_model: "factor" | "ensemble";
}

export interface ForecastResponse {
  metrics: Metrics;
  diagnostics: Diagnostics;
  historical_volatility: VolatilityPoint[];
  forecast: ForecastPoint[];
}

export type BacktestSource = "all" | "live" | "replay";

export interface BacktestRow {
  model: string;
  horizon: number;
  n: number;
  cov68: number;
  cov95: number;
  qlike: number;
  mse: number;
  brier: number | null;
  n_strikes: number;
  dm_mean_diff: number | null;
  dm_p_value: number | null;
}

export interface CalibrationBin {
  p_mean: number;
  obs_freq: number;
  n: number;
}

export interface Calibration {
  model: string;
  horizon: number;
  bins: CalibrationBin[];
}

export interface BacktestResponse {
  computed_at: string;
  source: BacktestSource;
  replay_progress: { done: number; total: number };
  rows: BacktestRow[];
  calibration: Calibration[];
  recommended_fit_window: string | null;
}

export interface LiveBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface LiveResponse {
  bar: LiveBar | null;
  last_closed_time: string | null;
  cone_origin_time: string | null;
  cone_origin_price: number | null;
}

/** Thrown on HTTP 503: the backend is up but still loading data or fitting. */
export class NotReadyError extends Error {}

export async function getJson<T>(url: string, signal: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
  if (res.status === 503) {
    const body = await res.json().catch(() => ({}));
    throw new NotReadyError(body.detail ?? "Backend not ready");
  }
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json() as Promise<T>;
}
