from datetime import datetime

from pydantic import BaseModel


class CandleSchema(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ForecastPoint(BaseModel):
    time: datetime
    mean: float
    p025: float
    p16: float
    p50: float
    p84: float
    p975: float


class VolatilityPoint(BaseModel):
    time: datetime
    total_bps: float
    idiosyncratic_bps: float
    typical_bps: float


class Metrics(BaseModel):
    alpha_bps: float
    beta: float
    capm_r2: float
    pca_explained_variance_ratio: float
    pca_loadings: dict[str, float]
    resid_std_bps: float
    factor_variance_share: float
    resid_garch_params: dict[str, float]
    factor_garch_params: dict[str, float]


class Diagnostics(BaseModel):
    n_obs: int
    last_bar_time: datetime
    last_close: float
    resid_converged: bool
    factor_converged: bool
    seasonality_active: bool
    fit_window_hours: float
    cone_model: str


class ForecastResponse(BaseModel):
    metrics: Metrics
    diagnostics: Diagnostics
    historical_volatility: list[VolatilityPoint]
    forecast: list[ForecastPoint]


class LiveBar(BaseModel):
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class LiveResponse(BaseModel):
    bar: LiveBar | None
    last_closed_time: datetime | None
    cone_origin_time: datetime | None
    cone_origin_price: float | None


class LadderRow(BaseModel):
    strike: float
    p_above: float
    p_below: float
    beyond_grid: str | None  # "low" / "high" when outside the model's 1-99% range
    max_yes_cents: float | None
    max_no_cents: float | None
    bands: list[str]
    is_current: bool


class LadderRanges(BaseModel):
    p68: list[float]
    p95: list[float]
    median: float


class LadderResponse(BaseModel):
    available: bool
    deadline: datetime
    which: str
    minutes_left: int
    reference_price: float
    reference_is_live: bool
    cone_origin_time: datetime
    cone_model: str
    edge_points: float
    horizon_minutes: int | None = None
    ranges: LadderRanges | None
    rows: list[LadderRow]
