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
    resid_converged: bool
    factor_converged: bool


class ForecastResponse(BaseModel):
    metrics: Metrics
    diagnostics: Diagnostics
    historical_volatility: list[VolatilityPoint]
    forecast: list[ForecastPoint]
