import dataclasses as dc
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dc.dataclass
class QuantModelResult:
    alpha: float
    beta: float
    pca_explained_variance_ratio: float
    pca_loadings: dict[str, float]
    residual_std_error: float
    garch_params: dict[str, float]
    forecast_mean_returns: list[float]
    forecast_volatility: list[float]
    residuals_series: pd.Series
    historical_volatility: list[dict[str, Any]]


class BtcFactorGarchModel:
    def __init__(self, scale_factor: float = 1000.0):
        # 1000.0 brings 1-minute return variance into [0.1, 10.0] for clean GARCH MLE convergence
        self.scale_factor = scale_factor
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1)
        self.capm_fit = None
        self.garch_fit = None

    def _prepare_aligned_returns(
        self,
        btc_df: pd.DataFrame,
        eth_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        time_col: str = "time",
        price_col: str = "close",
    ) -> pd.DataFrame:
        btc = btc_df[[time_col, price_col]].rename(columns={price_col: "btc"}).set_index(time_col)
        eth = eth_df[[time_col, price_col]].rename(columns={price_col: "eth"}).set_index(time_col)
        sp = sp500_df[[time_col, price_col]].rename(columns={price_col: "sp500"}).set_index(time_col)

        merged = btc.join([eth, sp], how="left").sort_index()
        merged["sp500"] = merged["sp500"].ffill()
        merged = merged.dropna()

        returns = np.log(merged / merged.shift(1)).dropna() * self.scale_factor
        # Replace any infinities from zero prices if they ever occur
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        return returns

    def fit_and_forecast(
        self,
        btc_df: pd.DataFrame,
        eth_df: pd.DataFrame,
        sp500_df: pd.DataFrame,
        horizon: int = 30,
    ) -> QuantModelResult:
        returns = self._prepare_aligned_returns(btc_df, eth_df, sp500_df)

        # --- Stage 1: PCA Market Factor & CAPM Regression ---
        factor_inputs = returns[["eth", "sp500"]].values
        factor_scaled = self.scaler.fit_transform(factor_inputs)
        pc1 = self.pca.fit_transform(factor_scaled).ravel()

        if np.sum(self.pca.components_) < 0:
            pc1 = -pc1
            loadings = -self.pca.components_[0]
        else:
            loadings = self.pca.components_[0]

        X = sm.add_constant(pc1)
        y = returns["btc"].values
        self.capm_fit = sm.OLS(y, X).fit()

        # Convert alpha back to percentage points (%) for display consistency
        alpha_pct = float(self.capm_fit.params[0]) * (100.0 / self.scale_factor)
        beta = float(self.capm_fit.params[1])

        residuals = pd.Series(self.capm_fit.resid, index=returns.index, name="capm_resid")
        residual_std_error = float(np.sqrt(self.capm_fit.mse_resid)) / self.scale_factor

        # --- Stage 2: AR(1) + GJR-GARCH(1,1) with Skew-t Innovations ---
        garch_spec = arch_model(
            residuals,
            mean="AR",
            lags=1,
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="skewt",
            rescale=False,
        )
        self.garch_fit = garch_spec.fit(disp="off", options={"maxiter": 500})

        # Use 'bootstrap' for multi-step non-linear GJR-GARCH to prevent extreme skew-t simulation blowups
        forecasts = self.garch_fit.forecast(
            horizon=horizon,
            method="bootstrap" if horizon > 1 else "analytic",
            simulations=500,
            reindex=False,
        )

        # Unscale back to raw log returns and guard against NaN/Inf
        raw_mean = np.nan_to_num(forecasts.mean.iloc[-1].values / self.scale_factor, nan=0.0, posinf=0.0, neginf=0.0)
        raw_var = np.nan_to_num(forecasts.variance.iloc[-1].values, nan=0.0, posinf=1.0, neginf=0.0)
        raw_vol = np.sqrt(np.maximum(raw_var, 0.0)) / self.scale_factor

        # Extract historical conditional volatility (sigma_t)
        cond_vol = (self.garch_fit.conditional_volatility / self.scale_factor).dropna()
        hist_vol_records = [
            {"time": ts.isoformat(), "volatility": float(val)}
            for ts, val in cond_vol.items()
            if np.isfinite(val)
        ]

        return QuantModelResult(
            alpha=alpha_pct,
            beta=beta,
            pca_explained_variance_ratio=float(self.pca.explained_variance_ratio_[0]),
            pca_loadings={"eth": float(loadings[0]), "sp500": float(loadings[1])},
            residual_std_error=residual_std_error,
            garch_params={k: float(v) for k, v in self.garch_fit.params.items() if np.isfinite(v)},
            forecast_mean_returns=raw_mean.tolist(),
            forecast_volatility=raw_vol.tolist(),
            residuals_series=residuals / self.scale_factor,
            historical_volatility=hist_vol_records,
        )

    def project_price_path(
        self,
        last_close: float,
        result: QuantModelResult,
        expected_pc1_return: float = 0.0,
    ) -> dict[str, list[float]]:
        unscaled_alpha = result.alpha / 100.0
        unscaled_beta = result.beta

        total_log_returns = [
            unscaled_alpha + (unscaled_beta * expected_pc1_return) + ar_resid
            for ar_resid in result.forecast_mean_returns
        ]

        # Clamp cumulative log returns to [-0.5, 0.5] (~ +/- 65% move) to guarantee finite exp()
        cum_returns = np.clip(np.cumsum(total_log_returns), -0.5, 0.5)
        cum_vol = np.clip(np.sqrt(np.cumsum(np.square(result.forecast_volatility))), 0.0, 0.5)

        expected_prices = (last_close * np.exp(cum_returns)).tolist()
        upper_band = (last_close * np.exp(cum_returns + cum_vol)).tolist()
        lower_band = (last_close * np.exp(cum_returns - cum_vol)).tolist()

        return {
            "expected_price": expected_prices,
            "upper_1sd": upper_band,
            "lower_1sd": lower_band,
        }
