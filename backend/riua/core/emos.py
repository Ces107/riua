"""Statistical post-processing: censored, shifted gamma EMOS (Scheuerer and Hamill 2015).

Raw ensembles are not probabilities. They are biased (coarse models cannot rain 150 mm
in an hour), under-dispersive (lagged runs of one model resemble each other) and their
errors depend on the amount. So the predictive distribution of the OBSERVED amount Y is
modelled explicitly and fitted to past forecast/observation pairs:

    Y = max(0, G + delta),   G ~ Gamma(shape k, scale theta),   delta <= 0

The shift delta moves part of the gamma below zero; that mass is the probability of no
rain. The gamma's mean mu and standard deviation sigma depend on the ensemble:

    mu    = a0 + a1 * m + a2 * q90          m   = weighted ensemble mean
    sigma = b0 * sqrt(mu) + b1 * (q90 - m)  q90 = weighted 90th percentile (the tail)

with k = mu^2 / sigma^2 and theta = sigma^2 / mu. The coefficients minimise the mean
continuous ranked probability score (CRPS), for which the censored shifted gamma has a
closed form. Exceedance probabilities then follow analytically:

    P(Y >= T) = 1 - GammaCDF(T - delta; k, theta)

Two amounts are judged for every level (1 h and 12 h). Their union needs their
dependence: a Gaussian copula with correlation rho estimated on the training pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize, special, stats
from scipy.interpolate import RegularGridInterpolator


def _sp(x):
    return np.logaddexp(0.0, x)


def _sp_inv(y):
    y = np.maximum(y, 1e-6)
    return np.where(y > 30, y, np.log(np.expm1(y)))


@dataclass
class CSGD:
    a0: float; a1: float; a2: float; b0: float; b1: float; delta: float
    n: int = 0
    crps: float = float("nan")
    crps_ref: float = float("nan")     # CRPS of the raw ensemble-as-point-mass reference (mean absolute error of m)

    def moments(self, m, q90):
        m = np.maximum(np.asarray(m, np.float64), 0.0)
        q90 = np.maximum(np.asarray(q90, np.float64), m)
        mu = self.a0 + self.a1 * m + self.a2 * q90
        sigma = self.b0 * np.sqrt(mu) + self.b1 * (q90 - m)
        return mu, np.maximum(sigma, 1e-3)

    def shape_scale(self, m, q90):
        mu, sigma = self.moments(m, q90)
        return mu ** 2 / sigma ** 2, sigma ** 2 / mu

    def exceed(self, m, q90, thr):
        """P(Y >= thr). thr scalar or array broadcastable to m."""
        k, th = self.shape_scale(m, q90)
        return special.gammaincc(k, np.maximum(np.asarray(thr, np.float64) - self.delta, 0.0) / th)

    def quantile(self, m, q90, p):
        k, th = self.shape_scale(m, q90)
        return np.maximum(special.gammaincinv(k, p) * th + self.delta, 0.0)

    def to_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in asdict(self).items()}


def crps_csgd(y, k, theta, delta):
    """Closed-form CRPS of the censored shifted gamma (Scheuerer and Hamill 2015, eq. 5... A.3)."""
    y = np.asarray(y, np.float64)
    yt = (y - delta) / theta
    ct = -delta / theta
    Gy, Gc = special.gammainc(k, yt), special.gammainc(k, ct)
    Gy1, Gc1 = special.gammainc(k + 1, yt), special.gammainc(k + 1, ct)
    G2c = special.gammainc(2 * k, 2 * ct)
    # B(1/2, k + 1/2) / pi, computed in logs for large k
    lb = special.betaln(0.5, k + 0.5) - np.log(np.pi)
    return (theta * yt * (2 * Gy - 1) - theta * ct * Gc ** 2
            + theta * k * (1 + 2 * Gc * Gc1 - Gc ** 2 - 2 * Gy1)
            - theta * k * np.exp(lb) * (1 - G2c))


def fit_csgd(m, q90, y, weights=None, x0=None, maxiter: int = 300) -> CSGD:
    """Fit the regression by minimum (weighted) mean CRPS."""
    m, q90, y = (np.asarray(a, np.float64).ravel() for a in (m, q90, y))
    ok = np.isfinite(m) & np.isfinite(q90) & np.isfinite(y)
    m, q90, y = m[ok], np.maximum(q90[ok], m[ok]), y[ok]
    w = np.ones_like(y) if weights is None else np.asarray(weights, np.float64).ravel()[ok]
    w = w / w.sum()

    def unpack(v):
        return CSGD(_sp(v[0]) + 1e-3, _sp(v[1]), _sp(v[2]), _sp(v[3]) + 1e-3, _sp(v[4]), -_sp(v[5]))

    def loss(v):
        c = unpack(v)
        k, th = c.shape_scale(m, q90)
        val = crps_csgd(y, k, th, c.delta)
        return float(np.sum(w * val))

    if x0 is None:
        x0 = _sp_inv(np.array([0.3, 0.7, 0.2, 0.8, 0.5, 0.5]))
    best = None
    for start in (x0, _sp_inv(np.array([0.1, 1.0, 0.05, 1.5, 0.2, 0.1])), _sp_inv(np.array([1.0, 0.4, 0.5, 0.5, 1.0, 1.5]))):
        res = optimize.minimize(loss, start, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res
    c = unpack(best.x)
    c.n = int(y.size)
    c.crps = float(best.fun)
    c.crps_ref = float(np.sum(w * np.abs(m - y)))
    return c


# ---------------------------------------------------------------------------------------
# union of two exceedances with a Gaussian copula


class UnionCopula:
    """P(A or B) from the two marginal probabilities and a Gaussian-copula correlation."""

    def __init__(self, rho: float):
        self.rho = float(np.clip(rho, -0.99, 0.99))
        z = np.linspace(-6.0, 6.0, 97)
        zz_a, zz_b = np.meshgrid(z, z, indexing="ij")
        # P(Za > a, Zb > b) = Phi2(-a, -b; rho); tabulated once on a grid
        mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, self.rho], [self.rho, 1.0]])
        tab = mvn.cdf(np.column_stack([zz_a.ravel(), zz_b.ravel()])).reshape(zz_a.shape)
        self._interp = RegularGridInterpolator((z, z), tab, bounds_error=False, fill_value=None)

    def union(self, pa, pb):
        pa = np.clip(np.asarray(pa, np.float64), 1e-9, 1 - 1e-9)
        pb = np.clip(np.asarray(pb, np.float64), 1e-9, 1 - 1e-9)
        za, zb = stats.norm.ppf(pa), stats.norm.ppf(pb)        # P(A) = Phi(za)
        both = self._interp(np.stack([np.clip(za, -6, 6), np.clip(zb, -6, 6)], axis=-1))
        both = np.clip(both, np.maximum(pa + pb - 1, 0), np.minimum(pa, pb))   # Frechet bounds
        return np.clip(pa + pb - both, np.maximum(pa, pb), np.minimum(pa + pb, 1.0))


def copula_rho(c1: CSGD, m1, q1, y1, c12: CSGD, m12, q12, y12) -> float:
    """Correlation of the normal scores of the two observed amounts given their forecasts.

    Uses pairs where at least one of the two is wet; censored (zero) observations get the
    mid-point of their probability mass (randomised PIT replaced by its expectation).
    """
    def pit(c, m, q, y):
        k, th = c.shape_scale(m, q)
        u = special.gammainc(k, (np.asarray(y, np.float64) - c.delta) / th)
        u0 = special.gammainc(k, -c.delta / th)
        return np.where(np.asarray(y) <= 0, 0.5 * u0, u)
    y1, y12 = np.asarray(y1).ravel(), np.asarray(y12).ravel()
    sel = (y1 > 0) | (y12 > 0)
    if sel.sum() < 50:
        return 0.7
    u1 = np.clip(pit(c1, np.asarray(m1).ravel()[sel], np.asarray(q1).ravel()[sel], y1[sel]), 1e-4, 1 - 1e-4)
    u2 = np.clip(pit(c12, np.asarray(m12).ravel()[sel], np.asarray(q12).ravel()[sel], y12[sel]), 1e-4, 1 - 1e-4)
    return float(np.corrcoef(stats.norm.ppf(u1), stats.norm.ppf(u2))[0, 1])
