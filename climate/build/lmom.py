"""Vectorised L-moments and GEV-by-L-moments (Hosking 1990; Hosking & Wallis 1997), numpy only.

Shape convention used everywhere in this project: ``xi`` = -k of Hosking (xi > 0 = heavy upper tail):

    F(x) = exp(-(1 + xi (x - loc) / scale)^(-1/xi)),    x_T = loc + scale / xi * ((-ln(1 - 1/T))^(-xi) - 1)

The main fits are cross-checked against ``lmoments3`` in ``build_eva.py``.
"""
from __future__ import annotations

from math import lgamma

import numpy as np

_vgamma = np.vectorize(lambda v: np.exp(lgamma(v)) if v > 0 else np.nan, otypes=[float])


def sample_lmom(x: np.ndarray, axis: int = 0):
    """Unbiased sample L-moments along ``axis``: (l1, l2, t3, t4)."""
    xs = np.sort(np.moveaxis(np.asarray(x, np.float64), axis, 0), axis=0)
    n = xs.shape[0]
    j = np.arange(n, dtype=np.float64).reshape((n,) + (1,) * (xs.ndim - 1))
    b0 = xs.mean(axis=0)
    b1 = (xs * j / (n - 1)).mean(axis=0)
    b2 = (xs * j * (j - 1) / ((n - 1) * (n - 2))).mean(axis=0)
    b3 = (xs * j * (j - 1) * (j - 2) / ((n - 1) * (n - 2) * (n - 3))).mean(axis=0)
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    with np.errstate(invalid="ignore", divide="ignore"):
        return l1, l2, l3 / l2, l4 / l2


def gev_k_from_t3(t3):
    """Hosking's k from L-skewness (approximation of Hosking et al. 1985, |error| < 9e-4 for -0.5 < t3 < 0.5),
    refined by Newton iterations on the exact relation t3 = 2 (1 - 3^-k) / (1 - 2^-k) - 3."""
    t3 = np.asarray(t3, np.float64)
    c = 2.0 / (3.0 + t3) - np.log(2.0) / np.log(3.0)
    k = 7.8590 * c + 2.9554 * c * c
    for _ in range(20):
        k = np.where(np.abs(k) < 1e-8, 1e-8, k)
        a, b = 1.0 - 3.0 ** (-k), 1.0 - 2.0 ** (-k)
        f = 2.0 * a / b - 3.0 - t3
        df = 2.0 * (np.log(3.0) * 3.0 ** (-k) * b - a * np.log(2.0) * 2.0 ** (-k)) / (b * b)
        k = k - f / df
    return k


def gev_t4_from_k(k):
    k = np.where(np.abs(np.asarray(k, np.float64)) < 1e-8, 1e-8, k)
    return (5.0 * (1 - 4.0 ** (-k)) - 10.0 * (1 - 3.0 ** (-k)) + 6.0 * (1 - 2.0 ** (-k))) / (1 - 2.0 ** (-k))


def gev_from_lmom(l1, l2, k):
    """(loc, scale, xi) from l1, l2 and Hosking's k (k may be at-site or regional)."""
    k = np.where(np.abs(np.asarray(k, np.float64)) < 1e-8, 1e-8, k)
    g = _vgamma(1.0 + k)
    scale = l2 * k / ((1.0 - 2.0 ** (-k)) * g)
    loc = l1 - scale * (1.0 - g) / k
    return loc, scale, -k


def gev_fit(x: np.ndarray, axis: int = 0):
    l1, l2, t3, _t4 = sample_lmom(x, axis)
    return gev_from_lmom(l1, l2, gev_k_from_t3(t3))


def gev_quantile(F, loc, scale, xi):
    F = np.asarray(F, np.float64)
    xi = np.where(np.abs(np.asarray(xi, np.float64)) < 1e-8, 1e-8, xi)
    return loc + scale / xi * ((-np.log(F)) ** (-xi) - 1.0)


def gev_cdf(x, loc, scale, xi):
    xi = np.where(np.abs(np.asarray(xi, np.float64)) < 1e-8, 1e-8, xi)
    z = 1.0 + xi * (np.asarray(x, np.float64) - loc) / scale
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        F = np.exp(-np.maximum(z, 0.0) ** (-1.0 / xi))
    return np.where(z <= 0, np.where(xi > 0, 0.0, 1.0), F)


def return_level(T, loc, scale, xi):
    return gev_quantile(1.0 - 1.0 / np.asarray(T, np.float64), loc, scale, xi)


def return_period(x, loc, scale, xi):
    F = gev_cdf(x, loc, scale, xi)
    with np.errstate(divide="ignore"):
        return 1.0 / np.maximum(1.0 - F, 1e-12)
