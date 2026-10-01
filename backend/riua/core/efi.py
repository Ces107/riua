"""Extremeness of an ensemble forecast relative to climate, in the style of ECMWF's
Extreme Forecast Index and Shift of Tails.

EFI (Lalaurette 2003, Anderson-Darling weighting as revised by Zsoter 2006):

    EFI = (2 / pi) * integral_0^1 (p - F_f(q_c(p))) / sqrt(p (1 - p)) dp

where q_c(p) is the climate quantile and F_f the forecast (ensemble) distribution.
EFI = +1 when every member is above the climate maximum; values above ~0.8 are read as
"very unusual".

SOT (Zsoter 2006), for the upper tail:

    SOT = (Q_f(90) - Q_c(99)) / (Q_c(99) - Q_c(90))

SOT > 0 means at least 10 % of the members exceed the 99th percentile of climate;
SOT of 1 or more means they exceed it by more than the distance between the climate's
own 90th and 99th percentiles: an event outside ordinary experience.

Difference from ECMWF's product: the reference here is a reanalysis climate (ERA5 daily
precipitation, same grid spacing as the ensemble, +-15 day window, 30 years), not the
model's own reforecast climate, because reforecasts are not openly distributed. The
ensemble and ERA5 share the IFS model family, which limits but does not remove the bias.
"""
from __future__ import annotations

import numpy as np

CLIM_P = np.concatenate([np.arange(1, 100) / 100.0, [0.995, 0.999]])  # quantile levels stored in the climatology


def efi(members: np.ndarray, clim_q: np.ndarray, clim_p: np.ndarray = CLIM_P) -> np.ndarray:
    """members (M, ...), clim_q (K, ...) climate quantiles at levels clim_p. Returns EFI (...)."""
    m = np.sort(np.asarray(members, np.float64), axis=0)
    M = m.shape[0]
    p = np.asarray(clim_p, np.float64)
    use = (p > 0) & (p < 1)
    integrand = []
    for k in np.nonzero(use)[0]:
        # F_f(q_c(p)): fraction of members <= climate quantile
        ff = (m <= clim_q[k][None]).sum(axis=0) / M
        integrand.append((p[k] - ff) / np.sqrt(p[k] * (1 - p[k])))
    integrand = np.stack(integrand)
    pp = p[use]
    val = np.trapezoid(integrand, pp, axis=0) if hasattr(np, "trapezoid") else np.trapz(integrand, pp, axis=0)
    return np.clip(2.0 / np.pi * val, -1.0, 1.0)


def sot(members: np.ndarray, clim_q: np.ndarray, clim_p: np.ndarray = CLIM_P, q: float = 0.9) -> np.ndarray:
    qf = np.quantile(np.asarray(members, np.float64), q, axis=0)
    k99 = int(np.argmin(np.abs(clim_p - 0.99)))
    k90 = int(np.argmin(np.abs(clim_p - 0.90)))
    c99, c90 = clim_q[k99], clim_q[k90]
    return (qf - c99) / np.maximum(c99 - c90, 0.5)   # 0.5 mm floor: dry-season climates have c99 ~ c90
