from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class FitResult:
    """Result from a fit.  Access fitted parameters via .params, raw scipy outputs via .popt/.pcov."""

    model: str
    params: Dict[str, float]
    popt: np.ndarray
    pcov: np.ndarray
    x: np.ndarray
    y: np.ndarray
    y_fit: np.ndarray

    @property
    def perr(self) -> np.ndarray:
        return np.sqrt(np.diag(self.pcov))

    def __repr__(self) -> str:
        lines = [f"FitResult({self.model}):"]
        for k, v in self.params.items():
            if k.endswith("_err"):
                continue
            err = self.params.get(k + "_err")
            try:
                v_str = f"{float(v):.6g}"
                err_str = f" ± {float(err):.6g}" if err is not None else ""
            except (TypeError, ValueError):
                v_str, err_str = str(v), ""
            lines.append(f"  {k:<20s} = {v_str}{err_str}")
        return "\n".join(lines)


class Fitter:
    """Pure-scipy fitting for NV experiment data.

    Usage::

        fitter = Fitter()

        # auto-guess initial parameters
        res = fitter.lorentzian(x, y)
        res = fitter.double_lorentzian(x, y)
        res = fitter.decaying_cosine(x, y)

        # provide your own p0 (same positional order as scipy curve_fit)
        res = fitter.lorentzian(x, y, p0=[x0, gamma, amp, baseline])

        # any scipy curve_fit keyword (bounds, sigma, maxfev, …) is forwarded
        res = fitter.lorentzian(x, y, bounds=(lower, upper))

    The model functions are also available as static methods for standalone use::

        y_model = Fitter.lorentzian_func(x, x0, gamma, amp, baseline)
    """

    # ------------------------------------------------------------------ #
    # Model functions                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def lorentzian_func(x: np.ndarray, x0: float, gamma: float, amp: float, baseline: float) -> np.ndarray:
        """y = baseline + amp * gamma² / ((x − x0)² + gamma²)"""
        return baseline + amp * gamma**2 / ((x - x0) ** 2 + gamma**2)

    @staticmethod
    def double_lorentzian_func(
        x: np.ndarray,
        x01: float, gamma1: float, amp1: float,
        x02: float, gamma2: float, amp2: float,
        baseline: float,
    ) -> np.ndarray:
        """y = baseline + L(x; x01, gamma1, amp1) + L(x; x02, gamma2, amp2)"""
        L1 = amp1 * gamma1**2 / ((x - x01) ** 2 + gamma1**2)
        L2 = amp2 * gamma2**2 / ((x - x02) ** 2 + gamma2**2)
        return baseline + L1 + L2

    @staticmethod
    def decaying_cosine_func(
        x: np.ndarray, amp: float, freq: float, phase: float, T: float, offset: float
    ) -> np.ndarray:
        """y = amp * cos(2π freq x + phase) * exp(−x / T) + offset"""
        return amp * np.cos(2 * np.pi * freq * x + phase) * np.exp(-x / T) + offset

    # ------------------------------------------------------------------ #
    # Auto-guess helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _edge_baseline(y: np.ndarray) -> float:
        n = len(y)
        edge = max(1, n // 10)
        return float(np.nanmedian(np.concatenate([y[:edge], y[-edge:]])))

    @staticmethod
    def _dominant_peak(x: np.ndarray, yc: np.ndarray) -> tuple[int, float, float]:
        """Return (idx, x0, hwhm) for the dominant peak or dip in baseline-subtracted yc."""
        i_max = int(np.nanargmax(yc))
        i_min = int(np.nanargmin(yc))
        idx = i_max if abs(yc[i_max]) >= abs(yc[i_min]) else i_min
        amp = float(yc[idx])
        half = amp / 2.0
        # points on same side of half-max as the peak itself
        above = np.where(np.sign(yc - half) == np.sign(amp))[0]
        if len(above) >= 2:
            hwhm = max(float(x[above[-1]] - x[above[0]]) / 2.0, 0.0)
        else:
            hwhm = float(np.nanmax(x) - np.nanmin(x)) / 10.0
        dx = abs(float(np.diff(x).mean())) if len(x) > 1 else 1.0
        return idx, float(x[idx]), max(hwhm, dx)

    def _guess_lorentzian(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        baseline = self._edge_baseline(y)
        yc = y - baseline
        idx, x0, gamma = self._dominant_peak(x, yc)
        return [x0, gamma, float(yc[idx]), baseline]

    def _guess_double_lorentzian(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        n = len(x)
        baseline = self._edge_baseline(y)
        yc = y - baseline

        idx1, x01, _ = self._dominant_peak(x, yc)

        excl = max(3, n // 20)
        yc2 = yc.copy().astype(float)
        yc2[max(0, idx1 - excl): min(n, idx1 + excl + 1)] = np.nan

        if np.all(~np.isfinite(yc2)):
            idx2, x02 = idx1, x01
        else:
            idx2, x02, _ = self._dominant_peak(x, yc2)

        if x01 > x02:
            x01, x02 = x02, x01
            idx1, idx2 = idx2, idx1

        span = float(np.nanmax(x) - np.nanmin(x))
        dx = abs(float(np.diff(x).mean())) if n > 1 else 1.0
        gamma0 = max(span / 20.0, dx)

        amp1 = float(yc[idx1]) or float(yc[idx2]) * 0.5
        amp2 = float(yc[idx2]) or amp1

        return [x01, gamma0, amp1, x02, gamma0, amp2, baseline]

    def _guess_decaying_cosine(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        y_mean = float(np.nanmean(y))
        y_amp = float((np.nanmax(y) - np.nanmin(y)) / 2.0)
        x_range = float(np.nanmax(x) - np.nanmin(x))
        min_idx = int(np.nanargmin(y))
        dx_min = float(x[min_idx]) - float(x[0])
        freq0 = (0.5 / dx_min) if dx_min > 0 else (1.0 / (2.0 * x_range) if x_range > 0 else 1.0)
        T0 = max(x_range * 5.0, 1e-12)
        return [y_amp, freq0, 0.0, T0, y_mean]

    # ------------------------------------------------------------------ #
    # Public fit methods                                                   #
    # ------------------------------------------------------------------ #

    def lorentzian(self, x, y, p0=None, **kwargs) -> FitResult:
        """Fit a single Lorentzian peak or dip.

        Model::

            y = baseline + amp * gamma² / ((x − x0)² + gamma²)

        p0 order: ``[x0, gamma, amp, baseline]``

        Any keyword accepted by :func:`scipy.optimize.curve_fit` (e.g. ``bounds``,
        ``sigma``, ``maxfev``) may be forwarded via ``**kwargs``.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if p0 is None:
            p0 = self._guess_lorentzian(x, y)

        fit_kw: dict = {"maxfev": 20000}
        fit_kw.update(kwargs)
        if "bounds" not in fit_kw:
            x_span = float(np.nanmax(x) - np.nanmin(x))
            y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
            dx = abs(float(np.diff(x).mean())) if len(x) > 1 else x_span / 10.0
            fit_kw["bounds"] = (
                [float(np.nanmin(x)) - x_span,  dx / 10.0,   -10 * y_span, float(np.nanmin(y)) - 2 * y_span],
                [float(np.nanmax(x)) + x_span,  x_span * 10, 10 * y_span,  float(np.nanmax(y)) + 2 * y_span],
            )

        popt, pcov = curve_fit(self.lorentzian_func, x, y, p0=p0, **fit_kw)
        perr = np.sqrt(np.diag(pcov))
        x0, gamma, amp, baseline = [float(v) for v in popt]
        params = {
            "x0":       x0,           "x0_err":       perr[0],
            "gamma":    abs(gamma),   "gamma_err":    perr[1],
            "fwhm":     2*abs(gamma), "fwhm_err":     2*perr[1],
            "amp":      amp,          "amp_err":      perr[2],
            "baseline": baseline,     "baseline_err": perr[3],
        }
        return FitResult("lorentzian", params, popt, pcov, x, y, self.lorentzian_func(x, *popt))

    def double_lorentzian(self, x, y, p0=None, **kwargs) -> FitResult:
        """Fit a sum of two Lorentzian peaks/dips with a shared baseline.

        Model::

            y = baseline + L(x; x01, gamma1, amp1) + L(x; x02, gamma2, amp2)
            L(x; x0, g, a) = a * g² / ((x − x0)² + g²)

        p0 order: ``[x01, gamma1, amp1, x02, gamma2, amp2, baseline]``

        Results are always ordered so that ``x01 < x02``.  The ``split`` key
        in :attr:`FitResult.params` gives ``|x02 − x01|``.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if p0 is None:
            p0 = self._guess_double_lorentzian(x, y)

        fit_kw: dict = {"maxfev": 20000}
        fit_kw.update(kwargs)
        if "bounds" not in fit_kw:
            x_span = float(np.nanmax(x) - np.nanmin(x))
            y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
            dx = abs(float(np.diff(x).mean())) if len(x) > 1 else x_span / 10.0
            amp_max = 10 * y_span
            lo = [float(np.nanmin(x)) - x_span, dx / 10.0, -amp_max,
                  float(np.nanmin(x)) - x_span, dx / 10.0, -amp_max,
                  float(np.nanmin(y)) - 2 * y_span]
            hi = [float(np.nanmax(x)) + x_span, x_span * 10, amp_max,
                  float(np.nanmax(x)) + x_span, x_span * 10, amp_max,
                  float(np.nanmax(y)) + 2 * y_span]
            fit_kw["bounds"] = (lo, hi)

        popt, pcov = curve_fit(self.double_lorentzian_func, x, y, p0=p0, **fit_kw)

        # Guarantee x01 < x02 by permuting popt/pcov if needed
        if popt[0] > popt[3]:
            idx = np.array([3, 4, 5, 0, 1, 2, 6])
            popt = popt[idx]
            pcov = pcov[np.ix_(idx, idx)]

        perr = np.sqrt(np.diag(pcov))
        x01, gamma1, amp1, x02, gamma2, amp2, baseline = [float(v) for v in popt]
        params = {
            "x01":      x01,            "x01_err":      perr[0],
            "gamma1":   abs(gamma1),    "gamma1_err":   perr[1],
            "fwhm1":    2*abs(gamma1),  "fwhm1_err":    2*perr[1],
            "amp1":     amp1,           "amp1_err":     perr[2],
            "x02":      x02,            "x02_err":      perr[3],
            "gamma2":   abs(gamma2),    "gamma2_err":   perr[4],
            "fwhm2":    2*abs(gamma2),  "fwhm2_err":    2*perr[4],
            "amp2":     amp2,           "amp2_err":     perr[5],
            "baseline": baseline,       "baseline_err": perr[6],
            "split":    abs(x02 - x01),
        }
        return FitResult("double_lorentzian", params, popt, pcov, x, y, self.double_lorentzian_func(x, *popt))

    def decaying_cosine(self, x, y, p0=None, **kwargs) -> FitResult:
        """Fit a decaying cosine oscillation (Power Rabi model).

        Model::

            y = amp * cos(2π freq x + phase) * exp(−x / T) + offset

        p0 order: ``[amp, freq, phase, T, offset]``

        Note: ``x`` should start near 0 (or be shifted by the caller) so the
        decay ``exp(−x/T)`` is well-defined.  ``T`` is the 1/e decay length in
        the same units as ``x``.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if p0 is None:
            p0 = self._guess_decaying_cosine(x, y)

        fit_kw: dict = {"maxfev": 20000}
        fit_kw.update(kwargs)
        if "bounds" not in fit_kw:
            x_range = float(np.nanmax(x) - np.nanmin(x))
            y_span = max(float(np.nanmax(y) - np.nanmin(y)), 1e-12)
            amp_max = 10 * y_span
            T_min = max(x_range / 100.0, 1e-12)
            T_max = max(x_range * 1e4, T_min * 10.0)
            off_range = 2 * y_span
            fit_kw["bounds"] = (
                [-amp_max, 0.0,    -4 * np.pi, T_min, float(np.nanmin(y)) - off_range],
                [amp_max,  np.inf,  4 * np.pi, T_max, float(np.nanmax(y)) + off_range],
            )

        popt, pcov = curve_fit(self.decaying_cosine_func, x, y, p0=p0, **fit_kw)
        perr = np.sqrt(np.diag(pcov))
        amp, freq, phase, T, offset = [float(v) for v in popt]
        params = {
            "amp":    amp,    "amp_err":    perr[0],
            "freq":   freq,   "freq_err":   perr[1],
            "phase":  phase,  "phase_err":  perr[2],
            "T":      T,      "T_err":      perr[3],
            "offset": offset, "offset_err": perr[4],
        }
        return FitResult("decaying_cosine", params, popt, pcov, x, y, self.decaying_cosine_func(x, *popt))
