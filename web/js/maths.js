// The numerical functions needed to reproduce, in the browser, the probabilities of the snapshot
// (backend/riua/core/emos.py and risk.py). No dependencies.

// ---- gamma ---------------------------------------------------------------------------------

const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = LANCZOS[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularised lower incomplete gamma P(a, x) by its series (good for x < a + 1). */
function gammaSeries(a, x) {
  let term = 1 / a, sum = term;
  for (let n = 1; n < 5000; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * 1e-15) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lgamma(a));
}

/** Regularised upper incomplete gamma Q(a, x) by Lentz's continued fraction (good for x >= a + 1). */
function gammaFraction(a, x) {
  const tiny = 1e-300;
  let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
  for (let i = 1; i < 5000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.exp(-x + a * Math.log(x) - lgamma(a)) * h;
}

/** Q(a, x) = 1 - P(a, x): regularised upper incomplete gamma function (scipy.special.gammaincc). */
export function gammaQ(a, x) {
  if (!(a > 0) || Number.isNaN(x)) return NaN;
  if (x <= 0) return 1;
  if (x < a + 1) return Math.min(1, Math.max(0, 1 - gammaSeries(a, x)));
  return Math.min(1, Math.max(0, gammaFraction(a, x)));
}

export const gammaP = (a, x) => 1 - gammaQ(a, x);

/** x such that P(a, x) = p (scipy.special.gammaincinv), by bisection. */
export function gammaPinv(a, p) {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  let lo = 0, hi = Math.max(a, 1);
  while (gammaP(a, hi) < p && hi < 1e7) hi *= 2;
  for (let i = 0; i < 200 && hi - lo > 1e-12 * Math.max(1, hi); i++) {
    const mid = 0.5 * (lo + hi);
    if (gammaP(a, mid) < p) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

// ---- normal --------------------------------------------------------------------------------

/** Complementary error function, relative error below 1.2e-7 everywhere (Numerical Recipes erfcc), then refined. */
function erfc(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

// Phi through the incomplete gamma function: erf(x) = P(1/2, x^2), accurate to ~1e-14.
export function normCdf(z) {
  if (z === 0) return 0.5;
  const half = 0.5 * gammaQ(0.5, 0.5 * z * z);       // = P(Z > |z|)
  const v = z > 0 ? 1 - half : half;
  return Number.isFinite(v) ? v : 0.5 * erfc(-z / Math.SQRT2);
}

export const normPdf = (z) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

/** Inverse of Phi (Acklam's rational approximation + one Halley step). */
export function normInv(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  let x;
  if (p < 0.02425) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p > 1 - 0.02425) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else {
    const q = p - 0.5, r = q * q;
    x = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }
  const e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
  return x - u / (1 + 0.5 * x * u);
}

/**
 * Phi2(a, b; rho) = P(Z1 <= a, Z2 <= b) for a standard bivariate normal with correlation rho,
 * from the one-dimensional integral  ∫_{-inf}^{a} phi(x) · Phi((b - rho·x) / sqrt(1 - rho²)) dx
 * (Simpson's rule, 600 intervals on [-8, a]).
 */
export function phi2(a, b, rho) {
  if (a <= -8 || b <= -8) return 0;
  const s = Math.sqrt(1 - rho * rho);
  const lo = -8, hi = Math.min(a, 8), n = 600, h = (hi - lo) / n;
  const f = (x) => normPdf(x) * normCdf((b - rho * x) / s);
  let sum = f(lo) + f(hi);
  for (let i = 1; i < n; i++) sum += f(lo + i * h) * (i % 2 ? 4 : 2);
  return (sum * h) / 3;
}

// ---- the calibrated distribution (censored shifted gamma) --------------------------------------

/** Parameters of the predictive distribution from the ensemble mean m and 90th percentile q (emos.CSGD). */
export function csgd(cal, m, q) {
  const mm = Math.max(m || 0, 0), qq = Math.max(q || 0, mm);
  const mu = cal.a0 + cal.a1 * mm + cal.a2 * qq;
  const sigma = Math.max(cal.b0 * Math.sqrt(mu) + cal.b1 * (qq - mm), 1e-3);
  return { m: mm, q: qq, mu, sigma, k: (mu * mu) / (sigma * sigma), theta: (sigma * sigma) / mu, delta: cal.delta };
}

/** P(amount >= T) = Q(k, (T - delta) / theta). */
export function exceed(d, T) {
  return gammaQ(d.k, Math.max(T - d.delta, 0) / d.theta);
}

/** Amount not exceeded with probability p: max(0, Pinv(k, p)·theta + delta). */
export function quantile(d, p) {
  return Math.max(gammaPinv(d.k, p) * d.theta + d.delta, 0);
}

/** P(A or B) with a Gaussian copula of correlation rho (emos.UnionCopula.union, same clipping). */
export function union(pa, pb, rho) {
  const clip = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
  const r = clip(rho, -0.99, 0.99);
  const a = clip(pa, 1e-9, 1 - 1e-9), b = clip(pb, 1e-9, 1 - 1e-9);
  const za = clip(normInv(a), -6, 6), zb = clip(normInv(b), -6, 6);
  const both = clip(phi2(za, zb, r), Math.max(a + b - 1, 0), Math.min(a, b));
  return { za, zb, both, union: clip(a + b - both, Math.max(a, b), Math.min(a + b, 1)) };
}

/** Weighted mean and weighted 90th percentile as risk.weighted_stats: scenarios without a value are left out. */
export function weightedStats(values, weights, q = 0.9) {
  const rows = [];
  let wsum = 0, acc = 0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] == null || !(weights[i] > 0)) continue;
    rows.push([values[i], weights[i]]);
    wsum += weights[i]; acc += weights[i] * values[i];
  }
  if (!(wsum > 0)) return { mean: null, q: null, n: 0 };
  rows.sort((x, y) => x[0] - y[0]);
  let cw = 0, quant = rows[rows.length - 1][0];
  for (const [v, w] of rows) { cw += w / wsum; if (cw >= q - 1e-9) { quant = v; break; } }
  return { mean: acc / wsum, q: quant, n: rows.length };
}

// ---- scenario dressing (risk.py::dressed_probabilities) -----------------------------------------

/**
 * Probability that one scenario reaches each of the four levels.
 *   a1, a12   its largest 1-h and 12-h amounts in the neighbourhood, mm (null = not available)
 *   fam       {s1h, s12h} representativeness factors of its family
 *   o12       mm already measured inside the 12-h amount of the frame
 *   k         kernel of the horizon: {sigma, bias, sigma1h, bias1h, sigmaObs, obs12Factor}
 *   thr       {t1:[4], t12:[4]} thresholds in mm
 * The 12-h term: the measured share phi of the amount carries the bias of the analysis (obs12Factor) and
 * almost no spread; only the forecast share carries the bias and spread of the horizon.
 *   A12 = s12h·a12      phi = min(1, o12 / A12)
 *   b12 = phi·obs12Factor + (1 − phi)·bias      sg12 = max(sigma·(1 − phi), min(sigma, sigmaObs))
 *   p12 = Φ(ln(b12·A12 / T12) / sg12)           p1 = Φ(ln(bias1h·s1h·a1 / T1) / sigma1h)      p = max(p1, p12)
 */
export function dressScenario(a1, a12, fam, o12, k, thr) {
  const A1 = a1 == null ? 0 : fam.s1h * a1;
  const A12 = a12 == null ? 0 : fam.s12h * a12;
  const phi = A12 > 0 ? Math.min(1, o12 / A12) : 0;
  const b12 = phi * k.obs12Factor + (1 - phi) * k.bias;
  const sg12 = Math.max(k.sigma * (1 - phi), Math.min(k.sigma, k.sigmaObs));
  const p1 = [0, 1, 2, 3].map((L) => (A1 > 0 ? normCdf(Math.log((k.bias1h * A1) / thr.t1[L]) / k.sigma1h) : 0));
  const p12 = [0, 1, 2, 3].map((L) => (A12 > 0 ? normCdf(Math.log((b12 * A12) / thr.t12[L]) / sg12) : 0));
  return { A1, A12, phi, b12, sg12, p1, p12, p: p1.map((v, L) => Math.max(v, p12[L])) };
}

/**
 * The dressing of one cell and frame: rows = [{a1, a12, family, w}] (the scenarios with weight),
 * famOf(family) -> {s1h, s12h}, tau = {"2":..,"5":..}, cap = highest level the horizon may publish (or null).
 * -> { per:[dressScenario result per row], P:[4] (nested), W, level }
 */
export function dressCell(rows, famOf, o12, k, thr, tau, cap = null) {
  const sum = [0, 0, 0, 0];
  let W = 0;
  const per = rows.map((r) => {
    const d = dressScenario(r.a1, r.a12, famOf(r.family), o12, k, thr);
    if (r.a1 != null || r.a12 != null) { W += r.w; d.p.forEach((p, L) => { sum[L] += r.w * p; }); }
    return d;
  });
  const P = sum.map((p) => (W > 0 ? p / W : 0));
  for (let L = 1; L < 4; L++) P[L] = Math.min(P[L], P[L - 1]);          // np.minimum.accumulate over the levels
  let level = 1;
  [2, 3, 4, 5].forEach((L, i) => { if (P[i] >= Number(tau[String(L)])) level = L; });
  if (cap != null) level = Math.min(level, cap);
  return { per, P, W, level };
}

/**
 * The whole chain for one cell and frame. `c` = cellFrame() numbers, `cal` = frame.cal,
 * `thr` = {t1:[4], t12:[4]} thresholds in mm (yellow, orange, red, extreme), tau = {"2":..,"5":..}.
 */
export function auditCell(c, frame, thr, tau) {
  const cal = frame.cal || {};
  const use1 = !!(frame.has_1h && cal.c1);
  const calA = use1 ? cal.c1 : cal.c1_from12;                // distribution of the 1-h amount
  const dA = calA ? csgd(calA, use1 ? c.m1 : c.m12, use1 ? c.q1 : c.q12) : null;
  const dB = cal.c12 ? csgd(cal.c12, c.m12, c.q12) : null;   // distribution of the 12-h amount
  const rho = cal.rho ?? 0.7;
  const levels = [];
  let prev = 1;
  for (let k = 0; k < 4; k++) {
    const pa = dA ? exceed(dA, thr.t1[k]) : 0;
    const pb = dB ? exceed(dB, thr.t12[k]) : 0;
    const u = union(pa, pb, rho);
    const nested = Math.min(prev, u.union);              // np.minimum.accumulate over the levels
    prev = nested;
    const t = Number(tau[String(k + 2)]);
    levels.push({ L: k + 2, T1: thr.t1[k], T12: thr.t12[k], pa, pb, ...u, p: nested, tau: t, reached: nested >= t, published: c.p[k] });
  }
  let level = 1;
  for (const l of levels) if (l.reached) level = l.L;
  return {
    use1, calA, calAName: use1 ? 'c1' : 'c1_from12', dA, dB, rho, levels, level,
    quant: {
      e1: dA ? [quantile(dA, 0.5), quantile(dA, 0.9)] : [null, null],
      e12: dB ? [quantile(dB, 0.5), quantile(dB, 0.9)] : [null, null],
    },
  };
}
