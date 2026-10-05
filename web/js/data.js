// Loading and decoding of snapshot.json and explain-<hz>.bin (contract: docs/SNAPSHOT.md).
// Every packed array is decoded from base64 once; frames are read through index arithmetic or
// subarray() views, never copied.

import { DATA_DIR, HORIZONS } from './config.js';

export function b64(s) {
  if (!s) return new Uint8Array(0);
  if (typeof Uint8Array.fromBase64 === 'function') return Uint8Array.fromBase64(s);
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export const mm = (v) => (v / 8) * (v / 8);          // packed amount -> mm
export const mmOrNull = (v) => (v === 255 ? null : mm(v));
export const prob = (v) => v / 200;                   // packed probability -> 0..1

/** Turns the raw JSON into the object the page works with. Throws a plain Error if it is unusable. */
export function prepare(raw, origin) {
  if (!raw || raw.v !== 1 || !raw.grid || !raw.horizons) throw new Error('formato de datos desconocido');
  const g = raw.grid;
  const total = g.nx * g.ny;
  const bits = b64(raw.mask);
  const cellToN = new Int32Array(total).fill(-1);    // flat grid index c -> position in the packed arrays
  const nToCell = new Int32Array(raw.n_cells);
  let n = 0;
  for (let c = 0; c < total; c++) {
    if (bits[c >> 3] & (0x80 >> (c & 7))) { cellToN[c] = n; nToCell[n] = c; n++; }
  }
  if (n !== raw.n_cells) throw new Error('la máscara de celdas no cuadra con n_cells');
  const N = n;

  const hz = {};
  for (const key of HORIZONS) {
    const h = raw.horizons[key];
    if (!h || !h.frames || !h.cells) continue;
    const F = h.frames.length;
    const level = b64(h.cells.level);
    if (level.length !== F * N) continue;
    const frames = h.frames.map((f) => ({ ...f, d0: new Date(f.t0), d1: new Date(f.t1) }));
    // worst level of the period per cell, and the frame where it happens
    const p = b64(h.cells.p);
    const maxLevel = new Uint8Array(N);
    const maxFrame = new Uint8Array(N);
    let top = 0;
    for (let i = 0; i < N; i++) {
      let best = 0, bf = 0, bp = -1;
      for (let f = 0; f < F; f++) {
        const L = level[f * N + i];
        const pk = p.length ? p[Math.max(L - 2, 0) * F * N + f * N + i] : 0;
        if (L > best || (L === best && L > 0 && pk > bp)) { best = L; bf = f; bp = pk; }
      }
      maxLevel[i] = best; maxFrame[i] = bf;
      if (best > top) top = best;
    }
    const sigma = Number(h.sigma), bias = Number(h.bias ?? 1);
    hz[key] = {
      key, F, N, frames, tau: h.tau, sigma: h.sigma, bias: h.bias ?? 1, fam: h.fam || null, level, p, top, maxLevel, maxFrame,
      // kernel of the scenario dressing. Fields a snapshot older than the one that introduced them does not
      // carry fall back to the values that reproduce the old formula (one bias and sigma, measured rain at 1).
      kernel: {
        sigma, bias, sigma1h: Number(h.sigma1h ?? sigma), bias1h: Number(h.bias1h ?? bias),
        sigmaObs: Number(h.sigma_obs ?? 0.15), obs12Factor: Number(h.obs12_factor ?? 1),
      },
      levelCap: h.level_cap ?? null,                    // the published level is min(level, cap)
      o12: h.cells.o12 ? b64(h.cells.o12) : new Uint8Array(0),   // mm measured inside the 12-h amount of each frame
      e1: b64(h.cells.e1), e12: b64(h.cells.e12), acc: h.cells.acc ? b64(h.cells.acc) : new Uint8Array(0), accTotal: h.cells.acc_total ? b64(h.cells.acc_total) : new Uint8Array(0),
      m1: b64(h.cells.m1), q1: b64(h.cells.q1), m12: b64(h.cells.m12), q12: b64(h.cells.q12),
      members: h.members || [], basins: h.basins || null, points: h.points || null,
      scale: h.scale || 'cell', zones: h.zones || null,   // days 2-7: level and P per warning zone and day
      explain: (raw.explain || {})[key] || null,
    };
    h.cells = null;                                    // free the base64 strings
  }
  const obs = raw.obs ? { hours: raw.obs.hours, last: raw.obs.last_hour, o1: b64(raw.obs.o1), o12: b64(raw.obs.o12), o24: b64(raw.obs.o24) } : null;
  return {
    origin, generated: new Date(raw.generated), generatedIso: raw.generated, paramsVersion: raw.params_version,
    grid: g, N, cellToN, nToCell, thresholds: raw.thresholds || { zones: {}, extreme: { x1h: 1.5, x12h: 1.67 } },
    hz, obs, gauges: raw.gauges || [], rivers: raw.rivers || [], warnings: raw.warnings || [],
    drivers: raw.drivers ?? null, reservoirs: raw.reservoirs || null, sources: raw.sources || [], notes: raw.notes || [],
  };
}

/** n of the cell that contains (lat, lon), or -1. */
export function cellAt(snap, lat, lon) {
  const g = snap.grid;
  const i = Math.floor((lon - g.lon0) / g.d), j = Math.floor((lat - g.lat0) / g.d);
  if (i < 0 || j < 0 || i >= g.nx || j >= g.ny) return -1;
  return snap.cellToN[j * g.nx + i];
}

export function cellCentre(snap, n) {
  const g = snap.grid, c = snap.nToCell[n];
  const j = Math.floor(c / g.nx), i = c % g.nx;
  return { j, i, c, lat: g.lat0 + (j + 0.5) * g.d, lon: g.lon0 + (i + 0.5) * g.d };
}

/** Everything the snapshot says about cell n in frame f of horizon h (plain numbers). */
export function cellFrame(h, f, n) {
  const FN = h.F * h.N, o = f * h.N + n;
  const two = (a) => (a.length === 2 * FN ? [mm(a[o]), mm(a[FN + o])] : [null, null]);
  return {
    level: h.level[o],
    p: [0, 1, 2, 3].map((k) => (h.p.length === 4 * FN ? prob(h.p[k * FN + o]) : null)),
    e1: two(h.e1), e12: two(h.e12), acc: two(h.acc),
    o12: h.o12.length === FN ? mm(h.o12[o]) : 0,
    m1: h.m1.length === FN ? mmOrNull(h.m1[o]) : null, q1: h.q1.length === FN ? mmOrNull(h.q1[o]) : null,
    m12: h.m12.length === FN ? mmOrNull(h.m12[o]) : null, q12: h.q12.length === FN ? mmOrNull(h.q12[o]) : null,
  };
}

/**
 * JSON.parse that also accepts the bare NaN / Infinity tokens Python's json module writes
 * (they are not JSON; an untuned calibration carries "crps": NaN). They become null.
 */
export function parseJson(text) {
  try { return JSON.parse(text); } catch (e) {
    return JSON.parse(text.replace(/([:[,])\s*(?:NaN|-?Infinity)(?=\s*[,\]}])/g, '$1null'));
  }
}

async function getJson(url, ms) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), ms);
  try {
    const r = await fetch(url, { cache: 'no-cache', signal: ctl.signal });
    if (!r.ok) throw new Error(`${r.status}`);
    return parseJson(await r.text());
  } finally { clearTimeout(timer); }
}

/**
 * The snapshot published with the page. `onSnap(snap)` is called if it is newer than `since` (ms);
 * `onFail(error)` if it cannot be downloaded or is unusable (the caller keeps whatever it already shows).
 * The first call uses the download that index.html starts before any module has loaded (window.riuaSnap).
 */
export async function loadSnapshots(onSnap, onFail, since = 0) {
  try {
    let raw = null;
    const early = window.riuaSnap;
    window.riuaSnap = null;
    if (early && !since) { const got = await early; if (got && got.text) raw = parseJson(got.text); }
    if (!raw) raw = await getJson(`${DATA_DIR}snapshot.json`, 30000);
    const t = Date.parse(raw && raw.generated);
    if (!Number.isFinite(t)) throw new Error('la predicción no trae fecha');
    if (t > since) onSnap(prepare(raw, 'static'));
    return true;
  } catch (e) {
    onFail(e);
    return false;
  }
}

/** First and last frame of the unbroken run of frames around f in which cell n keeps the level it has at f. */
export function levelRun(h, n, f) {
  const L = h.level[f * h.N + n];
  let a = f, b = f;
  while (a > 0 && h.level[(a - 1) * h.N + n] === L) a--;
  while (b < h.F - 1 && h.level[(b + 1) * h.N + n] === L) b++;
  return [a, b];
}

const explainCache = new Map();

/**
 * Layout of explain-<hz>.bin from its head (snapshot.explain[hz]).
 *   new:  a1[M1,F,N] (only the scenarios with has1[k], in order), a12[M,F,N], p1[4,F,N], p12[4,F,N]
 *   old (no M1 in the head): a1[M,F,N], a12[M,F,N], p1[4,F,N], p12[4,F,N]
 * -> { M, M1, slot: Int32Array[M] (position of scenario k in a1, or -1 = no 1-h amount), bytes }
 */
export function explainLayout(head) {
  const { M, F, N } = head;
  const slot = new Int32Array(M);
  let M1 = 0;
  if (head.M1 != null && Array.isArray(head.has1)) for (let k = 0; k < M; k++) slot[k] = head.has1[k] ? M1++ : -1;
  else for (let k = 0; k < M; k++) slot[k] = M1++;
  return { M, M1, slot, bytes: (M1 + M + 8) * F * N };
}

/** Lazy audit binary of one horizon -> {a1, a12, p1, p12, M, M1, slot} as views on one buffer. */
export function loadExplain(snap, key) {
  const id = `${snap.origin}|${snap.generatedIso}|${key}`;
  if (!explainCache.has(id)) {
    explainCache.set(id, (async () => {
      const h = snap.hz[key];
      const head = h.explain;
      if (!head || !head.M) throw new Error('esta predicción no trae el detalle por escenario');
      const url = `${DATA_DIR}explain-${key}.bin`;
      const r = await fetch(url, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`no se pudo descargar (${r.status})`);
      const buf = new Uint8Array(await r.arrayBuffer());
      const { M, F, N } = head;
      const lay = explainLayout(head);
      if (F !== h.F || N !== h.N || buf.length !== lay.bytes || (head.M1 != null && lay.M1 !== head.M1)) {
        throw new Error('el fichero de detalle no corresponde a esta predicción (se actualizó mientras tanto)');
      }
      const FN = F * N, a = lay.M1 * FN, b = a + M * FN;
      return {
        M, M1: lay.M1, F, N, slot: lay.slot,
        a1: buf.subarray(0, a), a12: buf.subarray(a, b),
        p1: buf.subarray(b, b + 4 * FN), p12: buf.subarray(b + 4 * FN, b + 8 * FN),
      };
    })());
    explainCache.get(id).catch(() => explainCache.delete(id));
  }
  return explainCache.get(id);
}
