// Loading and decoding of snapshot.json and explain-<hz>.bin (contract: docs/SNAPSHOT.md).
// Every packed array is decoded from base64 once; frames are read through index arithmetic or
// subarray() views, never copied.

import { API_BASE, DATA_DIR, DEV, HORIZONS } from './config.js';

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
    hz[key] = {
      key, F, N, frames, tau: h.tau, sigma: h.sigma, bias: h.bias ?? 1, fam: h.fam || null, level, p, top, maxLevel, maxFrame,
      e1: b64(h.cells.e1), e12: b64(h.cells.e12),
      m1: b64(h.cells.m1), q1: b64(h.cells.q1), m12: b64(h.cells.m12), q12: b64(h.cells.q12),
      members: h.members || [], basins: h.basins || null, points: h.points || null,
      explain: (raw.explain || {})[key] || null,
    };
    h.cells = null;                                    // free the base64 strings
  }
  const obs = raw.obs ? { hours: raw.obs.hours, last: raw.obs.last_hour, o1: b64(raw.obs.o1), o12: b64(raw.obs.o12), o24: b64(raw.obs.o24) } : null;
  return {
    origin, generated: new Date(raw.generated), generatedIso: raw.generated, paramsVersion: raw.params_version,
    grid: g, N, cellToN, nToCell, thresholds: raw.thresholds || { zones: {}, extreme: { x1h: 1.5, x12h: 1.67 } },
    hz, obs, gauges: raw.gauges || [], rivers: raw.rivers || [], warnings: raw.warnings || [],
    drivers: raw.drivers ?? null, sources: raw.sources || [], notes: raw.notes || [],
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
    e1: two(h.e1), e12: two(h.e12),
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
 * First the static copy that ships with the page, then (silently) the live API; `onSnap` is
 * called for each one that is newer than what is already shown. Returns when both attempts ended.
 */
export async function loadSnapshots(onSnap, onFail, since = 0) {
  let shown = since;
  let got = false;
  const offer = (raw, origin) => {
    const t = Date.parse(raw.generated);
    got = true;
    if (!(t > shown)) return;
    const snap = prepare(raw, origin);
    shown = t;
    onSnap(snap);
  };
  const first = getJson(`${DATA_DIR}snapshot.json`, 20000).then((raw) => offer(raw, 'static'))
    .catch((e) => { if (!shown && !got) onFail(e); });
  const live = DEV ? Promise.resolve() : getJson(`${API_BASE}/v1/snapshot`, 70000).then((raw) => offer(raw, 'api')).catch(() => {});
  await first;
  await live;
  return shown > 0;
}

const explainCache = new Map();

/** Lazy audit binary of one horizon -> {a1, a12, p1, p12, M} as views on one buffer. */
export function loadExplain(snap, key) {
  const id = `${snap.origin}|${snap.generatedIso}|${key}`;
  if (!explainCache.has(id)) {
    explainCache.set(id, (async () => {
      const h = snap.hz[key];
      const head = h.explain;
      if (!head || !head.M) throw new Error('esta predicción no trae el detalle por escenario');
      const url = snap.origin === 'api' ? `${API_BASE}/v1/explain/${key}` : `${DATA_DIR}explain-${key}.bin`;
      const r = await fetch(url, { cache: 'no-cache' });
      if (!r.ok) throw new Error(`no se pudo descargar (${r.status})`);
      const buf = new Uint8Array(await r.arrayBuffer());
      const { M, F, N } = head;
      if (F !== h.F || N !== h.N || buf.length !== (2 * M + 8) * F * N) {
        throw new Error('el fichero de detalle no corresponde a esta predicción (se actualizó mientras tanto)');
      }
      const MFN = M * F * N, FN = F * N;
      return {
        M, F, N,
        a1: buf.subarray(0, MFN), a12: buf.subarray(MFN, 2 * MFN),
        p1: buf.subarray(2 * MFN, 2 * MFN + 4 * FN), p12: buf.subarray(2 * MFN + 4 * FN, 2 * MFN + 8 * FN),
      };
    })());
    explainCache.get(id).catch(() => explainCache.delete(id));
  }
  return explainCache.get(id);
}
