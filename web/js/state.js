// The view as a URL hash, so that any view is a link:
//   #h=mid&t=202610020300&m=cuencas&p=39.428,-0.418&pt=poyo-paiporta&v=39.40,-0.60,9.5
// h horizon (now|mid|long) · f=max (worst of the period) or t = start of the frame, UTC yyyymmddhhmm
// (a time, not an index: the link still means the same hours after the next update; f=<index> is still read)
// m map mode (celdas|cuencas) · zi=1 flood zones on · p selected place (lat,lon) · pt selected control point · dm selected reservoir · v map centre and zoom

import { HORIZONS } from './config.js';

export const state = {
  hz: null,            // 'now' | 'mid' | 'long'
  f: 'max',            // 'max' or frame index
  mode: 'celdas',      // 'celdas' | 'cuencas'
  sel: null,           // {lat, lon}
  pt: null,            // control-point id
  view: null,          // {lat, lon, z}
  zi: false,           // official 500-year flood zones shown over the map
  dm: null,            // selected reservoir id
};

const r = (v, d) => Number(v).toFixed(d).replace(/\.?0+$/, '');

/** t0: ISO start of the selected frame ("2026-10-02T03:00Z"), needed when s.f is an index. */
export function encode(s = state, t0 = null) {
  const q = [];
  if (s.hz) q.push(`h=${s.hz}`);
  if (s.f === 'max' || !t0) q.push(`f=${s.f}`);
  else q.push(`t=${String(t0).replace(/\D/g, '').slice(0, 12)}`);
  if (s.mode !== 'celdas') q.push(`m=${s.mode}`);
  if (s.sel) q.push(`p=${r(s.sel.lat, 4)},${r(s.sel.lon, 4)}`);
  if (s.pt) q.push(`pt=${encodeURIComponent(s.pt)}`);
  if (s.dm) q.push(`dm=${encodeURIComponent(s.dm)}`);
  if (s.zi) q.push('zi=1');
  if (s.view) q.push(`v=${r(s.view.lat, 3)},${r(s.view.lon, 3)},${r(s.view.z, 2)}`);
  return `#${q.join('&')}`;
}

/** Reads a hash into a partial state ({hz, f | t, mode, sel, pt, view}); anything malformed is simply left out. */
export function decode(hash) {
  const out = {};
  const q = new URLSearchParams((hash || '').replace(/^#/, ''));
  if (HORIZONS.includes(q.get('h'))) out.hz = q.get('h');
  const f = q.get('f'), t = q.get('t');
  if (t && /^\d{12}$/.test(t)) out.t = `${t.slice(0, 4)}-${t.slice(4, 6)}-${t.slice(6, 8)}T${t.slice(8, 10)}:${t.slice(10, 12)}Z`;
  else if (f === 'max') out.f = 'max';
  else if (f != null && /^\d+$/.test(f)) out.f = Number(f);
  if (q.get('m') === 'cuencas' || q.get('m') === 'celdas') out.mode = q.get('m');
  const nums = (key, n) => {
    const a = (q.get(key) || '').split(',').map(Number);
    return a.length === n && a.every(Number.isFinite) ? a : null;
  };
  const p = nums('p', 2);
  if (p && Math.abs(p[0]) <= 90 && Math.abs(p[1]) <= 180) out.sel = { lat: p[0], lon: p[1] };
  if (q.get('pt')) out.pt = q.get('pt');
  if (q.get('dm')) out.dm = q.get('dm');
  if (q.get('zi') === '1') out.zi = true;
  const v = nums('v', 3);
  if (v && v[2] >= 5 && v[2] <= 16) out.view = { lat: v[0], lon: v[1], z: v[2] };
  return out;
}

let last = '';
/** Writes the hash without adding history entries. */
export function writeHash(t0 = null) {
  const h = encode(state, t0);
  if (h === last) return;
  last = h;
  try { history.replaceState(null, '', location.pathname + location.search + h); } catch (e) { /* sandboxed frame: the view simply is not linkable */ }
}
export const isOwnHash = (h) => h === last;
