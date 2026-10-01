// The view as a URL hash, so that any view is a link:
//   #h=mid&f=5&m=cuencas&p=39.428,-0.418&pt=poyo-paiporta&v=39.40,-0.60,9.5
// h horizon (now|mid|long) · f frame index or "max" · m map mode (celdas|cuencas) · p selected place (lat,lon)
// pt selected control point · v map centre and zoom

import { HORIZONS } from './config.js';

export const state = {
  hz: null,            // 'now' | 'mid' | 'long'
  f: 'max',            // 'max' or frame index
  mode: 'celdas',      // 'celdas' | 'cuencas'
  sel: null,           // {lat, lon}
  pt: null,            // control-point id
  view: null,          // {lat, lon, z}
};

const r = (v, d) => Number(v).toFixed(d).replace(/\.?0+$/, '');

export function encode(s = state) {
  const q = [];
  if (s.hz) q.push(`h=${s.hz}`);
  q.push(`f=${s.f}`);
  if (s.mode !== 'celdas') q.push(`m=${s.mode}`);
  if (s.sel) q.push(`p=${r(s.sel.lat, 4)},${r(s.sel.lon, 4)}`);
  if (s.pt) q.push(`pt=${encodeURIComponent(s.pt)}`);
  if (s.view) q.push(`v=${r(s.view.lat, 3)},${r(s.view.lon, 3)},${r(s.view.z, 2)}`);
  return `#${q.join('&')}`;
}

/** Reads a hash into a partial state; anything malformed is simply left out. */
export function decode(hash) {
  const out = {};
  const q = new URLSearchParams((hash || '').replace(/^#/, ''));
  if (HORIZONS.includes(q.get('h'))) out.hz = q.get('h');
  const f = q.get('f');
  if (f === 'max') out.f = 'max';
  else if (f != null && /^\d+$/.test(f)) out.f = Number(f);
  if (q.get('m') === 'cuencas' || q.get('m') === 'celdas') out.mode = q.get('m');
  const nums = (key, n) => {
    const a = (q.get(key) || '').split(',').map(Number);
    return a.length === n && a.every(Number.isFinite) ? a : null;
  };
  const p = nums('p', 2);
  if (p && Math.abs(p[0]) <= 90 && Math.abs(p[1]) <= 180) out.sel = { lat: p[0], lon: p[1] };
  if (q.get('pt')) out.pt = q.get('pt');
  const v = nums('v', 3);
  if (v && v[2] >= 5 && v[2] <= 16) out.view = { lat: v[0], lon: v[1], z: v[2] };
  return out;
}

let last = '';
/** Writes the hash without adding history entries. */
export function writeHash() {
  const h = encode();
  if (h === last) return;
  last = h;
  history.replaceState(null, '', location.pathname + location.search + h);
}
export const isOwnHash = (h) => h === last;
