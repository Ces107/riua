// The one line above the map, written from the data: level, where, from when to when.

import { HORIZONS, LEVEL } from './config.js';
import { geo } from './geo.js';
import { span, valid } from './time.js';

const lower = (s) => s.charAt(0).toLowerCase() + s.slice(1);

/** "a, b y c"; "y" becomes "e" before a word that starts with the sound i ("Valencia e Interior"). */
export function listEs(items) {
  if (items.length <= 1) return items.join('');
  const last = items[items.length - 1];
  return `${items.slice(0, -1).join(', ')} ${/^h?i(?!e)/i.test(last) ? 'e' : 'y'} ${last}`;
}

/**
 * Zone names ("Litoral sur de Valencia") as a short phrase:
 * one -> "el litoral sur de Valencia"; two of one province -> "el litoral sur y el interior sur de Valencia";
 * more -> the provinces, "Valencia y Castellón".
 */
export function zonesPhrase(names) {
  const split = names.map((n) => { const m = /^(.*) de ([^ ]+)$/.exec(n); return m ? { part: lower(m[1]), prov: m[2] } : { part: lower(n), prov: '' }; });
  const provs = [...new Set(split.map((s) => s.prov).filter(Boolean))];
  if (names.length === 1) return `el ${lower(names[0])}`;
  if (names.length === 2 && provs.length === 1) return `el ${split[0].part} y el ${split[1].part} de ${provs[0]}`;
  if (names.length === 2) return `el ${lower(names[0])} y el ${lower(names[1])}`;
  return provs.length ? listEs(provs) : `${names.length} zonas`;
}

/**
 * items = [{hz, fr:{d0,d1}, ...}] (frames that share a level) -> the first unbroken period among them, in time order.
 * Frames that are already over are dropped while a later one remains. Hourly and 3-hourly frames join;
 * the daily ones only among themselves.
 */
export function firstRun(items, now = new Date()) {
  const ahead = items.filter((it) => !valid(it.fr.d1) || it.fr.d1 > now);
  const list = (ahead.length ? ahead : items).slice().sort((a, b) => a.fr.d0 - b.fr.d0);
  const run = [list[0]];
  for (const it of list.slice(1)) {
    const end = run[run.length - 1].fr.d1;
    if ((it.hz === 'long') !== (list[0].hz === 'long') || it.fr.d0 > end) break;
    if (it.fr.d1 > end) run.push(it);
  }
  return run;
}

/**
 * -> { level, text, hz, frame } : the highest level anywhere, where (warning zones) and from when to when.
 * Only the cells are used: the basin and control-point products have their own table.
 * Frames that are already over do not count while a later one holds the same level.
 */
export function headline(snap, now = new Date()) {
  const present = HORIZONS.filter((k) => snap.hz[k]);
  if (!present.length) return { level: 0, text: 'Sin datos', hz: null, frame: null };
  let top = 0;
  for (const k of present) top = Math.max(top, snap.hz[k].top);
  const missing = HORIZONS.length - present.length;
  if (top <= 1) {
    return { level: top, text: top === 0 ? 'Sin datos' : `Sin riesgo${missing ? '' : ' en 7 días'}`, hz: present[0], frame: null };
  }
  // every frame that has the top level, with its cells per zone
  const items = [];
  for (const k of present) {
    const h = snap.hz[k];
    for (let f = 0; f < h.F; f++) {
      const zones = new Map();
      let count = 0, inside = 0;
      for (let n = 0; n < h.N; n++) {
        if (h.level[f * h.N + n] !== top) continue;
        count++;
        const zi = geo.cells ? geo.cells.zone[snap.nToCell[n]] : 255;
        if (zi !== 255) { inside++; zones.set(zi, (zones.get(zi) || 0) + 1); }
      }
      if (count) items.push({ hz: k, f, fr: h.frames[f], count, inside, zones });
    }
  }
  const run = firstRun(items, now);
  const first = run[0];
  const zoneCount = new Map();
  let inside = 0;
  for (const it of run) { inside += it.inside; for (const [zi, c] of it.zones) zoneCount.set(zi, (zoneCount.get(zi) || 0) + c); }
  let where = '';
  if (geo.cells && !inside) where = ' aguas arriba';
  else if (geo.cells) where = ` en ${zonesPhrase([...zoneCount.entries()].sort((a, b) => b[1] - a[1]).map(([zi]) => geo.cells.zoneNames[zi]))}`;
  const text = `${LEVEL[top].name}${where} · ${span(first.hz, first.fr.d0, run[run.length - 1].fr.d1)}`;
  return { level: top, text, hz: first.hz, frame: first.f };
}

/** Horizon to open: the one with the highest level; ties go to the nearest in time. */
export function defaultHorizon(snap) {
  let best = null;
  for (const k of HORIZONS) if (snap.hz[k] && (!best || snap.hz[k].top > snap.hz[best].top)) best = k;
  return best;
}
