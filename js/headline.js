// The one-sentence summary above the map, written from the data.

import { HORIZONS, LEVEL } from './config.js';
import { geo } from './geo.js';
import { parts, weekdayLong, valid } from './time.js';

/** "esta noche", "mañana por la tarde", "el viernes de madrugada", "el sábado" ... */
export function whenPhrase(hzKey, fr, now) {
  if (!valid(fr.d0) || !valid(fr.d1)) return '';
  const mid = new Date((fr.d0.getTime() + fr.d1.getTime()) / 2);
  const a = parts(mid), n = parts(now);
  const day = a.dayNo - n.dayNo;
  if (hzKey === 'long') return day === 1 ? 'mañana' : `el ${weekdayLong(mid)}`;
  if (fr.d0 <= now && fr.d1.getTime() - now.getTime() < 2 * 3600e3) return 'ahora';
  const h = a.h;
  if (day <= 0) return h < 6 ? 'esta madrugada' : h < 13 ? 'esta mañana' : h < 20 ? 'esta tarde' : 'esta noche';
  if (day === 1) return h < 6 ? (n.h >= 12 ? 'esta noche' : 'la próxima madrugada') : h < 13 ? 'mañana por la mañana' : h < 20 ? 'mañana por la tarde' : 'mañana por la noche';
  return `el ${weekdayLong(mid)} ${h < 6 ? 'de madrugada' : h < 13 ? 'por la mañana' : h < 20 ? 'por la tarde' : 'por la noche'}`;
}

function peakPhrase(hzKey, fr) {
  const a = parts(fr.d0), b = parts(fr.d1);
  if (hzKey === 'long') return `el ${weekdayLong(new Date((fr.d0.getTime() + fr.d1.getTime()) / 2))}`;
  return `entre las ${a.hh} y las ${b.hh} h del ${weekdayLong(fr.d0)}`;
}

const lower = (s) => s.charAt(0).toLowerCase() + s.slice(1);

function listEs(items) {
  if (items.length <= 1) return items.join('');
  return `${items.slice(0, -1).join(', ')} y ${items[items.length - 1]}`;
}

/**
 * Zone names ("Litoral sur de Valencia") as a short phrase:
 * one -> "el litoral sur de Valencia"; two of one province -> "el litoral sur y el interior sur de Valencia";
 * more -> "4 zonas de Valencia y Castellón".
 */
export function zonesPhrase(names) {
  const split = names.map((n) => { const m = /^(.*) de ([^ ]+)$/.exec(n); return m ? { part: lower(m[1]), prov: m[2] } : { part: lower(n), prov: '' }; });
  const provs = [...new Set(split.map((s) => s.prov).filter(Boolean))];
  if (names.length === 1) return `el ${lower(names[0])}`;
  if (names.length === 2 && provs.length === 1) return `el ${split[0].part} y el ${split[1].part} de ${provs[0]}`;
  if (names.length === 2) return `el ${lower(names[0])} y el ${lower(names[1])}`;
  return `${names.length} zonas de ${listEs(provs)}`;
}

/**
 * -> { level, text, hz, frame } : the highest level anywhere, where (warning zones) and when.
 * Only the cells are used: the basin and control-point products have their own sentence.
 */
export function headline(snap, now = new Date()) {
  const present = HORIZONS.filter((k) => snap.hz[k]);
  if (!present.length) return { level: 0, text: 'La última predicción no trae datos utilizables.', hz: null, frame: null };
  let top = 0;
  for (const k of present) top = Math.max(top, snap.hz[k].top);
  const missing = HORIZONS.length - present.length;
  if (top <= 1) {
    const span = missing ? 'en los plazos disponibles' : 'en los próximos 7 días';
    return { level: top, text: top === 0 ? 'No hay datos de predicción en este momento.' : `Sin riesgo apreciable ${span}.`, hz: present[0], frame: null };
  }
  // every frame that has the top level, in time order; count the cells per zone
  const zoneCount = new Map();
  let first = null, peak = null, outside = 0, inside = 0;
  for (const k of present) {
    const h = snap.hz[k];
    for (let f = 0; f < h.F; f++) {
      let count = 0;
      for (let n = 0; n < h.N; n++) {
        if (h.level[f * h.N + n] !== top) continue;
        count++;
        const zi = geo.cells ? geo.cells.zone[snap.nToCell[n]] : 255;
        if (zi === 255) outside++; else { inside++; zoneCount.set(zi, (zoneCount.get(zi) || 0) + 1); }
      }
      if (!count) continue;
      const item = { hz: k, f, fr: h.frames[f], count };
      if (!first || item.fr.d0 < first.fr.d0) first = item;
      if (!peak || count > peak.count) peak = item;
    }
  }
  const word = `${LEVEL[top].word} (${top})`;
  let where;
  if (!geo.cells) where = '';
  else if (!inside) where = ' aguas arriba, fuera de la Comunitat Valenciana, en cuencas que desaguan en ella';
  else {
    where = ` en ${zonesPhrase([...zoneCount.entries()].sort((a, b) => b[1] - a[1]).map(([zi]) => geo.cells.zoneNames[zi]))}`;  }
  let text = `Riesgo ${word} ${whenPhrase(first.hz, first.fr, now)}${where}.`;
  const samePeak = peak.hz === first.hz && peak.f === first.f;
  if (!samePeak || first.hz !== 'long') text += ` Máximo previsto ${peakPhrase(peak.hz, peak.fr)}.`;
  return { level: top, text, hz: first.hz, frame: first.f, peak };
}

/** Horizon to open: the one with the highest level; ties go to the nearest in time. */
export function defaultHorizon(snap) {
  let best = null;
  for (const k of HORIZONS) if (snap.hz[k] && (!best || snap.hz[k].top > snap.hz[best].top)) best = k;
  return best;
}
