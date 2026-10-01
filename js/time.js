// Dates and numbers as text. Every time shown to the reader is Europe/Madrid.

import { TZ } from './config.js';

const partsFmt = new Intl.DateTimeFormat('es-ES', { timeZone: TZ, weekday: 'short', day: 'numeric', month: 'short', year: 'numeric',
  hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
const longDay = new Intl.DateTimeFormat('es-ES', { timeZone: TZ, weekday: 'long' });
const utcFmt = new Intl.DateTimeFormat('es-ES', { timeZone: 'UTC', day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });

/** {wd:'jue', day:2, month:'oct', year:2026, hh:'05', mm:'00', h:5, dayNo: days since epoch in local time} */
export function parts(d) {
  const o = {};
  for (const p of partsFmt.formatToParts(d)) o[p.type] = p.value;
  const wd = o.weekday.replace('.', '');
  const month = o.month.replace('.', '');
  const dayNo = Math.floor(Date.UTC(Number(o.year), monthIndex(month), Number(o.day)) / 86400000);
  return { wd, day: Number(o.day), month, year: Number(o.year), hh: o.hour, mm: o.minute, h: Number(o.hour), dayNo };
}

const MONTHS = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sept', 'oct', 'nov', 'dic'];
function monthIndex(m) {
  const i = MONTHS.findIndex((x) => x === m || x.slice(0, 3) === m.slice(0, 3));
  return i < 0 ? 0 : i;
}

export const weekdayLong = (d) => longDay.format(d);
export const valid = (d) => d instanceof Date && !Number.isNaN(d.getTime());

/** "jue 2, 05–08 h" for hourly / 3-hourly frames, "sáb 4" for the daily ones. */
export function frameLabel(hzKey, fr) {
  if (!valid(fr.d0) || !valid(fr.d1)) return 'tramo sin fecha';
  const a = parts(fr.d0), b = parts(fr.d1);
  if (hzKey === 'long') return `${a.wd} ${a.day}`;
  return `${a.wd} ${a.day}, ${a.hh}–${b.hh} h`;
}

/** Exact window, local and UTC, for the audit: "jue 2 oct 05:00 – 08:00 (03:00–06:00 UTC)". */
export function frameExact(fr) {
  if (!valid(fr.d0) || !valid(fr.d1)) return 'sin fecha';
  const a = parts(fr.d0), b = parts(fr.d1);
  const sameDay = a.dayNo === b.dayNo;
  const utc = (d) => utcFmt.format(d).replace(',', '');
  return `${a.wd} ${a.day} ${a.month} ${a.hh}:${a.mm} – ${sameDay ? '' : `${b.wd} ${b.day} ${b.month} `}${b.hh}:${b.mm} (hora peninsular) = ${utc(fr.d0)} – ${utc(fr.d1)} UTC`;
}

/** "jue 2 14 h → sáb 4 08 h" */
export function spanLabel(frames) {
  if (!frames.length) return '';
  const a = parts(frames[0].d0), b = parts(frames[frames.length - 1].d1);
  return `${a.wd} ${a.day} ${a.hh} h → ${b.wd} ${b.day} ${b.hh} h`;
}

/** "hoy 12:05", "ayer 23:10", "mié 30 sept 08:00" */
export function dayTime(d, now = new Date()) {
  if (!valid(d)) return 'hora desconocida';
  const a = parts(d), n = parts(now);
  const hm = `${a.hh}:${a.mm}`;
  if (a.dayNo === n.dayNo) return `hoy ${hm}`;
  if (a.dayNo === n.dayNo - 1) return `ayer ${hm}`;
  if (a.dayNo === n.dayNo + 1) return `mañana ${hm}`;
  return `${a.wd} ${a.day} ${a.month} ${hm}`;
}

/** "jue 2 oct 2026, 12:05" */
export function stamp(d) {
  if (!valid(d)) return '';
  const a = parts(d);
  return `${a.wd} ${a.day} ${a.month} ${a.year}, ${a.hh}:${a.mm}`;
}

/** "hace 9 min", "hace 3 h", "hace 2 días" */
export function age(d, now = new Date()) {
  const min = Math.round((now - d) / 60000);
  if (!Number.isFinite(min)) return '';
  if (min < 1) return 'hace menos de 1 min';
  if (min < 60) return `hace ${min} min`;
  if (min < 48 * 60) return `hace ${Math.floor(min / 60)} h${min % 60 && min < 600 ? ` ${min % 60} min` : ''}`;
  return `hace ${Math.floor(min / 1440)} días`;
}

// ---- numbers ---------------------------------------------------------------------------------

const nf0 = new Intl.NumberFormat('es-ES', { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat('es-ES', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

export const num = (v, dec = 0) => (v == null || Number.isNaN(v) ? '—' : new Intl.NumberFormat('es-ES', { minimumFractionDigits: dec, maximumFractionDigits: dec, useGrouping: Math.abs(v) >= 10000 }).format(v));
/** Rain amount: one decimal under 10 mm, none above. */
export const mmTxt = (v) => (v == null || Number.isNaN(v) ? '—' : `${v < 9.95 ? nf1.format(v) : nf0.format(v)} mm`);
export const pct = (p, dec = 0) => (p == null || Number.isNaN(p) ? '—' : `${num(p * 100, dec)} %`);
export const sci = (v, sig = 4) => (v == null || Number.isNaN(v) ? '—' : Number(v.toPrecision(sig)).toString().replace('.', ','));

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}
