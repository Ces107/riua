// "Embalses": reservoirs, their state now and the forecast of filling and spill (snapshot block `reservoirs`,
// docs/SNAPSHOT.md). Everything here is optional: a snapshot without the block shows nothing.

import { INK, LEVEL, RIVER } from './config.js';
import { geo } from './geo.js';
import { dayTime, esc, num, parts, pct } from './time.js';

const TOP = 115;                       // the fill bar runs from empty to 115 % of the volume at the spill level

/** The block of the snapshot, or null. */
export const resv = (snap) => (snap && snap.reservoirs && Array.isArray(snap.reservoirs.dams) ? snap.reservoirs : null);

/** State of reservoir k in horizon hz for frame choice f (index or 'max'). */
export function damState(snap, hz, f, k) {
  const R = resv(snap);
  const d = R.dams[k], h = R.horizons && R.horizons[hz];
  const out = { d, k, L: 0, f: 0, pSpill: 0, tSpill: null, pct: [null, null, null], ok: false };
  if (!h || !h.level || !h.level.length || !h.ok || !h.ok[k]) return out;
  const F = h.level.length;
  let ff = 0;
  if (f === 'max') {
    let bl = -1, bp = -1;
    for (let j = 0; j < F; j++) { const L = h.level[j][k] || 0, p = h.p_spill[j][k] || 0; if (L > bl || (L === bl && p > bp)) { bl = L; bp = p; ff = j; } }
    // over the whole period the fill shown is the one at its end (the volume keeps what entered)
    ff = bl >= 2 ? ff : F - 1;
  } else ff = Math.min(f, F - 1);
  out.ok = true;
  out.f = ff;
  out.L = f === 'max' ? Math.max(...h.level.map((row) => row[k] || 0)) : h.level[ff][k] || 0;
  out.pSpill = f === 'max' ? h.p_spill_any[k] || 0 : h.p_spill[ff][k] || 0;
  out.tSpill = h.t_spill[k] ? new Date(h.t_spill[k]) : null;
  out.pct = [0, 1, 2].map((q) => (h.pct && h.pct[q] ? h.pct[q][ff][k] : null));
  out.qout = h.qout ? [h.qout[0][ff][k], h.qout[1][ff][k]] : [null, null];
  out.qin = h.qin ? [h.qin[0][ff][k], h.qin[1][ff][k]] : [null, null];
  return out;
}

/** Rows of every reservoir, worst first. */
export function damRows(snap, hz, f) {
  const R = resv(snap);
  if (!R) return [];
  return R.dams.map((d, k) => damState(snap, hz, f, k))
    .sort((a, b) => b.L - a.L || b.pSpill - a.pSpill || (b.d.now ? b.d.now.pct || 0 : 0) - (a.d.now ? a.d.now.pct || 0 : 0));
}

const x = (p) => `${(Math.max(0, Math.min(p, TOP)) / TOP * 100).toFixed(1)}%`;

/** The thin bar: fill now (ink), forecast range at the end of the frame (hatched), tick at the spill level, reserve tick. */
export function fillBar(s, big = false) {
  const now = s.d.now && s.d.now.pct != null ? s.d.now.pct : null;
  if (now == null) return '<span class="fill none" aria-hidden="true"></span>';
  const [lo, mid, hi] = s.pct;
  const rng = s.ok && lo != null && hi != null && hi - lo >= 0.5 ? `<s style="left:${x(lo)};width:calc(${x(hi)} - ${x(lo)})"></s>` : '';
  const med = s.ok && mid != null && Math.abs(mid - now) >= 0.5 ? `<u style="left:${x(mid)}"></u>` : '';
  const res = s.d.res != null && s.d.cap ? `<em style="left:${x(100 * s.d.res / s.d.cap)}"></em>` : '';
  const label = `${num(now)} % lleno${mid != null && s.ok ? `, previsto ${num(mid)} %` : ''}`;
  return `<span class="fill${big ? ' big' : ''}" role="img" aria-label="${esc(label)}"><i style="width:${x(now)}"></i>${rng}${med}${res}<b style="left:${x(100)}"></b></span>`;
}

const hourTxt = (d) => {
  const p = parts(d), today = parts(new Date());
  return esc(p.day === today.day ? `${p.hh} h` : `${p.wd} ${p.hh} h`);
};
// above 100 % the dam is over its spillway: say so, and how much it is letting through
const fillTxt = (n) => (n && n.pct != null ? (n.pct >= 99.5 ? `lleno${n.qout >= 1 ? ` · sale ${num(n.qout)} m³/s` : ''}` : `${num(n.pct)} %`) : '');
const spillTxt = (s) => (s.ok && s.pSpill >= 0.005 ? pct(s.pSpill) : s.ok ? '<span class="dim">0 %</span>' : '<span class="dim">—</span>');

function rowHtml(s, sel) {
  const n = s.d.now;
  return `<tr class="${sel ? 'open' : ''}"><td><span class="lv lv${s.L}">${s.L || '–'}</span></td>
<th class="wrap"><button type="button" class="link" data-dm="${esc(s.d.id)}"${sel ? ' aria-current="true"' : ''}>${esc(s.d.name)}</button><span class="town">${esc(s.d.river || '')}</span></th>
<td class="fillcell">${fillBar(s)}<span class="num">${fillTxt(n) || '—'}</span></td><td>${spillTxt(s)}</td><td>${s.ok && s.pSpill >= 0.1 && s.tSpill ? hourTxt(s.tSpill) : ''}</td></tr>`;
}

/** The "Embalses" block (appended to the Cauces section). */
export function damsHtml(snap, st, open) {
  const rows = damRows(snap, st.hz, st.f);
  if (!rows.length) return '';
  const key = (s) => s.L >= 2 || s.d.id === st.dm;
  const main = rows.filter(key), rest = rows.filter((s) => !key(s));
  const table = (list) => `<div class="scroll"><table class="data points dams"><thead><tr><th></th><th class="wrap">Embalse</th><th title="volumen ahora sobre el del aliviadero; la marca vertical es el aliviadero, el tramo rayado la previsión">Lleno</th><th title="probabilidad de verter por el aliviadero">Vierte</th><th>Hora</th></tr></thead><tbody>${list.map((s) => rowHtml(s, s.d.id === st.dm)).join('')}</tbody></table></div>`;
  const isOpen = open || rest.some((s) => s.d.id === st.dm);
  return `<h2 class="gap2">Embalses</h2>
${main.length ? table(main) : `<div class="lvl calm"><span class="lv lv1">1</span><div><b>Sin riesgo</b></div></div>`}
${rest.length ? `<details class="dams-all"${isOpen ? ' open' : ''}><summary>Todos (${rows.length})</summary>${table(rest)}</details>` : ''}`;
}

/** Reservoirs directly above a control point: [{s, share, lag, cuts}] (none when the snapshot has no block). */
export function damsAbove(snap, hz, f, pointId) {
  const R = resv(snap);
  if (!R || !R.points || !R.points[pointId]) return [];
  return R.points[pointId].map(([k, share, lag, cuts]) => ({ s: damState(snap, hz, f, k), share, lag, cuts }));
}

/** A compact line for a Cauces row or the place panel: only when a reservoir upstream matters now. */
export function damNote(snap, hz, f, pointId) {
  const list = damsAbove(snap, hz, f, pointId).filter(({ s, share }) => {
    const n = s.d.now;
    return share >= 0.1 && (s.L >= 2 || s.pSpill >= 0.05 || (n && n.pct != null && n.pct >= 85));
  });
  if (!list.length) return '';
  return list.map(({ s }) => {
    const n = s.d.now;
    const sp = s.ok && s.pSpill >= 0.05 ? ` · vierte ${pct(s.pSpill)}` : '';
    return `<button type="button" class="dam-note" data-dm="${esc(s.d.id)}"><span class="lv lv${s.L}">${s.L || '–'}</span>${fillBar(s)}<span>${esc(s.d.name)} ${fillTxt(n)}${sp}</span></button>`;
  }).join('');
}

function niceStep(max, ticks) {
  const raw = max / ticks, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

/** Filling curve: measured hours (ink), forecast band 10–90 % and median (river blue), the spill level, the reserve. */
function chart(snap, st, s) {
  const R = resv(snap), h = R.horizons && R.horizons[st.hz], k = s.k, d = s.d;
  const past = (d.now && d.now.sv ? d.now.sv : []).filter((p) => p[1] != null).map((p) => [new Date(p[0]).getTime(), p[1]]);
  const ft = h && h.t ? h.t.map((t) => new Date(t).getTime()) : [];
  const v = h && h.v && s.ok ? [0, 1, 2].map((q) => h.v[q].map((row) => row[k])) : null;
  if (!past.length && (!v || ft.length < 2)) return '';
  const W = 640, H = 220, mL = 44, mR = 12, mT = 14, mB = 32;
  const t0 = Math.min(...(past.length ? [past[0][0]] : []), ...(ft.length ? [ft[0]] : [])), t1 = Math.max(...(past.length ? [past[past.length - 1][0]] : []), ...(ft.length ? [ft[ft.length - 1]] : []));
  if (!(t1 > t0)) return '';
  const all = [...past.map((p) => p[1]), ...(v ? v[2] : []), 100];
  const top = Math.max(110, ...all) * 1.04, bot = Math.max(0, Math.min(...past.map((p) => p[1]), ...(v ? v[0] : [100])) - 10);
  const X = (t) => mL + ((t - t0) / (t1 - t0)) * (W - mL - mR), Y = (p) => H - mB - ((p - bot) / (top - bot)) * (H - mT - mB);
  let g = '';
  const step = niceStep(top - bot, 4);
  for (let p = Math.ceil(bot / step) * step; p <= top; p += step) {
    g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(p).toFixed(1)}" y2="${Y(p).toFixed(1)}" stroke="${INK}" stroke-opacity="0.15"/><text x="${mL - 6}" y="${(Y(p) + 4).toFixed(1)}" text-anchor="end">${num(p)}</text>`;
  }
  const hours = (t1 - t0) / 3600e3, every = hours <= 30 ? 3 : hours <= 72 ? 6 : 24;
  for (let ms = Math.ceil(t0 / 3600e3) * 3600e3; ms <= t1; ms += 3600e3) {
    const pp = parts(new Date(ms));
    if (pp.h % every) continue;
    const xx = X(ms).toFixed(1), mid = pp.h === 0;
    g += `<line x1="${xx}" x2="${xx}" y1="${H - mB}" y2="${H - mB + (mid ? 7 : 4)}" stroke="${INK}"/>`;
    if (mid) g += `<text x="${xx}" y="${H - 6}" text-anchor="middle">${pp.wd} ${pp.day}</text>`;
    else if (every < 24 && (hours <= 30 || pp.h === 12)) g += `<text x="${xx}" y="${H - mB + 15}" text-anchor="middle" class="minor">${pp.hh}</text>`;
  }
  g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(100).toFixed(1)}" y2="${Y(100).toFixed(1)}" stroke="${LEVEL[4].color}" stroke-width="1.6"/>`
    + `<text x="${W - mR - 4}" y="${(Y(100) - 5).toFixed(1)}" text-anchor="end" fill="${LEVEL[4].color}">aliviadero</text>`;
  if (d.res != null && d.cap) {
    const r = 100 * d.res / d.cap;
    if (r > bot) g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(r).toFixed(1)}" y2="${Y(r).toFixed(1)}" stroke="${INK}" stroke-dasharray="5 4"/><text x="${W - mR - 4}" y="${(Y(r) - 5).toFixed(1)}" text-anchor="end" class="minor">resguardo</text>`;
  }
  const now = Date.now();
  if (now > t0 && now < t1) g += `<line x1="${X(now).toFixed(1)}" x2="${X(now).toFixed(1)}" y1="${mT}" y2="${H - mB}" stroke="${INK}" stroke-dasharray="2 3"/>`;
  if (v && ft.length >= 2) {
    const path = (arr) => arr.map((p, j) => `${j ? 'L' : 'M'}${X(ft[j]).toFixed(1)} ${Y(p).toFixed(1)}`).join('');
    const band = `${path(v[2])}${v[0].map((p, j) => `L${X(ft[j]).toFixed(1)} ${Y(p).toFixed(1)}`).reverse().join('')}Z`;
    g += `<path d="${band}" fill="${RIVER}" fill-opacity="0.22"/><path d="${path(v[1])}" fill="none" stroke="${RIVER}" stroke-width="2.2"/>`;
  }
  if (past.length) g += `<path d="${past.map((p, j) => `${j ? 'L' : 'M'}${X(p[0]).toFixed(1)} ${Y(p[1]).toFixed(1)}`).join('')}" fill="none" stroke="${INK}" stroke-width="2"/>`;
  const label = `Llenado previsto del embalse de ${d.name}, en % del volumen del aliviadero.`;
  return `<figure class="hydro"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" preserveAspectRatio="xMidYMid meet">${g}</svg>
<figcaption>% lleno · — medido · banda 10–90 %</figcaption></figure>`;
}

/** Side panel of a selected reservoir. */
export function damPanel(snap, st) {
  const R = resv(snap);
  const k = R ? R.dams.findIndex((d) => d.id === st.dm) : -1;
  if (k < 0) return null;
  const s = damState(snap, st.hz, st.f, k), d = s.d, n = d.now;
  const meta = LEVEL[s.L];
  const flow = (v) => (v == null ? '—' : num(v, v < 10 ? 1 : 0));
  const kv = (a, b) => `<tr><th>${a}</th><td>${b}</td></tr>`;
  const nowRows = n ? [
    kv('Volumen', `${num(n.v, n.v < 10 ? 2 : 0)} hm³${d.cap ? ` <span class="dim">/ ${num(d.cap, d.cap < 10 ? 1 : 0)}</span>` : ''}`),
    n.level != null ? kv('Cota', `${num(n.level, 2)} m`) : '',
    n.qin != null ? kv('Entra', `${flow(n.qin)} m³/s`) : '',
    n.qout != null ? kv('Sale', `${flow(n.qout)} m³/s`) : '',
    n.rate != null ? kv('Ritmo', `${n.rate >= 0 ? '+' : '−'}${num(Math.abs(n.rate), Math.abs(n.rate) < 0.1 ? 3 : 2)} hm³/h`) : '',
  ].join('') : '';
  const fc = s.ok ? [
    kv('Vierte', `${spillTxt(s)}${s.pSpill >= 0.1 && s.tSpill ? ` · ${hourTxt(s.tSpill)}` : ''}`),
    kv('Entrada máx.', `${flow(s.qin[0])}${s.qin[1] > s.qin[0] ? `<span class="dim">–${flow(s.qin[1])}</span>` : ''} m³/s`),
    kv('Salida máx.', `${flow(s.qout[0])}${s.qout[1] > s.qout[0] ? `<span class="dim">–${flow(s.qout[1])}</span>` : ''} m³/s`),
  ].join('') : '';
  const below = [];
  for (const [pid, list] of Object.entries(R.points || {})) {
    if (!list.some((e) => e[0] === k)) continue;
    const pt = geo.points && geo.points.find((p) => p.id === pid);
    if (pt) below.push({ pt, lag: list.find((e) => e[0] === k)[2] });
  }
  below.sort((a, b) => a.lag - b.lag);
  const h = snap.hz[st.hz], pp = h && h.points;
  const lvOf = (pt) => {
    if (!pp || !pp.level || !geo.points) return 0;
    const i = geo.points.indexOf(pt);
    return st.f === 'max' ? Math.max(...pp.level.map((row) => row[i] || 0)) : (pp.level[Math.min(st.f, pp.level.length - 1)] || [])[i] || 0;
  };
  const down = below.map(({ pt, lag }) => `<li><button type="button" data-pt="${esc(pt.id)}"><span class="lv lv${lvOf(pt)}">${lvOf(pt) || '–'}</span><span class="t">${esc(pt.stream)}<small>${esc(pt.town || '')}</small></span><span class="r">${num(lag)} h</span></button></li>`).join('');
  const nxt = d.next ? R.dams.find((x) => x.id === d.next) : null;
  return `<div class="ttl"><h2>${esc(d.name)}</h2><button type="button" class="x" data-act="back" aria-label="Quitar la selección" title="Quitar la selección">×</button></div>
<p class="sub">Embalse · ${esc(d.river || '')}</p>
<div class="lvl"><span class="lv big lv${s.L}">${s.L || '–'}</span><div><b>${meta.name}</b>${fillBar(s, true)}</div></div>
${chart(snap, st, s)}
${nowRows ? `<h3>Ahora</h3><table class="data kv">${nowRows}</table>` : ''}
${fc ? `<h3>Previsión</h3><table class="data kv">${fc}</table>` : ''}
${down || nxt ? `<h3>Aguas abajo</h3><ul class="rows">${nxt ? `<li><button type="button" data-dm="${esc(nxt.id)}"><span class="dm-mark" aria-hidden="true"></span><span class="t">${esc(nxt.name)}<small>embalse</small></span><span class="r"></span></button></li>` : ''}${down}</ul>` : ''}
${n && n.t ? `<p class="dim num small">${esc(d.source === 'saih_segura' ? 'SAIH Segura' : 'SAIH Júcar')} · ${esc(dayTime(new Date(n.t)))}</p>` : ''}`;
}

/** Markers for the map: [{id, lat, lon, L, sel, name}] in the frame in view. */
export function damMarks(snap, st) {
  const R = resv(snap);
  if (!R) return null;
  return R.dams.map((d, k) => ({ id: d.id, lat: d.lat, lon: d.lon, name: d.name, L: damState(snap, st.hz, st.f, k).L, sel: st.dm === d.id }));
}
