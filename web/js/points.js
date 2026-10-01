// "Barrancos y ríos": the table of control points and the hydrograph of the selected one.
// Exists only when the snapshot carries `points` and web/geo has the control-point list.

import { INK, LEVEL, RIVER } from './config.js';
import { geo, nearest } from './geo.js';
import { dayTime, esc, frameLabel, num, parts, pct } from './time.js';

/** Channel capacity in m3/s: from the snapshot if it carries it (`points.cap`), else from the section file. */
export function capacity(pp, i) {
  if (pp && Array.isArray(pp.cap) && pp.cap[i] != null) return pp.cap[i];
  return geo.points && geo.points[i] ? geo.points[i].cap : null;
}

/** One row per point for horizon h and frame choice (index or 'max' = each point's worst frame). */
export function pointRows(h, fSel) {
  const pp = h && h.points;
  if (!pp || !pp.level || !geo.points) return [];
  const F = pp.level.length;
  const times = (pp.t || []).map((t) => new Date(t));
  return geo.points.map((pt, i) => {
    let f = 0;
    if (fSel === 'max') {
      let bl = -1, bp = -1;
      for (let k = 0; k < F; k++) {
        const L = pp.level[k][i] || 0, p = pp.p && pp.p[2] ? pp.p[2][k][i] ?? 0 : 0;
        if (L > bl || (L === bl && p > bp)) { bl = L; bp = p; f = k; }
      }
    } else f = Math.min(fSel, F - 1);
    // time of the median peak inside the frame (or the whole period)
    const fr = h.frames[f];
    const a = fSel === 'max' ? h.frames[0].d0 : fr.d0, b = fSel === 'max' ? h.frames[F - 1].d1 : fr.d1;
    let tPeak = null, qMax = -1;
    if (pp.q && pp.q[1]) {
      for (let k = 0; k < times.length; k++) {
        if (times[k] <= a || times[k] > b) continue;
        const q = pp.q[1][k][i];
        if (q != null && q > qMax) { qMax = q; tPeak = times[k]; }
      }
    }
    const two = (arr) => (arr && arr[0] && arr[0][f] ? [arr[0][f][i], arr[1][f][i]] : [null, null]);
    return { pt, i, f, level: pp.level[f][i] || 0, pOver: pp.p && pp.p[2] ? pp.p[2][f][i] : null, q: two(pp.qpeak), hover: two(pp.hover),
      cap: capacity(pp, i), tPeak: qMax > 0 ? tPeak : null, rp: two(pp.rp) };
  }).sort((x, y) => y.level - x.level || (y.pOver ?? 0) - (x.pOver ?? 0) || (x.pt.stream || '').localeCompare(y.pt.stream || '', 'es'));
}

export const rpTxt = (t) => (t == null ? '—' : t < 2 ? '<2 años' : t >= 1000 ? '>500 años' : `${num(t)} años`);

const signed = (v) => `${v >= 0 ? '+' : '−'}${num(Math.abs(v), 1)}`;

function row(r, open, snap, h) {
  const cap = r.cap == null ? '<span class="dim" title="sin capacidad del cauce: nivel por caudal específico">—</span>' : num(r.cap);
  const over = r.cap == null ? '<span class="dim">—</span>' : pct(r.pOver);
  const hgt = r.hover[0] == null ? '' : r.hover[0] >= 0 ? `${signed(r.hover[0])} m` : '';
  let out = `<tr class="${open ? 'open' : ''}"><td><span class="lv lv${r.level}">${r.level || '–'}</span></td>
<th class="wrap"><button type="button" class="link" data-pt="${esc(r.pt.id)}" aria-expanded="${open}">${esc(r.pt.stream || r.pt.id)}</button><span class="town">${esc(r.pt.town || '')}</span></th>
<td>${over}</td><td>${r.q[0] == null ? '—' : num(r.q[0])}</td><td>${cap}</td><td>${rpTxt(r.rp[0])}</td><td>${hgt}</td><td>${r.tPeak ? esc(dayTime(r.tPeak).replace(/:00$/, ' h')) : ''}</td></tr>`;
  if (open) out += `<tr class="chart-row"><td colspan="8">${hydrograph(snap, h, r)}</td></tr>`;
  return out;
}

const HEAD = '<thead><tr><th></th><th class="wrap">Cauce</th><th title="probabilidad de desbordar">Desborde</th><th title="caudal punta, escenario central">Punta m³/s</th><th title="capacidad del cauce">Cauce m³/s</th><th title="periodo de retorno de la punta central (CAUMAX, CEDEX)">Retorno</th><th title="altura del agua sobre el borde">Sobre borde</th><th>Punta</th></tr></thead>';

export function renderPoints(root, snap, st) {
  const h = snap && snap.hz[st.hz];
  const rows = pointRows(h, st.f);
  if (!rows.length) { root.hidden = true; root.innerHTML = ''; return; }
  root.hidden = false;
  const key = (r) => r.level >= 2 || st.pt === r.pt.id;
  const main = rows.filter(key), rest = rows.filter((r) => !key(r));
  const body = (list) => list.map((r) => row(r, st.pt === r.pt.id, snap, h)).join('');
  root.innerHTML = `<h2>Cauces</h2>
${main.length ? `<div class="scroll"><table class="data points">${HEAD}<tbody>${body(main)}</tbody></table></div>` : '<p class="dim">Ningún cauce en riesgo.</p>'}
${rest.length ? `<details${rest.some((r) => st.pt === r.pt.id) ? ' open' : ''}><summary>Todos (${rows.length})</summary><div class="scroll"><table class="data points">${HEAD}<tbody>${body(rest)}</tbody></table></div></details>` : ''}`;
}

function niceStep(max, ticks) {
  const raw = max / ticks, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

/** Inline SVG: p10–p90 band, median, channel capacity, the reading of a river gauge next to the point if there is one. */
function hydrograph(snap, h, r) {
  const pp = h.points, i = r.i;
  const t = (pp.t || []).map((x) => new Date(x));
  if (!pp.q || t.length < 2) return '<p>Esta actualización no trae el hidrograma de este punto.</p>';
  const q10 = pp.q[0].map((row) => row[i] ?? 0), q50 = pp.q[1].map((row) => row[i] ?? 0), q90 = pp.q[2].map((row) => row[i] ?? 0);
  const cap = r.cap;
  const gauge = nearest(snap.rivers.filter((g) => g.flow_m3s != null), r.pt.lat, r.pt.lon, 1, 2)[0];
  const W = 640, H = 260, mL = 54, mR = 12, mT = 16, mB = 34;
  const top = Math.max(...q90, cap || 0, gauge ? gauge.it.flow_m3s : 0, 1) * 1.08;
  const t0 = t[0].getTime(), t1 = t[t.length - 1].getTime();
  const X = (d) => mL + ((d - t0) / (t1 - t0)) * (W - mL - mR), Y = (q) => H - mB - (q / top) * (H - mT - mB);
  const path = (arr) => arr.map((q, k) => `${k ? 'L' : 'M'}${X(t[k].getTime()).toFixed(1)} ${Y(q).toFixed(1)}`).join('');
  const band = `${path(q90)}${q10.map((q, k) => `L${X(t[k].getTime()).toFixed(1)} ${Y(q).toFixed(1)}`).reverse().join('')}Z`;
  let g = '';
  // y axis
  const step = niceStep(top, 4);
  for (let v = 0; v <= top; v += step) {
    g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="${INK}" stroke-opacity="${v === 0 ? 1 : 0.15}"/>`
      + `<text x="${mL - 6}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${num(v)}</text>`;
  }
  // x axis: local midnights labelled with the day, plus 6-hourly ticks when the span is short
  const hours = (t1 - t0) / 3600e3;
  const every = hours <= 30 ? 3 : hours <= 72 ? 6 : 24;
  for (let ms = Math.ceil(t0 / 3600e3) * 3600e3; ms <= t1; ms += 3600e3) {
    const p = parts(new Date(ms));
    if (p.h % every !== 0) continue;
    const x = X(ms).toFixed(1), midnight = p.h === 0;
    g += `<line x1="${x}" x2="${x}" y1="${H - mB}" y2="${H - mB + (midnight ? 7 : 4)}" stroke="${INK}"/>`;
    if (midnight) g += `<line x1="${x}" x2="${x}" y1="${mT}" y2="${H - mB}" stroke="${INK}" stroke-opacity="0.15"/><text x="${x}" y="${H - 6}" text-anchor="middle">${p.wd} ${p.day}</text>`;
    else if (every < 24 && (hours <= 30 || p.h === 12)) g += `<text x="${x}" y="${H - mB + 15}" text-anchor="middle" class="minor">${p.hh}</text>`;
  }
  const now = Date.now();
  if (now > t0 && now < t1) g += `<line x1="${X(now).toFixed(1)}" x2="${X(now).toFixed(1)}" y1="${mT}" y2="${H - mB}" stroke="${INK}" stroke-dasharray="2 3"/><text x="${(X(now) + 4).toFixed(1)}" y="${mT + 9}">ahora</text>`;
  g += `<path d="${band}" fill="${RIVER}" fill-opacity="0.22"/><path d="${path(q50)}" fill="none" stroke="${RIVER}" stroke-width="2.2"/>`;
  if (cap) {
    g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(cap).toFixed(1)}" y2="${Y(cap).toFixed(1)}" stroke="${LEVEL[4].color}" stroke-width="1.6" stroke-dasharray="7 4"/>`
      + `<text x="${W - mR - 4}" y="${(Y(cap) - 5).toFixed(1)}" text-anchor="end" fill="${LEVEL[4].color}">cauce ${num(cap)}</text>`;
  }
  let gaugeNote = 'No hay un aforo con caudal medido junto a este punto.';
  if (gauge) {
    const gt = Date.parse(gauge.it.t_utc);
    if (gt >= t0 && gt <= t1) g += `<rect x="${(X(gt) - 4).toFixed(1)}" y="${(Y(gauge.it.flow_m3s) - 4).toFixed(1)}" width="8" height="8" fill="${INK}"/>`;
    gaugeNote = `Cuadrado negro: caudal medido en el aforo «${esc(gauge.it.name)}», ${num(gauge.it.flow_m3s, 1)} m³/s (${dayTime(new Date(gauge.it.t_utc))}; dato provisional).`;
  }
  const label = `Hidrograma previsto de ${r.pt.stream} en ${r.pt.town}. Punta mediana ${num(r.q[0])} metros cúbicos por segundo; capacidad del cauce ${cap ? num(cap) : 'desconocida'}.`;
  return `<figure class="hydro">
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" preserveAspectRatio="xMidYMid meet">${g}</svg>
<figcaption>m³/s · línea: central · banda: 1 de cada 10 por debajo / por encima${gauge ? ' · ■ aforo medido' : ''}</figcaption>
</figure>`;
}
