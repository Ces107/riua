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
      cap: capacity(pp, i), tPeak: qMax > 0 ? tPeak : null };
  }).sort((x, y) => y.level - x.level || (y.pOver ?? 0) - (x.pOver ?? 0) || (x.pt.stream || '').localeCompare(y.pt.stream || '', 'es'));
}

const signed = (v) => `${v >= 0 ? '+' : '−'}${num(Math.abs(v), 1)}`;

function heightText(h) {
  if (h[0] == null && h[1] == null) return '—';
  const main = h[0] == null ? '—' : h[0] >= 0 ? `${signed(h[0])} m` : `no desborda (${num(-h[0], 1)} m por debajo)`;
  const high = h[1] == null ? '' : h[1] >= 0 ? `<br><span class="dim">alto: ${signed(h[1])} m</span>` : '';
  return main + high;
}

export function renderPoints(root, snap, st) {
  const h = snap && snap.hz[st.hz];
  const rows = pointRows(h, st.f);
  if (!rows.length) { root.hidden = true; root.innerHTML = ''; return; }
  root.hidden = false;
  const period = st.f === 'max' ? 'lo peor del periodo' : frameLabel(st.hz, h.frames[Math.min(st.f, h.F - 1)]);
  let body = '';
  for (const r of rows) {
    const open = st.pt === r.pt.id;
    body += `<tr class="${open ? 'open' : ''}"><td>${`<span class="lv lv${r.level}">${r.level || '–'}</span>`}</td>
<th scope="row"><button type="button" class="link" data-pt="${esc(r.pt.id)}" aria-expanded="${open}">${esc(r.pt.stream || r.pt.id)}</button><br><span class="town">${esc(r.pt.town || '')}</span></th>
<td>${pct(r.pOver)}</td>
<td>${r.q[0] == null ? '—' : num(r.q[0])} / ${r.cap == null ? '—' : num(r.cap)}${r.q[1] == null ? '' : `<br><span class="dim">alto: ${num(r.q[1])}</span>`}</td>
<td>${heightText(r.hover)}</td>
<td>${r.tPeak ? dayTime(r.tPeak).replace(/:00$/, ' h') : '—'}</td></tr>`;
    if (open) body += `<tr class="chart-row"><td colspan="6">${hydrograph(snap, h, r)}</td></tr>`;
  }
  root.innerHTML = `<h2>Barrancos y ríos</h2>
<p>Caudal estimado a partir de la lluvia prevista en la cuenca de cada punto, comparado con lo que cabe en el cauce. ${esc(period.charAt(0).toUpperCase() + period.slice(1))}.
Toca un nombre para ver su hidrograma y situarlo en el mapa.</p>
<div class="scroll"><table class="data points"><thead><tr><th scope="col">Nivel</th><th scope="col">Barranco o río<br>municipio</th><th scope="col">P(desborde)</th>
<th scope="col">Punta prevista / capacidad (m³/s)</th><th scope="col">Altura sobre el borde</th><th scope="col">Hora de la punta</th></tr></thead><tbody>${body}</tbody></table></div>
<p class="dim">Es una estimación de escorrentía en régimen natural: no ve desembalses, puentes taponados ni el alcantarillado. Niveles: 2 el barranco baja con fuerza, 3 cerca del borde, 4 desborda, 5 el agua supera el borde en más de 1 m.</p>`;
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
      + `<text x="${W - mR - 4}" y="${(Y(cap) - 5).toFixed(1)}" text-anchor="end" fill="${LEVEL[4].color}">capacidad del cauce: ${num(cap)} m³/s</text>`;
  }
  let gaugeNote = 'No hay un aforo con caudal medido junto a este punto.';
  if (gauge) {
    const gt = Date.parse(gauge.it.t_utc);
    if (gt >= t0 && gt <= t1) g += `<rect x="${(X(gt) - 4).toFixed(1)}" y="${(Y(gauge.it.flow_m3s) - 4).toFixed(1)}" width="8" height="8" fill="${INK}"/>`;
    gaugeNote = `Cuadrado negro: caudal medido en el aforo «${esc(gauge.it.name)}», ${num(gauge.it.flow_m3s, 1)} m³/s (${dayTime(new Date(gauge.it.t_utc))}; dato provisional).`;
  }
  const label = `Hidrograma previsto de ${r.pt.stream} en ${r.pt.town}. Punta mediana ${num(r.q[0])} metros cúbicos por segundo; capacidad del cauce ${cap ? num(cap) : 'desconocida'}.`;
  return `<figure class="hydro"><figcaption><strong>${esc(r.pt.stream)}</strong> en ${esc(r.pt.town)} · caudal en m³/s</figcaption>
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" preserveAspectRatio="xMidYMid meet">${g}</svg>
<p class="dim">Línea: escenario central (mediana). Banda: entre el escenario bajo (1 de cada 10 queda por debajo) y el alto (1 de cada 10 lo supera). ${gaugeNote}</p></figure>`;
}
