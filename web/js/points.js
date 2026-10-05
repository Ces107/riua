// "Cauces": the table of control points (ravines and rivers at towns) and the hydrograph of the selected one.
// Exists only when the snapshot carries `points` and web/geo has the control-point list.

import { INK, LEVEL, RIVER } from './config.js';
import { geo, nearest } from './geo.js';
import { dayTime, esc, nice, num, parts, pct } from './time.js';
import { damNote, damsHtml } from './dams.js';

const GAUGE_KM = 2;       // a river gauge this close to a control point is taken to measure the same water

/** Channel capacity in m3/s: from the snapshot if it carries it (`points.cap`), else from the section file. */
export function capacity(pp, i) {
  if (pp && Array.isArray(pp.cap) && pp.cap[i] != null) return pp.cap[i];
  return geo.points && geo.points[i] ? geo.points[i].cap : null;
}

/** The river gauge with a measured flow next to a control point, or null: {it, d}. */
export function gaugeNear(snap, pt) {
  return nearest((snap.rivers || []).filter((g) => g.flow_m3s != null), pt.lat, pt.lon, 1, GAUGE_KM)[0] || null;
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

/** "32–143": central peak and the one that 1 scenario in 10 exceeds. */
export const peakTxt = (q) => (q[0] == null ? '—' : `${num(q[0])}${q[1] != null && Math.round(q[1]) > Math.round(q[0]) ? `<span class="dim">–${num(q[1])}</span>` : ''}`);

const signed = (v) => `${v >= 0 ? '+' : '−'}${num(Math.abs(v), 1)}`;
const hourTxt = (d) => esc(dayTime(d).replace(/:00$/, ' h'));

// columns marked "x" are dropped on a narrow screen (they are in the chart of the row)
function head(cols) {
  return `<thead><tr><th></th><th class="wrap">Cauce</th><th title="probabilidad de que el agua supere la capacidad del cauce">Desborde</th>`
    + '<th title="caudal punta: el central y el que supera 1 de cada 10 escenarios">Punta m³/s</th>'
    + '<th class="x" title="caudal que cabe en el cauce">Capacidad</th>'
    + '<th class="x" title="periodo de retorno de la punta central (CAUMAX, CEDEX)">Retorno</th>'
    + `${cols.hover ? '<th class="x" title="altura del agua sobre el borde del cauce">Sobre borde</th>' : ''}`
    + `${cols.gauge ? '<th class="x" title="caudal medido ahora en el aforo que hay junto al punto (dato provisional)">Medido</th>' : ''}<th>Hora</th></tr></thead>`;
}

function row(r, open, snap, h, cols, st) {
  const cap = r.cap == null ? '<span class="dim" title="sin capacidad del cauce: el nivel sale del caudal por km²">—</span>' : num(r.cap);
  const over = r.cap == null ? '<span class="dim">—</span>' : pct(r.pOver);
  const hgt = r.hover[0] != null && r.hover[0] >= 0 ? `${signed(r.hover[0])} m` : '';
  const g = cols.gauge ? gaugeNear(snap, r.pt) : null;
  let out = `<tr class="${open ? 'open' : ''}"><td><span class="lv lv${r.level}">${r.level || '–'}</span></td>
<th class="wrap"><button type="button" class="link" data-pt="${esc(r.pt.id)}" aria-expanded="${open}">${esc(r.pt.stream || r.pt.id)}</button><span class="town">${esc(r.pt.town || '')}</span></th>
<td>${over}</td><td>${peakTxt(r.q)}</td><td class="x">${cap}</td><td class="x">${rpTxt(r.rp[0])}</td>${cols.hover ? `<td class="x">${hgt}</td>` : ''}${cols.gauge ? `<td class="x">${g ? num(g.it.flow_m3s, g.it.flow_m3s < 10 ? 1 : 0) : ''}</td>` : ''}<td>${r.tPeak ? hourTxt(r.tPeak) : ''}</td></tr>`;
  const dn = damNote(snap, st.hz, st.f, r.pt.id);
  if (dn) out += `<tr class="dam-row"><td></td><td colspan="${6 + (cols.hover ? 1 : 0) + (cols.gauge ? 1 : 0)}">${dn}</td></tr>`;
  if (open) out += `<tr class="chart-row"><td colspan="${7 + (cols.hover ? 1 : 0) + (cols.gauge ? 1 : 0)}">${hydrograph(snap, h, r)}</td></tr>`;
  return out;
}

export function renderPoints(root, snap, st) {
  const h = snap && snap.hz[st.hz];
  const rows = pointRows(h, st.f);
  const damsOpen = !!root.querySelector('details.dams-all[open]');
  const dams = snap ? damsHtml(snap, st, damsOpen) : '';
  if (!rows.length && !dams) { root.hidden = true; root.innerHTML = ''; return; }
  root.hidden = false;
  const key = (r) => r.level >= 2 || st.pt === r.pt.id;
  const main = rows.filter(key), rest = rows.filter((r) => !key(r));
  const table = (list) => {
    const cols = { hover: list.some((r) => r.hover[0] != null && r.hover[0] >= 0), gauge: list.some((r) => gaugeNear(snap, r.pt)) };
    return `<div class="scroll"><table class="data points">${head(cols)}<tbody>${list.map((r) => row(r, st.pt === r.pt.id, snap, h, cols, st)).join('')}</tbody></table></div>`;
  };
  // the long list is only built while it is open: closed, it would be 50 rows rebuilt on every frame step
  const open = !!root.querySelector('details.pts-all[open]') || rest.some((r) => st.pt === r.pt.id);
  root.innerHTML = `${rows.length ? `<h2>Cauces</h2>
${main.length ? table(main) : `<div class="lvl calm"><span class="lv lv1">1</span><div><b>Sin riesgo</b></div></div>`}
${rest.length ? `<details class="pts-all"${open ? ' open' : ''}><summary>Todos (${rows.length})</summary>${open ? table(rest) : ''}</details>` : ''}` : ''}
${dams}`;
  const det = root.querySelector('details.pts-all');
  if (det && !open) det.addEventListener('toggle', () => { if (det.open && !det.querySelector('table')) det.insertAdjacentHTML('beforeend', table(rest)); });
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
  if (!pp.q || t.length < 2) return '<p class="dim">Sin hidrograma.</p>';
  const q10 = pp.q[0].map((row) => row[i] ?? 0), q50 = pp.q[1].map((row) => row[i] ?? 0), q90 = pp.q[2].map((row) => row[i] ?? 0);
  const cap = r.cap;
  const gauge = gaugeNear(snap, r.pt);
  const W = 640, H = 240, mL = 54, mR = 12, mT = 16, mB = 34;
  const peak = Math.max(...q90, gauge ? gauge.it.flow_m3s : 0, 1);
  // a capacity far above every scenario would flatten the curve: then it is only written, not drawn to scale
  const capIn = cap && cap <= 4 * peak;
  const top = Math.max(peak, capIn ? cap : 0) * 1.08;
  const t0 = t[0].getTime(), t1 = t[t.length - 1].getTime();
  const X = (d) => mL + ((d - t0) / (t1 - t0)) * (W - mL - mR), Y = (q) => H - mB - (q / top) * (H - mT - mB);
  const path = (arr) => arr.map((q, k) => `${k ? 'L' : 'M'}${X(t[k].getTime()).toFixed(1)} ${Y(q).toFixed(1)}`).join('');
  const band = `${path(q90)}${q10.map((q, k) => `L${X(t[k].getTime()).toFixed(1)} ${Y(q).toFixed(1)}`).reverse().join('')}Z`;
  let g = '';
  // y axis
  const step = niceStep(top, 4);
  for (let v = 0; v <= top; v += step) {
    g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="${INK}" stroke-opacity="${v === 0 ? 1 : 0.15}"/>`
      + `<text x="${mL - 6}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${num(v, step < 1 ? 1 : 0)}</text>`;
  }
  // x axis: local midnights labelled with the day, plus hour ticks when the span is short
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
  if (now > t0 && now < t1) g += `<line x1="${X(now).toFixed(1)}" x2="${X(now).toFixed(1)}" y1="${mT}" y2="${H - mB}" stroke="${INK}" stroke-dasharray="2 3"/>`;
  g += `<path d="${band}" fill="${RIVER}" fill-opacity="0.22"/><path d="${path(q50)}" fill="none" stroke="${RIVER}" stroke-width="2.2"/>`;
  if (capIn) {
    g += `<line x1="${mL}" x2="${W - mR}" y1="${Y(cap).toFixed(1)}" y2="${Y(cap).toFixed(1)}" stroke="${LEVEL[4].color}" stroke-width="1.6" stroke-dasharray="7 4"/>`
      + `<text x="${W - mR - 4}" y="${(Y(cap) - 5).toFixed(1)}" text-anchor="end" fill="${LEVEL[4].color}">capacidad ${num(cap)}</text>`;
  } else if (cap) {
    g += `<text x="${W - mR - 4}" y="${mT + 9}" text-anchor="end" fill="${LEVEL[4].color}">capacidad ${num(cap)} ↑</text>`;
  }
  let gaugeTxt = '';
  if (gauge) {
    // the reading is a few minutes old: it sits on the left edge when the curve starts after it
    const gt = Math.min(Math.max(Date.parse(gauge.it.t_utc) || t0, t0), t1);
    g += `<rect x="${(X(gt) - 4).toFixed(1)}" y="${(Y(gauge.it.flow_m3s) - 4).toFixed(1)}" width="8" height="8" fill="${INK}"/>`;
    gaugeTxt = ` · ■ ${esc(nice(gauge.it.name))} ${num(gauge.it.flow_m3s, gauge.it.flow_m3s < 10 ? 1 : 0)} m³/s, ${esc(dayTime(new Date(gauge.it.t_utc)))}`;
  }
  const label = `Caudal previsto de ${r.pt.stream} en ${r.pt.town}. Punta central ${num(r.q[0])} metros cúbicos por segundo${r.q[1] != null ? `, hasta ${num(r.q[1])} en 1 de cada 10 escenarios` : ''}; capacidad del cauce ${cap ? num(cap) : 'desconocida'}.`;
  return `<figure class="hydro">
<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}" preserveAspectRatio="xMidYMid meet">${g}</svg>
<figcaption>m³/s · banda 10–90 %${gaugeTxt}</figcaption>
</figure>`;
}
