// The side panel. Without a selection: the warning zones (our level next to the official AEMET one).
// With a place: its level and when, the strip of all frames, probabilities, rain fallen and expected,
// the ravines that affect it, the official warnings, and the collapsed details.

import { AEMET_URL, HORIZONS, LEVEL } from './config.js';
import { cellAt, cellCentre, cellFrame, levelRun, mmOrNull } from './data.js';
import { basinAt, cellBasin, cellZone, distKm, downstreamChain, geo, nearest, zoneAt } from './geo.js';
import { auditHtml, cellExport } from './auditview.js';
import { firstRun } from './headline.js';
import { peakTxt, pointRows, rpTxt } from './points.js';
import { dayTime, esc, frameLabel, nice, num, parts, pct, span } from './time.js';

export const lv = (L) => `<span class="lv lv${L}">${L === 0 ? '–' : L}</span>`;
// only networks whose data may be shown (AVAMET and Meteoclimatic are CC BY-NC-ND: never on the page)
const SOURCE = { saih_chj: 'SAIH Júcar', saih_segura: 'SAIH Segura', saih_ebro: 'SAIH Ebro', aemet: 'AEMET' };
const WARN = { yellow: 2, orange: 3, red: 4 };
const WARN_NAME = { yellow: 'amarillo', orange: 'naranja', red: 'rojo' };
const SEG = { now: 'Ahora', mid: '48 h', long: 'Días 2–7' };
const mm0 = (v) => (v == null ? '—' : num(v, v >= 0.05 && v < 9.95 ? 1 : 0));

export function context(snap, st) {
  if (!snap || !st.sel) return null;
  const { lat, lon } = st.sel;
  const n = cellAt(snap, lat, lon);
  const place = nearest(geo.places, lat, lon, 1)[0] || null;
  const c = n >= 0 ? cellCentre(snap, n) : null;
  const zone = zoneAt(lon, lat) || (c ? cellZone(c.c) : null);
  const basin = basinAt(lon, lat) || (c ? cellBasin(c.c) : null);
  const h = snap.hz[st.hz] || null;
  let f = null, isMax = false;
  if (h && n >= 0) {
    if (st.f === 'max') { f = h.maxFrame[n]; isMax = true; } else f = Math.min(st.f, h.F - 1);
  }
  return { snap, lat, lon, n, cell: c, place, zone, basin, hzKey: st.hz, h, f, isMax };
}

export function placeTitle(ctx) {
  if (!ctx.place) return `${num(ctx.lat, 2)}° N, ${num(Math.abs(ctx.lon), 2)}° ${ctx.lon < 0 ? 'O' : 'E'}`;
  return ctx.place.d <= 2.5 ? ctx.place.it.name : `Cerca de ${ctx.place.it.name}`;
}

// ---- official warnings, merged by zone / level / period -------------------------------------------

function activeWarnings(snap) {
  const now = Date.now();
  const groups = new Map();
  for (const w of snap.warnings || []) {
    if (w.expires && Date.parse(w.expires) <= now) continue;
    const L = WARN[w.level];
    if (!L) continue;
    const zc = String(w.zone_code || '').replace(/C$/, '');
    const key = `${zc}|${w.level}|${w.onset}|${w.expires}`;
    const g = groups.get(key) || { zc, zone: w.zone_name, level: w.level, L, onset: w.onset, expires: w.expires, h1: null, h12: null };
    const p = w.params || {};
    const t = String(w.text || '');
    const m1 = /una hora:\s*(\d+)/i.exec(t), m12 = /12 horas:\s*(\d+)/i.exec(t);
    if (p.accum_hours === 1 && p.value != null) g.h1 = Number(p.value); else if (m1) g.h1 = Number(m1[1]);
    if (p.accum_hours === 12 && p.value != null) g.h12 = Number(p.value); else if (m12) g.h12 = Number(m12[1]);
    groups.set(key, g);
  }
  return [...groups.values()].sort((a, b) => b.L - a.L || Date.parse(a.onset) - Date.parse(b.onset));
}

/** "hoy 18–24 h", "hoy 18 h – mañana 02 h": AEMET ends its warnings at hh:59:59, shown as the next full hour. */
function warnWhen(g) {
  const txt = (iso) => {
    const ms = Date.parse(iso);
    if (!Number.isFinite(ms)) return null;
    const d = new Date(Math.round(ms / 60000) * 60000), p = parts(d);
    return { day: dayTime(d).replace(/ \d\d:\d\d$/, ''), h: p.mm === '00' ? p.hh : `${p.hh}:${p.mm}` };
  };
  const a = txt(g.onset), b = txt(g.expires);
  if (!a || !b) return a ? `desde ${a.day} ${a.h} h` : b ? `hasta ${b.day} ${b.h} h` : '';
  return a.day === b.day ? `${a.day} ${a.h}–${b.h} h` : `${a.day} ${a.h} h – ${b.day} ${b.h} h`;
}

const swatch = (g) => (g ? `<span class="aw lv${g.L}" role="img" aria-label="Aviso AEMET ${WARN_NAME[g.level]}" title="Aviso AEMET ${WARN_NAME[g.level]}"></span>` : '<span class="aw none" aria-hidden="true"></span>');

function warnRow(g) {
  const amounts = [g.h1 != null ? `${num(g.h1)} mm/1 h` : null, g.h12 != null ? `${num(g.h12)} mm/12 h` : null].filter(Boolean).join(' · ');
  return `<li><div class="row">${swatch(g)}<span class="t">${esc(warnWhen(g))}</span><span class="r">${esc(amounts)}</span></div></li>`;
}

// ---- no selection: the zones -----------------------------------------------------------------------

/** Is the centre of cell n inside the polygon of warning zone zi (the cell is assigned to it by majority)? */
function centreIn(snap, n, zi) {
  const c = cellCentre(snap, n), z = zoneAt(c.lon, c.lat);
  return !!z && z.code === geo.cells.zoneCodes[zi];
}

// The list does not depend on the frame or horizon in view: it is worked out once per snapshot and minute
// (the minute, because frames and warnings expire), not on every step of the slider.
const summaries = new WeakMap();
function summary(snap) {
  const key = `${Math.floor(Date.now() / 60000)}|${geo.cells ? 1 : 0}|${geo.zones.length}`;
  const had = summaries.get(snap);
  if (had && had.key === key) return had.html;
  const html = buildSummary(snap);
  summaries.set(snap, { key, html });
  return html;
}

function buildSummary(snap) {
  // per warning zone: its highest level over the three horizons and the frames that hold it
  const zones = new Map();      // zone index -> {L, items}
  let outside = 0;
  for (const key of HORIZONS) {
    const h = snap.hz[key];
    if (!h || !geo.cells) continue;
    for (let f = 0; f < h.F; f++) {
      const item = { hz: key, f, fr: h.frames[f] };
      for (let n = 0; n < h.N; n++) {
        const L = h.level[f * h.N + n];
        if (L < 2) continue;
        const zi = geo.cells.zone[snap.nToCell[n]];
        if (zi === 255) { if (L > outside) outside = L; continue; }
        // the row leads to the worst cell of the zone: the one with the highest probability of that level
        const p = h.p.length ? h.p[(L - 2) * h.F * h.N + f * h.N + n] : 0;
        const z = zones.get(zi);
        // (a cell on a zone border may have its centre in the neighbouring zone: such a cell is only used
        // while there is no better one, so that the place opened says the same zone as the row)
        if (!z || L > z.L) zones.set(zi, { L, items: [item], n, p, hz: key, own: centreIn(snap, n, zi) });
        else if (L === z.L) {
          if (z.items[z.items.length - 1] !== item) z.items.push(item);
          if (p > z.p || !z.own) {
            const own = centreIn(snap, n, zi);
            if (own ? (!z.own || p > z.p) : (!z.own && p > z.p)) { z.p = p; z.n = n; z.hz = key; z.own = own; }
          }
        }
      }
    }
  }
  const official = new Map();   // zone code -> highest warning in force or announced
  for (const g of activeWarnings(snap)) if (!official.has(g.zc) || official.get(g.zc).L < g.L) official.set(g.zc, g);

  const rows = [];
  const codes = geo.cells ? geo.cells.zoneCodes : [];
  codes.forEach((code, zi) => {
    const z = zones.get(zi), g = official.get(code);
    if (!z && !g) return;
    const run = z ? firstRun(z.items) : null;
    rows.push({ code, name: geo.cells.zoneNames[zi], L: z ? z.L : 1, g, run, t: run ? run[0].fr.d0.getTime() : Infinity, n: z ? z.n : -1, hz: z ? z.hz : null });
  });
  for (const [code, g] of official) if (!codes.includes(code)) rows.push({ code, name: g.zone || code, L: 1, g, run: null, t: Infinity });
  rows.sort((a, b) => b.L - a.L || a.t - b.t || (b.g ? b.g.L : 0) - (a.g ? a.g.L : 0) || a.name.localeCompare(b.name, 'es'));

  const html = rows.map((z) => {
    const meta = geo.zones.find((x) => x.code === z.code);
    const worst = z.n >= 0 ? cellCentre(snap, z.n) : null;
    const ll = worst ? `${worst.lat.toFixed(3)},${worst.lon.toFixed(3)}` : meta && meta.label && meta.label[0] != null ? `${meta.label[1]},${meta.label[0]}` : '';
    const when = z.run ? span(z.run[0].hz, z.run[0].fr.d0, z.run[z.run.length - 1].fr.d1) : '';
    const inner = `${lv(z.L)}<span class="t">${esc(z.name)}</span><span class="r">${esc(when)}</span>${swatch(z.g)}`;
    return ll ? `<li><button type="button" data-ll="${ll}"${z.hz ? ` data-hz="${z.hz}"` : ''}>${inner}</button></li>` : `<li><div class="row">${inner}</div></li>`;
  }).join('') + (outside ? `<li><div class="row">${lv(outside)}<span class="t">Aguas arriba</span><span class="r"></span>${swatch(null)}</div></li>` : '');

  const aemet = `<a href="${AEMET_URL}">AEMET</a>`;
  return `${html
    ? `<h3 class="cols"><span>Zonas</span>${aemet}</h3><ul class="rows zones">${html}</ul>`
    : `<div class="lvl calm">${lv(1)}<div><b>Sin riesgo</b><span class="num dim">7 días · ${aemet} sin avisos</span></div></div>`}
${atmosphere(snap)}`;
}

// ---- atmosphere: the ingredients of heavy rain along the coast, every 6 h ---------------------------

const WIND = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
const card = (deg) => (deg == null ? '' : WIND[Math.round(deg / 45) % 8]);

function atmosphere(snap) {
  const d = snap.drivers;
  if (!d || !d.series || !Array.isArray(d.times)) return '';
  const now = Date.now();
  const idx = [];
  d.times.forEach((t, k) => { const ms = Date.parse(t); if (ms >= now - 3 * 3600e3 && idx.length < 6 && (!idx.length || ms - Date.parse(d.times[idx[idx.length - 1]]) >= 6 * 3600e3 - 1)) idx.push(k); });
  if (!idx.length) return '';
  const g = (k) => (d.series[k] && Array.isArray(d.series[k].strip) ? d.series[k].strip : null);
  const row = (label, title, f) => `<tr><th title="${title}">${label}</th>${idx.map((k) => `<td>${f(k)}</td>`).join('')}</tr>`;
  const val = (k, dec = 0) => (i) => { const a = g(k); return a && a[i] != null ? num(a[i], dec) : '—'; };
  const wind = (ks, kd) => (i) => { const a = g(ks), b = g(kd); return a && a[i] != null ? `${num(a[i])} ${card(b ? b[i] : null)}` : '—'; };
  const head = idx.map((k) => { const p = parts(new Date(d.times[k])); return `<th>${p.wd} ${p.hh}</th>`; }).join('');
  return `<details class="atm"><summary>Atmósfera</summary><div class="scroll"><table class="data"><thead><tr><th></th>${head}</tr></thead><tbody>
${row('Agua precipitable mm', 'vapor de agua en la columna (litoral, percentil 90)', val('pwat'))}
${row('Transporte IVT', 'transporte integrado de vapor, kg/m/s, y de dónde viene', wind('ivt', 'ivt_dir'))}
${row('CAPE J/kg', 'energía convectiva de la parcela más inestable', val('mucape'))}
${row('Nube cálida m', 'espesor entre la base de la nube y la isocero: lluvia eficiente por encima de 3000', val('wcd'))}
${row('Chorro bajo m/s', 'viento máximo en capas bajas y de dónde viene', wind('llj_speed', 'llj_dir'))}
${row('Corfidi m/s', 'velocidad del vector de propagación: por debajo de 5, tormentas casi estacionarias', val('corfidi_up', 1))}
</tbody></table></div></details>`;
}

// ---- place ----------------------------------------------------------------------------------------

function strip(ctx, st) {
  const now = Date.now();
  let out = '<div class="strip">';
  for (const key of HORIZONS) {
    const h = ctx.snap.hz[key];
    // a horizon may be missing from an update (the ensemble did not arrive): its place stays, empty and dashed
    out += `<div class="seg"><span>${SEG[key]}</span><div class="cells${h ? '' : ' none'}"${h ? '' : ' title="sin datos en esta actualización"'}>`;
    if (h) {
      for (let f = 0; f < h.F; f++) {
        const L = h.level[f * h.N + ctx.n];
        const on = key === st.hz && (st.f === f || (st.f === 'max' && ctx.f === f));
        const past = h.frames[f].d1 <= now;
        out += `<button type="button" class="lv${L}${on ? ' on' : ''}${past ? ' past' : ''}" data-hz="${key}" data-f="${f}"${on ? ' aria-current="true"' : ''} title="${esc(frameLabel(key, h.frames[f]))}: ${LEVEL[L].name}" aria-label="${esc(frameLabel(key, h.frames[f]))}: nivel ${L}, ${LEVEL[L].name}">${L}</button>`;
      }
    }
    out += '</div></div>';
  }
  return `${out}</div>`;
}

function bars(ctx, cf) {
  const tau = ctx.h.tau, cap = ctx.h.levelCap;
  return `<div class="bars">${[2, 3, 4, 5].map((L, k) => {
    const p = cf.p[k] ?? 0, t = Number(tau[String(L)]);
    // a horizon may stop at a level (days 2–7 at 3): above it the probability is shown, but never as reached
    const capped = cap != null && L > cap;
    return `${lv(L)}<div class="bar" title="${capped ? `en este plazo el nivel no pasa de ${cap}` : `mínimo para dar el nivel: ${pct(t)}`}"><i class="lv${L}" style="width:${(p * 100).toFixed(1)}%"></i>${capped ? '' : `<b style="left:${(t * 100).toFixed(1)}%"></b>`}</div>`
      + `<span class="num${p >= t && !capped ? '' : ' no'}">${pct(p)}</span>`;
  }).join('')}</div>`;
}

/**
 * Rain at this point, in the order of what the level looks at: what is expected (the level is about what
 * is still to come), the amounts that are compared with the thresholds, and last, as a plain fact, what
 * has already fallen (it only counts towards a level while more rain is on its way).
 */
function rain(ctx, cf) {
  const h = ctx.h, obs = ctx.snap.obs;
  let out = '';
  // expected: the whole period in the "máximo" view, the frame otherwise
  let a = cf.acc, label = frameLabel(ctx.hzKey, h.frames[ctx.f]);
  if (ctx.isMax && h.accTotal.length === 2 * h.N) {
    a = [mmOrNull(h.accTotal[ctx.n]), mmOrNull(h.accTotal[h.N + ctx.n])];
    label = { now: '6 h', mid: '6–48 h', long: 'días 2–7' }[ctx.hzKey] || span(ctx.hzKey, h.frames[0].d0, h.frames[h.F - 1].d1);
  }
  if (a && a[0] != null) {
    const hi = a[1] != null && a[1] > a[0] + 0.5 ? `<span class="hi" title="1 de cada 10 escenarios lo supera">hasta ${mm0(a[1])}</span>` : '';
    out += `<p class="rain"><b>${mm0(a[0])} mm</b>${hi}<span class="num dim">prevista · ${esc(label)}</span></p>`;
  }
  // what the level is decided on: the largest 1-h and 12-h amounts within a few km, central scenario
  if (cf.level >= 2 && cf.e12[0] != null) {
    const km = ctx.hzKey === 'now' ? 6 : 12;
    const one = h.frames[ctx.f].has_1h && cf.e1[0] != null ? `${mm0(cf.e1[0])} mm/1 h · ` : '';
    out += `<p class="rain near" title="la mayor lluvia a menos de ${km} km en el escenario central: es lo que se compara con los umbrales"><span class="hi">${one}${mm0(cf.e12[0])} mm/12 h</span><span class="num dim">máx. a ${km} km</span></p>`;
  }
  if (obs && obs.o12.length > ctx.n) {
    const o12 = mmOrNull(obs.o12[ctx.n]), o24 = obs.o24.length > ctx.n ? mmOrNull(obs.o24[ctx.n]) : null;
    if (o24 != null && o24 >= 0.95) {
      const until = obs.last ? ` hasta ${dayTime(new Date(obs.last))}` : '';
      out += `<p class="rain past" title="lluvia medida (radar y pluviómetros)${esc(until)}"><span class="hi">${mm0(o12)} mm${o24 > o12 + 0.95 ? ` · ${mm0(o24)} en 24 h` : ''}</span><span class="num dim">ya caída · 12 h</span></p>`;
    }
  }
  return out;
}

function worstIdx(levelFX, pFX, idx) {
  let best = 0, bf = 0, bp = -1;
  for (let f = 0; f < levelFX.length; f++) {
    const L = levelFX[f][idx] || 0, p = pFX ? (pFX[Math.max(L - 2, 0)][f][idx] ?? 0) : 0;
    if (L > best || (L === best && p > bp)) { best = L; bf = f; bp = p; }
  }
  return bf;
}

function upstreamRow(ctx, st, cellLevel) {
  const b = ctx.basin, bp = ctx.h && ctx.h.basins;
  if (!b || !bp || !bp.level || !bp.level.length) return '';
  const f = st.f === 'max' ? worstIdx(bp.level, bp.p, b.idx) : Math.min(st.f, bp.level.length - 1);
  const L = bp.level[f][b.idx] || 0;
  const up = bp.upstream && bp.upstream[f] ? bp.upstream[f][b.idx] : 0;
  if (L < 2 || (L <= cellLevel && up < 0.5)) return '';
  return `<li><div class="row">${lv(L)}<span class="t">${up >= 0.5 ? 'Agua de aguas arriba' : 'Su cuenca'}<small>${esc(b.name)}</small></span><span class="r">${esc(frameLabel(ctx.hzKey, ctx.h.frames[f]))}</span></div></li>`;
}

function ravineRow(ctx, st) {
  const h = ctx.h;
  if (!geo.points || !h || !h.points) return '';
  for (const p of geo.points) if (p._basin === undefined) p._basin = basinAt(p.lon, p.lat);
  const chain = ctx.basin ? downstreamChain(ctx.basin).map((b) => b.id) : [];
  const rows = pointRows(h, st.f);
  let best = rows.filter((r) => r.pt._basin && chain.includes(r.pt._basin.id)).sort((a, b) => chain.indexOf(a.pt._basin.id) - chain.indexOf(b.pt._basin.id))[0];
  if (!best) best = rows.map((r) => ({ r, d: distKm(ctx.lat, ctx.lon, r.pt.lat, r.pt.lon) })).filter((x) => x.d <= 12).sort((a, b) => a.d - b.d).map((x) => x.r)[0];
  if (!best) return '';
  return `<li><button type="button" data-pt="${esc(best.pt.id)}">${lv(best.level)}<span class="t">${esc(best.pt.stream)}<small>${esc(best.pt.town)}</small></span>`
    + `<span class="r">${best.q[0] != null ? `${peakTxt(best.q)} m³/s` : ''}${best.rp[0] != null && best.rp[0] >= 2 ? `<br>${rpTxt(best.rp[0])}` : ''}</span></button></li>`;
}

function gauges(ctx) {
  const list = (ctx.snap.gauges || []).filter((g) => SOURCE[g.source]);
  const near = nearest(list, ctx.lat, ctx.lon, 4, 25);
  if (!near.length) return '<p class="dim">Ninguno a menos de 25 km.</p>';
  const v = (x) => (x == null ? '—' : num(x, 1));
  return `<div class="scroll"><table class="data"><thead><tr><th>mm</th><th>km</th><th>1 h</th><th>12 h</th><th>24 h</th></tr></thead><tbody>${
    near.map(({ it, d }) => `<tr><th class="wrap">${esc(nice(it.name))}<br><span class="dim">${esc(SOURCE[it.source])}${it.t_utc ? ` · ${esc(dayTime(new Date(it.t_utc)))}` : ''}</span></th><td>${num(d, 1)}</td><td>${v(it.p_1h)}</td><td>${v(it.p_12h)}</td><td>${v(it.p_24h)}</td></tr>`).join('')
  }</tbody></table></div>`;
}

const shown = new WeakMap();           // panel element -> the zone list it is showing

export function renderPanel(root, snap, st, hooks) {
  const ctx = context(snap, st);
  const wasOpen = new Set([...root.querySelectorAll('details[open]')].map((d) => d.className));
  const reopen = () => root.querySelectorAll('details').forEach((d) => { if (wasOpen.has(d.className) && !d.classList.contains('audit')) d.open = true; });
  if (!ctx) {
    const html = summary(snap);
    if (shown.get(root) !== html) { root.innerHTML = html; reopen(); shown.set(root, html); }   // untouched when nothing changed
    return null;
  }
  shown.delete(root);
  const head = `<div class="ttl"><h2>${esc(placeTitle(ctx))}</h2><button type="button" class="x" data-act="back" aria-label="Quitar la selección" title="Quitar la selección">×</button></div>
<p class="sub">${ctx.zone ? esc(ctx.zone.name) : ''}</p>`;
  if (ctx.n < 0) { root.innerHTML = `${head}<p class="dim gap">Fuera de la zona calculada.</p>`; return ctx; }
  if (!ctx.h) { root.innerHTML = `${head}<p class="dim gap">Sin datos para este plazo.</p>`; return ctx; }

  const cf = cellFrame(ctx.h, ctx.f, ctx.n);
  const L = cf.level, meta = LEVEL[L];
  // when: in the "máximo" view, the whole unbroken period at that level; otherwise the frame
  let when = frameLabel(ctx.hzKey, ctx.h.frames[ctx.f]);
  if (ctx.isMax) { const [a, b] = levelRun(ctx.h, ctx.n, ctx.f); when = span(ctx.hzKey, ctx.h.frames[a].d0, ctx.h.frames[b].d1); }
  const warns = ctx.zone ? activeWarnings(snap).filter((g) => g.zc === ctx.zone.code) : [];
  const extra = [upstreamRow(ctx, st, L), ravineRow(ctx, st)].join('');
  const fallen = rain(ctx, cf);
  root.innerHTML = `${head}
<div class="lvl">${lv(L).replace('class="lv', 'class="lv big')}<div><b>${meta.name}</b><span class="num">${esc(when)}</span></div>${warns.length ? `<a class="off" href="${AEMET_URL}">${swatch(warns[0])}<small>AEMET</small></a>` : ''}</div>
${strip(ctx, st)}
${L > 0 && cf.p.some((p) => p >= 0.005) ? `<h3>Probabilidad</h3>${bars(ctx, cf)}` : ''}
${fallen ? `<h3>Lluvia</h3>${fallen}` : ''}
${extra ? `<h3>Cauces</h3><ul class="rows">${extra}</ul>` : ''}
${warns.length ? `<h3>Avisos AEMET</h3><ul class="rows">${warns.map(warnRow).join('')}</ul>` : ''}
<div class="more">
<details class="g"><summary>Pluviómetros</summary><div>${gauges(ctx)}</div></details>
<details class="audit"><summary>Cálculo</summary><div class="audit-body"></div></details>
<details class="dl"><summary>Datos</summary><div><button type="button" class="link" data-act="json">JSON de esta celda</button> <span class="dim" data-role="json-msg"></span></div></details>
</div>`;
  reopen();

  const det = root.querySelector('details.audit');
  const body = det.querySelector('.audit-body');
  const fill = async () => {
    if (!det.open || body.dataset.done) return;
    body.dataset.done = '1';
    body.innerHTML = '<p class="dim">…</p>';
    try { body.innerHTML = await auditHtml(ctx); } catch (e) { body.innerHTML = `<p class="warn">${esc(e.message)}</p>`; delete body.dataset.done; }
  };
  det.addEventListener('toggle', () => { hooks.onAudit(det.open); fill(); });
  // an ensemble is one row in the scenario table; pressing its name lists its members
  body.addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-grp]');
    if (!b) return;
    const open = b.getAttribute('aria-expanded') !== 'true';
    b.setAttribute('aria-expanded', String(open));
    body.querySelectorAll(`tr[data-of="${b.dataset.grp}"]`).forEach((tr) => { tr.hidden = !open; });
  });
  if (hooks.auditOpen()) { det.open = true; fill(); }

  root.querySelector('[data-act="json"]').addEventListener('click', async (ev) => {
    const msg = root.querySelector('[data-role="json-msg"]');
    ev.target.disabled = true;
    try {
      const blob = new Blob([JSON.stringify(await cellExport(ctx), null, 1)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `riua_${ctx.cell.lat.toFixed(3)}_${ctx.cell.lon.toFixed(3)}_${snap.generatedIso.replace(/[^0-9TZ]/g, '')}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    } catch (e) { msg.textContent = e.message; }
    ev.target.disabled = false;
  });
  return ctx;
}
