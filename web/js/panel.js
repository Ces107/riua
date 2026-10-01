// The side panel. Without a selection: where and when (zones, ravines, official warnings).
// With a place: its level, the strip of all frames, probabilities, rain, upstream, warnings, details.

import { AEMET_URL, HORIZONS, LEVEL } from './config.js';
import { cellAt, cellCentre, cellFrame, mmOrNull } from './data.js';
import { basinAt, cellBasin, cellZone, distKm, downstreamChain, geo, nearest, zoneAt } from './geo.js';
import { auditHtml, cellExport, thresholdsFor } from './auditview.js';
import { pointRows, rpTxt } from './points.js';
import { dayTime, esc, frameLabel, num, parts, pct } from './time.js';

export const lv = (L) => `<span class="lv lv${L}">${L === 0 ? '–' : L}</span>`;
const SOURCE = { saih_chj: 'SAIH Júcar', saih_segura: 'SAIH Segura', saih_ebro: 'SAIH Ebro', aemet: 'AEMET', avamet: 'AVAMET', meteoclimatic: 'Meteoclimatic' };
const WARN = { yellow: 2, orange: 3, red: 4 };
const WARN_NAME = { yellow: 'amarillo', orange: 'naranja', red: 'rojo' };
const SEG = { now: 'Ahora', mid: '48 h', long: 'Días 2–7' };
const mm0 = (v) => (v == null ? '—' : num(v, v < 10 ? 1 : 0));

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

const short = (d) => { const p = parts(d); return `${p.wd} ${p.hh} h`; };

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
    const t = String(w.text || '');
    const m1 = /una hora:\s*(\d+)/i.exec(t), m12 = /12 horas:\s*(\d+)/i.exec(t);
    if (m1) g.h1 = Number(m1[1]);
    if (m12) g.h12 = Number(m12[1]);
    groups.set(key, g);
  }
  return [...groups.values()].sort((a, b) => b.L - a.L || Date.parse(a.onset) - Date.parse(b.onset));
}

function warnRow(g, withZone) {
  const when = `${g.onset ? dayTime(new Date(g.onset)) : ''} – ${g.expires ? dayTime(new Date(g.expires)) : ''}`;
  const amounts = [g.h1 != null ? `${g.h1} mm/1 h` : null, g.h12 != null ? `${g.h12} mm/12 h` : null].filter(Boolean).join(' · ');
  return `<li><div class="row"><span class="aw lv${g.L}" title="Aviso ${WARN_NAME[g.level]}"></span>
<span class="t">${withZone ? esc(g.zone || '') : `Aviso ${WARN_NAME[g.level]}`}<small>${esc(when)}</small></span><span class="r">${esc(amounts)}</span></div></li>`;
}

// ---- no selection: where and when ------------------------------------------------------------------

function summary(snap) {
  const zones = new Map();      // zone index -> {L, hz, f, d0}
  for (const key of HORIZONS) {
    const h = snap.hz[key];
    if (!h || !geo.cells) continue;
    for (let n = 0; n < h.N; n++) {
      const L = h.maxLevel[n];
      if (L < 2) continue;
      const zi = geo.cells.zone[snap.nToCell[n]];
      if (zi === 255) continue;
      const fr = h.frames[h.maxFrame[n]];
      const z = zones.get(zi);
      if (!z || L > z.L || (L === z.L && fr.d0 < z.d0)) zones.set(zi, { L, hz: key, fr, d0: fr.d0, zi });
    }
  }
  const zrows = [...zones.values()].sort((a, b) => b.L - a.L || a.d0 - b.d0).slice(0, 8).map((z) => {
    const meta = geo.zones.find((x) => x.code === geo.cells.zoneCodes[z.zi]);
    const ll = meta && meta.label && meta.label[0] != null ? `${meta.label[1]},${meta.label[0]}` : '';
    return `<li><button type="button" data-ll="${ll}" data-hz="${z.hz}">${lv(z.L)}<span class="t">${esc(geo.cells.zoneNames[z.zi])}</span><span class="r">${esc(frameLabel(z.hz, z.fr))}</span></button></li>`;
  }).join('');

  let prow = '';
  const seen = new Set();
  for (const key of ['now', 'mid']) {
    const h = snap.hz[key];
    for (const r of pointRows(h, 'max')) {
      if (r.level < 2 || seen.has(r.pt.id)) continue;
      seen.add(r.pt.id);
      prow += `<li><button type="button" data-pt="${esc(r.pt.id)}">${lv(r.level)}<span class="t">${esc(r.pt.stream)}<small>${esc(r.pt.town)}</small></span>`
        + `<span class="r">${r.rp[0] != null && r.rp[0] >= 2 ? rpTxt(r.rp[0]) : r.cap != null && r.pOver != null ? pct(r.pOver) : ''}${r.tPeak ? `<br>${esc(short(r.tPeak))}` : ''}</span></button></li>`;
    }
  }
  const warns = activeWarnings(snap);
  const byZone = new Map();
  for (const g of warns) if (!byZone.has(g.zc) || byZone.get(g.zc).L < g.L) byZone.set(g.zc, g);
  const wrow = [...byZone.values()].sort((a, b) => b.L - a.L).map((g) => warnRow(g, true)).join('');

  return `${zrows ? `<h3>Zonas</h3><ul class="rows">${zrows}</ul>` : `<div class="lvl">${lv(1)}<div><b>Sin riesgo</b><span class="num dim">7 días</span></div></div>`}
${prow ? `<h3>Cauces</h3><ul class="rows">${prow}</ul>` : ''}
<h3>Avisos AEMET</h3>${wrow ? `<ul class="rows">${wrow}</ul>` : '<p class="dim">Ninguno de lluvia.</p>'}
<p class="num dim" style="margin-top:.4rem"><a href="${AEMET_URL}">aemet.es</a></p>`;
}

// ---- place ----------------------------------------------------------------------------------------

function strip(ctx, st) {
  let out = '<div class="strip">';
  for (const key of HORIZONS) {
    const h = ctx.snap.hz[key];
    out += `<div class="seg"><span>${SEG[key]}</span><div class="cells">`;
    if (h) {
      for (let f = 0; f < h.F; f++) {
        const L = h.level[f * h.N + ctx.n];
        const on = key === st.hz && (st.f === f || (st.f === 'max' && ctx.f === f));
        out += `<button type="button" class="lv${L}${on ? ' on' : ''}" data-hz="${key}" data-f="${f}" title="${esc(frameLabel(key, h.frames[f]))}: ${LEVEL[L].name}" aria-label="${esc(frameLabel(key, h.frames[f]))}: nivel ${L}">${L}</button>`;
      }
    }
    out += '</div></div>';
  }
  return `${out}</div>`;
}

function bars(ctx, cf) {
  const tau = ctx.h.tau;
  return `<div class="bars">${[2, 3, 4, 5].map((L, k) => {
    const p = cf.p[k] ?? 0, t = Number(tau[String(L)]);
    return `${lv(L)}<div class="bar" title="mínimo para dar el nivel: ${pct(t)}"><i class="lv${L}" style="width:${(p * 100).toFixed(1)}%"></i><b style="left:${(t * 100).toFixed(1)}%"></b></div>`
      + `<span class="num${p >= t ? '' : ' no'}">${pct(p)}</span>`;
  }).join('')}</div>`;
}

function rain(ctx, cf) {
  const thr = thresholdsFor(ctx.snap, ctx.zone);
  const o = ctx.snap.obs;
  const fallen = o && o.o1.length ? [mmOrNull(o.o1[ctx.n]), mmOrNull(o.o12[ctx.n])] : [null, null];
  return `<table class="data"><thead><tr><th>mm</th><th>1 h</th><th>12 h</th></tr></thead><tbody>
<tr><th>típica</th><td>${mm0(cf.e1[0])}</td><td>${mm0(cf.e12[0])}</td></tr>
<tr><th title="1 de cada 10 escenarios la supera">alta</th><td>${mm0(cf.e1[1])}</td><td>${mm0(cf.e12[1])}</td></tr>
<tr><th>umbral rojo</th><td>${num(thr.t1[2])}</td><td>${num(thr.t12[2])}</td></tr>
<tr><th>ya caída</th><td>${mm0(fallen[0])}</td><td>${mm0(fallen[1])}</td></tr></tbody></table>`;
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
  return `<li><div class="row">${lv(L)}<span class="t">${up >= 0.5 ? 'Agua desde aguas arriba' : 'Su cuenca'}<small>${esc(b.name)}</small></span><span class="r">${esc(frameLabel(ctx.hzKey, ctx.h.frames[f]))}</span></div></li>`;
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
    + `<span class="r">${best.q[0] != null ? `${num(best.q[0])} m³/s` : ''}${best.rp[0] != null ? `<br>${rpTxt(best.rp[0])}` : ''}</span></button></li>`;
}

function zoneWarnings(ctx) {
  if (!ctx.zone) return '';
  const ws = activeWarnings(ctx.snap).filter((g) => g.zc === ctx.zone.code);
  return ws.length ? `<h3>Aviso AEMET</h3><ul class="rows">${ws.map((g) => warnRow(g, false)).join('')}</ul>` : '';
}

function gauges(ctx) {
  const near = nearest(ctx.snap.gauges || [], ctx.lat, ctx.lon, 4, 25);
  if (!near.length) return '<p class="dim">Ninguno a menos de 25 km.</p>';
  const v = (x) => (x == null ? '—' : num(x, 1));
  return `<div class="scroll"><table class="data"><thead><tr><th>mm</th><th>km</th><th>1 h</th><th>12 h</th><th>24 h</th></tr></thead><tbody>${
    near.map(({ it, d }) => `<tr><th class="wrap">${esc(it.name)}<br><span class="dim">${esc(SOURCE[it.source] || it.source || '')} · ${it.t_utc ? esc(dayTime(new Date(it.t_utc))) : ''}</span></th><td>${num(d, 1)}</td><td>${v(it.p_1h)}</td><td>${v(it.p_12h)}</td><td>${v(it.p_24h)}</td></tr>`).join('')
  }</tbody></table></div>`;
}

export function renderPanel(root, snap, st, hooks) {
  const ctx = context(snap, st);
  if (!ctx) { root.innerHTML = summary(snap); return null; }
  const head = `<h2>${esc(placeTitle(ctx))}</h2><p class="sub">${ctx.zone ? esc(ctx.zone.name) : ''}</p>`;
  if (ctx.n < 0) { root.innerHTML = `${head}<p class="dim" style="margin-top:.6rem">Fuera de la zona calculada.</p>`; return ctx; }
  if (!ctx.h) { root.innerHTML = `${head}<p class="dim" style="margin-top:.6rem">Sin datos para este plazo.</p>`; return ctx; }

  const cf = cellFrame(ctx.h, ctx.f, ctx.n);
  const L = cf.level, meta = LEVEL[L];
  const extra = [upstreamRow(ctx, st, L), ravineRow(ctx, st)].join('');
  root.innerHTML = `${head}
<div class="lvl">${lv(L).replace('class="lv', 'class="lv big')}<div><b>${meta.name}${meta.aemet ? ` <span class="dim" style="font-weight:400">${meta.aemet}</span>` : ''}</b>
<span class="num">${esc(frameLabel(ctx.hzKey, ctx.h.frames[ctx.f]))}</span></div></div>
${strip(ctx, st)}
<h3>Probabilidad</h3>${bars(ctx, cf)}
<h3>Lluvia</h3>${rain(ctx, cf)}
${extra ? `<h3>Cauces</h3><ul class="rows">${extra}</ul>` : ''}
${zoneWarnings(ctx)}
<details class="g"><summary>Pluviómetros</summary><div>${gauges(ctx)}</div></details>
<details class="audit"><summary>Cálculo</summary><div class="audit-body"></div></details>
<details class="dl"><summary>Datos</summary><div><button type="button" class="link" data-act="json">Descargar JSON de esta celda</button> <span class="dim" data-role="json-msg"></span></div></details>`;

  const det = root.querySelector('details.audit');
  const body = det.querySelector('.audit-body');
  const fill = async () => {
    if (!det.open || body.dataset.done) return;
    body.dataset.done = '1';
    body.innerHTML = '<p class="dim">…</p>';
    try { body.innerHTML = await auditHtml(ctx); } catch (e) { body.innerHTML = `<p class="warn">${esc(e.message)}</p>`; delete body.dataset.done; }
  };
  det.addEventListener('toggle', () => { hooks.onAudit(det.open); fill(); });
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
