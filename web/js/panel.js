// The detail panel of a place: level, timeline, probabilities, expected and observed rain, basin unit,
// control points downstream, official warning, nearby gauges, and the audit (collapsed).

import { AEMET_URL, HORIZONS, HZ_LABEL, HZ_PLAIN, LEVEL } from './config.js';
import { cellAt, cellCentre, cellFrame, mmOrNull } from './data.js';
import { basinAt, cellBasin, cellZone, distKm, downstreamChain, geo, nearest, zoneAt } from './geo.js';
import { auditHtml, cellExport, thresholdsFor } from './auditview.js';
import { dayTime, esc, frameLabel, mmTxt, num, pct, spanLabel } from './time.js';

export const lv = (L, extra = '') => `<span class="lv lv${L}${extra}">${L === 0 ? '–' : L}</span>`;
const SOURCE = { saih_chj: 'SAIH Júcar', saih_segura: 'SAIH Segura', saih_ebro: 'SAIH Ebro', aemet: 'AEMET', avamet: 'AVAMET', meteoclimatic: 'Meteoclimatic' };
const WARN = { yellow: ['amarillo', 2], orange: ['naranja', 3], red: ['rojo', 4] };

/** Everything known about the selected point, independent of what gets drawn. */
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
  if (!ctx.place) return `${num(ctx.lat, 3)}° N, ${num(ctx.lon, 3)}° E`;
  const { it, d } = ctx.place;
  return d <= 2.5 ? it.name : `Cerca de ${it.name}`;
}

/** Frame with the highest level in a [F][X] table (ties: highest probability of that level). */
function worst(levelFX, pFX, idx) {
  let best = 0, bf = 0, bp = -1;
  for (let f = 0; f < levelFX.length; f++) {
    const L = levelFX[f][idx] || 0;
    const p = pFX ? (pFX[Math.max(L - 2, 0)][f][idx] ?? 0) : 0;
    if (L > best || (L === best && p > bp)) { best = L; bf = f; bp = p; }
  }
  return bf;
}

function timeline(ctx, st) {
  let out = '<div class="tl">';
  for (const key of HORIZONS) {
    const h = ctx.snap.hz[key];
    out += `<div class="tl-row"><span class="tl-name">${HZ_LABEL[key].split(' · ')[0]}</span>`;
    if (!h) { out += '<span class="dim">sin datos</span></div>'; continue; }
    out += '<span class="tl-cells">';
    for (let f = 0; f < h.F; f++) {
      const L = h.level[f * h.N + ctx.n];
      const on = key === st.hz && (st.f === f || (st.f === 'max' && ctx.f === f));
      out += `<button type="button" class="lv lv${L}${on ? ' on' : ''}" data-hz="${key}" data-f="${f}" title="${esc(frameLabel(key, h.frames[f]))}: ${LEVEL[L].name}"`
        + ` aria-label="${esc(frameLabel(key, h.frames[f]))}: nivel ${L}, ${LEVEL[L].name}"${on ? ' aria-current="true"' : ''}>${L === 0 ? '–' : L}</button>`;
    }
    out += `</span><span class="tl-span num">${esc(spanLabel(h.frames))}</span></div>`;
  }
  return `${out}</div>`;
}

function probBars(ctx, cf) {
  const tau = ctx.h.tau;
  return `<table class="bars"><tbody>${[2, 3, 4, 5].map((L, k) => {
    const p = cf.p[k], t = Number(tau[String(L)]);
    const ok = p != null && p >= t;
    return `<tr><th scope="row">${lv(L)} ${LEVEL[L].name}</th>
<td class="bar-cell"><div class="bar" role="img" aria-label="Probabilidad ${pct(p)}; hace falta ${pct(t)}">
<span class="fill lv${L}" style="width:${p == null ? 0 : (p * 100).toFixed(1)}%"></span><span class="tau" style="left:${(t * 100).toFixed(1)}%"></span></div></td>
<td class="num">${pct(p)}</td><td class="num need">${ok ? 'llega a' : 'no llega a'} ${pct(t)}</td></tr>`;
  }).join('')}</tbody></table>
<p class="dim">Probabilidad de que la lluvia alcance cada nivel o más. La raya marca la probabilidad que hace falta para darlo: cuanto más grave, menos se exige.</p>`;
}

function rainTable(ctx, cf) {
  const thr = thresholdsFor(ctx.snap, ctx.zone);
  // the amount, and the level whose threshold it reaches (if any)
  const amt = (v, t) => {
    if (v == null) return '—';
    let k = 1;
    t.forEach((x, i) => { if (v >= x) k = i + 2; });
    return `${mmTxt(v)}${k >= 2 ? ` ${lv(k)}` : ''}`;
  };
  const row = (label, e, t) => `<tr><th scope="row">${label}</th><td>${amt(e[0], t)}</td><td>${amt(e[1], t)}</td></tr>`;
  const list = (t) => `${t.map((v) => num(v)).join(' / ')} mm`;
  return `<table class="data"><thead><tr><th scope="col"></th><th scope="col">Típico</th><th scope="col">Escenario alto</th></tr></thead>
<tbody>${row('En&nbsp;1&nbsp;h', cf.e1, thr.t1)}${row('En&nbsp;12&nbsp;h', cf.e12, thr.t12)}</tbody></table>
<p class="num thr">Umbrales de la zona (niveles 2 / 3 / 4 / 5):<br>en 1 h, ${list(thr.t1)}<br>en 12 h, ${list(thr.t12)}</p>
<p class="dim">El número en color indica el umbral que alcanza esa cantidad. Típico: la mitad de las veces llueve menos. Escenario alto: solo se supera 1 de cada 10 veces. Es la mayor cantidad en 1 h y en 12 h dentro del tramo, en el punto más lluvioso de la celda.</p>`;
}

function observed(ctx) {
  const o = ctx.snap.obs;
  if (!o || !o.o1.length) return '<p>No hay análisis de lluvia observada en esta actualización.</p>';
  const until = o.last ? dayTime(new Date(o.last)) : null;
  return `<p><span class="num">Última hora: ${mmTxt(mmOrNull(o.o1[ctx.n]))} · últimas 12 h: ${mmTxt(mmOrNull(o.o12[ctx.n]))} · últimas 24 h: ${mmTxt(mmOrNull(o.o24[ctx.n]))}</span>
<span class="dim">Radar ajustado con pluviómetros${until ? `, hasta ${until}` : ''}${o.hours < 24 ? `; solo hay ${o.hours} h de análisis` : ''}.</span></p>`;
}

function basinBlock(ctx, st) {
  const b = ctx.basin, bp = ctx.h && ctx.h.basins;
  if (!b) return '<p>Este punto no pertenece a ninguna de las cuencas que se siguen.</p>';
  const next = b.next ? geo.basinById.get(b.next) : null;
  let out = `<p><strong>${esc(b.name)}</strong> <span class="num">(${num(b.area)} km²${b.upArea > b.area * 1.02 ? `; ${num(b.upArea)} km² con todo lo que desagua en ella` : ''})</span>.
${next ? `Desagua hacia: ${esc(next.name)}.` : 'Es el último tramo antes del mar o de una cuenca cerrada.'}</p>`;
  if (!bp || !bp.level || !bp.level.length) return `${out}<p>Esta actualización no trae el cálculo por cuencas.</p>`;
  const f = st.f === 'max' ? worst(bp.level, bp.p, b.idx) : Math.min(st.f, bp.level.length - 1);
  const L = bp.level[f][b.idx] || 0;
  const g = (a, i) => (a && a[i] && a[i][f] ? a[i][f][b.idx] : null);
  const up = bp.upstream && bp.upstream[f] ? bp.upstream[f][b.idx] : null;
  out += `<p>${lv(L)} <strong>Nivel ${L} · ${LEVEL[L].name}</strong> en la cuenca, ${esc(frameLabel(ctx.hzKey, ctx.h.frames[f]))}${st.f === 'max' ? ' (su peor tramo)' : ''}.</p>
<table class="data wrap"><thead><tr><th scope="col">Lluvia media en 12 h</th><th scope="col">Típico</th><th scope="col">Escenario alto</th></tr></thead><tbody>
<tr><th scope="row">Sobre esta cuenca</th><td>${mmTxt(g(bp.own12, 0))}</td><td>${mmTxt(g(bp.own12, 1))}</td></tr>
<tr><th scope="row">Sobre ella y todo lo de aguas arriba</th><td>${mmTxt(g(bp.up12, 0))}</td><td>${mmTxt(g(bp.up12, 1))}</td></tr></tbody></table>`;
  const q = g(bp.q, 0), qh = g(bp.q, 1);
  if (q != null) out += `<p class="num">Caudal punta estimado por km²: ${num(q, 2)} m³/s/km² (alto: ${num(qh, 2)}).</p>`;
  if (L >= 2 && up != null) {
    out += up >= 0.5
      ? `<p class="up"><strong>La lluvia cae aguas arriba.</strong> En el ${pct(up)} de los escenarios el peligro llega por el cauce desde más arriba, aunque aquí llueva poco.</p>`
      : `<p>El peligro viene sobre todo de la lluvia sobre la propia cuenca (solo en el ${pct(up)} de los escenarios domina la de aguas arriba).</p>`;
  }
  return out;
}

function pointsBlock(ctx, st) {
  const pp = ctx.h && ctx.h.points;
  if (!geo.points || !pp || !pp.level) return '';
  for (const p of geo.points) if (p._basin === undefined) p._basin = basinAt(p.lon, p.lat);
  const chain = ctx.basin ? downstreamChain(ctx.basin).map((b) => b.id) : [];
  let list = geo.points.map((p, i) => ({ p, i, order: p._basin ? chain.indexOf(p._basin.id) : -1, d: distKm(ctx.lat, ctx.lon, p.lat, p.lon) }))
    .filter((x) => x.order >= 0).sort((a, b) => a.order - b.order || a.d - b.d).slice(0, 4);
  let title = 'Barrancos y ríos vigilados aguas abajo';
  if (!list.length) {
    list = geo.points.map((p, i) => ({ p, i, d: distKm(ctx.lat, ctx.lon, p.lat, p.lon) })).filter((x) => x.d <= 15).sort((a, b) => a.d - b.d).slice(0, 3);
    title = 'Barrancos y ríos vigilados cerca';
  }
  if (!list.length) return '';
  const rows = list.map(({ p, i, d }) => {
    const f = st.f === 'max' ? worst(pp.level, pp.p, i) : Math.min(st.f, pp.level.length - 1);
    const L = pp.level[f][i] || 0;
    const po = pp.p && pp.p[2] && pp.p[2][f] ? pp.p[2][f][i] : null;
    return `<li><button type="button" class="link" data-pt="${esc(p.id)}">${lv(L)} ${esc(p.stream)} en ${esc(p.town)}</button>
<span class="num">a ${num(d, d < 10 ? 1 : 0)} km · P(desborde) ${pct(po)}</span></li>`;
  }).join('');
  return `<section><h3>${title}</h3><ul class="plain">${rows}</ul></section>`;
}

function warningBlock(ctx) {
  if (!ctx.zone) return `<p>Este punto queda fuera de las zonas de aviso de la Comunitat Valenciana. Consulta los avisos de AEMET de su comunidad: <a href="https://www.aemet.es/es/eltiempo/prediccion/avisos">aemet.es/avisos</a>.</p>`;
  const now = Date.now();
  const ws = ctx.snap.warnings.filter((w) => String(w.zone_code || '').replace(/C$/, '') === ctx.zone.code && (!w.expires || Date.parse(w.expires) > now));
  const link = `<a href="${AEMET_URL}">Ver los avisos en aemet.es</a>`;
  const src = ctx.snap.sources.find((s) => s.id === 'aemet');
  if (src && src.ok === false) return `<p>No se han podido leer los avisos de AEMET en esta actualización. ${link}.</p>`;
  if (!ws.length) return `<p>AEMET no tiene ningún aviso de lluvias en vigor para «${esc(ctx.zone.name)}». ${link}.</p>`;
  return `<div class="official">${ws.map((w) => {
    const [name, L] = WARN[w.level] || [w.level || 'sin nivel', 0];
    const pr = w.params && w.params.probability ? ` Probabilidad: ${esc(w.params.probability)}.` : '';
    return `<p><span class="lv lv${L}">&nbsp;</span> <strong>Aviso ${esc(name)} de AEMET</strong> por lluvias en «${esc(w.zone_name || ctx.zone.name)}»,
<span class="num">de ${w.onset ? dayTime(new Date(w.onset)) : '—'} a ${w.expires ? dayTime(new Date(w.expires)) : '—'}</span>. ${esc(w.text || '')}${pr}</p>`;
  }).join('')}<p>Este es el aviso oficial, el que vale. ${link}.</p></div>`;
}

function gaugesBlock(ctx) {
  const src = ctx.snap.sources.find((s) => s.id === 'gauges');
  if (!ctx.snap.gauges.length) return `<p>${src && src.ok === false ? 'No se han podido leer los pluviómetros en esta actualización.' : 'Esta actualización no trae lecturas de pluviómetros.'}</p>`;
  const near = nearest(ctx.snap.gauges, ctx.lat, ctx.lon, 3, 25);
  if (!near.length) return '<p>No hay ningún pluviómetro con datos a menos de 25 km.</p>';
  const v = (x) => (x == null ? '—' : num(x, 1));
  return `<div class="scroll"><table class="data"><thead><tr><th scope="col">Pluviómetro</th><th scope="col">Dist.</th><th scope="col">1 h</th><th scope="col">12 h</th><th scope="col">24 h</th><th scope="col">Leído</th></tr></thead><tbody>${
    near.map(({ it, d }) => `<tr><th scope="row">${esc(it.name)} <span class="dim">${esc(SOURCE[it.source] || it.source || '')}</span></th><td>${num(d, 1)} km</td>
<td>${v(it.p_1h)}</td><td>${v(it.p_12h)}</td><td>${v(it.p_24h)}</td><td>${it.t_utc ? dayTime(new Date(it.t_utc)) : '—'}</td></tr>`).join('')
  }</tbody></table></div><p class="dim">Lluvia medida, en mm. Datos provisionales de las redes de origen.</p>`;
}

/** The optional `drivers` block: shown only if it carries readable text. */
export function driversText(d) {
  if (!d) return null;
  if (typeof d === 'string') return d;
  for (const k of ['text', 'summary', 'resumen', 'headline', 'description']) if (typeof d[k] === 'string' && d[k].trim()) return d[k];
  for (const k of ['lines', 'bullets', 'sentences']) if (Array.isArray(d[k]) && d[k].every((x) => typeof x === 'string')) return d[k].join(' ');
  return null;
}

export function renderPanel(root, snap, st, hooks) {
  const ctx = context(snap, st);
  if (!ctx) {
    root.innerHTML = '<p class="hint">Toca el mapa, busca tu municipio o usa «Mi ubicación» para ver aquí el riesgo de un lugar, cuándo y por qué.</p>';
    return null;
  }
  const where = `<p class="num coords">${num(ctx.lat, 3)}° N, ${num(Math.abs(ctx.lon), 3)}° ${ctx.lon < 0 ? 'O' : 'E'}${ctx.zone ? ` · zona de avisos: ${esc(ctx.zone.name)}` : ''}</p>`;
  if (ctx.n < 0) {
    root.innerHTML = `<h2>${esc(placeTitle(ctx))}</h2>${where}<p>Este punto queda fuera de la zona que se calcula: la Comunitat Valenciana y las cuencas que desaguan en ella.</p>`;
    return ctx;
  }
  if (!ctx.h) {
    root.innerHTML = `<h2>${esc(placeTitle(ctx))}</h2>${where}<p>No hay datos para ${HZ_PLAIN[st.hz] || 'este plazo'} en esta actualización.</p>`;
    return ctx;
  }
  const cf = cellFrame(ctx.h, ctx.f, ctx.n);
  const fr = ctx.h.frames[ctx.f];
  const L = cf.level, meta = LEVEL[L];
  const when = `${esc(frameLabel(ctx.hzKey, fr))}${ctx.isMax ? ` <span class="dim">(lo peor de ${HZ_PLAIN[ctx.hzKey]})</span>` : ''}`;
  const drv = driversText(snap.drivers);

  root.innerHTML = `<h2>${esc(placeTitle(ctx))}</h2>${where}
<div class="level"><span class="lv big lv${L}">${L === 0 ? '–' : L}</span>
<p><strong>Nivel ${L} · ${meta.name}</strong>${meta.aemet ? ` <span class="dim">(${meta.aemet})</span>` : ''}<br>${meta.note}<br><span class="num">${when}</span></p></div>
${!fr.ok ? '<p class="warn">Este tramo no tiene datos válidos en esta actualización.</p>' : ''}
<section><h3>Cómo evoluciona</h3>${timeline(ctx, st)}</section>
<section><h3>Probabilidad de cada nivel</h3>${probBars(ctx, cf)}</section>
<section><h3>Lluvia prevista</h3>${rainTable(ctx, cf)}</section>
<section><h3>Lo que ya ha caído</h3>${observed(ctx)}</section>
<section><h3>Su cuenca</h3>${basinBlock(ctx, st)}</section>
${pointsBlock(ctx, st)}
<section><h3>Aviso oficial de AEMET</h3>${warningBlock(ctx)}</section>
<section><h3>Pluviómetros cercanos</h3>${gaugesBlock(ctx)}</section>
${drv ? `<section><h3>Situación meteorológica</h3><p>${esc(drv)}</p></section>` : ''}
<details class="audit"><summary>Cómo se ha calculado</summary><div class="audit-body"><p>Cargando el detalle del cálculo…</p></div></details>
<p><button type="button" class="link" data-act="json">Descargar datos (JSON)</button> <span class="dim" data-role="json-msg"></span></p>`;

  const det = root.querySelector('details.audit');
  const body = det.querySelector('.audit-body');
  const fill = async () => {
    if (!det.open || body.dataset.done) return;
    body.dataset.done = '1';
    try { body.innerHTML = await auditHtml(ctx); } catch (e) {
      body.innerHTML = `<p class="warn">No se ha podido montar el detalle del cálculo: ${esc(e.message)}.</p>`;
      delete body.dataset.done;
    }
  };
  det.addEventListener('toggle', () => { hooks.onAudit(det.open); fill(); });
  if (hooks.auditOpen()) { det.open = true; fill(); }

  root.querySelector('[data-act="json"]').addEventListener('click', async (ev) => {
    const msg = root.querySelector('[data-role="json-msg"]');
    ev.target.disabled = true; msg.textContent = 'Preparando…';
    try {
      const blob = new Blob([JSON.stringify(await cellExport(ctx), null, 1)], { type: 'application/json' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `riua_${ctx.cell.lat.toFixed(3)}_${ctx.cell.lon.toFixed(3)}_${snap.generatedIso.replace(/[^0-9TZ]/g, '')}.json`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
      msg.textContent = 'Descargado.';
    } catch (e) { msg.textContent = `No se ha podido preparar el fichero: ${e.message}.`; }
    ev.target.disabled = false;
  });
  return ctx;
}
