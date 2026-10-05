// "Cómo se ha calculado": the full audit of one cell and frame, and the JSON export of a cell.
// The maths is in maths.js; this file only lays the numbers out.

import { DEFAULT_THR, HORIZONS, HZ_LABEL, LEVEL } from './config.js';
import { cellCentre, cellFrame, loadExplain, mmOrNull, prob } from './data.js';
import { auditCell, dressCell, weightedStats } from './maths.js';
import { esc, frameExact, frameLabel, num, pct } from './time.js';

const FAMILY = { radar: 'radar (extrapolación)', cp: 'alta resolución', regional: 'regional', global: 'global', ens: 'conjunto (ensemble)', eps: 'conjunto regional' };
const AEMET_NAME = ['amarillo', 'naranja', 'rojo', 'extremo'];
const fx = (v, d = 3) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d).replace('.', ',').replace('-', '−'));
const mmv = (v) => (v == null ? 'n/d' : fx(v, 1));

/** Thresholds of a zone in mm: yellow, orange, red and the extreme level (multiples of red). */
export function thresholdsFor(snap, zone) {
  const z = (zone && snap.thresholds.zones && snap.thresholds.zones[zone.code]) || DEFAULT_THR;
  const ex = snap.thresholds.extreme || {};
  const x1 = ex.x1h ?? 1.5, x12 = ex.x12h ?? 1.67;
  return { t1: [...z['1h'], z['1h'][2] * x1], t12: [...z['12h'], z['12h'][2] * x12], x1, x12, fromZone: !!(zone && snap.thresholds.zones && snap.thresholds.zones[zone.code]) };
}

/** Rows of the scenario table for (frame f, cell n), read from the explain binary. */
export function scenarios(h, ex, f, n) {
  const o = f * h.N + n, FN = h.F * h.N;
  return h.members.map((m, k) => ({
    name: m.name, family: m.family, model: m.model, run: m.run, step_h: m.step_h, neigh_km: m.neigh_km, w_raw: m.w_raw,
    w: Array.isArray(m.w) ? m.w[f] : null,
    // the 1-h block only holds the scenarios that have a 1-h amount (ex.slot, -1 = none)
    a1: k < ex.M && ex.slot[k] >= 0 ? mmOrNull(ex.a1[ex.slot[k] * FN + o]) : null,
    a12: k < ex.M ? mmOrNull(ex.a12[k * FN + o]) : null,
  }));
}

/** The dressing of one cell and frame from its scenario rows (maths.js::dressCell with this horizon's numbers). */
export function dressing(h, rows, c, thr) {
  const famOf = (family) => (h.fam && h.fam[family]) || { s1h: 1, s12h: 1 };
  return { famOf, ...dressCell(rows, famOf, c.o12 || 0, h.kernel, thr, h.tau, h.levelCap) };
}

function calTable(a, names) {
  const keys = ['a0', 'a1', 'a2', 'b0', 'b1', 'delta'];
  const head = keys.map((k) => `<th scope="col">${k === 'delta' ? 'δ (mm)' : k}</th>`).join('');
  const rows = names.map(([key, label, used]) => {
    const c = a[key];
    if (!c) return `<tr><th scope="row">${label}</th><td colspan="8">no disponible</td></tr>`;
    return `<tr><th scope="row">${label}${used ? '' : ' <span class="dim">(no se usa aquí)</span>'}</th>${keys.map((k) => `<td>${fx(c[k], 3)}</td>`).join('')}`
      + `<td>${c.n ? num(c.n) : '—'}</td><td>${Number.isFinite(c.crps) ? fx(c.crps, 3) : '—'}</td></tr>`;
  }).join('');
  return `<div class="scroll"><table class="data"><thead><tr><th scope="col">Cantidad</th>${head}<th scope="col">n.º de casos</th><th scope="col">CRPS</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function distBlock(title, d, calName) {
  if (!d) return `${title}: sin coeficientes, no se evalúa.\n`;
  const c = d.cal;
  return `${title} (coeficientes ${calName}): m = ${fx(d.m, 1)} mm, q90 = ${fx(d.q, 1)} mm\n`
    + `  mu    = a0 + a1·m + a2·q90 = ${fx(c.a0)} + ${fx(c.a1)}·${fx(d.m, 1)} + ${fx(c.a2)}·${fx(d.q, 1)} = ${fx(d.mu)} mm\n`
    + `  sigma = b0·√mu + b1·(q90 − m) = ${fx(c.b0)}·√${fx(d.mu)} + ${fx(c.b1)}·(${fx(d.q, 1)} − ${fx(d.m, 1)}) = ${fx(d.sigma)} mm\n`
    + `  k = mu²/sigma² = ${fx(d.k, 4)}     θ = sigma²/mu = ${fx(d.theta, 4)} mm     δ = ${fx(d.delta)} mm\n`;
}

function levelBlock(l, a) {
  const line = (label, d, T, p) => (d
    ? `  P(${label} ≥ ${fx(T, 0)} mm) = Q(k, (T − δ)/θ) = Q(${fx(d.k, 4)}, (${fx(T, 1)} − (${fx(d.delta)}))/${fx(d.theta, 4)}) = Q(${fx(d.k, 4)}, ${fx(Math.max(T - d.delta, 0) / d.theta, 4)}) = ${fx(p, 4)}\n`
    : `  P(${label} ≥ ${fx(T, 0)} mm) = 0 (sin coeficientes)\n`);
  return `Nivel ${l.L} (${AEMET_NAME[l.L - 2]}): ${fx(l.T1, 0)} mm en 1 h o ${fx(l.T12, 0)} mm en 12 h\n`
    + line('1 h', a.dA, l.T1, l.pa) + line('12 h', a.dB, l.T12, l.pb)
    + `  las dos a la vez = Φ₂(Φ⁻¹(P₁), Φ⁻¹(P₁₂); ρ) = Φ₂(${fx(l.za)}, ${fx(l.zb)}; ${fx(a.rho, 2)}) = ${fx(l.both, 4)}\n`
    + `  P(≥${l.L}) = P₁ + P₁₂ − ambas = ${fx(l.pa, 4)} + ${fx(l.pb, 4)} − ${fx(l.both, 4)} = ${fx(l.union, 4)}`
    + `${l.p < l.union - 1e-12 ? `  → limitada a ${fx(l.p, 4)} (no puede superar la del nivel anterior)` : ''}\n`
    + `  publicado en el snapshot: ${l.published == null ? 'n/d' : fx(l.published, 3)}     tau = ${fx(l.tau, 2)}     ${l.reached ? 'SE ALCANZA' : 'no se alcanza'}\n`;
}

const GROUP_FROM = 5;       // this many scenarios of one model and run are shown as one row (an ensemble)

/**
 * Rows for the scenario tables: the members of an ensemble become one group, everything else stays alone.
 * Members are recognised by their name, "ENS m07 · 01/10 12Z", "Radar STEPS m03 → AROME-HD …": the name
 * without the member number is the group (the radar members carry the family and model of the run they
 * blend into, so the family cannot tell them apart from that run).
 * -> [{label|null, items:[{r, k}]}] in the order of first appearance.
 */
export function grouped(rows) {
  const keyOf = (r) => { const name = String(r.name); return /\sm\d+\b/.test(name) ? name.replace(/\s+m\d+\b/, '') : null; };
  const by = new Map();
  rows.forEach((r, k) => { const key = keyOf(r); if (key == null) return; if (!by.has(key)) by.set(key, []); by.get(key).push({ r, k }); });
  const out = [], done = new Set();
  rows.forEach((r, k) => {
    const key = keyOf(r), g = key == null ? null : by.get(key);
    if (!g || g.length < GROUP_FROM) out.push({ label: null, items: [{ r, k }] });
    else if (!done.has(key)) { done.add(key); out.push({ label: key, items: g }); }
  });
  return out;
}

/** "12,3 0,0–48,1": median and range of the members' amounts (mm). */
function spread(vals) {
  const v = vals.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return 'n/d';
  const med = v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
  return `${fx(med, 1)} <span class="dim">${fx(v[0], 1)}–${fx(v[v.length - 1], 1)}</span>`;
}

/**
 * Scenario-based probabilities (risk.py::dressed_probabilities; formulas in maths.js::dressScenario):
 * each scenario gives P₁ from its 1-h amount and P₁₂ from its 12-h amount, the larger of the two counts,
 * and the weighted mean over the scenarios is the probability of the level.
 */
async function auditDressing(ctx) {
  const { snap, hzKey, f, n } = ctx;
  const h = snap.hz[hzKey];
  const fr = h.frames[f];
  const c = cellFrame(h, f, n);
  const thr = thresholdsFor(snap, ctx.zone);
  const k = h.kernel;
  const rows = scenarios(h, await loadExplain(snap, hzKey), f, n).filter((r) => r.w > 0);
  // the sum runs over every scenario; only the table groups the members of an ensemble
  const d = dressing(h, rows, c, thr);
  const sTxt = (fam) => (fam.s1h === 1 && fam.s12h === 1 ? '1' : `${fx(fam.s1h, 1)}/${fx(fam.s12h, 1)}`);
  const phiTxt = (v) => (v > 0 ? fx(v, 2) : '0');
  const line = (r, i, of) => `<tr${of ? ` class="mem" data-of="${of}" hidden` : ''}><td>${i + 1}</td><th class="wrap">${esc(r.name)}</th><td>${pct(r.w, 1)}</td><td>${mmv(r.a1)}</td><td>${mmv(r.a12)}</td>`
    + `<td>${sTxt(d.famOf(r.family))}</td><td>${phiTxt(d.per[i].phi)}</td>${d.per[i].p.map((p) => `<td>${fx(p, 2)}</td>`).join('')}</tr>`;
  const body = grouped(rows).map((g, gi) => {
    if (!g.label) return line(g.items[0].r, g.items[0].k, null);
    const w = g.items.reduce((s, it) => s + it.r.w, 0);
    const mean = (get) => (w > 0 ? g.items.reduce((s, it) => s + it.r.w * get(d.per[it.k]), 0) / w : 0);
    return `<tr class="grp"><td>${g.items.length}</td><th class="wrap"><button type="button" class="link" data-grp="g${gi}" aria-expanded="false" title="ver los ${g.items.length} miembros">${esc(g.label)}</button></th>`
      + `<td>${pct(w, 1)}</td><td>${spread(g.items.map((it) => it.r.a1))}</td><td>${spread(g.items.map((it) => it.r.a12))}</td><td>${sTxt(d.famOf(g.items[0].r.family))}</td><td>${phiTxt(mean((x) => x.phi))}</td>`
      + `${[0, 1, 2, 3].map((L) => `<td>${fx(mean((x) => x.p[L]), 2)}</td>`).join('')}</tr>${g.items.map((it) => line(it.r, it.k, `g${gi}`)).join('')}`;
  }).join('');
  const blank = '<td></td><td></td><td></td><td></td><td></td>';
  const head = `<tr><th title="número del escenario, o cuántos miembros tiene el conjunto">n.º</th><th class="wrap">Escenario</th><th>Peso</th><th>1 h</th><th>12 h</th><th title="factor de representatividad 1 h / 12 h">s</th>`
    + `<th title="parte ya medida de la cantidad en 12 h del escenario">φ</th>${[2, 3, 4, 5].map((L) => `<th>P≥${L}</th>`).join('')}</tr>`;
  const tot = `<tr><th></th><th class="wrap"><b>Total</b></th>${blank}${d.P.map((p) => `<td><b>${fx(p, 3)}</b></td>`).join('')}</tr>`
    + `<tr><th></th><th class="wrap">publicado</th>${blank}${c.p.map((p) => `<td>${fx(p, 3)}</td>`).join('')}</tr>`
    + `<tr><th></th><th class="wrap">mínimo τ</th>${blank}${[2, 3, 4, 5].map((L) => `<td>${fx(Number(h.tau[String(L)]), 2)}</td>`).join('')}</tr>`;
  const cc = cellCentre(snap, n);
  const sw = (L) => `<span class="lv lv${L}">${L || '–'}</span>`;
  // days 2-7: the published colour is the zone's (P that the level is reached somewhere in the zone that day)
  const zi = h.scale === 'zone' && h.zones && ctx.zone ? h.zones.codes.indexOf(ctx.zone.code) : -1;
  const zoneBlock = zi < 0 ? '' : `<table class="data" style="margin:.3rem 0"><tbody>`
    + `<tr><th class="wrap">${esc(ctx.zone.name)} · día</th>${[0, 1, 2, 3].map((k) => `<td><b>${fx(h.zones.p[k][f][zi] / 200, 2)}</b></td>`).join('')}<td>${sw(h.zones.level[f][zi])}</td></tr>`
    + `<tr><th class="wrap">τ</th>${[2, 3, 4, 5].map((L) => `<td>${fx(Number(h.zones.tau[String(L)]), 2)}</td>`).join('')}<td></td></tr></tbody></table>`;
  return `<p class="num">${fx(cc.lat, 3)}° N ${fx(cc.lon, 3)}° · ${esc(frameExact(fr))}</p>${zoneBlock}
<pre class="calc">A₁ = s₁·lluvia1h     A₁₂ = s₁₂·lluvia12h
φ = min(1, medido / A₁₂)     medido = ${fx(c.o12 || 0, 1)} mm
P₁  = Φ( ln(b₁·A₁ / U₁) / σ₁ )
P₁₂ = Φ( ln(b₁₂·A₁₂ / U₁₂) / σ₁₂ )
b₁₂ = φ·${fx(k.obs12Factor, 2)} + (1 − φ)·b
σ₁₂ = max(σ·(1 − φ), ${fx(Math.min(k.sigma, k.sigmaObs), 2)})
b = ${fx(k.bias, 2)}  σ = ${fx(k.sigma, 2)}  b₁ = ${fx(k.bias1h, 2)}  σ₁ = ${fx(k.sigma1h, 2)}
P(≥nivel) = Σ peso · max(P₁, P₁₂)
U₁  = ${thr.t1.map((v) => fx(v, 0)).join(' / ')} mm
U₁₂ = ${thr.t12.map((v) => fx(v, 0)).join(' / ')} mm</pre>
<div class="scroll tall"><table class="data"><thead>${head}</thead><tbody>${body}${tot}</tbody></table></div>
<p style="margin-top:.4rem">Nivel: el más alto con P ≥ τ${h.levelCap != null ? `, como mucho ${h.levelCap} en este plazo` : ''} → ${sw(d.level)} <span class="dim">publicado</span> ${sw(c.level)}</p>`;
}

/** HTML of the audit. ctx = {snap, hzKey, f, n, zone, cellInfo} */
export async function auditHtml(ctx) {
  const { snap, hzKey, f, n } = ctx;
  const h = snap.hz[hzKey];
  const fr = h.frames[f];
  if (fr.method === 'dressing' || !fr.cal) return auditDressing(ctx);
  const c = cellFrame(h, f, n);
  const thr = thresholdsFor(snap, ctx.zone);
  const a = auditCell(c, fr, thr, h.tau);
  if (a.dA) a.dA.cal = a.calA;
  if (a.dB) a.dB.cal = fr.cal.c12;
  const cc = cellCentre(snap, n);
  let out = '';

  out += `<h4>1. Qué se calcula</h4>
<p>Celda de ${num(snap.grid.d, 2)}° (fila <span class="num">j = ${cc.j}</span>, columna <span class="num">i = ${cc.i}</span>, índice <span class="num">c = ${cc.c}</span>),
centro <span class="num">${fx(cc.lat, 3)}° N, ${fx(cc.lon, 3)}° E</span>. Horizonte «${HZ_LABEL[hzKey]}», tramo ${f + 1} de ${h.F}:
<span class="num">${esc(frameExact(fr))}</span>. Ventana (t0, t1].</p>
<p>Para cada escenario se toman dos cantidades, las mismas en que AEMET escribe sus umbrales: la mayor acumulación en 1 h que termina dentro del tramo
y la mayor acumulación en 12 h que termina dentro del tramo (la ventana de 12 h puede empezar antes del tramo, y entonces incluye lluvia ya observada).</p>`;
  if (!fr.ok) out += '<p class="warn">Este tramo no tiene datos válidos: ningún escenario lo cubre. Nivel 0 = sin datos.</p>';

  // scenarios
  let rows = null, exErr = null;
  try { rows = scenarios(h, await loadExplain(snap, hzKey), f, n); } catch (e) { exErr = e; }
  out += `<h4>2. Escenarios${rows ? ` (${rows.length})` : ''}</h4>`;
  if (rows) {
    out += `<p>Cada fila es una pasada de un modelo, un miembro de un conjunto o un miembro de la extrapolación del radar.
«Vecindad» es el radio dentro del cual se toma el máximo del escenario (una tormenta colocada a unos kilómetros cuenta).
«n/d» en 1 h: ese escenario no informa de la intensidad horaria. El peso es el del tramo, ya normalizado.</p>
<div class="scroll tall"><table class="data"><thead><tr><th scope="col">n.º</th><th scope="col">Escenario</th><th scope="col">Familia</th><th scope="col">Pasada (UTC)</th>
<th scope="col">Paso</th><th scope="col">Vecindad</th><th scope="col">1 h (mm)</th><th scope="col">12 h (mm)</th><th scope="col">Peso</th></tr></thead><tbody>`
      + grouped(rows).map((g, gi) => {
        const one = (r, k, of) => `<tr${of ? ` class="mem" data-of="${of}" hidden` : ''}><td>${k + 1}</td><th scope="row">${esc(r.name)}</th><td>${esc(FAMILY[r.family] || r.family)}</td><td>${esc((r.run || '').replace('T', ' ').replace('Z', ''))}</td>`
          + `<td>${r.step_h} h</td><td>${r.neigh_km ? `${num(r.neigh_km)} km` : '—'}</td><td>${mmv(r.a1)}</td><td>${mmv(r.a12)}</td><td>${r.w == null ? '—' : pct(r.w, 2)}</td></tr>`;
        if (!g.label) return one(g.items[0].r, g.items[0].k, null);
        const r0 = g.items[0].r;
        return `<tr class="grp"><td>${g.items.length}</td><th scope="row"><button type="button" class="link" data-grp="e${gi}" aria-expanded="false" title="ver los ${g.items.length} miembros">${esc(g.label)}</button></th>`
          + `<td>${esc(FAMILY[r0.family] || r0.family)}</td><td>${esc((r0.run || '').replace('T', ' ').replace('Z', ''))}</td><td>${r0.step_h} h</td><td>${r0.neigh_km ? `${num(r0.neigh_km)} km` : '—'}</td>`
          + `<td>${spread(g.items.map((it) => it.r.a1))}</td><td>${spread(g.items.map((it) => it.r.a12))}</td><td>${pct(g.items.reduce((s, it) => s + (it.r.w || 0), 0), 2)}</td></tr>`
          + g.items.map((it) => one(it.r, it.k, `e${gi}`)).join('');
      }).join('')
      + '</tbody></table></div>';
  } else {
    out += `<p class="warn">No se ha podido cargar la tabla de escenarios (${esc(exErr.message)}). El resto del cálculo se muestra con el resumen publicado.</p>`;
  }

  // ensemble summary
  out += '<h4>3. Resumen del conjunto</h4>';
  const pub = [['1 h', c.m1, c.q1, 'a1'], ['12 h', c.m12, c.q12, 'a12']];
  out += `<div class="scroll"><table class="data"><thead><tr><th scope="col">Cantidad</th><th scope="col">Media ponderada m (publicada)</th><th scope="col">Percentil 90 q90 (publicado)</th>
<th scope="col">m recalculada con la tabla</th><th scope="col">q90 recalculado</th><th scope="col">Escenarios con dato</th></tr></thead><tbody>`
    + pub.map(([label, m, q, key]) => {
      const st = rows ? weightedStats(rows.map((r) => r[key]), rows.map((r) => r.w)) : null;
      return `<tr><th scope="row">${label}</th><td>${mmv(m)} mm</td><td>${mmv(q)} mm</td><td>${st ? `${mmv(st.mean)} mm` : '—'}</td><td>${st ? `${mmv(st.q)} mm` : '—'}</td><td>${st ? st.n : '—'}</td></tr>`;
    }).join('') + '</tbody></table></div>'
    + `<p class="dim">El snapshot guarda cada cantidad en 1 byte (mm = (v/8)², pasos de unos 0,5 mm a 16 mm y de 2 mm a 60 mm; P = v/200): lo recalculado aquí puede diferir de lo publicado en ese redondeo.
El cálculo original usa los valores sin redondear.</p>`;

  // calibration
  out += `<h4>4. Calibración</h4>
<p>El conjunto no es una probabilidad: está sesgado y es poco disperso. La cantidad que se observará se modela con una gamma desplazada y censurada en cero,
cuyos parámetros dependen de m y q90. Los coeficientes se ajustan con predicciones pasadas y lo que después se midió.</p>`
    + calTable(fr.cal || {}, [['c1', '1 h, con escenarios horarios', a.use1], ['c1_from12', '1 h, deducida de la de 12 h', !a.use1], ['c12', '12 h', true]])
    + `<p>Correlación entre las dos cantidades (cópula gaussiana): <span class="num">ρ = ${fx(a.rho, 3)}</span>.
${fr.has_1h ? 'Este tramo tiene escenarios con dato horario.' : 'Ningún escenario de este tramo informa de la intensidad horaria: la probabilidad de la cantidad en 1 h se deduce de m y q90 de 12 h (coeficientes c1_from12).'}
Versión de los parámetros: <span class="num">${esc(snap.paramsVersion || 'sin indicar')}</span>.</p>`;

  // thresholds
  out += `<h4>5. Umbrales</h4>
<div class="scroll"><table class="data"><thead><tr><th scope="col">Nivel</th><th scope="col">En 1 h</th><th scope="col">En 12 h</th><th scope="col">Probabilidad necesaria (tau)</th></tr></thead><tbody>`
    + a.levels.map((l) => `<tr><th scope="row"><span class="lv lv${l.L}">${l.L}</span> ${LEVEL[l.L].name}</th><td>${fx(l.T1, 1)} mm</td><td>${fx(l.T12, 1)} mm</td><td>${fx(l.tau, 2)}</td></tr>`).join('')
    + `</tbody></table></div>
<p>${ctx.zone ? `Zona de avisos «${esc(ctx.zone.name)}» (${esc(ctx.zone.code)}): umbrales amarillo, naranja y rojo de AEMET Meteoalerta.` : 'Esta celda queda fuera de las zonas de aviso de la Comunitat Valenciana: se aplican los umbrales comunes a sus once zonas.'}
El nivel 5 es ${fx(thr.x1, 2)} × el rojo en 1 h y ${fx(thr.x12, 2)} × el rojo en 12 h.
${snap.thresholds.source && snap.thresholds.source.url ? `Fuente: <a href="${esc(snap.thresholds.source.url)}">${esc(snap.thresholds.source.document || 'AEMET')}</a>.` : ''}
Cuanto más grave el nivel, menos probabilidad se exige para darlo.</p>`;

  // step by step
  out += `<h4>6. Probabilidad de cada nivel, paso a paso</h4>
<p>Fórmula: <span class="num">P(cantidad ≥ T) = Q(k, (T − δ)/θ)</span>, con Q la función gamma incompleta superior regularizada.
Un nivel se da si se supera el umbral de 1 h <em>o</em> el de 12 h; las dos probabilidades se unen con la cópula.</p>
<pre class="calc">${esc(distBlock('Cantidad en 1 h', a.dA, a.calAName) + distBlock('Cantidad en 12 h', a.dB, 'c12'))}
${esc(a.levels.map((l) => levelBlock(l, a)).join('\n'))}</pre>`;

  out += `<div class="scroll"><table class="data"><thead><tr><th scope="col">Nivel</th><th scope="col">P por 1 h</th><th scope="col">P por 12 h</th><th scope="col">P del nivel, recalculada</th>
<th scope="col">P publicada</th><th scope="col">Diferencia</th><th scope="col">tau</th><th scope="col">¿Se alcanza?</th></tr></thead><tbody>`
    + a.levels.map((l) => `<tr><th scope="row">≥ ${l.L}</th><td>${fx(l.pa, 4)}</td><td>${fx(l.pb, 4)}</td><td>${fx(l.p, 4)}</td><td>${l.published == null ? '—' : fx(l.published, 3)}</td>`
      + `<td>${l.published == null ? '—' : fx(l.p - l.published, 4)}</td><td>${fx(l.tau, 2)}</td><td>${l.reached ? 'sí' : 'no'}</td></tr>`).join('')
    + '</tbody></table></div>';
  if (rows) {
    try {
      const ex = await loadExplain(snap, hzKey);
      const FN = h.F * h.N, o = f * h.N + n;
      out += `<p class="dim">Probabilidades por separado publicadas en el fichero de detalle (antes de unirlas): por 1 h ${[0, 1, 2, 3].map((k) => fx(prob(ex.p1[k * FN + o]), 3)).join(' · ')};
por 12 h ${[0, 1, 2, 3].map((k) => fx(prob(ex.p12[k * FN + o]), 3)).join(' · ')} (niveles 2 a 5).</p>`;
    } catch (e) { /* already reported above */ }
  }

  // quantiles and decision
  out += `<h4>7. Cantidad esperada</h4>
<p>Mediana y cuantil 0,9 de las dos distribuciones: en 1 h <span class="num">${mmv(a.quant.e1[0])} / ${mmv(a.quant.e1[1])} mm</span>
(publicado ${mmv(c.e1[0])} / ${mmv(c.e1[1])}); en 12 h <span class="num">${mmv(a.quant.e12[0])} / ${mmv(a.quant.e12[1])} mm</span> (publicado ${mmv(c.e12[0])} / ${mmv(c.e12[1])}).
Cuantil = max(0, P⁻¹(k, p)·θ + δ).</p>
<h4>8. Decisión</h4>
<p>El nivel es el más alto cuya probabilidad alcanza su tau. Recalculado aquí: <span class="lv lv${a.level}">${a.level}</span> ${LEVEL[a.level].name}.
Publicado: <span class="lv lv${c.level}">${c.level}</span> ${LEVEL[c.level].name}.${a.level !== c.level && c.level > 0
    ? ' <strong>No coinciden</strong>: alguna probabilidad está a menos de un paso de redondeo de su tau; vale el publicado, que se calcula sin redondear.' : ''}</p>`;
  return out;
}

/** Everything the snapshot holds for one cell, as a plain object ready for JSON.stringify. */
export async function cellExport(ctx) {
  const { snap, n } = ctx;
  const cc = cellCentre(snap, n);
  const thr = thresholdsFor(snap, ctx.zone);
  const out = {
    about: 'riuà — datos de una celda. Herramienta no oficial; las fuentes oficiales son AEMET y el 112 Comunitat Valenciana.',
    generated: snap.generatedIso, params_version: snap.paramsVersion, origin: snap.origin,
    cell: { j: cc.j, i: cc.i, c: cc.c, lat: cc.lat, lon: cc.lon, size_deg: snap.grid.d },
    selected_point: { lat: ctx.lat, lon: ctx.lon }, nearest_town: ctx.place ? ctx.place.it.name : null,
    zone: ctx.zone ? { code: ctx.zone.code, name: ctx.zone.name } : null,
    basin_unit: ctx.basin ? { id: ctx.basin.id, name: ctx.basin.name, area_km2: ctx.basin.area, up_area_km2: ctx.basin.upArea, next: ctx.basin.next } : null,
    thresholds_mm: { '1h': thr.t1, '12h': thr.t12, order: ['yellow', 'orange', 'red', 'extreme'], source: snap.thresholds.source || null },
    observed_mm: snap.obs ? { last_hour: snap.obs.last, o1_cell_max: mmOrNull(snap.obs.o1[n]), o12_cell_mean: mmOrNull(snap.obs.o12[n]), o24_cell_mean: mmOrNull(snap.obs.o24[n]) } : null,
    horizons: {},
  };
  for (const key of HORIZONS) {
    const h = snap.hz[key];
    if (!h) { out.horizons[key] = null; continue; }
    let ex = null, exErr = null;
    try { ex = await loadExplain(snap, key); } catch (e) { exErr = e.message; }
    out.horizons[key] = {
      tau: h.tau, level_cap: h.levelCap,
      kernel: { sigma: h.kernel.sigma, bias: h.kernel.bias, sigma1h: h.kernel.sigma1h, bias1h: h.kernel.bias1h, sigma_obs: h.kernel.sigmaObs, obs12_factor: h.kernel.obs12Factor },
      family_factors: h.fam,
      scenarios_note: ex ? 'a1_mm / a12_mm: per-scenario amounts after neighbourhood; null = not available. phi = measured share of the 12-h amount; p_1h, p_12h, p = probability of levels 2..5 from that scenario (scenarios with weight only)'
        : `scenario table not available: ${exErr}`,
      frames: h.frames.map((fr, f) => {
        const c = cellFrame(h, f, n);
        const rows = ex ? scenarios(h, ex, f, n) : null;
        // the method of the frame decides which calculation is repeated here, as in the audit view
        const dressed = fr.method === 'dressing' || !fr.cal;
        let recomputed = null, levelRe = null, per = null;
        if (dressed && rows) {
          const used = rows.map((r, k) => ({ r, k })).filter((x) => x.r.w > 0);
          const d = dressing(h, used.map((x) => x.r), c, thr);
          recomputed = Object.fromEntries(d.P.map((p, L) => [`>=${L + 2}`, { p }]));
          levelRe = d.level;
          per = new Map(used.map((x, i) => [x.k, d.per[i]]));
        } else if (!dressed) {
          const a = auditCell(c, fr, thr, h.tau);
          recomputed = Object.fromEntries(a.levels.map((l) => [`>=${l.L}`, { p_1h: l.pa, p_12h: l.pb, both: l.both, p: l.p }]));
          levelRe = h.levelCap != null ? Math.min(a.level, h.levelCap) : a.level;
        }
        return {
          t0: fr.t0, t1: fr.t1, label: frameLabel(key, fr), ok: fr.ok, has_1h: fr.has_1h, method: dressed ? 'dressing' : 'emos', cal: fr.cal,
          level: c.level, p_published: { '>=2': c.p[0], '>=3': c.p[1], '>=4': c.p[2], '>=5': c.p[3] },
          p_recomputed: recomputed, level_recomputed: levelRe,
          measured_in_12h_mm: c.o12,
          ensemble_mm: { m1: c.m1, q1: c.q1, m12: c.m12, q12: c.q12 },
          expected_mm: { '1h': { median: c.e1[0], high_1_in_10: c.e1[1] }, '12h': { median: c.e12[0], high_1_in_10: c.e12[1] } },
          scenarios: rows ? rows.map((r, k) => {
            const x = per && per.get(k);
            return { name: r.name, family: r.family, run: r.run, weight: r.w, a1_mm: r.a1, a12_mm: r.a12, ...(x ? { phi: x.phi, p_1h: x.p1, p_12h: x.p12, p: x.p } : {}) };
          }) : null,
        };
      }),
    };
  }
  return out;
}
