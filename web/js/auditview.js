// "Cómo se ha calculado": the full audit of one cell and frame, and the JSON export of a cell.
// The maths is in maths.js; this file only lays the numbers out.

import { DEFAULT_THR, HORIZONS, HZ_LABEL, LEVEL } from './config.js';
import { cellCentre, cellFrame, loadExplain, mmOrNull, prob } from './data.js';
import { auditCell, normCdf, weightedStats } from './maths.js';
import { esc, frameExact, frameLabel, num, pct } from './time.js';

const FAMILY = { radar: 'radar (extrapolación)', cp: 'alta resolución', regional: 'regional', global: 'global', ens: 'conjunto (ensemble)' };
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
    a1: k < ex.M ? mmOrNull(ex.a1[k * FN + o]) : null,
    a12: k < ex.M ? mmOrNull(ex.a12[k * FN + o]) : null,
  }));
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

/**
 * Scenario-based probabilities (what the backend uses until a calibration is fitted):
 *   r_m,L = max(s1h·a1/T1h_L, s12h·a12/T12h_L)
 *   P(>=L) = Σ w_m Φ(ln(b·r_m,L)/σ) / Σ w_m
 */
async function auditDressing(ctx) {
  const { snap, hzKey, f, n } = ctx;
  const h = snap.hz[hzKey];
  const fr = h.frames[f];
  const c = cellFrame(h, f, n);
  const thr = thresholdsFor(snap, ctx.zone);
  const sigma = Number(h.sigma), bias = Number(h.bias || 1);
  const rows = scenarios(h, await loadExplain(snap, hzKey), f, n).filter((r) => r.w > 0);
  const P = [0, 0, 0, 0];
  let W = 0;
  const body = rows.map((r, k) => {
    const fam = (h.fam && h.fam[r.family]) || { s1h: 1, s12h: 1 };
    const ps = [0, 1, 2, 3].map((L) => {
      const r1 = r.a1 == null ? 0 : (fam.s1h * r.a1) / thr.t1[L];
      const r12 = r.a12 == null ? 0 : (fam.s12h * r.a12) / thr.t12[L];
      const rr = Math.max(r1, r12);
      return rr > 0 ? normCdf(Math.log(bias * rr) / sigma) : 0;
    });
    if (r.a1 != null || r.a12 != null) { W += r.w; ps.forEach((p, L) => { P[L] += r.w * p; }); }
    return `<tr><td>${k + 1}</td><th class="wrap">${esc(r.name)}</th><td>${pct(r.w, 1)}</td><td>${mmv(r.a1)}</td><td>${mmv(r.a12)}</td>`
      + `<td>${fam.s1h === 1 && fam.s12h === 1 ? '1' : `${fx(fam.s1h, 1)}/${fx(fam.s12h, 1)}`}</td>${ps.map((p) => `<td>${fx(p, 2)}</td>`).join('')}</tr>`;
  }).join('');
  const Pn = P.map((p) => (W > 0 ? p / W : 0));
  for (let L = 1; L < 4; L++) Pn[L] = Math.min(Pn[L], Pn[L - 1]);
  let level = 1;
  [2, 3, 4, 5].forEach((L, k) => { if (Pn[k] >= Number(h.tau[String(L)])) level = L; });
  const blank = '<td></td><td></td><td></td><td></td>';
  const head = `<tr><th>#</th><th class="wrap">Escenario</th><th>Peso</th><th>1 h</th><th>12 h</th><th title="factor de representatividad 1 h / 12 h">s</th>${[2, 3, 4, 5].map((L) => `<th>P≥${L}</th>`).join('')}</tr>`;
  const tot = `<tr><th></th><th class="wrap"><b>Total</b></th>${blank}${Pn.map((p) => `<td><b>${fx(p, 2)}</b></td>`).join('')}</tr>`
    + `<tr><th></th><th class="wrap">publicado</th>${blank}${c.p.map((p) => `<td>${fx(p, 2)}</td>`).join('')}</tr>`
    + `<tr><th></th><th class="wrap">mínimo τ</th>${blank}${[2, 3, 4, 5].map((L) => `<td>${fx(Number(h.tau[String(L)]), 2)}</td>`).join('')}</tr>`;
  const cc = cellCentre(snap, n);
  const sw = (L) => `<span class="lv lv${L}">${L || '–'}</span>`;
  return `<p class="num">${fx(cc.lat, 3)}° N ${fx(cc.lon, 3)}° · ${esc(frameExact(fr))}</p>
<pre class="calc">r = max(s₁·lluvia1h / U₁ , s₁₂·lluvia12h / U₁₂)
P(≥nivel) = Σ peso · Φ( ln(r) / σ )        σ = ${fx(sigma, 2)}
U₁  = ${thr.t1.map((v) => fx(v, 0)).join(' / ')} mm      (niveles 2 / 3 / 4 / 5)
U₁₂ = ${thr.t12.map((v) => fx(v, 0)).join(' / ')} mm</pre>
<div class="scroll tall"><table class="data"><thead>${head}</thead><tbody>${body}${tot}</tbody></table></div>
<p style="margin-top:.4rem">Nivel: el más alto con P ≥ τ → ${sw(level)} <span class="dim">publicado</span> ${sw(c.level)}</p>`;
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
      + rows.map((r, k) => `<tr><td>${k + 1}</td><th scope="row">${esc(r.name)}</th><td>${esc(FAMILY[r.family] || r.family)}</td><td>${esc((r.run || '').replace('T', ' ').replace('Z', ''))}</td>`
        + `<td>${r.step_h} h</td><td>${r.neigh_km ? `${num(r.neigh_km)} km` : '—'}</td><td>${mmv(r.a1)}</td><td>${mmv(r.a12)}</td><td>${r.w == null ? '—' : pct(r.w, 2)}</td></tr>`).join('')
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
      tau: h.tau,
      scenarios_note: ex ? 'a1_mm / a12_mm: per-scenario amounts after neighbourhood; null = not available' : `scenario table not available: ${exErr}`,
      frames: h.frames.map((fr, f) => {
        const c = cellFrame(h, f, n);
        const a = auditCell(c, fr, thr, h.tau);
        return {
          t0: fr.t0, t1: fr.t1, label: frameLabel(key, fr), ok: fr.ok, has_1h: fr.has_1h, cal: fr.cal,
          level: c.level, p_published: { '>=2': c.p[0], '>=3': c.p[1], '>=4': c.p[2], '>=5': c.p[3] },
          p_recomputed: Object.fromEntries(a.levels.map((l) => [`>=${l.L}`, { p_1h: l.pa, p_12h: l.pb, both: l.both, p: l.p }])),
          ensemble_mm: { m1: c.m1, q1: c.q1, m12: c.m12, q12: c.q12 },
          expected_mm: { '1h': { median: c.e1[0], high_1_in_10: c.e1[1] }, '12h': { median: c.e12[0], high_1_in_10: c.e12[1] } },
          scenarios: ex ? scenarios(h, ex, f, n).map((r) => ({ name: r.name, family: r.family, run: r.run, weight: r.w, a1_mm: r.a1, a12_mm: r.a12 })) : null,
        };
      }),
    };
  }
  return out;
}
