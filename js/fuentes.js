// fuentes.html: the state of every data source in the latest snapshot.

import { loadSnapshots } from './data.js';
import { age, dayTime, esc, num } from './time.js';

const $ = (id) => document.getElementById(id);
const FAMILY = { cp: 'alta resolución', regional: 'regional', global: 'global', ens: 'conjunto' };

function runTxt(iso) {                      // "2026-10-01T06:00Z" -> "jue 1 oct 08:00 (06 UTC)"
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return esc(iso);
  return `${dayTime(d)} <span class="dim">(${esc(String(iso).slice(11, 13))} UTC)</span>`;
}

function detail(s) {
  const out = [];
  if (Array.isArray(s.runs) && s.runs.length) out.push(`pasadas: ${s.runs.map(runTxt).join(', ')}`);
  if (s.n != null) out.push(`${num(s.n)} ${s.id === 'gauges' ? 'estaciones' : s.id === 'aemet' ? 'avisos de lluvia' : 'miembros'}`);
  if (s.members != null) out.push(`${num(s.members)} miembros`);
  if (s.frames != null) out.push(`${num(s.frames)} imágenes`);
  if (s.last) out.push(`última: ${dayTime(new Date(s.last))}`);
  if (s.source) out.push(`origen: ${esc(s.source)}`);
  if (Array.isArray(s.sources)) out.push(`redes: ${s.sources.map(esc).join(', ')}`);
  if (s.method) out.push(`método: ${esc(s.method)}`);
  if (s.family) out.push(FAMILY[s.family] || esc(s.family));
  if (s.error) out.push(`error: ${esc(s.error)}`);
  return out.join(' · ') || '—';
}

function show(snap) {
  $('src-when').textContent = `Predicción de ${dayTime(snap.generated)} (${age(snap.generated)}).`;
  if (!snap.sources.length) { $('src-table').innerHTML = '<p>Esta actualización no trae la lista de fuentes.</p>'; return; }
  $('src-table').innerHTML = `<div class="scroll"><table class="data wrap"><thead><tr><th scope="col">Fuente</th><th scope="col">Estado</th><th scope="col">Detalle</th></tr></thead><tbody>${
    snap.sources.map((s) => `<tr><th scope="row">${esc(s.label || s.id)}</th><td>${s.ok ? 'bien' : '<strong>FALLA</strong>'}</td><td class="left">${detail(s)}</td></tr>`).join('')
  }</tbody></table></div>`;
  $('src-notes').innerHTML = snap.notes.length ? `<p>Notas de esta actualización:</p><ul>${snap.notes.map((n) => `<li class="num">${esc(n)}</li>`).join('')}</ul>` : '';
}

loadSnapshots(show, () => { $('src-when').textContent = 'No se ha podido cargar la última predicción, así que no se puede mostrar el estado de las fuentes.'; });
