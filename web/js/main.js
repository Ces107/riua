// Page controller: loads data and geodata, keeps the view state, and re-renders every part on change.

import { DEV, HORIZONS, HZ_LABEL, LEVEL, STALE_MIN } from './config.js';
import { cellAt, loadSnapshots } from './data.js';
import { geo, loadGeo, loadPoints, searchPlaces } from './geo.js';
import { defaultHorizon, headline } from './headline.js';
import { createMap } from './map.js';
import { context, placeTitle, renderPanel } from './panel.js';
import { renderPoints } from './points.js';
import { shareCard } from './share.js';
import { decode, isOwnHash, state, writeHash } from './state.js';
import { age, dayTime, esc, frameLabel, spanLabel } from './time.js';

const $ = (id) => document.getElementById(id);
let snap = null;
let mapApi = null;
let auditOpen = false;
let head = null;
let loadFailed = false;
const maxCache = new WeakMap();        // per horizon object: worst level per basin unit / control point

// ---- what the map paints -------------------------------------------------------------------------

function maxOverFrames(table) {         // [F][X] -> [X]
  if (!table || !table.length) return null;
  const out = table[0].map((v) => v || 0);
  for (let f = 1; f < table.length; f++) for (let k = 0; k < out.length; k++) if ((table[f][k] || 0) > out[k]) out[k] = table[f][k];
  return out;
}

function frameIndex(h) { return state.f === 'max' ? 'max' : Math.min(state.f, h.F - 1); }

function scene() {
  const h = snap && snap.hz[state.hz];
  const o = { snap, mode: state.mode, cellLevels: null, basinLevels: null, pointLevels: null, selection: null, selPoint: state.pt };
  if (h) {
    const f = frameIndex(h);
    if (!maxCache.has(h)) maxCache.set(h, { basins: maxOverFrames(h.basins && h.basins.level), points: maxOverFrames(h.points && h.points.level) });
    const mc = maxCache.get(h);
    o.cellLevels = f === 'max' ? h.maxLevel : h.level.subarray(f * h.N, (f + 1) * h.N);
    o.basinLevels = f === 'max' ? mc.basins : h.basins && h.basins.level ? h.basins.level[f] : null;
    o.pointLevels = f === 'max' ? mc.points : h.points && h.points.level ? h.points.level[f] : null;
  }
  if (state.sel) {
    const ctx = context(snap, state);
    o.selection = { lat: state.sel.lat, lon: state.sel.lon, cell: ctx && ctx.cell ? { i: ctx.cell.i, j: ctx.cell.j } : null, basin: ctx ? ctx.basin : null };
  }
  return o;
}

// ---- rendering -----------------------------------------------------------------------------------

function renderHeadline() {
  const el = $('headline');
  if (!snap) { el.textContent = loadFailed ? 'No se ha podido cargar la predicción. Comprueba la conexión y vuelve a cargar la página.' : 'Cargando la última predicción…'; return; }
  head = headline(snap);
  el.innerHTML = `${head.level >= 2 ? `<span class="lv lv${head.level}">${head.level}</span> ` : ''}${esc(head.text)}`;
}

function renderControls() {
  for (const key of HORIZONS) {
    const tab = $(`tab-${key}`);
    const h = snap && snap.hz[key];
    const on = key === state.hz;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    tab.disabled = !!snap && !h;
    tab.innerHTML = `${h ? `<span class="lv lv${h.top}">${h.top || '–'}</span>` : ''}${HZ_LABEL[key]}`;
    tab.setAttribute('aria-label', `${HZ_LABEL[key]}${h ? `: nivel máximo ${h.top}, ${LEVEL[h.top].name}` : ': sin datos'}`);
  }
  const h = snap && snap.hz[state.hz];
  const range = $('frame'), label = $('frame-label');
  if (!h) { range.disabled = true; $('prev').disabled = true; $('next').disabled = true; label.textContent = snap ? 'Sin datos para este plazo.' : ' '; return; }
  const f = frameIndex(h);
  range.disabled = false;
  range.max = String(h.F);
  range.value = String(f === 'max' ? 0 : f + 1);
  $('prev').disabled = f === 'max';
  $('next').disabled = f !== 'max' && f >= h.F - 1;
  let text;
  if (f === 'max') text = `Lo peor del periodo · ${spanLabel(h.frames)}`;
  else text = `${frameLabel(state.hz, h.frames[f])} · tramo ${f + 1} de ${h.F}${h.frames[f].ok ? '' : ' · sin datos'}`;
  label.textContent = text;
  range.setAttribute('aria-valuetext', text);
  $('mode-celdas').setAttribute('aria-pressed', String(state.mode === 'celdas'));
  $('mode-cuencas').setAttribute('aria-pressed', String(state.mode === 'cuencas'));
}

function renderPick(ctx) {
  const el = $('pick');
  if (!ctx) { el.innerHTML = '<span class="dim">Toca el mapa o busca tu municipio.</span>'; return; }
  let L = null;
  if (ctx.h && ctx.n >= 0) L = ctx.h.level[ctx.f * ctx.h.N + ctx.n];
  el.innerHTML = `${L == null ? '' : `<span class="lv lv${L}">${L || '–'}</span>`}<a href="#detalle"><strong>${esc(placeTitle(ctx))}</strong>${L == null ? ': fuera de la zona calculada' : ` · ${LEVEL[L].name}`} — ver el detalle</a>`;
}

function renderFresh() {
  const el = $('fresh');
  if (!snap) { el.textContent = loadFailed ? 'Sin datos: no se ha podido cargar la predicción.' : 'Cargando…'; return; }
  const now = new Date();
  const min = (now - snap.generated) / 60000;
  const stale = min > STALE_MIN;
  el.classList.toggle('stale', stale);
  const dev = DEV ? 'DATOS INVENTADOS PARA PRUEBAS · ' : '';
  el.textContent = stale
    ? `${dev}ATENCIÓN: la última predicción es de ${dayTime(snap.generated, now)} (${age(snap.generated, now)}). Puede no reflejar la situación actual.`
    : `${dev}Actualizado ${dayTime(snap.generated, now)} · ${age(snap.generated, now)}`;
}

function render(parts = {}) {
  renderHeadline();
  renderControls();
  if (mapApi) mapApi.redraw();
  const ctx = snap ? renderPanel($('place'), snap, state, { auditOpen: () => auditOpen, onAudit: (v) => { auditOpen = v; } }) : null;
  renderPick(ctx);
  renderPoints($('barrancos'), snap, state);
  renderFresh();
  if (snap && state.hz) writeHash();
  if (parts.scrollToPoint && state.pt) {
    const row = document.querySelector('#barrancos tr.open');
    if (row) row.scrollIntoView({ block: 'center' });
  }
}

// ---- actions -------------------------------------------------------------------------------------

function say(text) { $('msg').textContent = text || ''; }

function setHorizon(key) {
  if (!snap || !snap.hz[key]) return;
  state.hz = key; state.f = 'max';
  render();
}

function setFrame(v) {
  const h = snap && snap.hz[state.hz];
  if (!h) return;
  state.f = v === 'max' || v < 0 ? 'max' : Math.min(v, h.F - 1);
  render();
}

function select(lat, lon, ptId = null, reveal = 0) {
  state.sel = { lat, lon };
  state.pt = ptId;
  say('');
  if (reveal && mapApi) mapApi.reveal(lat, lon, reveal);
  render();
}

function selectPoint(id, scroll) {
  const pt = geo.points && geo.points.find((p) => p.id === id);
  if (!pt) return;
  if (state.pt === id && !scroll) { state.pt = null; render(); return; }
  state.mode = 'cuencas';
  state.sel = { lat: pt.lat, lon: pt.lon };
  state.pt = id;
  mapApi.reveal(pt.lat, pt.lon, 10.5);
  render({ scrollToPoint: scroll });
}

function wire() {
  // tabs: click, and arrow keys as a tablist
  const tabs = HORIZONS.map((k) => $(`tab-${k}`));
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => setHorizon(tab.dataset.hz));
    tab.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      for (let s = 1; s <= 3; s++) {
        const t = tabs[(i + d * s + 3) % 3];
        if (!t.disabled) { t.focus(); setHorizon(t.dataset.hz); break; }
      }
    });
  });
  $('frame').addEventListener('input', (e) => setFrame(Number(e.target.value) - 1));
  $('prev').addEventListener('click', () => setFrame(state.f === 'max' ? 'max' : state.f - 1));
  $('next').addEventListener('click', () => setFrame(state.f === 'max' ? 0 : state.f + 1));
  $('mode-celdas').addEventListener('click', () => { state.mode = 'celdas'; render(); });
  $('mode-cuencas').addEventListener('click', () => { state.mode = 'cuencas'; render(); });

  // panel: timeline cells and control-point links
  $('place').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.hz) { state.hz = b.dataset.hz; state.f = Number(b.dataset.f); render(); const again = $('place').querySelector('.tl-cells button.on'); if (again) again.focus(); }
    else if (b.dataset.pt) selectPoint(b.dataset.pt, true);
  });
  $('barrancos').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-pt]');
    if (b) selectPoint(b.dataset.pt, false);
  });

  // search
  const q = $('q'), sug = $('sug');
  let hits = [];
  const choose = (p) => { q.value = p.name; sug.hidden = true; sug.innerHTML = ''; select(p.lat, p.lon, null, 10); };
  const suggest = () => {
    hits = geo.places.length ? searchPlaces(q.value) : [];
    if (!q.value.trim()) { sug.hidden = true; sug.innerHTML = ''; return; }
    sug.hidden = false;
    sug.innerHTML = hits.length
      ? hits.map((p, i) => `<li><button type="button" data-i="${i}"><span>${esc(p.name)}${p.alt ? ` <span class="dim">/ ${esc(p.alt)}</span>` : ''}</span><span class="dim">${p.zone ? '' : 'fuera de la C. Valenciana'}</span></button></li>`).join('')
      : `<li><button type="button" disabled>${geo.places.length ? 'Ningún municipio con ese nombre.' : 'La lista de municipios aún no ha cargado.'}</button></li>`;
  };
  q.addEventListener('input', suggest);
  q.addEventListener('keydown', (e) => { if (e.key === 'Escape') { sug.hidden = true; } });
  sug.addEventListener('click', (e) => { const b = e.target.closest('button[data-i]'); if (b) choose(hits[Number(b.dataset.i)]); });
  $('search').addEventListener('submit', (e) => { e.preventDefault(); suggest(); if (hits.length) choose(hits[0]); });

  $('locate').addEventListener('click', () => {
    if (!navigator.geolocation) { say('Este navegador no ofrece la ubicación.'); return; }
    say('Buscando tu ubicación…');
    navigator.geolocation.getCurrentPosition((pos) => {
      const { latitude: lat, longitude: lon } = pos.coords;
      if (snap && cellAt(snap, lat, lon) < 0) { say('Tu ubicación queda fuera de la zona que se calcula (Comunitat Valenciana y cuencas que desaguan en ella).'); return; }
      select(lat, lon, null, 10);
    }, (err) => {
      say(err.code === 1 ? 'No has dado permiso para usar tu ubicación. Puedes buscar tu municipio.' : 'No se ha podido obtener tu ubicación. Puedes buscar tu municipio.');
    }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 });
  });

  // sharing
  $('copy').addEventListener('click', async () => {
    writeHash();
    const url = location.href;
    try { await navigator.clipboard.writeText(url); say('Enlace copiado: lleva a esta misma vista.'); } catch (e) {
      const t = document.createElement('textarea');
      t.value = url; document.body.appendChild(t); t.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
      t.remove();
      say(ok ? 'Enlace copiado: lleva a esta misma vista.' : `No se ha podido copiar. El enlace es: ${url}`);
    }
  });
  $('share').addEventListener('click', async () => {
    if (!snap || !head) { say('Todavía no hay predicción que compartir.'); return; }
    const btn = $('share');
    btn.disabled = true; say('Preparando la imagen…');
    try {
      const h = snap.hz[state.hz];
      const f = h ? frameIndex(h) : 'max';
      const ctx = context(snap, state);
      let sub = h ? `${HZ_LABEL[state.hz]} · ${f === 'max' ? 'lo peor del periodo' : frameLabel(state.hz, h.frames[f])}` : '';
      if (state.mode === 'cuencas') sub += ' · cuencas';
      if (ctx && ctx.h && ctx.n >= 0) sub += ` · ${placeTitle(ctx)}: nivel ${ctx.h.level[ctx.f * ctx.h.N + ctx.n]}`;
      writeHash();
      const site = (location.host + location.pathname).replace(/index\.html$/, '').replace(/\/$/, '');
      const res = await shareCard({ headline: head.text, level: head.level, sub, generated: snap.generated, url: site, scene: scene() }, location.href);
      say(res === 'downloaded' ? 'Imagen descargada (1080 × 1350).' : res === 'shared' ? 'Compartido.' : '');
    } catch (e) { say(`No se ha podido crear la imagen: ${e.message}.`); }
    btn.disabled = false;
  });

  window.addEventListener('hashchange', () => {
    if (isOwnHash(location.hash)) return;
    applyHash(decode(location.hash));
    render();
  });

  // the fixed bar must never cover content: reserve exactly its height
  const bar = $('bar');
  const fit = () => document.documentElement.style.setProperty('--bar-h', `${bar.offsetHeight + 6}px`);
  if (window.ResizeObserver) new ResizeObserver(fit).observe(bar);
  fit();
}

function applyHash(p) {
  if (p.hz) state.hz = p.hz;
  if (p.f != null) state.f = p.f;
  if (p.mode) state.mode = p.mode;
  state.sel = p.sel || null;
  state.pt = p.pt || null;
  if (p.view && mapApi) { state.view = p.view; mapApi.setView(p.view); }
}

function onSnapshot(s) {
  const first = !snap;
  snap = s;
  loadFailed = false;
  if (!state.hz || !snap.hz[state.hz]) { state.hz = defaultHorizon(snap); state.f = 'max'; }
  const h = snap.hz[state.hz];
  if (h && state.f !== 'max' && state.f >= h.F) state.f = 'max';
  if (HORIZONS.some((k) => snap.hz[k] && snap.hz[k].points)) loadPoints().then((pts) => { if (pts) render(); });
  render();
  if (first && DEV) console.info('riuà: datos inventados de web/dev (modo ?dev=1)');
}

async function refresh() {
  await loadSnapshots(onSnapshot, () => { if (!snap) { loadFailed = true; render(); } }, snap ? snap.generated.getTime() : 0);
}

function start() {
  const initial = decode(location.hash);
  if (initial.hz) state.hz = initial.hz;
  if (initial.f != null) state.f = initial.f;
  if (initial.mode) state.mode = initial.mode;
  if (initial.sel) state.sel = initial.sel;
  if (initial.pt) state.pt = initial.pt;

  mapApi = createMap($('map'), scene, {
    onPick: (lat, lon, ptId) => { if (ptId) selectPoint(ptId, false); else select(lat, lon, null); },
    onView: () => { state.view = mapApi.view(); if (snap) writeHash(); },
  });
  if (initial.view) { state.view = initial.view; mapApi.setView(initial.view); }
  wire();
  render();

  // geodata arrives file by file: repaint the map each time, and the texts once names and zones are in
  let pending = null;
  loadGeo((name, err) => {
    if (err) say('Parte de la cartografía no ha cargado; el mapa puede verse incompleto.');
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = null; render(); });
  });
  refresh();
  setInterval(renderFresh, 30000);
  setInterval(refresh, 5 * 60000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { renderFresh(); if (snap && Date.now() - snap.generated > 20 * 60000) refresh(); } });
}

if (window.L) start();
else $('headline').textContent = 'No se ha podido cargar el mapa. Vuelve a cargar la página.';
