// Page controller: loads data and geodata, keeps the view state, and re-renders every part on change.

import { DEV, HORIZONS, HZ_LABEL, LATE_MIN, LEVEL, STALE_MIN } from './config.js';
import { cellAt, loadSnapshots } from './data.js';
import { geo, loadCatchments, loadGeo, loadPoints, searchPlaces } from './geo.js';
import { defaultHorizon, headline } from './headline.js';
import { createMap } from './map.js';
import { context, placeTitle, renderPanel } from './panel.js';
import { renderPoints } from './points.js';
import { shareCard } from './share.js';
import { decode, isOwnHash, state, writeHash } from './state.js';
import { age, dayTime, esc, frameLabel } from './time.js';

const $ = (id) => document.getElementById(id);
let snap = null;
let mapApi = null;
let auditOpen = false;
let head = null;
let loadFailed = false;
let catchmentsAsked = false;
let wantT = null;                     // frame start asked for by the link (ISO), resolved when the data arrives
const maxCache = new WeakMap();        // per horizon object: worst level per basin unit / control point
const wide = window.matchMedia('(min-width: 980px)');
const calm = window.matchMedia('(prefers-reduced-motion: reduce)');

// ---- what the map paints -------------------------------------------------------------------------

function maxOverFrames(table) {         // [F][X] -> [X]
  if (!table || !table.length) return null;
  const out = table[0].map((v) => v || 0);
  for (let f = 1; f < table.length; f++) for (let k = 0; k < out.length; k++) if ((table[f][k] || 0) > out[k]) out[k] = table[f][k];
  return out;
}

function frameIndex(h) { return state.f === 'max' ? 'max' : Math.min(state.f, h.F - 1); }

/** ISO start of the selected frame, or null in the "máximo" view: what the link carries. */
function frameT0() {
  const h = snap && snap.hz[state.hz];
  return h && state.f !== 'max' ? h.frames[Math.min(state.f, h.F - 1)].t0 : null;
}

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

function setHtml(el, html) { if (el.dataset.html !== html) { el.dataset.html = html; el.innerHTML = html; } }

function renderHeadline() {
  const el = $('headline');
  if (!snap) { setHtml(el, loadFailed ? 'Sin datos. <button type="button" class="link" id="retry">Reintentar</button>' : 'Cargando…'); return; }
  head = headline(snap);
  // "level and where · when": the hours never break across two lines
  const cut = head.text.lastIndexOf(' · ');
  const text = cut < 0 ? esc(head.text) : `${esc(head.text.slice(0, cut))} <span class="nw">· ${esc(head.text.slice(cut + 3))}</span>`;
  setHtml(el, `${head.level >= 2 ? `<span class="lv lv${head.level}">${head.level}</span> ` : ''}${text}`);
}

function renderControls() {
  for (const key of HORIZONS) {
    const tab = $(`tab-${key}`);
    const h = snap && snap.hz[key];
    const on = key === state.hz;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on || (!state.hz && key === 'now') ? 0 : -1;
    tab.disabled = !!snap && !h;
    setHtml(tab, `${h ? `<span class="lv lv${h.top}" aria-hidden="true">${h.top || '–'}</span>` : ''}${HZ_LABEL[key]}`);
    tab.setAttribute('aria-label', `${HZ_LABEL[key]}${h ? `: nivel máximo ${h.top}, ${LEVEL[h.top].name}` : snap ? ': sin datos' : ''}`);
  }
  $('mode-celdas').setAttribute('aria-pressed', String(state.mode === 'celdas'));
  $('mode-cuencas').setAttribute('aria-pressed', String(state.mode === 'cuencas'));
  $('zi').setAttribute('aria-pressed', String(!!state.zi));
  const h = snap && snap.hz[state.hz];
  const range = $('frame'), label = $('frame-label');
  if (!h) { range.disabled = true; $('prev').disabled = true; $('next').disabled = true; label.textContent = snap ? 'sin datos' : ' '; return; }
  const f = frameIndex(h);
  range.disabled = false;
  range.max = String(h.F);
  range.value = String(f === 'max' ? 0 : f + 1);
  $('prev').disabled = f === 'max';
  $('next').disabled = f !== 'max' && f >= h.F - 1;
  let text;
  if (f === 'max') text = 'máximo';
  else text = `${frameLabel(state.hz, h.frames[f])}${h.frames[f].ok ? '' : ' · sin datos'}`;
  label.textContent = text;
  range.setAttribute('aria-valuetext', text);
}

function renderFresh() {
  const el = $('fresh');
  if (!snap) { el.textContent = loadFailed ? '' : ' '; el.className = 'num'; return; }
  const now = new Date();
  const min = (now - snap.generated) / 60000;
  const stale = min > STALE_MIN, late = !stale && min > LATE_MIN;
  el.className = `num${stale ? ' stale' : late ? ' late' : ''}`;
  const dev = DEV ? 'PRUEBA · ' : '';
  const when = dayTime(snap.generated, now);
  el.textContent = stale ? `${dev}Desactualizado · ${age(snap.generated, now)}` : `${dev}${when} · ${age(snap.generated, now)}`;
  el.title = `Última actualización: ${when}`;
}

function render(parts = {}) {
  renderHeadline();
  renderControls();
  if (state.pt && !geo.catchments && !catchmentsAsked) { catchmentsAsked = true; loadCatchments().then((c) => { if (c && mapApi) mapApi.redraw(); }); }
  if (mapApi) mapApi.redraw();
  if (snap) renderPanel($('place'), snap, state, { auditOpen: () => auditOpen, onAudit: (v) => { auditOpen = v; } });
  renderPoints($('barrancos'), snap, state);
  renderFresh();
  if (snap && state.hz) writeHash(frameT0());
  if (parts.scrollToPoint && state.pt) {
    const row = document.querySelector('#barrancos tr.open');
    if (row) row.scrollIntoView({ block: 'center' });
  }
}

// ---- actions -------------------------------------------------------------------------------------

let sayTimer = 0;
function say(text, keep = false) {
  $('msg').textContent = text || '';
  clearTimeout(sayTimer);
  if (text && !keep) sayTimer = setTimeout(() => { $('msg').textContent = ''; }, 6000);
}

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

/** One column (phone, tablet): after choosing a place, bring its name and level into view with the least scroll. */
function peek() {
  if (wide.matches) return;
  const el = document.querySelector('#place .lvl') || document.querySelector('#place h2');
  if (el) el.scrollIntoView({ block: 'nearest', behavior: calm.matches ? 'auto' : 'smooth' });
}

function select(lat, lon, ptId = null, reveal = 0) {
  const inPanel = $('place').contains(document.activeElement);
  state.sel = { lat, lon };
  state.pt = ptId;
  say('');
  if (reveal && mapApi) mapApi.reveal(lat, lon, reveal);
  render();
  if (inPanel) $('detalle').focus({ preventScroll: true });   // the button that was pressed no longer exists
  peek();
}

function clearSelection() {
  state.sel = null; state.pt = null;
  $('q').value = '';
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
  const btn = [...document.querySelectorAll('#barrancos tr.open button[data-pt]')].find((b) => b.dataset.pt === id);
  if (btn && scroll) btn.focus({ preventScroll: true });
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
  $('prev').addEventListener('click', () => { setFrame(state.f === 'max' ? 'max' : state.f - 1); if ($('prev').disabled) $('next').focus(); });
  $('next').addEventListener('click', () => { setFrame(state.f === 'max' ? 0 : state.f + 1); if ($('next').disabled) $('prev').focus(); });
  $('mode-celdas').addEventListener('click', () => { state.mode = 'celdas'; render(); });
  $('mode-cuencas').addEventListener('click', () => { state.mode = 'cuencas'; render(); });
  $('zi').addEventListener('click', () => { state.zi = !state.zi; mapApi.floodZones(state.zi); render(); });
  $('headline').addEventListener('click', (e) => { if (e.target.id === 'retry') { loadFailed = false; render(); refresh(); } });

  // panel: zones, timeline cells, control-point links, back to the zones
  $('place').addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.act === 'back') { clearSelection(); $('q').focus({ preventScroll: true }); return; }
    if (b.dataset.ll) {
      const [la, lo] = b.dataset.ll.split(',').map(Number);
      if (b.dataset.hz && snap.hz[b.dataset.hz]) { state.hz = b.dataset.hz; state.f = 'max'; }
      if (Number.isFinite(la)) select(la, lo, null, 9);
      return;
    }
    if (b.dataset.hz) { state.hz = b.dataset.hz; state.f = Number(b.dataset.f); render(); const again = $('place').querySelector('.strip button.on'); if (again) again.focus(); }
    else if (b.dataset.pt) selectPoint(b.dataset.pt, true);
  });
  $('barrancos').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-pt]');
    if (b) selectPoint(b.dataset.pt, false);
  });

  // search: a list of buttons under the box; arrows move through it, Escape closes it
  const q = $('q'), sug = $('sug'), form = $('search');
  let hits = [];
  const close = () => { sug.hidden = true; sug.innerHTML = ''; q.setAttribute('aria-expanded', 'false'); };
  const choose = (p) => { q.value = p.name; close(); select(p.lat, p.lon, null, 10); };
  const suggest = () => {
    hits = geo.places.length ? searchPlaces(q.value) : [];
    if (!q.value.trim()) { close(); return; }
    sug.hidden = false;
    q.setAttribute('aria-expanded', 'true');
    sug.innerHTML = hits.length
      ? hits.map((p, i) => `<li><button type="button" data-i="${i}"><span>${esc(p.name)}${p.alt ? ` <span class="dim">/ ${esc(p.alt)}</span>` : ''}</span><span class="dim">${p.zone ? '' : 'fuera de la C. Valenciana'}</span></button></li>`).join('')
      : `<li><button type="button" disabled>${geo.places.length ? 'Sin resultados' : 'Cargando…'}</button></li>`;
  };
  q.addEventListener('input', suggest);
  form.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { if (!sug.hidden) { close(); q.focus(); } return; }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const items = [...sug.querySelectorAll('button[data-i]')];
    if (!items.length) return;
    e.preventDefault();
    const at = items.indexOf(document.activeElement);
    const to = e.key === 'ArrowDown' ? at + 1 : at - 1;
    if (to < 0) q.focus(); else items[Math.min(to, items.length - 1)].focus();
  });
  form.addEventListener('focusout', (e) => { if (!form.contains(e.relatedTarget)) setTimeout(() => { if (!form.contains(document.activeElement)) close(); }, 150); });
  sug.addEventListener('click', (e) => { const b = e.target.closest('button[data-i]'); if (b) choose(hits[Number(b.dataset.i)]); });
  form.addEventListener('submit', (e) => { e.preventDefault(); suggest(); if (hits.length) choose(hits[0]); });

  $('locate').addEventListener('click', () => {
    if (!navigator.geolocation) { say('Este navegador no da la ubicación.'); return; }
    say('Buscando…', true);
    navigator.geolocation.getCurrentPosition((pos) => {
      const { latitude: lat, longitude: lon } = pos.coords;
      if (snap && cellAt(snap, lat, lon) < 0) { say('Tu ubicación queda fuera del mapa.'); return; }
      select(lat, lon, null, 10);
    }, (err) => {
      say(err.code === 1 ? 'Sin permiso de ubicación.' : 'No se ha podido obtener la ubicación.');
    }, { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 });
  });

  // sharing
  $('copy').addEventListener('click', async () => {
    writeHash(frameT0());
    const url = location.href;
    try { await navigator.clipboard.writeText(url); say('Enlace copiado.'); } catch (e) {
      const t = document.createElement('textarea');
      t.value = url; document.body.appendChild(t); t.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (e2) { ok = false; }
      t.remove();
      say(ok ? 'Enlace copiado.' : url, !ok);
    }
  });
  $('share').addEventListener('click', async () => {
    if (!snap || !head) return;
    const btn = $('share');
    btn.disabled = true;
    try {
      const h = snap.hz[state.hz];
      const f = h ? frameIndex(h) : 'max';
      const ctx = context(snap, state);
      let sub = h ? `${HZ_LABEL[state.hz]} · ${f === 'max' ? 'máximo' : frameLabel(state.hz, h.frames[f])}` : '';
      if (state.mode === 'cuencas') sub += ' · cauces';
      if (ctx && ctx.h && ctx.n >= 0) sub += ` · ${placeTitle(ctx)}: nivel ${ctx.h.level[ctx.f * ctx.h.N + ctx.n]}`;
      writeHash(frameT0());
      const site = (location.host + location.pathname).replace(/index\.html$/, '').replace(/\/$/, '');
      const res = await shareCard({ headline: head.text, level: head.level, sub, generated: snap.generated, url: site, scene: scene() }, location.href);
      say(res === 'downloaded' ? 'Imagen descargada.' : '');
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
  const fit = () => {
    document.documentElement.style.setProperty('--bar-h', `${bar.offsetHeight}px`);
    if (mapApi && mapApi.invalidate) mapApi.invalidate();
  };
  if (window.ResizeObserver) { const ro = new ResizeObserver(fit); ro.observe(bar); ro.observe($('map')); }
  fit();
}

/** Frame index for a frame start (ISO) in the current horizon, or 'max' when that frame no longer exists. */
function frameAt(t0) {
  const h = snap && snap.hz[state.hz];
  if (!h || !t0) return 'max';
  const ms = Date.parse(t0);
  const f = h.frames.findIndex((fr) => fr.d0.getTime() === ms);
  return f < 0 ? 'max' : f;
}

function applyHash(p) {
  if (p.hz) state.hz = p.hz;
  if (p.t) { wantT = p.t; if (snap) { state.f = frameAt(wantT); wantT = null; } } else if (p.f != null) state.f = p.f;
  if (p.mode) state.mode = p.mode;
  state.zi = !!p.zi; if (mapApi) mapApi.floodZones(state.zi);
  state.sel = p.sel || null;
  state.pt = p.pt || null;
  if (p.view && mapApi) { state.view = p.view; mapApi.setView(p.view); }
}

function onSnapshot(s) {
  const first = !snap;
  // the selected frame is a time, not a position: keep the same hours when the frames move on
  const keepT = wantT || frameT0();
  wantT = null;
  snap = s;
  loadFailed = false;
  if (!state.hz || !snap.hz[state.hz]) { state.hz = defaultHorizon(snap); state.f = 'max'; }
  else if (keepT) state.f = frameAt(keepT);
  const h = snap.hz[state.hz];
  if (h && state.f !== 'max' && state.f >= h.F) state.f = 'max';
  if (HORIZONS.some((k) => snap.hz[k] && snap.hz[k].points)) loadPoints().then((pts) => { if (pts) render(); });
  render();
  if (first && DEV) console.info('riuà: datos inventados de web/dev (modo ?dev=1)');
}

let busy = false;
async function refresh() {
  if (busy) return;
  busy = true;
  await loadSnapshots(onSnapshot, (e) => { if (!snap) { loadFailed = true; render(); } console.warn('riuà: no se ha podido actualizar', e); }, snap ? snap.generated.getTime() : 0);
  busy = false;
}

function start() {
  const initial = decode(location.hash);
  if (initial.hz) state.hz = initial.hz;
  if (initial.t) wantT = initial.t; else if (initial.f != null) state.f = initial.f;
  if (initial.mode) state.mode = initial.mode;
  if (initial.sel) state.sel = initial.sel;
  if (initial.pt) state.pt = initial.pt;

  mapApi = createMap($('map'), scene, {
    onPick: (lat, lon, ptId) => { if (ptId) selectPoint(ptId, false); else select(lat, lon, null); },
    onView: () => { state.view = mapApi.view(); if (snap) writeHash(frameT0()); },
  });
  // the flood-zone toggle appears only once its tiles are published
  fetch('geo/zi500/index.json', { method: 'HEAD' }).then((r) => { if (r.ok) $('zi').hidden = false; }).catch(() => {});
  if (initial.zi) { state.zi = true; mapApi.floodZones(true); }
  if (initial.view) { state.view = initial.view; mapApi.setView(initial.view); }
  wire();
  render();

  // geodata arrives file by file: repaint the map each time, and the texts once names and zones are in
  let pending = null;
  loadGeo((name, err) => {
    if (err) say('Parte del mapa no ha cargado.', true);
    if (pending) return;
    pending = requestAnimationFrame(() => { pending = null; render(); });
  });
  refresh();
  setInterval(renderFresh, 30000);
  setInterval(() => { if (!document.hidden) refresh(); }, 3 * 60000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) { renderFresh(); if (!snap || Date.now() - snap.generated > 10 * 60000) refresh(); } });
}

if (window.L) start();
else $('headline').textContent = 'No se ha podido cargar el mapa.';
