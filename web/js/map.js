// The Leaflet map: an optional street base map (only when zoomed in) and one canvas layer of our own
// that paints the whole scene with draw.js.

import { IGN_TILES, TILE_MIN_ZOOM } from './config.js';
import { drawLevels, drawLines, drawTop, drawUnder, geoStamp } from './draw.js';
import { geo } from './geo.js';

const L = window.L;
const PAD = 0.3;          // the canvas is this much larger than the map on every side, so a short drag never shows its edge
const IGN = '© <a href="https://www.ign.es/">IGN</a>';
const SHEET = [-2.7, 37.3, 1.1, 41.3];      // w, s, e, n of what the cartography covers (land_box of geo/boundary.geojson)
const CV = [-1.55, 37.83, 0.55, 40.8];      // the Comunitat Valenciana with a hair of margin
const mercDeg = (lat) => (Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) * 180) / Math.PI;   // Mercator y in degrees of longitude
const latOf = (m) => (Math.atan(Math.exp((m * Math.PI) / 180)) * 360) / Math.PI - 90;

const SceneLayer = L.Layer.extend({
  initialize(getScene) { this._getScene = getScene; this._fonts = 0; },

  onAdd(map) {
    this._canvas = L.DomUtil.create('canvas', 'riua-scene leaflet-zoom-animated');
    this._canvas.setAttribute('aria-hidden', 'true');
    map.getPane('overlayPane').appendChild(this._canvas);
    map.on('moveend zoomend resize', this.redraw, this);
    map.on('zoomanim', this._animateZoom, this);
    this.redraw();
  },

  onRemove(map) {
    this._canvas.remove();
    map.off('moveend zoomend resize', this.redraw, this);
    map.off('zoomanim', this._animateZoom, this);
  },

  // same arithmetic as L.Renderer: keep the bitmap glued to the ground while the zoom animation runs
  _animateZoom(e) {
    const map = this._map;
    const scale = map.getZoomScale(e.zoom, this._zoom);
    const viewHalf = map.getSize().multiplyBy(0.5 + PAD);
    const offset = viewHalf.multiplyBy(-scale).add(map.project(this._center, e.zoom)).subtract(map._getNewPixelOrigin(e.center, e.zoom));
    L.DomUtil.setTransform(this._canvas, offset, scale);
  },

  /** Rectangles (canvas pixels) covered by the legend and the map controls: labels keep out of them. */
  _covered(pad) {
    const el = this._map.getContainer(), base = el.getBoundingClientRect();
    const out = [];
    for (const c of [...el.querySelectorAll('.leaflet-control'), ...el.parentNode.querySelectorAll('.legend')]) {
      const r = c.getBoundingClientRect();
      if (r.width && r.height) out.push([r.left - base.left + pad.x - 4, r.top - base.top + pad.y - 4, r.width + 8, r.height + 8]);
    }
    return out;
  },

  redraw() {
    const map = this._map;
    if (!map) return;
    const size = map.getSize();
    if (!size.x || !size.y) return;
    const pad = size.multiplyBy(PAD).round();
    const topLeft = map.containerPointToLayerPoint([-pad.x, -pad.y]).round();
    const w = size.x + 2 * pad.x, h = size.y + 2 * pad.y;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cv = this._canvas;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    cv.style.width = `${w}px`; cv.style.height = `${h}px`;
    L.DomUtil.setTransform(cv, topLeft, 1);
    this._center = map.getCenter(); this._zoom = map.getZoom();

    // layer point of (lon, lat) = projected pixel - pixel origin; written out so that tens of thousands
    // of vertices do not each allocate Leaflet objects
    const zoom = map.getZoom(), world = 256 * Math.pow(2, zoom), origin = map.getPixelOrigin();
    const ox = origin.x + topLeft.x, oy = origin.y + topLeft.y;
    const nw = map.containerPointToLatLng([-pad.x, -pad.y]), se = map.containerPointToLatLng([size.x + pad.x, size.y + pad.y]);
    const P = {
      zoom,
      view: [nw.lng, se.lat, se.lng, nw.lat],
      x: (lon) => (world * (lon + 180)) / 360 - ox,
      y: (lat) => {
        const s = Math.sin((lat * Math.PI) / 180);
        return world * (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) - oy;
      },
    };
    const streets = zoom >= TILE_MIN_ZOOM;
    const o = { ...this._getScene(), base: !streets, labels: !streets, k: 1, blocked: this._covered(pad), frame: [pad.x, pad.y, size.x, size.y] };
    // everything but the levels only changes with the view, the geodata or the selection: what goes under
    // them and what goes over them are kept as two bitmaps, so that stepping through the frames only
    // repaints the levels
    const view = `${zoom}|${ox},${oy}|${cv.width}x${cv.height}|${streets}|${o.mode}|${geoStamp()}|${this._fonts}`;
    const s = o.selection;
    const top = `${view}|${o.selPoint}|${s ? `${s.lat},${s.lon},${s.cell ? `${s.cell.i}.${s.cell.j}` : ''},${s.basin ? s.basin.id : ''}` : ''}`
      + `|${o.mode === 'cuencas' && o.pointLevels ? o.pointLevels.join('') : ''}|${o.blocked.map((b) => b.map(Math.round).join(',')).join(';')}`;
    const under = this._kept('under', view, cv, dpr, (c) => drawUnder(c, P, o));
    const over = this._kept('over', top, cv, dpr, (c) => { drawLines(c, P, o); drawTop(c, P, o); });
    const ctx = cv.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (o.base) ctx.drawImage(under, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawLevels(ctx, P, o);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(over, 0, 0);
  },

  /** An off-screen bitmap the size of the canvas, repainted only when its key changes. */
  _kept(name, key, cv, dpr, paint) {
    const all = this._layers || (this._layers = {});
    const it = all[name] || (all[name] = { cv: document.createElement('canvas'), key: null });
    if (it.key !== key) {
      it.cv.width = cv.width; it.cv.height = cv.height;            // also wipes it
      const c = it.cv.getContext('2d');
      c.setTransform(dpr, 0, 0, dpr, 0, 0);
      paint(c);
      it.key = key;
    }
    return it.cv;
  },

  /** The web font arrived: what was written with the fallback is repainted. */
  fontsReady() { this._fonts += 1; this.redraw(); },
});

/**
 * handlers: onPick(lat, lon, pointId|null), onView()            getScene(): what to draw (see draw.js)
 */
export function createMap(el, getScene, handlers) {
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const map = L.map(el, {
    // no zoom snapping: the home view must fit the box exactly. The map cannot leave the sheet.
    zoomSnap: 0, zoomDelta: 0.5, minZoom: 5, maxZoom: 15, wheelPxPerZoomLevel: 90,
    maxBounds: [[SHEET[1], SHEET[0]], [SHEET[3], SHEET[2]]], maxBoundsViscosity: 1,
    attributionControl: false, zoomControl: false, fadeAnimation: false, markerZoomAnimation: false, zoomAnimation: !still,
    // the page must scroll: the wheel only zooms with Ctrl, and on touch one finger scrolls the page
    scrollWheelZoom: false, dragging: !L.Browser.mobile, touchZoom: true, tap: false,
  });
  el.addEventListener('wheel', (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    map.setZoomAround(map.mouseEventToLatLng(e), map.getZoom() + (e.deltaY < 0 ? 0.5 : -0.5), { animate: false });
  }, { passive: false });
  L.control.zoom({ position: 'topright', zoomInTitle: 'Acercar', zoomOutTitle: 'Alejar' }).addTo(map);
  // the only third-party imagery is the street map, and it is only there when zoomed in: credit it only then
  // (every other credit, Leaflet included, is on the "Fuentes" page)
  const credit = L.control.attribution({ position: 'bottomright', prefix: false }).addTo(map);
  L.control.scale({ position: 'bottomright', imperial: false, maxWidth: 90 }).addTo(map);
  L.tileLayer(IGN_TILES, { minZoom: TILE_MIN_ZOOM, maxZoom: 15, className: 'riua-tiles' }).addTo(map);
  let credited = false;
  const creditNow = () => {
    const on = map.getZoom() >= TILE_MIN_ZOOM;
    if (on === credited) return;
    credited = on;
    if (on) credit.addAttribution(IGN); else credit.removeAttribution(IGN);
  };
  map.on('zoomend', creditNow);
  const layer = new SceneLayer(getScene);

  // Home view: the Comunitat Valenciana from top to bottom. The page sizes the map (css, --mapw) so that at
  // this scale the sheet is at least as wide as the map: no blank paper beside it. Where the map is wider than
  // the sheet anyway (a short phone), the sheet is centred.
  const homeView = () => {
    const s = map.getSize();
    const top = mercDeg(CV[3]), bottom = mercDeg(CV[1]);
    const ppd = Math.min((s.y - 12) / (top - bottom), (s.x - 12) / (CV[2] - CV[0]));      // px per degree of longitude
    const half = s.x / 2 / ppd;
    const lon = SHEET[0] + half <= SHEET[2] - half ? Math.min(Math.max((CV[0] + CV[2]) / 2, SHEET[0] + half), SHEET[2] - half) : (SHEET[0] + SHEET[2]) / 2;
    return { lat: latOf((top + bottom) / 2), lon, z: Math.log2((ppd * 360) / 256) };
  };
  let homeAt = null;                    // the home view last applied
  let homeSize = { x: 0, y: 0 };        // ... and the size of the map it was worked out for
  let atHome = true;                    // false once the reader (or a link) has moved the map
  const home = () => {
    const s = map.getSize();
    if (!s.x || !s.y) return;           // born without a size (tab opened hidden): fitted when it first gets one
    homeAt = homeView();
    homeSize = s;
    map.options.minZoom = Math.min(homeAt.z, map.getZoom() ?? homeAt.z);   // (set directly: setMinZoom would animate a zoom)
    map.setView([homeAt.lat, homeAt.lon], homeAt.z, { animate: false });
    map.setMinZoom(homeAt.z);           // nothing to see further out
    atHome = true;
  };
  const isHome = () => {
    if (!homeAt) return true;
    const d = map.latLngToContainerPoint([homeAt.lat, homeAt.lon]).distanceTo(map.getSize().divideBy(2));
    return Math.abs(map.getZoom() - homeAt.z) < 1e-3 && d < 3;
  };
  home();
  map.on('moveend', () => { atHome = isHome(); });
  creditNow();
  layer.addTo(map);
  if (document.fonts && document.fonts.addEventListener) document.fonts.addEventListener('loadingdone', () => layer.fontsReady());

  const pick = (latlng, containerPoint) => {
    let hit = null;
    if (geo.points && getScene().mode === 'cuencas') {
      let best = 16;                                   // px
      for (const pt of geo.points) {
        const d = map.latLngToContainerPoint([pt.lat, pt.lon]).distanceTo(containerPoint);
        if (d < best) { best = d; hit = pt; }
      }
    }
    if (hit) handlers.onPick(hit.lat, hit.lon, hit.id);
    else handlers.onPick(latlng.lat, latlng.lng, null);
  };
  map.on('click', (e) => pick(e.latlng, e.containerPoint));
  // keyboard: arrows move the map, + and − zoom (Leaflet); Enter chooses what is at its centre
  el.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.target !== el) return;
    e.preventDefault();
    pick(map.getCenter(), map.getSize().divideBy(2));
  });
  map.on('moveend', () => handlers.onView());

  return {
    map,
    redraw: () => layer.redraw(),
    /** The map box changed size: the home view follows it; a view chosen by the reader is kept. */
    invalidate: () => {
      map.invalidateSize({ pan: false });
      const s = map.getSize();
      // (compared with the size the home view was made for: Leaflet has usually seen the new size already)
      if (s.x && s.y && (s.x !== homeSize.x || s.y !== homeSize.y)) {
        if (atHome || !homeAt) home();
        else { homeAt = homeView(); homeSize = s; map.setMinZoom(Math.min(homeAt.z, map.getZoom())); }
      }
      layer.redraw();
    },
    home,
    /** null while the map shows the home view, so that a link does not freeze somebody else's screen size into it. */
    view() { if (atHome) return null; const c = map.getCenter(); return { lat: c.lat, lon: c.lng, z: map.getZoom() }; },
    setView(v) { map.setView([v.lat, v.lon], v.z, { animate: false }); atHome = isHome(); },
    /** Brings a place into view without changing the zoom more than needed. */
    reveal(lat, lon, minZoom) {
      const z = Math.max(map.getZoom(), minZoom || 0);
      if (z !== map.getZoom() || !map.getBounds().pad(-0.15).contains([lat, lon])) map.setView([lat, lon], z, { animate: false });
    },
  };
}
