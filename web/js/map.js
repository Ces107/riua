// The Leaflet map: an optional street base map (only when zoomed in) and one canvas layer of our own
// that paints the whole scene with draw.js.

import { IGN_TILES, TILE_MIN_ZOOM } from './config.js';
import { drawScene } from './draw.js';
import { geo } from './geo.js';

const L = window.L;
const PAD = 0.3;          // the canvas is this much larger than the map on every side, so a short drag never shows its edge

const SceneLayer = L.Layer.extend({
  initialize(getScene) { this._getScene = getScene; },

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

  redraw() {
    const map = this._map;
    if (!map) return;
    const size = map.getSize();
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
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const scene = this._getScene();
    const streets = zoom >= TILE_MIN_ZOOM;
    drawScene(ctx, P, { ...scene, base: !streets, labels: !streets, k: 1 });
  },
});

/**
 * handlers: onPick(lat, lon, pointId|null), onView()            getScene(): what to draw (see draw.js)
 */
export function createMap(el, getScene, handlers) {
  const map = L.map(el, {
    zoomSnap: 0.25, zoomDelta: 0.5, minZoom: 6.5, maxZoom: 15, wheelPxPerZoomLevel: 90,
    maxBounds: [[36.6, -4.2], [42.0, 2.6]], maxBoundsViscosity: 0.8,
    attributionControl: false, zoomControl: false, fadeAnimation: false, markerZoomAnimation: false,
    // the page must scroll: the wheel only zooms with Ctrl, and on touch one finger scrolls the page
    scrollWheelZoom: false, dragging: !L.Browser.mobile, touchZoom: true, tap: false,
  });
  el.addEventListener('wheel', (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    map.setZoomAround(map.mouseEventToLatLng(e), map.getZoom() + (e.deltaY < 0 ? 0.5 : -0.5), { animate: false });
  }, { passive: false });
  L.control.zoom({ position: 'topright', zoomInTitle: 'Acercar', zoomOutTitle: 'Alejar' }).addTo(map);
  L.control.attribution({ position: 'bottomright', prefix: '<a href="https://leafletjs.com">Leaflet</a>' }).addTo(map);
  L.control.scale({ position: 'topleft', imperial: false, maxWidth: 110 }).addTo(map);
  L.tileLayer(IGN_TILES, { minZoom: TILE_MIN_ZOOM, maxZoom: 15, className: 'riua-tiles',
    attribution: '© <a href="https://www.ign.es/">Instituto Geográfico Nacional</a>' }).addTo(map);
  const layer = new SceneLayer(getScene);

  const home = () => map.fitBounds([[37.83, -1.55], [40.8, 0.55]], { animate: false, padding: [6, 6] });
  home();
  layer.addTo(map);

  map.on('click', (e) => {
    let hit = null;
    if (geo.points && getScene().mode === 'cuencas') {
      let best = 16;                                   // px
      for (const pt of geo.points) {
        const d = map.latLngToContainerPoint([pt.lat, pt.lon]).distanceTo(e.containerPoint);
        if (d < best) { best = d; hit = pt; }
      }
    }
    if (hit) handlers.onPick(hit.lat, hit.lon, hit.id);
    else handlers.onPick(e.latlng.lat, e.latlng.lng, null);
  });
  map.on('moveend', () => handlers.onView());

  return {
    map,
    redraw: () => layer.redraw(),
    invalidate: () => { map.invalidateSize({ pan: false }); layer.redraw(); },
    home,
    view() { const c = map.getCenter(); return { lat: c.lat, lon: c.lng, z: map.getZoom() }; },
    setView(v) { map.setView([v.lat, v.lon], v.z, { animate: false }); },
    /** Brings a place into view without changing the zoom more than needed. */
    reveal(lat, lon, minZoom) {
      const z = Math.max(map.getZoom(), minZoom || 0);
      if (z !== map.getZoom() || !map.getBounds().pad(-0.15).contains([lat, lon])) map.setView([lat, lon], z, { animate: false });
    },
  };
}
