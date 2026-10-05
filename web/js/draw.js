// One painter for the whole scene (sea, land, cells or basin units, rivers, boundaries, towns,
// control points, selection). The Leaflet layer and the share card both call it; they only differ in
// the projection they hand over:  P = { x(lon), y(lat), view:[w,s,e,n] in degrees, zoom }.

import { CELL_ALPHA, INK, LAND_OUT, LEVEL, MONO, PAPER, RIVER, SEA, SERIF } from './config.js';
import { geo } from './geo.js';

const STREET_ALPHA = 0.56;
let hatchTile = null;

/** Diagonal paper-coloured lines: laid over level 5 so it is unmistakable without colour. */
export function hatch(ctx) {
  if (!hatchTile) {
    hatchTile = document.createElement('canvas');
    hatchTile.width = hatchTile.height = 7;
    const c = hatchTile.getContext('2d');
    c.strokeStyle = PAPER;
    c.lineWidth = 1.4;
    c.lineCap = 'square';
    c.beginPath();
    c.moveTo(-1, 8); c.lineTo(8, -1);
    c.moveTo(-1, 1); c.lineTo(1, -1);
    c.moveTo(6, 8); c.lineTo(8, 6);
    c.stroke();
  }
  return ctx.createPattern(hatchTile, 'repeat');
}

const seen = (b, v) => !(b[2] < v[0] || b[0] > v[2] || b[3] < v[1] || b[1] > v[3]);

function trace(ctx, P, rings, close) {
  for (const r of rings) {
    ctx.moveTo(P.x(r[0]), P.y(r[1]));
    for (let k = 2; k < r.length; k += 2) ctx.lineTo(P.x(r[k]), P.y(r[k + 1]));
    if (close) ctx.closePath();
  }
}

function strokeShapes(ctx, P, shapes, style, width, dash) {
  ctx.beginPath();
  for (const s of shapes) if (s && seen(s.bbox, P.view)) trace(ctx, P, s.rings, false);
  ctx.strokeStyle = style; ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * The sheet: sea, land and a half-degree graticule, only inside the rectangle the cartography covers
 * (geo.land's box), closed by a neatline. Outside it nothing is known, so nothing is drawn: plain paper,
 * not a sea that would lie over Albacete or Tarragona.
 */
function drawBase(ctx, P) {
  if (!geo.land) return;
  const [w, s, e, n] = geo.land.bbox;
  const x0 = Math.round(P.x(w)), x1 = Math.round(P.x(e)), y0 = Math.round(P.y(n)), y1 = Math.round(P.y(s));
  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, x1 - x0, y1 - y0); ctx.clip();
  ctx.fillStyle = SEA;
  ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  ctx.beginPath(); trace(ctx, P, geo.land.rings, true); ctx.fillStyle = LAND_OUT; ctx.fill('evenodd');
  if (geo.region) { ctx.beginPath(); trace(ctx, P, geo.region.rings, true); ctx.fillStyle = PAPER; ctx.fill('evenodd'); }
  const v = P.view;
  const a = Math.max(w, v[0]), b = Math.min(e, v[2]), c = Math.max(s, v[1]), d = Math.min(n, v[3]);
  ctx.beginPath();
  for (let lon = Math.ceil(a * 2) / 2; lon <= b; lon += 0.5) { ctx.moveTo(Math.round(P.x(lon)) + 0.5, y0); ctx.lineTo(Math.round(P.x(lon)) + 0.5, y1); }
  for (let lat = Math.ceil(c * 2) / 2; lat <= d; lat += 0.5) { ctx.moveTo(x0, Math.round(P.y(lat)) + 0.5); ctx.lineTo(x1, Math.round(P.y(lat)) + 0.5); }
  ctx.strokeStyle = 'rgba(27,26,23,0.10)'; ctx.lineWidth = 1; ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = INK; ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
}

/** Cells of one frame. Runs of equal level along a row are painted as one rectangle on whole pixels: no seams. */
function drawCells(ctx, P, snap, levels, alpha) {
  const g = snap.grid;
  const xs = new Float64Array(g.nx + 1), ys = new Float64Array(g.ny + 1);
  for (let i = 0; i <= g.nx; i++) xs[i] = Math.round(P.x(g.lon0 + i * g.d));
  for (let j = 0; j <= g.ny; j++) ys[j] = Math.round(P.y(g.lat0 + j * g.d));
  const i0 = Math.max(0, Math.floor((P.view[0] - g.lon0) / g.d)), i1 = Math.min(g.nx, Math.ceil((P.view[2] - g.lon0) / g.d));
  const j0 = Math.max(0, Math.floor((P.view[1] - g.lat0) / g.d)), j1 = Math.min(g.ny, Math.ceil((P.view[3] - g.lat0) / g.d));
  const pat = hatch(ctx);
  for (let pass = 2; pass <= 6; pass++) {            // 2..5 the colours, 6 the hatch of level 5
    const L = pass === 6 ? 5 : pass;
    if (pass === 6) { ctx.globalAlpha = 0.6; ctx.fillStyle = pat; } else { ctx.globalAlpha = alpha; ctx.fillStyle = LEVEL[L].color; }
    for (let j = j0; j < j1; j++) {
      let start = -1;
      for (let i = i0; i <= i1; i++) {
        const n = i < i1 ? snap.cellToN[j * g.nx + i] : -1;
        const hit = n >= 0 && levels[n] === L;
        if (hit && start < 0) start = i;
        if (!hit && start >= 0) { ctx.fillRect(xs[start], ys[j + 1], xs[i] - xs[start], ys[j] - ys[j + 1]); start = -1; }
      }
    }
  }
  ctx.globalAlpha = 1;
}

function drawBasins(ctx, P, levels, alpha) {
  const pat = hatch(ctx);
  for (let pass = 2; pass <= 6; pass++) {
    const L = pass === 6 ? 5 : pass;
    ctx.beginPath();
    let any = false;
    for (const b of geo.basins) {
      if ((levels ? levels[b.idx] : 0) !== L || !seen(b.bbox, P.view)) continue;
      trace(ctx, P, b.rings, true); any = true;
    }
    if (!any) continue;
    ctx.globalAlpha = pass === 6 ? 0.6 : alpha;
    ctx.fillStyle = pass === 6 ? pat : LEVEL[L].color;
    ctx.fill('evenodd');
  }
  ctx.globalAlpha = 1;
}

function drawBasinEdges(ctx, P) {
  ctx.beginPath();
  for (const b of geo.basins) if (seen(b.bbox, P.view)) trace(ctx, P, b.rings, true);
  ctx.strokeStyle = 'rgba(27,26,23,0.2)'; ctx.lineWidth = 0.6; ctx.stroke();
}

function drawRivers(ctx, P, more) {
  const z = P.zoom + (more ? 1 : 0);
  const minLen = z < 8 ? 60 : z < 9 ? 30 : z < 10 ? 14 : z < 11 ? 6 : 0;
  ctx.lineJoin = 'round';
  for (const [lo, hi, w] of [[150, Infinity, 1.7], [60, 150, 1.2], [0, 60, 0.8]]) {
    ctx.beginPath();
    for (const r of geo.rivers) {
      if (r.len < minLen || r.len < lo || r.len >= hi || !seen(r.bbox, P.view)) continue;
      trace(ctx, P, r.rings, false);
    }
    ctx.strokeStyle = RIVER; ctx.lineWidth = w; ctx.stroke();
  }
}

/** Greedy label placement: a label is drawn only where it does not cover one already drawn. */
function labeller(ctx, frame) {
  const boxes = [];
  const free = (x, y, w, h) => !boxes.some((b) => x < b[0] + b[2] && x + w > b[0] && y < b[1] + b[3] && y + h > b[1]);
  return {
    block(x, y, w, h) { boxes.push([x, y, w, h]); },
    put(text, x, y, font, color, sides = ['r', 'l', 't', 'b'], size = 12) {
      ctx.font = font;
      const w = ctx.measureText(text).width, h = size + 2;
      for (const side of sides) {
        const bx = side === 'r' ? x + 5 : side === 'l' ? x - 5 - w : x - w / 2;
        const by = side === 't' ? y - 5 - h : side === 'b' ? y + 5 : y - h / 2;
        if (!free(bx - 1, by - 1, w + 2, h + 2)) continue;
        // a label of something that is in sight is never cut by the edge of the map: it goes on another side
        if (frame && x >= frame[0] && x <= frame[0] + frame[2] && y >= frame[1] && y <= frame[1] + frame[3]
          && (bx < frame[0] + 2 || bx + w > frame[0] + frame[2] - 2 || by < frame[1] + 2 || by + h > frame[1] + frame[3] - 2)) continue;
        boxes.push([bx - 1, by - 1, w + 2, h + 2]);
        ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
        ctx.lineJoin = 'round'; ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(243,239,230,0.9)';
        ctx.strokeText(text, bx, by + h / 2);
        ctx.fillStyle = color;
        ctx.fillText(text, bx, by + h / 2);
        return true;
      }
      return false;
    },
  };
}

function drawTowns(ctx, P, lab, k) {
  const z = P.zoom;
  const minPop = z < 7.6 ? 90000 : z < 8.5 ? 40000 : z < 9.5 ? 12000 : z < 10.5 ? 2500 : 0;
  for (const p of geo.places) {                        // sorted by population, largest first
    if (p.pop < minPop) break;
    if (p.lon < P.view[0] || p.lon > P.view[2] || p.lat < P.view[1] || p.lat > P.view[3]) continue;
    const x = P.x(p.lon), y = P.y(p.lat), big = p.pop >= 150000;
    const size = Math.round((big ? 14 : 12) * k);
    if (lab.put(p.name.split('/')[0], x, y, `italic ${big ? '700 ' : ''}${size}px ${SERIF}`, INK, ['r', 'l', 't', 'b'], size)) {
      const s = Math.round((big ? 5 : 3) * k);
      ctx.fillStyle = INK;
      ctx.fillRect(Math.round(x - s / 2), Math.round(y - s / 2), s, s);
      lab.block(x - s, y - s, 2 * s, 2 * s);
    }
  }
}

function drawRiverNames(ctx, P, lab, k) {
  if (P.zoom < 7.6) return;
  const done = new Set();
  for (const r of geo.rivers) {
    if (r.len < (P.zoom < 9 ? 100 : 45)) break;       // sorted by length, longest first
    if (done.has(r.name)) continue;
    const ring = r.rings[0];
    // a vertex of the line that is on screen, as close to its middle as possible
    let best = -1;
    const mid = (ring.length >> 2) << 1;
    for (let k2 = 0; k2 < ring.length; k2 += 2) {
      if (ring[k2] < P.view[0] || ring[k2] > P.view[2] || ring[k2 + 1] < P.view[1] || ring[k2 + 1] > P.view[3]) continue;
      if (best < 0 || Math.abs(k2 - mid) < Math.abs(best - mid)) best = k2;
    }
    if (best < 0) continue;
    const size = Math.round(11 * k);
    if (lab.put(r.name.replace(/^Río /, ''), P.x(ring[best]), P.y(ring[best + 1]), `italic ${size}px ${SERIF}`, RIVER, ['t', 'b', 'r'], size)) done.add(r.name);
  }
}

/**
 * Control points. With the whole region in view the sixty of them would be a heap of squares: there, only
 * the points with a level (2+) and the selected one are numbered squares, the rest are small marks.
 * Higher levels are drawn last, so they are never hidden.
 */
function drawPoints(ctx, P, o, lab, k) {
  if (!geo.points) return;
  const far = P.zoom < 9;
  const sel0 = o.selPoint && geo.catchments ? geo.catchments.get(o.selPoint) : null;
  if (sel0) {                                          // the land that drains to the selected point
    ctx.beginPath(); trace(ctx, P, sel0.rings, true);
    ctx.fillStyle = 'rgba(59,106,143,0.10)'; ctx.fill('evenodd');
    ctx.strokeStyle = PAPER; ctx.lineWidth = 3.5; ctx.stroke();
    ctx.strokeStyle = INK; ctx.lineWidth = 1.2; ctx.setLineDash([5, 3]); ctx.stroke(); ctx.setLineDash([]);
  }
  const pat = hatch(ctx);
  const order = geo.points.map((pt, i) => ({ pt, L: o.pointLevels ? o.pointLevels[i] || 0 : 0, sel: o.selPoint === pt.id }))
    .filter(({ pt }) => !(pt.lon < P.view[0] || pt.lon > P.view[2] || pt.lat < P.view[1] || pt.lat > P.view[3]))
    .sort((a, b) => (a.sel - b.sel) || (a.L - b.L));
  for (const { pt, L, sel } of order) {
    const small = far && L < 2 && !sel;
    const s = Math.round((sel ? 17 : small ? 6 : 13) * k), x = Math.round(P.x(pt.lon) - s / 2), y = Math.round(P.y(pt.lat) - s / 2);
    ctx.fillStyle = L >= 2 ? LEVEL[L].color : PAPER;
    ctx.fillRect(x, y, s, s);
    if (L === 5) { ctx.globalAlpha = 0.6; ctx.fillStyle = pat; ctx.fillRect(x, y, s, s); ctx.globalAlpha = 1; }
    ctx.strokeStyle = INK; ctx.lineWidth = sel ? 2.5 : small ? 1 : 1.5;
    ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
    if (L >= 1 && !small) {
      ctx.font = `700 ${Math.round(10 * k)}px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillStyle = L >= 2 ? LEVEL[L].text : INK;
      ctx.fillText(String(L), x + s / 2, y + s / 2 + 1);
    }
    lab.block(x - 2, y - 2, s + 4, s + 4);
  }
  if (P.zoom >= 9.5 || o.selPoint) {
    for (const pt of geo.points) {
      if (P.zoom < 9.5 && pt.id !== o.selPoint) continue;
      if (pt.lon < P.view[0] || pt.lon > P.view[2] || pt.lat < P.view[1] || pt.lat > P.view[3]) continue;
      const size = Math.round(11 * k);
      lab.put(pt.stream || pt.id, P.x(pt.lon) + 4, P.y(pt.lat), `italic ${size}px ${SERIF}`, RIVER, ['r', 'l', 'b', 't'], size);
    }
  }
}

/*
 * Reservoirs (Cauces mode): a short thick bar across the river, the shape of a dam wall seen from above,
 * filled with its level; the selected one larger. Names from zoom 9.5, or always for level 2+ and the selected one.
 */
function drawDams(ctx, P, o, lab, k) {
  if (!o.dams) return;
  const pat = hatch(ctx);
  const list = o.dams.filter((d) => !(d.lon < P.view[0] || d.lon > P.view[2] || d.lat < P.view[1] || d.lat > P.view[3]))
    .sort((a, b) => (a.sel - b.sel) || (a.L - b.L));
  for (const d of list) {
    const small = P.zoom < 9 && d.L < 2 && !d.sel;
    const w = Math.round((d.sel ? 22 : small ? 9 : 16) * k), h = Math.round((d.sel ? 8 : small ? 4 : 6) * k);
    const x = Math.round(P.x(d.lon) - w / 2), y = Math.round(P.y(d.lat) - h / 2);
    ctx.fillStyle = d.L >= 2 ? LEVEL[d.L].color : PAPER;
    ctx.fillRect(x, y, w, h);
    if (d.L === 5) { ctx.globalAlpha = 0.6; ctx.fillStyle = pat; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1; }
    ctx.strokeStyle = INK; ctx.lineWidth = d.sel ? 2.5 : 1.5;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    lab.block(x - 2, y - 2, w + 4, h + 4);
  }
  for (const d of list) {
    if (P.zoom < 9.5 && d.L < 2 && !d.sel) continue;
    const size = Math.round(10.5 * k);
    lab.put(d.name, P.x(d.lon) + 12 * k, P.y(d.lat), `${size}px ${SERIF}`, INK, ['r', 'l', 'b', 't'], size);
  }
}

function drawSelection(ctx, P, o) {
  const s = o.selection;
  if (!s) return;
  const twice = (fn) => { ctx.strokeStyle = PAPER; ctx.lineWidth = 5; fn(); ctx.strokeStyle = INK; ctx.lineWidth = 2; fn(); };
  if (o.mode === 'cuencas') {
    // a selected control point is shown by its own catchment (drawPoints); otherwise the basin unit of the place
    if (s.basin && !(o.selPoint && geo.catchments && geo.catchments.has(o.selPoint))) twice(() => { ctx.beginPath(); trace(ctx, P, s.basin.rings, true); ctx.stroke(); });
  } else if (s.cell && o.snap) {
    const g = o.snap.grid;
    const x0 = Math.round(P.x(g.lon0 + s.cell.i * g.d)), x1 = Math.round(P.x(g.lon0 + (s.cell.i + 1) * g.d));
    const y0 = Math.round(P.y(g.lat0 + (s.cell.j + 1) * g.d)), y1 = Math.round(P.y(g.lat0 + s.cell.j * g.d));
    twice(() => ctx.strokeRect(x0, y0, x1 - x0, y1 - y0));
  }
  const x = Math.round(P.x(s.lon)) + 0.5, y = Math.round(P.y(s.lat)) + 0.5, a = 9;
  twice(() => { ctx.beginPath(); ctx.moveTo(x - a, y); ctx.lineTo(x + a, y); ctx.moveTo(x, y - a); ctx.lineTo(x, y + a); ctx.stroke(); });
}

/*
 * The scene is four layers, bottom to top. Only `levels` changes from one frame to the next, so the
 * map keeps the other three as bitmaps (map.js) and stepping through the frames repaints almost nothing.
 *
 * o = { snap, mode:'celdas'|'cuencas', cellLevels:Uint8Array|null, basinLevels:Array|null, pointLevels:Array|null,
 *       selection:{lat,lon,cell:{i,j}|null,basin}|null, selPoint:id|null, base:bool, labels:bool, k:label scale,
 *       blocked:[[x,y,w,h]] rectangles where no label may be written }
 */

/** 1. The sheet: sea, land, graticule (nothing over the street map). */
export function drawUnder(ctx, P, o) {
  if (o.base) drawBase(ctx, P);
}

/** 2. The levels: cells or basin units. */
export function drawLevels(ctx, P, o) {
  // over the street base map the colours are a little thinner, so that street names stay readable
  const alpha = o.base ? CELL_ALPHA : STREET_ALPHA;
  if (o.mode === 'cuencas') { if (geo.basins.length) drawBasins(ctx, P, o.basinLevels, alpha); }
  else if (o.snap && o.cellLevels) drawCells(ctx, P, o.snap, o.cellLevels, alpha);
}

/** 3. Line work: basin edges and control-point streams (Cauces), rivers, warning zones, provinces, region, coast. */
export function drawLines(ctx, P, o) {
  const k = o.k || 1;
  if (o.mode === 'cuencas' && geo.basins.length) drawBasinEdges(ctx, P);
  ctx.lineWidth = 1;
  if (geo.rivers.length) drawRivers(ctx, P, false);
  strokeShapes(ctx, P, geo.zones, 'rgba(27,26,23,0.55)', 0.7 * k, [3 * k, 3 * k]);
  strokeShapes(ctx, P, geo.provinces, INK, 0.9 * k);
  strokeShapes(ctx, P, [geo.region], INK, 1.3 * k);
  strokeShapes(ctx, P, [geo.coast], INK, 1.8 * k);
  if (o.mode === 'cuencas' && geo.points && geo.streams) strokeShapes(ctx, P, geo.streams, RIVER, P.zoom < 9 ? 1.6 : 2.2);
}

/** 4. Control points, names, selection. */
export function drawTop(ctx, P, o) {
  const k = o.k || 1;
  const lab = labeller(ctx, o.frame || null);      // frame: [x, y, w, h] of what is in sight (the canvas is larger)
  // nothing is written under the legend or the map buttons (rectangles in canvas pixels, from map.js)
  for (const b of o.blocked || []) lab.block(b[0], b[1], b[2], b[3]);
  if (o.mode === 'cuencas') { drawPoints(ctx, P, o, lab, k); drawDams(ctx, P, o, lab, k); }
  if (o.selection) lab.block(P.x(o.selection.lon) - 10, P.y(o.selection.lat) - 10, 20, 20);
  if (o.labels) {
    drawTowns(ctx, P, lab, k);
    drawRiverNames(ctx, P, lab, k);
    if (o.base && geo.land && P.zoom < 9.5) {
      // starts well off the coast of the Gulf of València and runs east: it can never lie on land
      // ... and it is only written if it fits in the sea that is in view
      const size = Math.round(12 * k), font = `italic ${size}px ${SERIF}`;
      const x = P.x(0.12), room = P.x(Math.min(geo.land.bbox[2], P.view[2])) - x - 12;
      ctx.font = font;
      const text = ['M a r   M e d i t e r r á n e o', 'Mediterráneo'].find((t) => ctx.measureText(t).width <= room);
      if (text) lab.put(text, x, P.y(39.12), font, RIVER, ['r'], size);
    }
  }
  drawSelection(ctx, P, o);
}

/** The four layers in one go (the share card). */
export function drawScene(ctx, P, o) {
  drawUnder(ctx, P, o);
  drawLevels(ctx, P, o);
  drawLines(ctx, P, o);
  drawTop(ctx, P, o);
}

/** What the three cached layers depend on besides the view: which geodata has arrived, and for the top one, the selection. */
export function geoStamp() {
  return [geo.land ? 1 : 0, geo.region ? 1 : 0, geo.coast ? 1 : 0, geo.provinces.length, geo.zones.length, geo.basins.length, geo.rivers.length,
    geo.places.length, geo.points ? geo.points.length : 0, geo.streams ? geo.streams.length : 0, geo.catchments ? geo.catchments.size : 0].join('.');
}

/** Web-Mercator projection at a given scale: px per degree of longitude, with (lon0, lat0) at (x0, y0). */
export function mercator(pxPerDeg, lon0, lat0, x0, y0) {
  const R = (pxPerDeg * 180) / Math.PI;
  const my = (lat) => R * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  const m0 = my(lat0);
  return { x: (lon) => x0 + (lon - lon0) * pxPerDeg, y: (lat) => y0 - (my(lat) - m0), my, R };
}
