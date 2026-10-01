// Static geodata (web/geo, built by web/dev/build_geo.py) and the geometry helpers that use it.
// A "ring" is a Float64Array [lon0, lat0, lon1, lat1, ...]; a shape is {rings, bbox:[w,s,e,n]}.

import { GEO_DIR, POINTS_URL } from './config.js';
import { b64 } from './data.js';

export const geo = {
  land: null, region: null, provinces: [], coast: null,   // boundary.geojson
  zones: [], basins: [], basinById: new Map(), rivers: [], places: [], cells: null,
  points: null, streams: null, catchments: null,           // control-point network (may never arrive)
  failed: [],
};

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function bboxOf(rings) {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  for (const r of rings) {
    for (let k = 0; k < r.length; k += 2) {
      if (r[k] < w) w = r[k]; if (r[k] > e) e = r[k];
      if (r[k + 1] < s) s = r[k + 1]; if (r[k + 1] > n) n = r[k + 1];
    }
  }
  return [w, s, e, n];
}

const shape = (rings) => ({ rings, bbox: bboxOf(rings) });

function flat(coords) {
  const a = new Float64Array(coords.length * 2);
  for (let k = 0; k < coords.length; k++) { a[2 * k] = coords[k][0]; a[2 * k + 1] = coords[k][1]; }
  return a;
}

/** GeoJSON geometry -> list of rings (holes and parts all together: fill with the even-odd rule). */
export function ringsOf(g) {
  if (!g) return [];
  switch (g.type) {
    case 'Polygon': return g.coordinates.map(flat);
    case 'MultiPolygon': return g.coordinates.flatMap((p) => p.map(flat));
    case 'LineString': return [flat(g.coordinates)];
    case 'MultiLineString': return g.coordinates.map(flat);
    default: return [];
  }
}

function undelta(r, scale) {
  const a = new Float64Array(r.length);
  let x = 0, y = 0;
  for (let k = 0; k < r.length; k += 2) {
    x += r[k]; y += r[k + 1];
    a[k] = x / scale; a[k + 1] = y / scale;
  }
  return a;
}

/** Accent- and case-insensitive key: "València" -> "valencia", "l'Alcúdia" -> "l alcudia". */
export function norm(s) {
  return (s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

export function pointInRings(rings, lon, lat) {
  let inside = false;
  for (const r of rings) {
    for (let k = 0, j = r.length - 2; k < r.length; j = k, k += 2) {
      const yi = r[k + 1], yj = r[j + 1];
      if ((yi > lat) !== (yj > lat) && lon < ((r[j] - r[k]) * (lat - yi)) / (yj - yi) + r[k]) inside = !inside;
    }
  }
  return inside;
}

const inBox = (b, lon, lat) => lon >= b[0] && lon <= b[2] && lat >= b[1] && lat <= b[3];

export function basinAt(lon, lat) {
  for (const b of geo.basins) if (inBox(b.bbox, lon, lat) && pointInRings(b.rings, lon, lat)) return b;
  return null;
}

export function zoneAt(lon, lat) {
  for (const z of geo.zones) if (inBox(z.bbox, lon, lat) && pointInRings(z.rings, lon, lat)) return z;
  return null;
}

export function distKm(lat1, lon1, lat2, lon2) {
  const k = Math.PI / 180;
  const x = (lon2 - lon1) * k * Math.cos(((lat1 + lat2) / 2) * k), y = (lat2 - lat1) * k;
  return 6371 * Math.hypot(x, y);
}

export function nearest(list, lat, lon, count = 1, maxKm = Infinity) {
  const out = [];
  for (const it of list) {
    if (it.lat == null || it.lon == null) continue;
    const d = distKm(lat, lon, it.lat, it.lon);
    if (d <= maxKm) out.push({ d, it });
  }
  out.sort((a, b) => a.d - b.d);
  return out.slice(0, count);
}

/** Towns matching what the user typed: prefix matches first, then word starts, then anywhere; larger towns first. */
export function searchPlaces(text, limit = 8) {
  const q = norm(text);
  if (!q) return [];
  const rank = (key) => (key === q ? 0 : key.startsWith(q) ? 1 : key.includes(' ' + q) ? 2 : key.includes(q) ? 3 : 9);
  const hits = [];
  for (const p of geo.places) {
    const r = Math.min(rank(p.key), p.altKey ? rank(p.altKey) : 9);
    if (r < 9) hits.push({ r, p });
  }
  hits.sort((a, b) => a.r - b.r || b.p.pop - a.p.pop);
  return hits.slice(0, limit).map((h) => h.p);
}

/** Zone / basin unit that the backend assigned to grid cell c (the one covering most of the cell). */
export function cellZone(c) {
  if (!geo.cells) return null;
  const k = geo.cells.zone[c];
  return k === 255 ? null : geo.zones.find((z) => z.code === geo.cells.zoneCodes[k]) || { code: geo.cells.zoneCodes[k], name: geo.cells.zoneNames[k] };
}

export function cellBasin(c) {
  if (!geo.cells) return null;
  const k = geo.cells.basin[c];
  return k === 65535 ? null : geo.basins[k] || null;
}

/** Units from `b` down to the sea (b first). */
export function downstreamChain(b) {
  const out = [];
  const seen = new Set();
  while (b && !seen.has(b.id)) { out.push(b); seen.add(b.id); b = b.next ? geo.basinById.get(b.next) : null; }
  return out;
}

/**
 * Starts every download at once. `onPart(name)` is called as each file becomes usable, so the map
 * can be redrawn progressively; a file that fails is listed in geo.failed and the rest carries on.
 */
export function loadGeo(onPart) {
  const part = (name, url, use, optional = false) => json(url).then((d) => { use(d); onPart(name); })
    .catch((e) => { if (!optional) { geo.failed.push(name); onPart(name, e); } });
  return Promise.all([
    part('boundary', `${GEO_DIR}boundary.geojson`, (d) => {
      for (const f of d.features) {
        const s = shape(ringsOf(f.geometry));
        const kind = f.properties.kind;
        if (kind === 'land_box') geo.land = s;
        else if (kind === 'region') geo.region = s;
        else if (kind === 'coastline') geo.coast = s;
        else if (kind === 'province') geo.provinces.push({ ...s, name: f.properties.name });
      }
    }),
    part('zones', `${GEO_DIR}zones.geojson`, (d) => {
      geo.zones = d.features.map((f) => ({ ...shape(ringsOf(f.geometry)), code: f.properties.code, name: f.properties.name,
        province: f.properties.province, label: [f.properties.label_lon, f.properties.label_lat] }));
    }),
    part('basins', `${GEO_DIR}basins.json`, (d) => {
      geo.basins = d.units.map((u) => ({ ...shape(u[9].map((r) => undelta(r, d.scale))), idx: u[0], id: u[1], name: u[2], river: u[3],
        town: u[4], area: u[5], upArea: u[6], next: u[7], label: u[8] }));
      geo.basinById = new Map(geo.basins.map((b) => [b.id, b]));
    }),
    part('rivers', `${GEO_DIR}rivers.json`, (d) => {
      geo.rivers = d.lines.map((l) => ({ ...shape(l[3].map((r) => undelta(r, d.scale))), name: l[0], kind: l[1], len: l[2] }));
    }),
    part('places', `${GEO_DIR}places.json`, (d) => {
      geo.places = d.places.map((p) => ({ name: p[0], alt: p[1], lat: p[2], lon: p[3], pop: p[4] || 0, basin: p[5], zone: p[6],
        province: p[7], key: norm(p[0]), altKey: p[1] ? norm(p[1]) : null }));
    }),
    part('cells', `${GEO_DIR}cells.json`, (d) => {
      const bz = b64(d.basin);
      geo.cells = { zoneCodes: d.zone_codes, zoneNames: d.zone_names, zone: b64(d.zone), cv: b64(d.cv),
        basin: new Uint16Array(bz.buffer, bz.byteOffset, bz.length >> 1) };
    }),
  ]);
}

let pointsAsked = null;

/**
 * Control points (ravines and rivers at towns). Only asked for when the snapshot carries a
 * `points` block, so a site without that network never requests a file that is not there.
 * points.json says whether streams.geojson / catchments.geojson exist next to it.
 */
export function loadPoints() {
  if (!pointsAsked) {
    pointsAsked = (async () => {
      try {
        const d = await json(POINTS_URL);
        geo.points = Array.isArray(d) ? d : d.points;
        if (d.streams) {
          const s = await json(`${GEO_DIR}streams.geojson`);
          geo.streams = s.features.map((f) => ({ ...shape(ringsOf(f.geometry)), name: f.properties.name || f.properties.stream || '' }));
        }
        if (d.catchments) {
          const c = await json(`${GEO_DIR}catchments.geojson`);
          geo.catchments = c.features.map((f) => ({ ...shape(ringsOf(f.geometry)), id: f.properties.id || f.properties.point_id }));
        }
      } catch (e) { /* no control points: that part of the page stays hidden */ }
      return geo.points;
    })();
  }
  return pointsAsked;
}
