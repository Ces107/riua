// Constants shared by every module. Nothing here is computed from data.

// `?dev=1` opens the FABRICATED storm of web/dev/fake_snapshot.py (never the live data).
export const DEV = new URLSearchParams(location.search).has('dev');
export const DATA_DIR = DEV ? 'dev/data/' : 'data/';
export const GEO_DIR = 'geo/';
export const POINTS_URL = DEV ? 'dev/data/points.json' : 'geo/points.json';

export const TZ = 'Europe/Madrid';
// The pipeline publishes every ~10 min. Older than LATE_MIN: the update line stops being discreet;
// older than STALE_MIN: it becomes a warning.
export const LATE_MIN = 30;
export const STALE_MIN = 90;

export const HORIZONS = ['now', 'mid', 'long'];
export const HZ_LABEL = { now: 'Ahora', mid: '48 h', long: 'Días 2–7' };
export const HZ_PLAIN = { now: 'las próximas 6 horas', mid: 'las próximas 48 horas', long: 'los días 2 a 7' };

export const INK = '#1b1a17';
export const PAPER = '#f3efe6';
export const SEA = '#dfe5e3';
export const LAND_OUT = '#ebe6da';      // land outside the Comunitat Valenciana
export const RIVER = '#3b6a8f';

// index = level. 0 = no data.
export const LEVEL = [
  { n: 0, name: 'Sin datos', word: 'SIN DATOS', color: null, text: INK, aemet: '', note: 'No hay datos para este lugar y momento.' },
  { n: 1, name: 'Sin riesgo', word: 'SIN RIESGO', color: null, text: INK, aemet: '', note: 'No se espera lluvia que alcance los umbrales de aviso.' },
  { n: 2, name: 'Medio', word: 'MEDIO', color: '#f2cf3b', text: INK, aemet: '≈ aviso amarillo AEMET', note: 'Lluvia fuerte posible. Atención en barrancos, pasos inundables y carreteras.' },
  { n: 3, name: 'Alto', word: 'ALTO', color: '#ee8a1d', text: INK, aemet: '≈ naranja', note: 'Lluvia muy fuerte probable. Evita desplazamientos innecesarios y aléjate de cauces.' },
  { n: 4, name: 'Muy alto', word: 'MUY ALTO', color: '#d1261c', text: '#ffffff', aemet: '≈ rojo', note: 'No se recomienda permanecer en el exterior.' },
  { n: 5, name: 'EXTREMO', word: 'EXTREMO', color: '#5a1183', text: '#ffffff', aemet: '', note: 'Considerablemente más peligroso que un aviso rojo.' },
];
export const CELL_ALPHA = 0.72;

// Used for cells outside every warning zone (upstream land in other regions): the backend applies
// the values common to the eleven Valencian zones (backend/riua/static.py, DEFAULT_T1H / DEFAULT_T12H).
export const DEFAULT_THR = { '1h': [20, 40, 90], '12h': [60, 100, 180] };

export const AEMET_URL = 'https://www.aemet.es/es/eltiempo/prediccion/avisos?w=hoy&k=val';
// Layer "Callejero gris". Its identifier in the service's GetCapabilities is IGNBase-gris (IGNBaseTodo-gris answers 400).
export const IGN_TILES = 'https://www.ign.es/wmts/ign-base?layer=IGNBase-gris&style=default&tilematrixset=GoogleMapsCompatible&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix={z}&TileCol={x}&TileRow={y}';
export const TILE_MIN_ZOOM = 11;        // the street base map appears from this zoom on

export const SERIF = '"Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif';
export const MONO = '"Courier Prime", "Courier New", Courier, monospace';
