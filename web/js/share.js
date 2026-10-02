// The share card: a 1080 x 1350 PNG drawn from the data (not a screenshot of the map).

import { INK, LEVEL, MONO, PAPER, SERIF } from './config.js';
import { drawScene, hatch, mercator } from './draw.js';
import { geo } from './geo.js';
import { stamp } from './time.js';

const W = 1080, H = 1350, M = 56;

function wrap(ctx, text, maxW) {
  const lines = [];
  let line = '';
  for (const word of text.split(/\s+/)) {
    const trial = line ? `${line} ${word}` : word;
    if (ctx.measureText(trial).width > maxW && line) { lines.push(line); line = word; } else line = trial;
  }
  if (line) lines.push(line);
  return lines;
}

function swatch(ctx, x, y, s, L) {
  ctx.fillStyle = L >= 2 ? LEVEL[L].color : PAPER;
  ctx.fillRect(x, y, s, s);
  if (L === 5) { ctx.save(); ctx.globalAlpha = 0.6; ctx.fillStyle = hatch(ctx); ctx.fillRect(x, y, s, s); ctx.restore(); }
  ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(x + 1, y + 1, s - 2, s - 2);
  ctx.fillStyle = L >= 2 ? LEVEL[L].text : INK;
  ctx.font = `700 ${Math.round(s * 0.62)}px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(String(L), x + s / 2, y + s / 2 + 2);
  ctx.textAlign = 'left';
}

/**
 * info = { headline, level, sub (horizon / frame line), generated: Date, url (text shown), scene (as draw.js) }
 * -> canvas
 */
export async function drawCard(info) {
  if (document.fonts && document.fonts.load) {
    try { await Promise.all([document.fonts.load(`700 20px ${MONO}`), document.fonts.load(`20px ${MONO}`)]); } catch (e) { /* system fallback */ }
  }
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = PAPER; ctx.fillRect(0, 0, W, H);
  const rule = (y, w = 2) => { ctx.fillStyle = INK; ctx.fillRect(M, y, W - 2 * M, w); };

  // masthead
  ctx.fillStyle = INK; ctx.textBaseline = 'alphabetic';
  ctx.font = `700 54px ${SERIF}`;
  let x = M;
  for (const ch of 'RIUÀ') { ctx.fillText(ch, x, 96); x += ctx.measureText(ch).width + 10; }
  ctx.font = `italic 24px ${SERIF}`;
  ctx.fillText('riuà (val.) f. Crecida súbita de un río o barranco.', x + 18, 94);
  rule(114, 3);

  // headline
  ctx.font = `700 46px ${SERIF}`;
  let lines = wrap(ctx, info.headline, W - 2 * M);
  let size = 46;
  if (lines.length > 4) { size = 38; ctx.font = `700 ${size}px ${SERIF}`; lines = wrap(ctx, info.headline, W - 2 * M); }
  lines = lines.slice(0, 5);
  let y = 114 + 24 + size;
  for (const l of lines) { ctx.fillText(l, M, y); y += Math.round(size * 1.2); }
  y -= Math.round(size * 0.55);
  ctx.font = `24px ${MONO}`;
  ctx.fillText(info.sub, M, y + 26);
  y += 48;

  // map: the Comunitat Valenciana from top to bottom; the frame is the edge of the sheet (the rectangle the
  // cartography covers), so nothing outside it is ever shown
  const footH = 118;
  const lat0 = 37.8, lat1 = 40.82;
  const sheet = geo.land ? geo.land.bbox : [-2.7, 37.3, 1.1, 41.3];
  const maxH = H - y - footH - 26;
  let P = mercator(1, 0, lat1, 0, 0);
  const span = P.my(lat1) - P.my(lat0);
  const k = Math.min(maxH / span, (W - 2 * M) / (sheet[2] - sheet[0]));
  const box = { w: Math.round(k * (sheet[2] - sheet[0])), h: Math.round(k * span), y };
  box.x = Math.round((W - box.w) / 2);
  const cx = (sheet[0] + sheet[2]) / 2;
  P = mercator(k, cx, lat1, box.x + box.w / 2, box.y);
  const proj = { x: P.x, y: P.y, zoom: Math.log2((k * 360) / 256), view: [sheet[0], lat0, sheet[2], lat1] };
  // legend inside the map, in the sea off Alicante (bottom right corner)
  const names = ['', '', 'Medio', 'Alto', 'Muy alto', 'Extremo'];
  const lg = { w: 196, h: 4 * 46 + 12 };
  lg.x = box.x + box.w - lg.w - 14; lg.y = box.y + box.h - lg.h - 14;
  ctx.save();
  ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
  drawScene(ctx, proj, { ...info.scene, base: true, labels: true, k: 1.7, blocked: [[lg.x - 6, lg.y - 6, lg.w + 12, lg.h + 12]] });
  ctx.restore();
  ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(box.x, box.y, box.w, box.h);
  ctx.fillStyle = PAPER; ctx.fillRect(lg.x, lg.y, lg.w, lg.h);
  ctx.lineWidth = 1; ctx.strokeRect(lg.x + 0.5, lg.y + 0.5, lg.w - 1, lg.h - 1);
  for (let L = 2; L <= 5; L++) {
    const sy = lg.y + 9 + (L - 2) * 46;
    swatch(ctx, lg.x + 9, sy, 36, L);
    ctx.fillStyle = INK; ctx.font = `25px ${SERIF}`; ctx.textBaseline = 'middle';
    ctx.fillText(names[L], lg.x + 57, sy + 19);
  }

  // foot
  const fy = H - footH;
  rule(fy, 2);
  ctx.textBaseline = 'alphabetic'; ctx.fillStyle = INK;
  ctx.font = `700 27px ${SERIF}`;
  ctx.fillText('riuà — herramienta no oficial', M, fy + 42);
  ctx.font = `23px ${SERIF}`;
  ctx.fillText('Las fuentes oficiales son AEMET y el 112 Comunitat Valenciana.', M, fy + 74);
  ctx.font = `22px ${MONO}`;
  ctx.fillText(`Predicción de ${stamp(info.generated)}`, M, fy + 104);
  ctx.textAlign = 'right';
  ctx.fillText(info.url, W - M, fy + 104);
  ctx.textAlign = 'left';
  return cv;
}

const toBlob = (cv) => new Promise((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error('no se pudo crear la imagen'))), 'image/png'));

/** Shares the card with the system sheet when files can be shared; otherwise downloads it. -> 'shared' | 'downloaded' | 'cancelled' */
export async function shareCard(info, link) {
  const blob = await toBlob(await drawCard(info));
  const name = `riua-${info.generated.toISOString().slice(0, 16).replace(/[-:T]/g, '')}.png`;
  const file = new File([blob], name, { type: 'image/png' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: 'riuà', text: `${info.headline} ${link}` });
      return 'shared';
    } catch (e) {
      if (e && e.name === 'AbortError') return 'cancelled';
    }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  return 'downloaded';
}
