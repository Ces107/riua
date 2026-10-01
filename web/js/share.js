// The share card: a 1080 x 1350 PNG drawn from the data (not a screenshot of the map).

import { INK, LEVEL, MONO, PAPER, SERIF } from './config.js';
import { drawScene, hatch, mercator } from './draw.js';
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

  // map
  const legendH = 150, footH = 118;
  const box = { x: M, y, w: W - 2 * M, h: H - y - legendH - footH };
  const b = [-1.62, 37.8, 0.62, 40.82];                     // Comunitat Valenciana with a margin
  let P = mercator(1, b[0], b[3], 0, 0);
  const k = Math.min(box.w / (b[2] - b[0]), box.h / (P.my(b[3]) - P.my(b[1])));
  const cx = (b[0] + b[2]) / 2;
  P = mercator(k, cx, b[3], box.x + box.w / 2, box.y + (box.h - k * (P.my(b[3]) - P.my(b[1]))) / 2);
  const inv = { lon: (px) => cx + (px - (box.x + box.w / 2)) / k };
  const yTop = box.y, yBot = box.y + box.h;
  const latAt = (py) => { const m = P.my(b[3]) - (py - P.y(b[3])); return (Math.atan(Math.exp(m / P.R)) * 360) / Math.PI - 90; };
  const proj = { x: P.x, y: P.y, zoom: Math.log2((k * 360) / 256), view: [inv.lon(box.x), latAt(yBot), inv.lon(box.x + box.w), latAt(yTop)] };
  ctx.save();
  ctx.beginPath(); ctx.rect(box.x, box.y, box.w, box.h); ctx.clip();
  drawScene(ctx, proj, { ...info.scene, base: true, labels: true, k: 1.7 });
  ctx.restore();
  ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(box.x, box.y, box.w, box.h);
  // degree labels on the frame
  ctx.font = `17px ${MONO}`; ctx.fillStyle = INK; ctx.textBaseline = 'top';
  for (let lat = Math.ceil(proj.view[1] * 2) / 2; lat < proj.view[3]; lat += 0.5) {
    const py = P.y(lat);
    if (py > box.y + 30 && py < yBot - 30 && Number.isInteger(lat)) ctx.fillText(`${lat}° N`, box.x + 8, py + 4);
  }

  // legend
  let ly = yBot + 26;
  const names = ['', 'Sin riesgo', 'Medio (≈ amarillo AEMET)', 'Alto (≈ naranja)', 'Muy alto (≈ rojo)', 'EXTREMO (más que un aviso rojo)'];
  const pos = [[M, ly], [M + 330, ly], [M + 690, ly], [M, ly + 58], [M + 330, ly + 58]];
  for (let L = 1; L <= 5; L++) {
    const [sx, sy] = pos[L - 1];
    swatch(ctx, sx, sy, 40, L);
    ctx.fillStyle = INK; ctx.font = `25px ${SERIF}`; ctx.textBaseline = 'middle';
    ctx.fillText(names[L], sx + 52, sy + 21);
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
