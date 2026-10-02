"""The link-preview image (data/og.png): the worst level of the next 48 h on a plain map."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from . import static
from .core import grid

PAPER, INK = (243, 239, 230), (27, 26, 23)
COLORS = {2: (242, 207, 59), 3: (238, 138, 29), 4: (209, 38, 28), 5: (90, 17, 131)}
NAMES = {1: "Sin riesgo apreciable", 2: "Riesgo MEDIO (2)", 3: "Riesgo ALTO (3)", 4: "Riesgo MUY ALTO (4)", 5: "Riesgo EXTREMO (5)"}
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", "C:/Windows/Fonts/georgia.ttf"]
FONTS_B = ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "C:/Windows/Fonts/georgiab.ttf"]


def _font(size: int, bold: bool = False):
    for f in (FONTS_B if bold else FONTS):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _hatch(img: Image.Image, boxes: list) -> None:
    """Diagonal paper-coloured lines over the given rectangles: level 5 is recognisable without colour,
    as on the web map (css .lv5, draw.js hatch)."""
    if not boxes:
        return
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    for b in boxes:
        md.rectangle(b, fill=255)
    lines = Image.new("L", (w, h), 0)
    ld = ImageDraw.Draw(lines)
    for k in range(-h, w, 7):
        ld.line([(k, h), (k + h, 0)], fill=150, width=2)          # 150/255: the 0.6 opacity of the web hatch
    img.paste(PAPER, mask=ImageChops.multiply(mask, lines))


def render(level_max: np.ndarray, generated: datetime, out: Path, w: int = 1200, h: int = 630) -> None:
    """level_max (NY, NX) uint8: worst level per cell over the period shown."""
    st = static.load()
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    # map panel on the right, plate carree stretched by cos(lat): the whole Comunitat Valenciana, from
    # Pilar de la Horadada (37.84 N) to the Sénia (40.79 N), with a margin
    lon0, lat0, lon1, lat1 = -1.75, 37.7, 0.75, 40.9
    mh = h - 40
    mw = int(mh * (lon1 - lon0) * np.cos(np.deg2rad(39.3)) / (lat1 - lat0))
    ox, oy = w - mw - 30, 20
    X = lambda lon: ox + (lon - lon0) / (lon1 - lon0) * mw
    Y = lambda lat: oy + (lat1 - lat) / (lat1 - lat0) * mh
    extreme = []
    for j in range(grid.NY):
        for i in range(grid.NX):
            L = int(level_max[j, i])
            if L >= 2 and st.mask[j, i]:
                x0, x1 = X(grid.LON0 + i * grid.D), X(grid.LON0 + (i + 1) * grid.D)
                y1, y0 = Y(grid.LAT0 + j * grid.D), Y(grid.LAT0 + (j + 1) * grid.D)
                d.rectangle([x0, y0, x1 + 1, y1 + 1], fill=COLORS[L])
                if L >= 5:
                    extreme.append([x0, y0, x1 + 1, y1 + 1])
    _hatch(img, extreme)
    d = ImageDraw.Draw(img)
    try:
        gj = json.loads((static.GEO / "out" / "boundary.geojson").read_text(encoding="utf-8"))
        for f in gj["features"]:
            # only the Comunitat Valenciana (region, provinces, its coast). The "land_box" feature is the
            # coast of the neighbouring regions: drawn in ink it ran off the bottom edge and looked as if
            # the south of Alicante had been cut.
            if (f.get("properties") or {}).get("kind") == "land_box":
                continue
            g = f["geometry"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]] if g["type"] == "Polygon" else []
            lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]] if g["type"] == "LineString" else []
            for ring in [r for p in polys for r in p] + lines:
                pts = [(X(x), Y(y)) for x, y in ring]
                if len(pts) > 1:
                    d.line(pts, fill=INK, width=2)
    except Exception:
        pass
    top = int(level_max[st.mask].max()) if st.mask.any() else 1
    d.text((40, 44), "riuà", font=_font(64, True), fill=INK)
    d.text((40, 124), "Riesgo de lluvia extrema e inundación", font=_font(26), fill=INK)
    d.text((40, 158), "Comunitat Valenciana · próximas 48 h", font=_font(26), fill=INK)
    d.line([(40, 210), (ox - 40, 210)], fill=INK, width=2)
    def swatch(box, L):
        nonlocal d
        d.rectangle(box, fill=COLORS[L])
        if L >= 5:
            _hatch(img, [box])
            d = ImageDraw.Draw(img)
        d.rectangle(box, outline=INK)

    if top >= 2:
        swatch([40, 238, 76, 274], top)
    d.text((92 if top >= 2 else 40, 236), NAMES[max(top, 1)], font=_font(34, True), fill=INK)
    y = 310
    for L in (2, 3, 4, 5):
        swatch([40, y, 62, y + 22], L)
        d.text((74, y - 3), {2: "2 Medio", 3: "3 Alto", 4: "4 Muy alto", 5: "5 Extremo"}[L], font=_font(22), fill=INK)
        y += 34
    loc = generated.astimezone(ZoneInfo("Europe/Madrid"))
    d.text((40, h - 108), f"Actualizado {loc:%d/%m/%Y %H:%M}", font=_font(22), fill=INK)
    d.text((40, h - 76), "Herramienta no oficial. Fuentes oficiales:", font=_font(20), fill=INK)
    d.text((40, h - 50), "AEMET y 112 Comunitat Valenciana.", font=_font(20), fill=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)


def from_snapshot(snap: dict, out: Path) -> None:
    import base64
    g = snap["grid"]
    mask = np.unpackbits(np.frombuffer(base64.b64decode(snap["mask"]), np.uint8))[: g["nx"] * g["ny"]].astype(bool)
    worst = np.zeros(int(mask.sum()), np.uint8)
    for hz in ("now", "mid"):
        blk = snap["horizons"].get(hz)
        if blk:
            lv = np.frombuffer(base64.b64decode(blk["cells"]["level"]), np.uint8).reshape(len(blk["frames"]), -1)
            worst = np.maximum(worst, lv.max(axis=0))
    full = np.zeros(mask.size, np.uint8)
    full[mask] = worst
    t = datetime.strptime(snap["generated"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    render(full.reshape(g["ny"], g["nx"]), t, out)
