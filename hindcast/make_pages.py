"""Write the validation tables of web/validacion.html and README.md from hindcast/results.json.

    py -3.11 hindcast/make_pages.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "hindcast" / "results.json").read_text(encoding="utf-8"))
PARAMS = json.loads((ROOT / "backend" / "riua" / "params.json").read_text(encoding="utf-8")) if (ROOT / "backend" / "riua" / "params.json").exists() else {}
HZ = {"now": ("Ahora (0–6 h)", "Now (0–6 h)"), "mid": ("48 h (predicción del día anterior)", "48 h (day-ahead runs)"),
      "long": ("Días 2–7 (emitida 3 y 5 días antes)", "Days 2–7 (issued 3 and 5 days before)")}
LV = {"2": "2 medio", "3": "3 alto", "4": "4 muy alto", "5": "5 extremo"}
pc = lambda v: "—" if v is None else f"{round(100 * v)} %"
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
# The Poyo at the A-3 gauge on 29 Oct 2024: the last reading before the sensor was lost, and the CEDEX
# estimate of the real peak. One figure, used wherever that flood is quoted.
POYO_MEASURED, POYO_PEAK = 2283, 3200
try:
    POINTS = {p["id"]: p for p in json.loads((ROOT / "web" / "geo" / "points.json").read_text(encoding="utf-8"))["points"]}
except Exception:
    POINTS = {}


def es(v, dec=None) -> str:
    """A number the Spanish way: decimal comma, a thin space every three digits from 10 000 on."""
    if v is None:
        return "—"
    s = f"{v:.{dec}f}" if dec is not None else f"{v:g}"
    whole, _, frac = s.partition(".")
    sign, digits = ("−", whole[1:]) if whole.startswith("-") else ("", whole)
    if len(digits) > 4:
        digits = re.sub(r"(?<=\d)(?=(\d{3})+$)", " ", digits)
    return sign + digits + ("," + frac if frac else "")


def point_name(pid: str) -> str:
    """'poyo-ribarroja' -> 'Rambla del Poyo, Riba-roja de Túria' (names of web/geo/points.json)."""
    p = POINTS.get(pid)
    if not p:
        return pid
    town = re.sub(r"\s*\(.*?\)", "", p.get("town") or "").split(" / ")[0].strip()
    stream = re.sub(r"\s*\(.*?\)", "", p.get("stream") or pid).split(" / ")[0].strip()
    return f"{stream}, {town}" if town else stream


def month(case: str) -> str:
    """'2025-03-03' -> 'mar 2025'."""
    m = re.match(r"(\d{4})-(\d{2})", case)
    return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}" if m else case


def counts(res: dict, key: str):
    """Scores of the thresholds in use, cross-validated when there are enough cases."""
    tau = (PARAMS.get("tau") or {}).get(key)
    block = res.get("deployed") or res.get("cross_validated") or res.get("in_sample")
    return block, tau, "deployed" if res.get("deployed") else "cross_validated" if res.get("cross_validated") else "in_sample"


def html() -> str:
    out = ['<p>Predicciones que existían entonces, comparadas con la lluvia medida (radar + pluviómetros). '
           'Episodios: 28 oct–14 nov 2024, 2–6 mar 2025, 27–30 sep 2025, 13–15 y 27–29 dic 2025.</p>'
           '<p>«Detectado»: de las veces que ocurrió, cuántas se avisó. «Falsa alarma»: de las veces que se avisó, cuántas no ocurrió.</p>']
    for key in ("now", "mid", "long"):
        res = R.get(key)
        if not res:
            continue
        block, tau, kind = counts(res, key)
        t = res["tuned"]
        out.append(f"<h2>{HZ[key][0]}</h2>")
        out.append(f'<p class="n">{len(res["cases"])} casos · {es(res["n_frames"])} tramos · σ {es(t["sigma"])} · sesgo {es(t["bias"])}</p>')
        for scope, title in (("zone_day", "Por zona de aviso y día"), ("cell", "Por celda y tramo")):
            rows = "".join(
                f'<tr><td>{LV[L]}</td><td class="n">{pc(float(tau[L])) if tau else "—"}</td><td class="n">{es(v["hits"])}</td><td class="n">{es(v["misses"])}</td>'
                f'<td class="n">{es(v["false_alarms"])}</td><td class="n">{pc(v["POD"])}</td><td class="n">{pc(v["FAR"])}</td></tr>'
                for L, v in block[scope].items())
            out.append(f"<h3>{title}</h3><table><tr><th>Nivel</th><th>Mínimo P</th><th>Aciertos</th><th>Fallos</th>"
                       f"<th>Falsas alarmas</th><th>Detectado</th><th>Falsa alarma</th></tr>{rows}</table>")
        rel = res["reliability"]["3"]
        out.append("<h3>Fiabilidad (nivel 3 o más)</h3><table><tr><th>Probabilidad dada</th>"
                   + "".join(f'<td class="n">{pc(b["p_forecast"])}</td>' for b in rel)
                   + "</tr><tr><th>Ocurrió</th>" + "".join(f'<td class="n">{pc(b["observed_freq"])}</td>' for b in rel) + "</tr></table>")
    h = R.get("hydrology_poyo_2024")
    if h:
        run = next((x for x in h["runs"] if x["p0_mm"] == 25.0), h["runs"][0])
        out.append("<h2>Rambla del Poyo, 29-oct-2024</h2><table><tr><th>Punto</th><th>Punta calculada</th><th>Medido</th></tr>"
                   f'<tr><td>A-3 (aforo SAIH)</td><td class="n">{es(run["poyo-ribarroja"]["peak_m3s"])} m³/s</td>'
                   f'<td class="n">{es(POYO_MEASURED)} m³/s al perderse el sensor · punta real ≈ {es(POYO_PEAK)} (CEDEX)</td></tr>'
                   f'<tr><td>Paiporta</td><td class="n">{es(run["poyo-paiporta"]["peak_m3s"])} m³/s</td><td class="n">sin aforo</td></tr></table>'
                   "<p>La hora de la punta no se puede comprobar: radar de Cullera atenuado durante el máximo y sin datos horarios abiertos de pluviómetros de 2024.</p>")
    fit = ROOT / "hindcast" / "hydro_fit.json"
    if fit.exists():
        hf = json.loads(fit.read_text(encoding="utf-8"))

        def measured(e):
            # the 2024 flood of the Poyo: the gauge was lost on the way up, so what was measured is a lower bound
            if e["point"] == "poyo-ribarroja" and e["case"].startswith("2024-10"):
                return f"&gt; {es(POYO_MEASURED)}"
            return es(e["observed"])

        rows = "".join(f'<tr><td>{point_name(e["point"])}</td><td class="n">{month(e["case"])}</td><td class="n">{es(e["rain_mm"]) if e["rain_mm"] else "—"}</td>'
                       f'<td class="n">{measured(e)}</td><td class="n">{es(e["after"])}</td><td class="n">{es(e["before"])}</td></tr>'
                       for e in sorted([e for e in hf["events"] if max(e["observed"], e["after"]) >= 20],
                                       key=lambda e: -e["observed"])[:30])
        out.append("<h2>Caudal: medido y calculado</h2>"
                   f'<p class="n">{hf["n"]} crecidas · aforos SAIH Júcar · umbral de escorrentía {es(hf["p0_mm"])} mm · retención {es(hf["s_mm"])} mm</p>'
                   "<table><tr><th>Punto</th><th>Episodio</th><th>Lluvia (mm)</th><th>Medido (m³/s)</th><th>Calculado</th>"
                   f"<th>Antes (25 mm)</th></tr>{rows}</table>")
    out.append("<h2>Límites</h2><ul><li>Pocos casos de nivel 5: las cifras de ese nivel son orientativas.</li>"
               "<li>Se comprueba «lluvia de ese nivel a menos de 12 km del punto» (6 km en «Ahora»).</li>"
               "<li>«Ahora» usa series de AROME de muy corto plazo: resultado optimista.</li>"
               "<li>No hay archivo libre de modelos de alta resolución a más de un día.</li></ul>")
    return "\n    ".join(out)


def markdown() -> str:
    out = []
    for key in ("now", "mid", "long"):
        res = R.get(key)
        if not res:
            continue
        block, tau, kind = counts(res, key)
        t = res["tuned"]
        out.append(f"**{HZ[key][1]}** — {len(res['cases'])} cases, {res['n_frames']} frames, σ = {t['sigma']}, bias = {t['bias']}; per warning zone and day:\n")
        out.append("| Level | Min. P | Hits | Misses | False alarms | Detected | False-alarm ratio |\n|---|---|---|---|---|---|---|")
        for L, v in block["zone_day"].items():
            out.append(f"| {L} | {pc(float(tau[L])) if tau else '—'} | {v['hits']} | {v['misses']} | {v['false_alarms']} | {pc(v['POD'])} | {pc(v['FAR'])} |")
        out.append("")
    h = R.get("hydrology_poyo_2024")
    if h:
        run = next((x for x in h["runs"] if x["p0_mm"] == 25.0), h["runs"][0])
        out.append(f"**Rambla del Poyo, 29 Oct 2024** (observed rain as input): {run['poyo-ribarroja']['peak_m3s']} m³/s at the A-3 gauge "
                   f"(measured 2283 m³/s when the sensor was lost), {run['poyo-paiporta']['peak_m3s']} m³/s at Paiporta. Peak timing cannot be "
                   "verified: the Cullera radar was attenuated during the maximum and no open sub-daily gauge data exist for 2024.")
    return "\n".join(out)


if __name__ == "__main__":
    f = ROOT / "web" / "validacion.html"
    s = f.read_text(encoding="utf-8")
    a, b = s.index('<h1 class="title">Validación</h1>') + len('<h1 class="title">Validación</h1>'), s.index("  </article>")
    f.write_text(s[:a] + "\n    " + html() + "\n" + s[b:], encoding="utf-8")
    f = ROOT / "README.md"
    s = f.read_text(encoding="utf-8")
    block = "<!-- VALIDATION-TABLE -->\n" + markdown() + "\n<!-- /VALIDATION-TABLE -->"
    s = re.sub(r"<!-- VALIDATION-TABLE -->(.*?<!-- /VALIDATION-TABLE -->)?", lambda m: block, s, count=1, flags=re.S)
    f.write_text(s, encoding="utf-8")
    print("pages written")
