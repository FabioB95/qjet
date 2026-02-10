# wpd_to_obs.py — robusto a CSV con ; e virgola decimale
import csv, argparse, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True, help="CSV da WebPlotDigitizer (colonne X,Y)")
ap.add_argument("--out_csv", default="./data/obs/M87_raw.csv", help="CSV output con colonne z_mas,r_mas")
ap.add_argument("--x_scale", type=float, default=1.0, help="fattore di scala su X (0.001 se X è in µas->mas)")
ap.add_argument("--y_scale", type=float, default=1.0, help="fattore di scala su Y (0.001 se Y è in µas->mas)")
ap.add_argument("--y_half", action="store_true", help="se Y è larghezza (diametro/FWHM), divide per 2 per ottenere il raggio")
ap.add_argument("--guess_locale", action="store_true", help="accetta formato con ; come separatore e virgola decimale")
args = ap.parse_args()

def to_float(s):
    try: return float(s)
    except: return None

rows = []

if args.guess_locale:
    # Leggi grezzo e normalizza: sostituisci separatore decimale ',' -> '.'
    # e separatore di colonna ';' -> ','
    with open(args.in_csv, "r", encoding="utf-8") as f:
        text = f.read()
    # normalizza: togli spazi, rimpiazza ';' con ',' e virgole decimali con '.'
    # Primo, sostituiamo ';' con ',' per avere colonne, poi convertiamo numeri "1,234" -> "1.234"
    text = text.replace(";", ",")
    # Se ci sono numeri con virgola decimale, sostituiamoli con punto
    # (attenzione a non toccare le virgole separatrici: ma ormai tutte sono ',')
    def repl_decimal(m):
        return m.group(0).replace(",", ".")
    text = re.sub(r"\d+,\d+", repl_decimal, text)
    # Ora possiamo usare csv.reader
    rdr = csv.reader(text.splitlines())
    # Se la prima riga ha header, saltala
    header = next(rdr, None)
    if header and (to_float(header[0]) is None or to_float(header[1]) is None):
        pass
    else:
        # prima riga era dati
        if header:
            try:
                x0 = to_float(header[0]); y0 = to_float(header[1])
                if x0 is not None and y0 is not None:
                    rows.append((x0, y0))
            except:
                pass
    for row in rdr:
        if len(row) < 2: continue
        x = to_float(row[0]); y = to_float(row[1])
        if x is None or y is None: continue
        rows.append((x, y))
else:
    with open(args.in_csv, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header and (to_float(header[0]) is not None and to_float(header[1]) is not None):
            f.seek(0); rdr = csv.reader(f); header=None
        for row in rdr:
            if len(row) < 2: continue
            x = to_float(row[0]); y = to_float(row[1])
            if x is None or y is None: continue
            rows.append((x, y))

# Applica scale e metà larghezza se richiesto
out = []
for x, y in rows:
    z_mas = x * args.x_scale
    r_val = y * args.y_scale
    if args.y_half:
        r_val *= 0.5
    out.append((z_mas, r_val))

os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["z_mas","r_mas"])
    for z, r in out:
        w.writerow([z, r])

print(f"[OK] Saved {args.out_csv}  (N={len(out)}, x_scale={args.x_scale}, y_scale={args.y_scale}, y_half={args.y_half}, guess_locale={args.guess_locale})")
