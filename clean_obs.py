# salva come clean_obs.py nella cartella src
import csv
import sys
import statistics as st

inp = r"data/obs/M87_raw.csv"
out = r"data/obs/M87_raw_clean.csv"

Z, R = [], []

with open(inp) as f:
    rd = csv.DictReader(f)
    for r in rd:
        z_str = r.get("z_mas", "").strip()
        r_str = r.get("r_mas", "").strip()
        # Salta righe con valori mancanti o non convertibili
        if not z_str or not r_str:
            continue
        try:
            z_val = float(z_str)
            r_val = float(r_str)
            if r_val <= 0:  # opzionale: salta anche r <= 0 qui
                continue
            Z.append(z_val)
            R.append(r_val)
        except ValueError:
            continue  # ignora righe con dati non numerici

if not R:
    print("[ERROR] Nessun dato valido trovato.")
    sys.exit(1)

# rolling median (finestra 5) + sigma-clip locale (3.5 MAD)
def roll_med(a, k=5):
    half = k // 2
    out = []
    for i in range(len(a)):
        j0 = max(0, i - half)
        j1 = min(len(a), i + half + 1)
        out.append(st.median(a[j0:j1]))
    return out

Rm = roll_med(R, 5)
mad = st.median([abs(r - m) for r, m in zip(R, Rm)]) or 1e-9
threshold = 3.5 * 1.4826 * mad

clean = [(z, r) for z, r, m in zip(Z, R, Rm) if abs(r - m) <= threshold]

with open(out, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["z_mas", "r_mas"])
    wr.writerows(clean)

print(f"[OK] wrote {out}  (kept {len(clean)}/{len(R) + len([x for x in R if x <= 0])})")