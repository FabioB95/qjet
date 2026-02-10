# plot_external_compare_cli.py — overlay CLI
import argparse, csv, numpy as np, matplotlib.pyplot as plt
from benchmark import load_field, grid_like, normalize, core_profile

def load_ref(csvpath):
    import csv, numpy as np
    z_ref, r_up_ref, r_dn_ref = [], [], []
    with open(csvpath, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        name_map = {c.lower(): c for c in rdr.fieldnames}
        cols = set(k.lower() for k in rdr.fieldnames)

        # Caso 1: formato wide (z, r_up_ref, r_down_ref)
        if {"z", "r_up_ref", "r_down_ref"} <= cols:
            zc  = name_map["z"]
            uc  = name_map["r_up_ref"]
            dc  = name_map["r_down_ref"]
            for row in rdr:
                z_ref.append(float(row[zc]))
                u = row[uc].strip(); d = row[dc].strip()
                r_up_ref.append(float(u) if u != "" else np.nan)
                r_dn_ref.append(float(d) if d != "" else np.nan)
            return np.array(z_ref, float), np.array(r_up_ref, float), np.array(r_dn_ref, float)

        # Caso 2: formato long (z, r, side)
        elif {"z", "r", "side"} <= cols:
            zc = name_map["z"]; rc = name_map["r"]; sc = name_map["side"]
            by_z = {}
            for row in rdr:
                try:
                    z = float(row[zc])
                except Exception:
                    continue
                side = (row[sc] or "").strip().lower()
                try:
                    rval = float(row[rc])
                except Exception:
                    rval = float("nan")
                bucket = by_z.setdefault(z, {"up": [], "down": []})
                if side in ("up", "down"):
                    bucket[side].append(rval)
                else:
                    bucket["up" if z >= 0 else "down"].append(rval)

            z_sorted = sorted(by_z.keys())
            rup, rdn = [], []
            for z in z_sorted:
                ups = [x for x in by_z[z]["up"] if np.isfinite(x)]
                dns = [x for x in by_z[z]["down"] if np.isfinite(x)]
                rup.append(np.median(ups) if ups else np.nan)
                rdn.append(np.median(dns) if dns else np.nan)
            return np.array(z_sorted, float), np.array(rup, float), np.array(rdn, float)

        else:
            raise ValueError(f"Formato ref non riconosciuto in {csvpath}. Colonne: {list(rdr.fieldnames)}")




def metrics(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0: return np.nan, np.nan
    diff = a[mask] - b[mask]
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    return mae, bias

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True, help="cartella modello (con Sx_opt.npy,...)")
    ap.add_argument("--ref", required=True, help="CSV riferimento con z,r_up_ref,r_down_ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Model vs Observations")
    args = ap.parse_args()

    # modello
    Sx, Sy, Sz = load_field(args.A)
    x, y, z_model, _, _, _ = grid_like(Sx)
    bx, by, bz = normalize(Sx, Sy, Sz)
    r_up_model, r_dn_model = core_profile(bx, by, bz, x, y, z_model)

    # riferimento
    z_ref, r_up_ref, r_dn_ref = load_ref(args.ref)

    # interpola modello su z_ref per confronto visivo
    r_up_m_on_ref = np.interp(z_ref, z_model, r_up_model, left=np.nan, right=np.nan)
    r_dn_m_on_ref = np.interp(z_ref, z_model, r_dn_model, left=np.nan, right=np.nan)

    mae_up, bias_up = metrics(r_up_m_on_ref, r_up_ref)
    mae_dn, bias_dn = metrics(r_dn_m_on_ref, r_dn_ref)
    mae = np.nanmean([mae_up, mae_dn])
    bias = np.nanmean([bias_up, bias_dn])

    plt.figure(figsize=(6.5,4.2))
    # modello (linee)
    plt.plot(z_model, r_up_model, lw=2, label="Model (up)")
    plt.plot(z_model, r_dn_model, lw=2, ls="--", label="Model (down)")
    # osservazioni (punti)
    plt.scatter(z_ref, r_up_ref, s=10, alpha=0.8, label="Obs (up)")
    plt.scatter(z_ref, r_dn_ref, s=10, alpha=0.8, label="Obs (down)")

    plt.xlabel("z (model units)")
    plt.ylabel("r (model units)")
    plt.title(args.title, fontsize=12, pad=10)

    if np.isfinite(mae):
        plt.text(0.03, 0.96, f"MAE = {mae:.3f}\nBias = {bias:.3f}",
                 transform=plt.gca().transAxes, fontsize=10,
                 va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()
    print(f"[OK] Saved {args.out}")
