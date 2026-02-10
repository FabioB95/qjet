# obs_to_ref.py — converte osservazioni (mas) in un profilo di riferimento in unità del modello
# Output: CSV "wide" con colonne [z, r_up_ref, r_down_ref] + meta YAML
import argparse, numpy as np, pandas as pd
import yaml
from pathlib import Path

# costanti fisiche
G = 6.67430e-11
c = 2.99792458e8
MSUN = 1.98847e30
PC = 3.085677581491367e16
MAS2RAD = np.deg2rad(1/3600/1e3)  # 1 mas in rad

def mas_to_rg(x_mas, D_Mpc, M_Msun):
    D_m = D_Mpc * 1e6 * PC
    rg = G * M_Msun * MSUN / c**2
    return x_mas * MAS2RAD * D_m / rg

def guess_cols(df):
    zc = None; rc = None; sc = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("z_mas","zmas","z","z_arcsec","z_pix","zmas_up","zmas_down") and zc is None:
            zc = c
        if cl in ("r_mas","r","width_mas","fwhm_mas","sigma_mas") and rc is None:
            rc = c
        if cl in ("side","beam","jet_side") and sc is None:
            sc = c
    if zc is None or rc is None:
        raise ValueError(f"Colonne z/r non trovate. Vedo: {list(df.columns)}")
    return zc, rc, sc

def main():
    ap = argparse.ArgumentParser()
    # supporta sia --obs_csv che posizionale
    ap.add_argument("in_csv", nargs="?", help="CSV osservazioni")
    ap.add_argument("--obs_csv", dest="obs_csv", help="CSV osservazioni (alias)")
    ap.add_argument("--mass_Msun", type=float, required=True)
    ap.add_argument("--distance_Mpc", type=float, required=True)
    ap.add_argument("--modelA", type=str, required=False, help="cartella modello (solo per coerenza pipeline)")
    ap.add_argument("--mask", choices=["core","all"], default="core")
    ap.add_argument("--optimize_s", action="store_true", help="stima anche s; default: s fisso")
    ap.add_argument("--s0", type=float, default=0.50, help="scala radiale s di default")
    ap.add_argument("--zmin", type=float, default=-4.0)
    ap.add_argument("--zmax", type=float, default= 4.0)
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()

    obs_csv = args.obs_csv or args.in_csv
    if not obs_csv:
        ap.error("devi specificare il CSV: --obs_csv IN oppure argomento posizionale IN")

    obs_csv = Path(obs_csv)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) leggi e individua colonne
    df = pd.read_csv(obs_csv)
    zc, rc, sc = guess_cols(df)

    # 2) converte a r_g
    z_rg = mas_to_rg(df[zc].to_numpy(float), args.distance_Mpc, args.mass_Msun)
    r_rg = mas_to_rg(df[rc].to_numpy(float), args.distance_Mpc, args.mass_Msun)

    # 3) stima L0 e (opz.) s nel core
    r_target = 0.17
    s = float(args.s0)

    core_mask_phys = np.isfinite(z_rg) & np.isfinite(r_rg) & (np.abs(z_rg) <= 2e4) & (r_rg > 0)
    if not np.any(core_mask_phys):
        core_mask_phys = np.isfinite(z_rg) & np.isfinite(r_rg) & (r_rg > 0)

    med_r = np.nanmedian(r_rg[core_mask_phys])
    if args.optimize_s:
        # quick two-step robust: prima L0 con s0, poi ritaro s
        L0 = max(100.0, (s * med_r) / r_target)
        s = max(1e-3, (r_target * L0) / (med_r + 1e-12))
    else:
        L0 = max(100.0, (s * med_r) / r_target)

    # 4) proietta in unità del modello
    z_unit = z_rg / L0
    r_unit = s * r_rg / L0

    # 5) maschera finale
    keep = np.isfinite(z_unit) & np.isfinite(r_unit) & (r_unit >= 0)
    if args.mask == "core":
        keep &= (z_unit >= args.zmin) & (z_unit <= args.zmax) & (np.abs(z_unit) <= 2.0)
    else:
        keep &= (z_unit >= args.zmin) & (z_unit <= args.zmax)

    # 6) lato (up/down)
    if sc and sc in df.columns:
        side = df[sc].astype(str).str.lower().where(df[sc].notna(), None)
        side = np.where(side.isin(["up","down"]), side, np.where(z_unit>=0, "up", "down"))
    else:
        side = np.where(z_unit>=0, "up", "down")

    out_long = pd.DataFrame({
        "z": z_unit[keep],
        "r": r_unit[keep],
        "side": side[keep]
    })

    # 7) formato wide richiesto da plot_external_compare_cli.py
    if len(out_long) == 0:
        wide = pd.DataFrame({"z": [], "r_up_ref": [], "r_down_ref": []})
    else:
        wide_piv = out_long.pivot_table(index="z", columns="side", values="r", aggfunc=np.nanmedian)
        wide_piv = wide_piv.sort_index()
        up  = wide_piv["up"]   if "up"   in wide_piv else pd.Series(index=wide_piv.index, data=np.nan)
        dwn = wide_piv["down"] if "down" in wide_piv else pd.Series(index=wide_piv.index, data=np.nan)
        wide = pd.DataFrame({
            "z": wide_piv.index.values,
            "r_up_ref":  up.values.astype(float),
            "r_down_ref": dwn.values.astype(float),
        })

    wide = wide.sort_values("z").reset_index(drop=True)
    wide.to_csv(out_csv, index=False)

    # diagnostica
    core_sel = (np.abs(out_long["z"]) <= 2.0) if len(out_long) else pd.Series([], dtype=bool)
    mae = float(np.mean(np.abs(out_long.loc[core_sel, "r"] - r_target))) if np.any(core_sel) else float("nan")
    nvalid = int(core_sel.sum()) if len(out_long) else 0

    z_fin = z_rg[np.isfinite(z_rg)]
    r_fin = r_rg[np.isfinite(r_rg)]
    print(f"[DEBUG] z_rg range: {z_fin.min():.2f} – {z_fin.max():.2f} r_g, "
          f"r_rg range: {r_fin.min():.2f} – {r_fin.max():.2f} r_g")
    print(f"[OBS2REF] Best: L0 ≈ {L0:.1f} r_g/unit, s ≈ {s:.3f}, MAE ≈ {mae:.4f}, "
          f"mask={'core' if args.mask=='core' else 'all'}, nvalid={nvalid}")
    print(f"[OK] Saved {out_csv}")

    # 8) META accanto al CSV
    meta = {
        "L0_rg_per_unit": float(L0),
        "s": float(s),
        "mask": args.mask,
        "nvalid": int(nvalid),
    }
    meta_path = out_csv.with_name(out_csv.stem + "_meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    print(f"[OK] Saved {meta_path}")

if __name__ == "__main__":
    main()
