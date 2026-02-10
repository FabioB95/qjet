# run_benchmarks.py — runner batch robusto e compatibile coi tuoi script
import yaml, subprocess, csv, pathlib, sys, os
import shutil
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "figures" / "K16_optimized"   # cartella con Sx_opt.npy, ... (esiste nel tuo repo)

# trova benchmarks.yaml in src o src/data
CANDIDATES = [ROOT / "benchmarks.yaml", ROOT / "data" / "benchmarks.yaml"]
YAML_PATH = next((p for p in CANDIDATES if p.exists()), None)
if YAML_PATH is None:
    raise FileNotFoundError(f"benchmarks.yaml non trovato in: {', '.join(map(str,CANDIDATES))}")

def abspath(p):
    p = pathlib.Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()

def run_py(script, *args):
    script_path = (ROOT / script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script non trovato: {script_path}")
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    cfg = yaml.safe_load(open(YAML_PATH, "r", encoding="utf-8"))
    common = cfg["common"]
    FIG_DIR.mkdir(exist_ok=True)
    rows = []

    for s in cfg["sources"]:
        sid = s["id"]
        fits_path = abspath(s["fits"])
        axis_reg  = abspath(s["axis_reg"])

        # skip se manca qualcosa
        missing = []
        if not fits_path.exists(): missing.append(f"FITS {fits_path}")
        if not axis_reg.exists():  missing.append(f"axis_reg {axis_reg}")
        if missing:
            print(f"[SKIP] {sid}: manca {', '.join(missing)}")
            continue

        base = DATA_DIR / sid
        base.mkdir(parents=True, exist_ok=True)
        raw     = base / f"{sid}_raw.csv"
        clean   = base / f"{sid}_raw_clean.csv"      # destinazione desiderata
        refp    = base / f"{sid}_ref_profile.csv"
        metrics = base / "metrics.csv"
        fig     = FIG_DIR / f"{sid}_overlay.png"

        # 1) Estrazione profili (la tua versione di fits_to_profile.py accetta --out, non --out_csv)
        try:
            run_py("fits_to_profile.py",
                   "--fits", fits_path,
                   "--axis_reg", axis_reg,
                   "--step_mas", common["step_mas"],
                   "--half_cut_mas", common["half_cut_mas"],
                   "--ncut", common["ncut"],
                   "--out", raw)
        except subprocess.CalledProcessError:
            # fallback per eventuale versione che usa --out_csv
            run_py("fits_to_profile.py",
                   "--fits", fits_path,
                   "--axis_reg", axis_reg,
                   "--step_mas", common["step_mas"],
                   "--half_cut_mas", common["half_cut_mas"],
                   "--ncut", common["ncut"],
                   "--out_csv", raw)

       
        # 2) Pulizia disattivata: la tua clean_obs.py è hard-coded su M87 e inquina le altre sorgenti
        obs_csv = raw




        # 3) Conversione in r_g + ref_profile (obs_to_ref.py richiede --obs_csv)
        
        run_py("obs_to_ref.py",
               "--obs_csv", obs_csv,
               "--mass_Msun", s["mass_Msun"],
               "--distance_Mpc", s["distance_Mpc"],
               "--modelA", MODEL_DIR,
               "--out", refp)


        # 4) Metriche esterne (external_benchmark usa --A e --ref)
        run_py("external_benchmark.py",
               "--A", MODEL_DIR,
               "--ref", refp,
               "--out", metrics)

        # 5) Overlay (usa la nuova versione CLI qui sotto)
        run_py("plot_external_compare_cli.py",
               "--A", MODEL_DIR,
               "--ref", refp,
               "--out", fig,
               "--title", f"{sid} Jet: Model vs Observations")

        # raccogli una riga di summary (se hai bisogno di L0, puoi aggiungerlo in ref_profile.csv)
        # metrics.csv del tuo script ha due righe (up/down); mettiamo la media per compattezza
        try:
            import pandas as pd
            df = pd.read_csv(metrics)
            mae = df["MAE"].mean() if "MAE" in df else float("nan")
            bias = df["bias"].mean() if "bias" in df else float("nan")
            npts = sum(1 for _ in open(refp, "r", encoding="utf-8")) - 1  # tolta header
        except Exception:
            mae = bias = float("nan"); npts = ""
        rows.append(dict(id=sid, MAE=mae, bias=bias, MSE="", L0="", n_points=npts, notes=""))

  
            
            
    # summary globale ricco
    rows = []
    for s in cfg["sources"]:
        sid = s["id"]
        base = DATA_DIR / sid
        metrics = base / "metrics.csv"
        refp    = base / f"{sid}_ref_profile.csv"
        meta_y  = base / f"{sid}_ref_profile_meta.yaml"

        if not metrics.exists(): continue
        dfm = pd.read_csv(metrics)
        def pick(col, side): 
            try: return float(dfm.loc[dfm["side"]==side, col].iloc[0])
            except: return float("nan")
        mae_up  = pick("MAE","up");   mae_dn  = pick("MAE","down")
        bias_up = pick("bias","up");  bias_dn = pick("bias","down")

        L0 = s_ = None
        if meta_y.exists():
            meta = yaml.safe_load(open(meta_y, "r", encoding="utf-8"))
            L0 = meta.get("L0_rg_per_unit", None)
            s_ = meta.get("s", None)

        # numero punti utili (per lato)
        try:
            dfr = pd.read_csv(refp)
            n_up = int(pd.Series(dfr.get("r_up_ref", [])).notna().sum())
            n_dn = int(pd.Series(dfr.get("r_down_ref", [])).notna().sum())
        except:
            n_up = n_dn = ""

        rows.append(dict(
            id=sid,
            MAE_up=mae_up, MAE_down=mae_dn, MAE_mean=pd.Series([mae_up,mae_dn]).mean(),
            bias_up=bias_up, bias_down=bias_dn, bias_mean=pd.Series([bias_up,bias_dn]).mean(),
            L0=L0, s=s_, n_up=n_up, n_down=n_dn
        ))

    pd.DataFrame(rows).to_csv(ROOT / "benchmark_summary.csv", index=False)
    print("[OK] wrote benchmark_summary.csv")


if __name__ == "__main__":
    main()
