# multi_benchmark.py (robusto per Windows / percorsi assoluti)
import yaml, subprocess, csv, pathlib, sys

# --- Localizzazione root progetto ---
THIS = pathlib.Path(__file__).resolve()
# Se è dentro .../src/data -> ROOT = .../src
ROOT = THIS.parent.parent if THIS.parent.name.lower() == "data" else THIS.parent
SCRIPTS_DIR = ROOT          # dove stanno fits_to_profile.py & co.
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
YAML_PATH = ROOT / "benchmarks.yaml"
SUMMARY_PATH = ROOT / "benchmark_summary.csv"

def abspath_from_root(p):
    # Consente sia "data/..." sia path assoluti nel YAML
    p = pathlib.Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()

def run_py(script_name, *args):
    script = (SCRIPTS_DIR / script_name).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Script non trovato: {script}")
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Config YAML non trovato: {YAML_PATH}")

    cfg = yaml.safe_load(open(YAML_PATH, "r", encoding="utf-8"))
    common = cfg["common"]
    rows = []

    FIG_DIR.mkdir(exist_ok=True)

    for s in cfg["sources"]:
        sid = s["id"]

        # Percorsi input dal YAML risolti dalla root progetto
        fits_path = abspath_from_root(s["fits"])
        axis_reg = abspath_from_root(s["axis_reg"])

        # Cartelle output standardizzate
        base = DATA_DIR / sid
        base.mkdir(parents=True, exist_ok=True)

        raw = base / f"{sid}_raw.csv"
        clean = base / f"{sid}_raw_clean.csv"
        refp = base / f"{sid}_ref_profile.csv"
        metrics = base / "metrics.csv"
        fig = FIG_DIR / f"{sid}_overlay.png"

        # Check input esistenti (errori early & chiari)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS non trovato: {fits_path}")
        if not axis_reg.exists():
            raise FileNotFoundError(f"Axis .reg non trovato: {axis_reg}")

        # 1) Estrazione profili
        run_py("fits_to_profile.py",
               "--fits", fits_path,
               "--axis_reg", axis_reg,
               "--step_mas", common["step_mas"],
               "--half_cut_mas", common["half_cut_mas"],
               "--ncut", common["ncut"],
               "--out", raw)

        # 2) Pulizia
        run_py("clean_obs.py", raw, "-o", clean)

        # 3) Conversione r_g / model units
        run_py("obs_to_ref.py",
               "--in_csv", clean,
               "--mass_Msun", s["mass_Msun"],
               "--distance_Mpc", s["distance_Mpc"],
               "--optimize_L0",
               "--mask", common["mask"],
               "--out", refp)

        # 4) Metriche
        run_py("external_benchmark.py",
               "--ref_csv", refp,
               "--out", metrics)

        # 5) Overlay
        run_py("plot_external_compare.py",
               "--ref_csv", refp,
               "--out", fig)

        # Raccogli metriche principali (se disponibili)
        try:
            with open(metrics, newline="", encoding="utf-8") as f:
                r = next(csv.DictReader(f))
        except Exception:
            r = {}
        r["id"] = sid
        r["L0"] = r.get("L0", "")
        rows.append(r)

    # Tabella riassuntiva
    fieldnames = ["id", "MAE", "MSE", "bias", "L0", "n_points", "notes"]
    with open(SUMMARY_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

if __name__ == "__main__":
    main()
