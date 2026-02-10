import pandas as pd, matplotlib.pyplot as plt, pathlib, glob

def main():
    for mfile in glob.glob("data/*/metrics.csv"):
        df = pd.read_csv(mfile)
        sid = pathlib.Path(mfile).parts[1]
        if "z_model" in df and "r_model" in df and "r_obs" in df:
            res = df["r_obs"] - df["r_model"]
            z = df["z_model"]
            plt.figure()
            plt.scatter(z, res, s=10)
            plt.axhline(0, ls="--")
            plt.xlabel("z (model units or r_g)")
            plt.ylabel("Residual (r_obs - r_model)")
            pathlib.Path("figures").mkdir(exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"figures/{sid}_residuals.png", dpi=180)
            plt.close()

if __name__=="__main__":
    main()
