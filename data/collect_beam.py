from astropy.io import fits
import pathlib, csv, yaml

YAML = "benchmarks.yaml"
OUT = "beam_info.csv"

def get_beam(h):
    def g(k, default=None):
        return h.get(k, h.get(k.encode(), default))
    return (g("BMAJ",""), g("BMIN",""), g("BPA",""))

def main():
    cfg = yaml.safe_load(open(YAML,"r",encoding="utf-8"))
    rows = []
    for s in cfg["sources"]:
        sid = s["id"]; fits_path = pathlib.Path(s["fits"])
        with fits.open(fits_path) as hdul:
            hdr = hdul[0].header
            bmaj,bmin,bpa = get_beam(hdr)
            rows.append(dict(id=sid, fits=str(fits_path),
                             BMAJ=bmaj, BMIN=bmin, BPA=bpa))
    with open(OUT,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["id","fits","BMAJ","BMIN","BPA"])
        w.writeheader(); w.writerows(rows)

if __name__=="__main__":
    main()
