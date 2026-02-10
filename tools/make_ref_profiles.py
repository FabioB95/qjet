# src/tools/make_ref_profiles.py

import os
import csv

BASE = os.path.join("src", "LATEX_IMG")

def main():
    if not os.path.isdir(BASE):
        print(f"[ERROR] Folder not found: {BASE}")
        return

    sources = sorted(
        d for d in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, d))
    )

    print(f"[INFO] Found {len(sources)} source folders in {BASE}")
    for src in sources:
        folder = os.path.join(BASE, src)
        prof = os.path.join(folder, f"{src}_profile.csv")
        if not os.path.exists(prof):
            print(f"[WARN] No profile CSV for {src} at {prof}, skipping.")
            continue

        raw = os.path.join(folder, f"{src}_profile_raw.csv")
        if not os.path.exists(raw):
            os.rename(prof, raw)
            print(f"[OK] Renamed {prof} -> {raw} (backup created)")
        else:
            print(f"[INFO] Backup already exists for {src}, reusing {raw}")

        rows = []
        with open(raw, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            print(f"[WARN] Empty CSV in {raw}, skipping.")
            continue

        # Determine z and r keys
        sample = rows[0]
        z_key = None
        r_key = None

        # Possible names z, z_mas
        for cand in ["z", "z_mas", "z_rg"]:
            if cand in sample:
                z_key = cand
                break

        # Possible names r, r_mas
        for cand in ["r", "r_mas"]:
            if cand in sample:
                r_key = cand
                break

        if z_key is None or r_key is None:
            print(f"[ERROR] {raw}: cannot find z/r columns (have keys: {list(sample.keys())}), skipping.")
            continue

        out_rows = []
        for row in rows:
            try:
                z = float(row[z_key])
                r = float(row[r_key])
            except Exception:
                continue
            out_rows.append((z, r, r))

        if not out_rows:
            print(f"[WARN] {raw}: no valid numeric rows, skipping.")
            continue

        # Write NEW profile in the format expected by external_benchmark
        with open(prof, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["z", "r_up_ref", "r_down_ref"])
            for z, r_up, r_dn in out_rows:
                w.writerow([z, r_up, r_dn])

        print(f"[OK] Wrote normalized ref profile for {src}: {prof}")

if __name__ == "__main__":
    main()
