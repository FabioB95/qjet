#!/usr/bin/env python3
"""
Batch runner for width–profile extraction.

Run from the project root as:

  python -m src.tools.batch_profiles

It will:
  * call fits_to_profile.run(...) for each AGN
  * read the DS9 ridge files you created
  * write one CSV per source in src/data/obs/
"""

import os
from pathlib import Path

# import the main function from your existing script
from src.fits_to_profile import run as fits_to_profile_run


def main():
    # base folders (you already created "data" folders)
    base_data = Path("src/data")
    out_dir   = base_data / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- IMPORTANT ---
    # Here we list ONLY the sources for which you gave me DS9 files
    # (those inside ds9.zip).
    #
    # You just need to adjust the FITS filenames so they match what you
    # actually downloaded (the .fits in each folder).
    #
    # Example: if for M87 the FITS file is called "M87_43GHz.fits"
    # and sits in src/data/M87/, then put that exact name below.

    sources = [
        {
            "name": "M87",
            "fits": base_data / "M87" / "M87.fits",          # <-- change if needed
            "reg" : base_data / "M87" / "M87_axis.reg",
        },
        {
            "name": "3C_273",
            "fits": base_data / "3C_273" / "3C_273.fits",    # <-- change
            "reg" : base_data / "3C_273" / "3C_273_axis.reg",
        },
        {
            "name": "3C_120",
            "fits": base_data / "3C_120" / "3C_120.fits",    # <-- change
            "reg" : base_data / "3C_120" / "3C_120_axis.reg",
        },
        {
            "name": "3C_111",
            "fits": base_data / "3C_111" / "3C_111.fits",    # <-- change
            "reg" : base_data / "3C_111" / "3C_111_axis.reg",
        },
        {
            "name": "3C_84",
            "fits": base_data / "3C_84" / "3C_84.fits",      # <-- change
            "reg" : base_data / "3C_84" / "3C_84_axis.reg",
        },
        {
            "name": "3C_264",
            "fits": base_data / "3C_264" / "3C_264.fits",    # <-- change
            "reg" : base_data / "3C_264" / "3C_264_axis.reg",
        },
        {
            # Note: in the region zip it was "3C_3453_axis.reg".
            # Keep the same spelling you used in your folders.
            "name": "3C_3453",
            "fits": base_data / "3C_3453" / "3C_3453.fits",  # <-- change or remove if you don't use this one
            "reg" : base_data / "3C_3453" / "3C_3453_axis.reg",
        },
        {
            "name": "BL_Lac",
            "fits": base_data / "BL_Lac" / "BL_Lac.fits",    # <-- change
            "reg" : base_data / "BL_Lac" / "BL_Lac_axis.reg",
        },
        {
            # region file in the zip was "NCG_625.reg" (typo).
            # Keep the same filename that you actually saved.
            "name": "NGC_6251",
            "fits": base_data / "NGC_6251" / "NGC_6251.fits",  # <-- change
            "reg" : base_data / "NGC_6251" / "NCG_625.reg",
        },
        {
            "name": "NGC_1052",
            "fits": base_data / "NGC_1052" / "NGC_1052.fits",  # <-- change
            "reg" : base_data / "NGC_1052" / "NGC_1052.reg",
        },
    ]

    # common extraction parameters (you can tune them later if needed)
    step_mas      = 0.05
    half_cut_mas  = 1.0
    ncut          = 200

    print("[BATCH] Starting profile extraction for all sources...\n")

    for src in sources:
        name    = src["name"]
        fits_fn = src["fits"]
        reg_fn  = src["reg"]
        out_csv = out_dir / f"{name}_raw.csv"

        print(f"[{name}]")
        print(f"  FITS: {fits_fn}")
        print(f"  REG : {reg_fn}")
        print(f"  OUT : {out_csv}")

        if not fits_fn.exists():
            print(f"  [WARN] FITS file not found, skipping this source.")
            print()
            continue

        if not reg_fn.exists():
            print(f"  [WARN] DS9 region file not found, skipping this source.")
            print()
            continue

        # call your existing function from fits_to_profile.py
        fits_to_profile_run(
            str(fits_fn),
            str(reg_fn),
            step_mas,
            half_cut_mas,
            ncut,
            str(out_csv),
        )

        print(f"  [OK] Finished {name}\n")

    print("[BATCH] Done.")


if __name__ == "__main__":
    main()
