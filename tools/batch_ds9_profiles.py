#!/usr/bin/env python3
"""
Batch extractor for transverse jet-width profiles.

For each DS9 axis region in:
    src/data/ds9/*.reg

we:

  1. Infer the AGN name from the region filename (e.g., M87_axis.reg → "M87").
  2. Map it to the correct data folder in src/data/<AGN_DIR>.
  3. Find a FITS image in that folder.
  4. Call src.fits_to_profile.run(...) to extract a z_mas / r_mas profile.
  5. Save the CSV and a simple PNG plot into:
        src/LATEX_IMG/<AGN_DIR>/<AGN_DIR>_profile.csv
        src/LATEX_IMG/<AGN_DIR>/<AGN_DIR>_profile.png

Run from the project root as:

  python -m src.tools.batch_ds9_profiles
"""

import os
import csv
import pathlib
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Import the existing profile extractor
from src.fits_to_profile import run as extract_profile


def _norm(s: str) -> str:
    """Normalize a name to alphanumeric uppercase (for fuzzy matching)."""
    return "".join(ch.upper() for ch in s if ch.isalnum())


def infer_base_name(region_file: pathlib.Path) -> str:
    """
    Extract base AGN name from region filename.
    Examples:
        M87_axis.reg          → "M87"
        3C_111_axis.reg       → "3C_111"
        NGC_6251.reg          → "NGC_6251"
        3C273_axis_long.reg   → "3C273"
    """
    stem = region_file.stem  # removes .reg
    # Remove common suffixes like "_axis", "_long", etc. (keep only leading part before first "_axis" or similar)
    if "_axis" in stem:
        base = stem.split("_axis")[0]
    else:
        base = stem
    return base


def build_region_to_dir_map(data_dir: pathlib.Path, region_files: List[pathlib.Path]) -> Dict[str, pathlib.Path]:
    """
    Given a list of region files, infer their base names and map each to
    a concrete data subdirectory that contains FITS images.

    We ONLY use AGN that:
      - have a DS9 axis region, AND
      - have at least one FITS file in one of the data/<AGN_DIR> folders.
    """
    ds9_dir = data_dir / "ds9"
    if not ds9_dir.is_dir():
        raise SystemExit(f"[ERROR] Missing DS9 directory: {ds9_dir}")

    # All available data subdirectories (except ds9 itself)
    all_dirs: List[pathlib.Path] = [
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.lower() != "ds9"
    ]

    # Build base names from region files
    region_bases: List[str] = []
    for f in region_files:
        base = infer_base_name(f)
        region_bases.append(base)

    region_bases = sorted(set(region_bases))

    # Manual overrides for ambiguous/mismatched names
    manual_map = {
        # Same galaxy, different naming
        "NGC_625": "NGC_6251",
        "NGC_6251": "NGC_6251",
        # 3C345 vs 3C3453 both present
        "3C_345": "3C_345",
        "3C_3453": "3C_3453",
        # OC_272 region saved, data folder actually OC_270
        "OC_272": "OC_270",
        # Add more if needed
    }

    # Build normalization dictionary for folders
    dir_norm_map = {d.name: _norm(d.name) for d in all_dirs}

    mapping: Dict[str, pathlib.Path] = {}

    for base in region_bases:
        nb = _norm(base)

        # Manual override first
        if base in manual_map:
            target_name = manual_map[base]
            candidate = data_dir / target_name
            if candidate.is_dir():
                mapping[base] = candidate
                continue
            else:
                print(f"[WARN] Manual map for {base} -> {target_name}, but folder not found.")

        # Fuzzy matching: exact normalized name, or substring
        candidates = []
        for d in all_dirs:
            nd = dir_norm_map[d.name]
            if nb == nd or nb in nd or nd in nb:
                candidates.append(d)

        if not candidates:
            print(f"[WARN] No matching data folder for region base '{base}'. Skipping.")
            continue

        # If multiple, prefer exact string match on the raw name, then shortest name
        chosen = None
        for d in candidates:
            if d.name == base:
                chosen = d
                break
        if chosen is None:
            # Pick the one with the shortest name as a reasonable heuristic
            chosen = min(candidates, key=lambda d: len(d.name))

        mapping[base] = chosen

    return mapping


def find_first_fits(folder: pathlib.Path) -> pathlib.Path:
    """
    Return the first FITS file in 'folder'. Raise if none.
    """
    fits_candidates = [
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in (".fits", ".fit", ".fits.gz")
    ]
    if not fits_candidates:
        raise FileNotFoundError(f"No FITS file found in {folder}")
    # Sort for determinism
    fits_candidates.sort()
    return fits_candidates[0]


def load_profile_csv(csv_path: pathlib.Path):
    """
    Load z_mas and r_mas columns from the CSV written by fits_to_profile.run.
    Returns (z, r) as numpy arrays with NaN rows removed.
    """
    z_list = []
    r_list = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            z_val = row.get("z_mas", "")
            r_val = row.get("r_mas", "")
            if z_val == "" or r_val == "":
                continue
            try:
                z = float(z_val)
                r = float(r_val)
            except ValueError:
                continue
            if np.isfinite(z) and np.isfinite(r):
                z_list.append(z)
                r_list.append(r)

    if not z_list:
        return np.array([]), np.array([])
    return np.array(z_list), np.array(r_list)


def plot_profile(z: np.ndarray, r: np.ndarray, out_png: pathlib.Path, title: str):
    """
    Simple profile plot: r_mas vs z_mas (both in mas).
    """
    if z.size == 0:
        print(f"[WARN] Empty profile for {title}, no plot produced.")
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.plot(z, r, marker=".", linestyle="-")
    plt.xlabel("z (mas)")
    plt.ylabel("half-width r (mas)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved plot {out_png}")


def main():
    # Project root = .../QUANTUM_BH
    this_file = pathlib.Path(__file__).resolve()
    project_root = this_file.parents[2]

    data_dir = project_root / "src" / "data"
    ds9_dir = data_dir / "ds9"
    latex_img_root = project_root / "src" / "LATEX_IMG"  # ✅ Now points to src/LATEX_IMG

    if not data_dir.is_dir():
        raise SystemExit(f"[ERROR] data dir not found: {data_dir}")
    if not ds9_dir.is_dir():
        raise SystemExit(f"[ERROR] ds9 dir not found: {ds9_dir}")

    latex_img_root.mkdir(parents=True, exist_ok=True)  # ✅ Create if missing

    # Find ALL .reg files in ds9_dir
    region_files = [
        f for f in ds9_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".reg"
    ]

    if not region_files:
        print(f"[WARN] No .reg files found in {ds9_dir}")
        return

    print(f"Found {len(region_files)} region file(s) in {ds9_dir}")

    # Map region base names to data folders
    base_to_dir = build_region_to_dir_map(data_dir, region_files)

    total = 0
    ok_count = 0
    skipped = []

    # Process in sorted order
    for reg_path in sorted(region_files):
        base = infer_base_name(reg_path)
        total += 1

        if base not in base_to_dir:
            print(f"[SKIP] {base}: no mapped data folder.")
            skipped.append(base)
            continue

        agn_dir = base_to_dir[base]
        agn_name = agn_dir.name

        # Ensure we have a FITS
        try:
            fits_path = find_first_fits(agn_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {base}: {e}")
            skipped.append(base)
            continue

        print(f"\n[INFO] Processing {base} -> folder '{agn_name}'")
        print(f"       FITS: {fits_path}")
        print(f"       DS9 : {reg_path}")

        # Output paths under src/LATEX_IMG/<AGN_DIR>/
        out_dir = latex_img_root / agn_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_dir / f"{agn_name}_profile.csv"
        out_png = out_dir / f"{agn_name}_profile.png"

        # 1) Extract profile
        try:
            extract_profile(
                str(fits_path),
                str(reg_path),
                step_mas=0.05,
                half_cut_mas=1.0,
                ncut=200,
                out_csv=str(out_csv),
            )
        except Exception as e:
            print(f"[ERROR] Profile extraction failed for {base}: {e}")
            skipped.append(base)
            continue

        # 2) Load CSV and plot
        z, r = load_profile_csv(out_csv)
        plot_profile(z, r, out_png, title=agn_name)

        ok_count += 1

    print("\n================ SUMMARY ================")
    print(f"Total regions found : {total}")
    print(f"Successfully processed: {ok_count}")
    if skipped:
        print(f"Skipped ({len(skipped)}): {', '.join(sorted(set(skipped)))}")


if __name__ == "__main__":
    main()