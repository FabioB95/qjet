import os
import time
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

# Force on-screen rendering (disable off-screen for testing)
mlab.options.offscreen = False

def plot_field(Sx, Sy, Sz, title_prefix, out_prefix, rotate_gif=False):
    # Deduce shape (known case 100x100x136)
    L = len(Sx)
    if L == 1360000:
        nx, ny, nz = 100, 100, 136
    else:
        raise ValueError(f"Shape non riconosciuta (L={L})")

    Sx = Sx.reshape((nx, ny, nz))
    Sy = Sy.reshape((nx, ny, nz))
    Sz = Sz.reshape((nx, ny, nz))

    # Debug: Check raw reshaped data
    print(f"  [plot_field] Raw reshaped data shapes: Sx={Sx.shape}, Sy={Sy.shape}, Sz={Sz.shape}")
    print(f"  [plot_field] Raw reshaped Sx - Min: {Sx.min():.4f}, Max: {Sx.max():.4f}, Mean: {Sx.mean():.4f}")
    print(f"  [plot_field] Raw reshaped Sy - Min: {Sy.min():.4f}, Max: {Sy.max():.4f}, Mean: {Sy.mean():.4f}")
    print(f"  [plot_field] Raw reshaped Sz - Min: {Sz.min():.4f}, Max: {Sz.max():.4f}, Mean: {Sz.mean():.4f}")
    print(f"  [plot_field] Raw reshaped Sx has NaN: {np.any(np.isnan(Sx))}, Inf: {np.any(np.isinf(Sx))}")
    print(f"  [plot_field] Raw reshaped Sy has NaN: {np.any(np.isnan(Sy))}, Inf: {np.any(np.isinf(Sy))}")
    print(f"  [plot_field] Raw reshaped Sz has NaN: {np.any(np.isnan(Sz))}, Inf: {np.any(np.isinf(Sz))}")

    # Compute magnitude and normalize
    Bmag_raw = np.sqrt(Sx**2 + Sy**2 + Sz**2)
    print(f"  [plot_field] Bmag_raw - Min: {Bmag_raw.min():.4f}, Max: {Bmag_raw.max():.4f}, Mean: {Bmag_raw.mean():.4f}")
    print(f"  [plot_field] Bmag_raw has NaN: {np.any(np.isnan(Bmag_raw))}, Inf: {np.any(np.isinf(Bmag_raw))}")

    # Add small value to avoid division by zero
    Bmag = Bmag_raw + 1e-12
    print(f"  [plot_field] Bmag (after adding 1e-12) - Min: {Bmag.min():.4f}, Max: {Bmag.max():.4f}, Mean: {Bmag.mean():.4f}")
    print(f"  [plot_field] Bmag (after adding 1e-12) has NaN: {np.any(np.isnan(Bmag))}, Inf: {np.any(np.isinf(Bmag))}")

    # Normalize vectors, handle zero magnitudes
    mask = Bmag < 1e-10  # Identify near-zero magnitudes
    Sx_norm = np.where(mask, 1e-6, Sx / Bmag)  # Replace invalid with small value
    Sy_norm = np.where(mask, 0, Sy / Bmag)
    Sz_norm = np.where(mask, 0, Sz / Bmag)

    print(f"  [plot_field] Normalized Sx_norm - Min: {Sx_norm.min():.4f}, Max: {Sx_norm.max():.4f}, Mean: {Sx_norm.mean():.4f}")
    print(f"  [plot_field] Normalized Sy_norm - Min: {Sy_norm.min():.4f}, Max: {Sy_norm.max():.4f}, Mean: {Sy_norm.mean():.4f}")
    print(f"  [plot_field] Normalized Sz_norm - Min: {Sz_norm.min():.4f}, Max: {Sz_norm.max():.4f}, Mean: {Sz_norm.mean():.4f}")
    print(f"  [plot_field] Normalized Sx_norm has NaN: {np.any(np.isnan(Sx_norm))}, Inf: {np.any(np.isinf(Sx_norm))}")
    print(f"  [plot_field] Normalized Sy_norm has NaN: {np.any(np.isnan(Sy_norm))}, Inf: {np.any(np.isinf(Sy_norm))}")
    print(f"  [plot_field] Normalized Sz_norm has NaN: {np.any(np.isnan(Sz_norm))}, Inf: {np.any(np.isinf(Sz_norm))}")

    Sx = Sx_norm
    Sy = Sy_norm
    Sz = Sz_norm

    src = mlab.pipeline.vector_field(Sx, Sy, Sz, scalars=Bmag)

    seed_types = ["line", "sphere", "plane"]

    for seed in seed_types:
        print(f"  [plot_field] Creating figure for seed: {seed}")
        fig = mlab.figure(bgcolor=(0,0,0), size=(900,700))
        stream = mlab.pipeline.streamline(
            src,
            integration_direction='both',
            seedtype=seed,
            seed_scale=1.0,
            seed_resolution=10,
            seed_visible=True
        )
        stream.stream_tracer.maximum_propagation = max(nx, ny, nz) * 2
        stream.streamline_type = 'tube'
        try:
            stream.tube_filter.radius = 0.5
            stream.tube_filter.number_of_sides = 8
        except AttributeError:
            print(f"    Warning: Could not set tube_filter properties for {seed}")

        surf = mlab.pipeline.surface(stream, colormap='jet', vmin=0, vmax=1)
        mlab.view(azimuth=45, elevation=70, distance=max(nx,ny,nz)*1.8)
        mlab.title(f"{title_prefix} ({seed})", size=0.5, color=(1,1,1))

        if rotate_gif:
            for angle in range(0, 360, 10):
                print(f"    Rendering scene for {seed} at angle {angle}...")
                mlab.view(azimuth=angle, elevation=70, distance=max(nx,ny,nz)*1.8)
                mlab.draw()
                fig.scene.render()
                mlab.process_ui_events()
                time.sleep(0.5)
                fname = f"figures/{out_prefix}_{seed}_angle{angle}.png"
                print(f"    Saving {fname}...")
                mlab.savefig(fname)
                print(f"    Saved {fname}.")
        else:
            print(f"    Rendering scene for {seed}...")
            mlab.draw()
            fig.scene.render()
            mlab.process_ui_events()
            mlab.show()  # Force interactive rendering
            time.sleep(3)  # Wait for rendering
            fname = f"figures/{out_prefix}_{seed}.png"
            print(f"    Saving {fname}...")
            mlab.savefig(fname)
            print(f"    Saved {fname}.")
        # Avoid closing the figure to prevent engine issues
        # mlab.close(fig)  # Commented out to avoid TypeError
        print(f"[OK] Salvato {fname}")

def make_collage(before_prefix, after_prefix, out_file):
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    seeds = ["line", "sphere", "plane"]

    for i, seed in enumerate(seeds):
        before_img = Image.open(f"figures/{before_prefix}_{seed}.png")
        after_img = Image.open(f"figures/{after_prefix}_{seed}.png")

        axs[i,0].imshow(before_img)
        axs[i,0].set_title(f"Before ({seed})")
        axs[i,0].axis("off")

        axs[i,1].imshow(after_img)
        axs[i,1].set_title(f"After ({seed})")
        axs[i,1].axis("off")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Salvato collage: {out_file}")

def make_gif(before_prefix, after_prefix, out_file, n_frames=36):
    frames = []
    seeds = ["sphere"]  # Use sphere for GIF
    if rotate_gif:
        for angle in range(0, 360, 10):
            for prefix in [before_prefix, after_prefix]:
                img_path = f"figures/{prefix}_{seeds[0]}_angle{angle}.png"
                if os.path.exists(img_path):
                    print(f"Loading {img_path} for GIF...")
                    img = Image.open(img_path).convert("RGB")
                    frames.append(img.copy())
    else:
        for prefix in [before_prefix, after_prefix]:
            img_path = f"figures/{prefix}_{seeds[0]}.png"
            if os.path.exists(img_path):
                print(f"Loading {img_path} for GIF...")
                img = Image.open(img_path).convert("RGB")
                frames.append(img.copy())

    if frames:
        frames[0].save(
            out_file,
            save_all=True,
            append_images=frames[1:],
            duration=100 if rotate_gif else 500,
            loop=0
        )
        print(f"[OK] GIF saved in {out_file}")
    else:
        print(f"[ERROR] No frames found for GIF {out_file}")

if __name__ == "__main__":
    # Test Mayavi rendering
    print("Testing Mayavi rendering...")
    test_fig = mlab.figure(bgcolor=(0,0,0), size=(900,700))
    mlab.test_contour3d()
    mlab.savefig("figures/test_contour.png")
    mlab.close(test_fig)  # Safe to close test figure
    print("[OK] Saved test_contour.png - Check if this image is not black")

    os.makedirs("figures", exist_ok=True)

    # Load initial data
    Sx = np.load("../data/spins/data3D0000_Sx.npy")
    Sy = np.load("../data/spins/data3D0000_Sy.npy")
    Sz = np.load("../data/spins/data3D0000_Sz.npy")

    print("Initial Data Check:")
    print(f"  Loaded Sx shape: {Sx.shape}")
    print(f"  Loaded Sx - Min: {Sx.min():.4f}, Max: {Sx.max():.4f}, Mean: {Sx.mean():.4f}")
    print(f"  Loaded Sx has NaN: {np.any(np.isnan(Sx))}, Inf: {np.any(np.isinf(Sx))}")
    print(f"  Loaded Sy shape: {Sy.shape}")
    print(f"  Loaded Sy - Min: {Sy.min():.4f}, Max: {Sy.max():.4f}, Mean: {Sy.mean():.4f}")
    print(f"  Loaded Sy has NaN: {np.any(np.isnan(Sy))}, Inf: {np.any(np.isinf(Sy))}")
    print(f"  Loaded Sz shape: {Sz.shape}")
    print(f"  Loaded Sz - Min: {Sz.min():.4f}, Max: {Sz.max():.4f}, Mean: {Sz.mean():.4f}")
    print(f"  Loaded Sz has NaN: {np.any(np.isnan(Sz))}, Inf: {np.any(np.isinf(Sz))}")
    print("\n" + "="*50 + "\n")

    # Load optimized data
    Sx_opt = np.load("figures/K16_optimized/Sx_opt.npy")
    Sy_opt = np.load("figures/K16_optimized/Sy_opt.npy")
    Sz_opt = np.load("figures/K16_optimized/Sz_opt.npy")

    print("Optimized Data Check:")
    print(f"  Loaded Sx_opt shape: {Sx_opt.shape}")
    print(f"  Loaded Sx_opt - Min: {Sx_opt.min():.4f}, Max: {Sx_opt.max():.4f}, Mean: {Sx_opt.mean():.4f}")
    print(f"  Loaded Sx_opt has NaN: {np.any(np.isnan(Sx_opt))}, Inf: {np.any(np.isinf(Sx_opt))}")
    print(f"  Loaded Sy_opt shape: {Sy_opt.shape}")
    print(f"  Loaded Sy_opt - Min: {Sy_opt.min():.4f}, Max: {Sy_opt.max():.4f}, Mean: {Sy_opt.mean():.4f}")
    print(f"  Loaded Sy_opt has NaN: {np.any(np.isnan(Sy_opt))}, Inf: {np.any(np.isinf(Sy_opt))}")
    print(f"  Loaded Sz_opt shape: {Sz_opt.shape}")
    print(f"  Loaded Sz_opt - Min: {Sz_opt.min():.4f}, Max: {Sz_opt.max():.4f}, Mean: {Sz_opt.mean():.4f}")
    print(f"  Loaded Sz_opt has NaN: {np.any(np.isnan(Sz_opt))}, Inf: {np.any(np.isinf(Sz_opt))}")
    print("\n" + "="*50 + "\n")

    # Plot initial field
    print("Plotting Initial Field...")
    plot_field(Sx, Sy, Sz, "Initial Field", "field_before", rotate_gif=False)

    # Plot optimized field
    print("\nPlotting Optimized Field...")
    plot_field(Sx_opt, Sy_opt, Sz_opt, "Optimized Field", "field_after", rotate_gif=False)

    # Create collage
    print("\nCreating Collage...")
    make_collage("field_before", "field_after", "figures/field_comparison.png")

    # Create GIF
    print("\nCreating GIF...")
    make_gif("field_before", "field_after", "figures/field_evolution.gif")

    print("\nScript completed.")