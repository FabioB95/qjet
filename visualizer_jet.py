import numpy as np
from mayavi import mlab
import os

mlab.options.offscreen = False

def plot_blackhole_with_simjet(Sx, Sy, Sz, out_file="figures/bh_sim_jet.png", vertical_boost=18.0):
    nx, ny, nz = 100, 100, 136
    Sx = Sx.reshape((nx, ny, nz))
    Sy = Sy.reshape((nx, ny, nz))
    Sz = Sz.reshape((nx, ny, nz))

    # Griglia per torus
    x = np.linspace(-2, 2, nx)
    y = np.linspace(-2, 2, ny)
    z = np.linspace(-4, 4, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Disco di accrescimento
    r = np.sqrt(X**2 + Y**2)
    torus = np.exp(-((r - 1.0)**2) / 0.05) * np.exp(-(Z**2) / 0.1)

    fig = mlab.figure(bgcolor=(0.02, 0.02, 0.08), size=(1600, 1600))
    
    mlab.contour3d(X, Y, Z, torus,
                   contours=[0.1, 0.25, 0.4, 0.55, 0.7],
                   opacity=0.35,
                   colormap="hot")

    # Buco nero
    mlab.points3d(0, 0, 0, scale_factor=0.55,
                  color=(0.0, 0.0, 0.05), resolution=100)

    # Campo vettoriale potenziato
    Z_sign = np.sign(Z + 0.01)
    distance_from_equator = np.abs(Z) / 4.0
    
    # Componente rotazionale
    theta = np.arctan2(Y, X)
    Sx_twisted = Sx - 0.3 * np.sin(theta) * (1.0 - distance_from_equator)
    Sy_twisted = Sy + 0.3 * np.cos(theta) * (1.0 - distance_from_equator)
    Sz_boosted = Sz + vertical_boost * Z_sign * (1.0 - distance_from_equator)
    
    Bmag = np.sqrt(Sx_twisted**2 + Sy_twisted**2 + Sz_boosted**2) + 1e-12
    Sx_norm = Sx_twisted / Bmag
    Sy_norm = Sy_twisted / Bmag
    Sz_norm = Sz_boosted / Bmag
    
    print(f"Campo originale - Sz range: [{Sz.min():.3f}, {Sz.max():.3f}]")
    print(f"Campo finale - Sz range: [{Sz_norm.min():.3f}, {Sz_norm.max():.3f}]")

    src = mlab.pipeline.vector_field(X, Y, Z, Sx_norm, Sy_norm, Sz_norm)

    # JET CONFIGURATION
    n_seeds = 20
    radius_inner = 0.7
    radius_outer = 1.1
    
    for ring_radius in [radius_inner, radius_outer]:
        angles = np.linspace(0, 2*np.pi, n_seeds, endpoint=False)
        
        # JET SUPERIORE - va verso +Z (SU)
        for i, angle in enumerate(angles):
            x0 = ring_radius * np.cos(angle)
            y0 = ring_radius * np.sin(angle)
            z0 = 0.2 if ring_radius == radius_inner else 0.3
            
            flow_up = mlab.pipeline.streamline(
                src,
                seedtype='point',
                integration_direction='forward'  # GIUSTO: forward da z>0 va SU
            )
            flow_up.seed.widget.position = [x0, y0, z0]
            flow_up.seed.widget.enabled = False
            flow_up.stream_tracer.maximum_propagation = 40
            flow_up.stream_tracer.initial_integration_step = 0.07
            flow_up.stream_tracer.terminal_speed = 1e-7
            
            intensity = 0.8 + 0.2 * (i / n_seeds)
            thickness = 4.0 if ring_radius == radius_inner else 3.0
            flow_up.actor.property.line_width = thickness
            flow_up.actor.property.color = (1.0, 0.4 * intensity, 0.05)
        
        # JET INFERIORE - va verso -Z (GIÙ)
        for i, angle in enumerate(angles):
            x0 = ring_radius * np.cos(angle)
            y0 = ring_radius * np.sin(angle)
            z0 = -0.2 if ring_radius == radius_inner else -0.3
            
            flow_down = mlab.pipeline.streamline(
                src,
                seedtype='point',
                integration_direction='forward'  # CORREZIONE: forward da z<0 va GIÙ!
            )
            flow_down.seed.widget.position = [x0, y0, z0]
            flow_down.seed.widget.enabled = False
            flow_down.stream_tracer.maximum_propagation = 40
            flow_down.stream_tracer.initial_integration_step = 0.07
            flow_down.stream_tracer.terminal_speed = 1e-7
            
            intensity = 0.8 + 0.2 * (i / n_seeds)
            thickness = 4.0 if ring_radius == radius_inner else 3.0
            flow_down.actor.property.line_width = thickness
            flow_down.actor.property.color = (0.05, 0.5 * intensity, 1.0)

    # EVENTO HORIZON GLOW
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:30j]
    for r_glow in [0.58, 0.62, 0.66]:
        x_sphere = r_glow * np.cos(u) * np.sin(v)
        y_sphere = r_glow * np.sin(u) * np.sin(v)
        z_sphere = r_glow * np.cos(v)
        alpha = 0.4 / r_glow
        mlab.mesh(x_sphere, y_sphere, z_sphere, 
                  color=(1.0, 0.9, 0.3), opacity=alpha)

    # PARTICELLE LUMINOSE - SOLO LUNGO I JET (non sparse)
    n_particles = 80
    for i in range(n_particles):
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0.6, 1.2)
        z_part = np.random.uniform(1.5, 3.8)  # Solo nella zona jet
        x_part = r * np.cos(angle)
        y_part = r * np.sin(angle)
        
        # Particelle SOPRA (arancio)
        mlab.points3d(x_part, y_part, z_part,
                     scale_factor=0.04,
                     color=(1.0, 0.6, 0.1),
                     opacity=0.8)
        
        # Particelle SOTTO (blu) - SOLO se nella zona jet
        if i % 2 == 0:  # Meno particelle sotto
            mlab.points3d(x_part, y_part, -z_part,
                         scale_factor=0.04,
                         color=(0.1, 0.6, 1.0),
                         opacity=0.8)

    # Titolo
    mlab.title("Black Hole Quantum Jet Field", 
               size=0.5, color=(0.9, 0.9, 1.0), height=0.95)
    
    # Vista ottimale
    mlab.view(azimuth=40, elevation=68, distance=12, focalpoint=(0, 0, 0.5))
    
    fig.scene.light_manager.light_mode = 'vtk'

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mlab.savefig(out_file, magnification=2)
    print(f"[OK] Saved SPECTACULAR jet visualization to {out_file}")

    mlab.show()


if __name__ == "__main__":
    Sx = np.load("figures/K16_optimized/Sx_opt.npy")
    Sy = np.load("figures/K16_optimized/Sy_opt.npy")
    Sz = np.load("figures/K16_optimized/Sz_opt.npy")

    plot_blackhole_with_simjet(Sx, Sy, Sz, "figures/bh_sim_jet_final.png", vertical_boost=20.0)