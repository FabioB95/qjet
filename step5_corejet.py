# step5_corejet.py
import os, argparse, numpy as np
os.environ.setdefault("ETS_TOOLKIT","qt"); os.environ.setdefault("QT_API","pyqt5")
from mayavi import mlab

p=argparse.ArgumentParser()
p.add_argument("--optdir", default="figures/K16_optimized")
p.add_argument("--show", action="store_true")
p.add_argument("--outfile", default="figures/bh_corejet.png")
args=p.parse_args()

OPT=args.optdir
Sx=np.load(os.path.join(OPT,"Sx_opt.npy"))
Sy=np.load(os.path.join(OPT,"Sy_opt.npy"))
Sz=np.load(os.path.join(OPT,"Sz_opt_recon.npy"))
if Sx.ndim==1:
    nx,ny,nz=100,100,136
    Sx=Sx.reshape(nx,ny,nz); Sy=Sy.reshape(nx,ny,nz); Sz=Sz.reshape(nx,ny,nz)
else:
    nx,ny,nz=Sx.shape

x=np.linspace(-2,2,nx); y=np.linspace(-2,2,ny); z=np.linspace(-4,4,nz)
X,Y,Z=np.meshgrid(x,y,z,indexing="ij"); R=np.sqrt(X**2+Y**2)
B=np.sqrt(Sx**2+Sy**2+Sz**2)+1e-30
bx,by,bz=Sx/B,Sy/B,Sz/B
print(f"[COREJET] |bz| median={np.median(np.abs(bz)):.3e}  max={np.max(np.abs(bz)):.3e}")

def interp_trilinear(A,px,py,pz):
    tx=(px-x[0])/(x[-1]-x[0])*(nx-1); ty=(py-y[0])/(y[-1]-y[0])*(ny-1); tz=(pz-z[0])/(z[-1]-z[0])*(nz-1)
    tx=np.clip(tx,0,nx-1-1e-6); ty=np.clip(ty,0,ny-1-1e-6); tz=np.clip(tz,0,nz-1-1e-6)
    i0=np.floor(tx).astype(int); j0=np.floor(ty).astype(int); k0=np.floor(tz).astype(int)
    i1=i0+1; j1=j0+1; k1=k0+1
    fx=tx-i0; fy=ty-j0; fz=tz-k0
    c000=A[i0,j0,k0]; c100=A[i1,j0,k0]; c010=A[i0,j1,k0]; c110=A[i1,j1,k0]
    c001=A[i0,j0,k1]; c101=A[i1,j0,k1]; c011=A[i0,j1,k1]; c111=A[i1,j1,k1]
    c00=c000*(1-fx)+c100*fx; c01=c001*(1-fx)+c101*fx
    c10=c010*(1-fx)+c110*fx; c11=c011*(1-fx)+c111*fx
    c0=c00*(1-fy)+c10*fy; c1=c01*(1-fy)+c11*fy
    return c0*(1-fz)+c1*fz

def B_interp(px,py,pz):
    return np.array([interp_trilinear(bx,px,py,pz),
                     interp_trilinear(by,px,py,pz),
                     interp_trilinear(bz,px,py,pz)],float)

def trace_line(p0,h=0.04,nsteps=1400,direction=+1,r_clip=0.38,vmin=1e-6,h_max=0.06):
    pts=[np.array(p0,float)]; p=np.array(p0,float)
    for _ in range(nsteps):
        if not (x[0]<=p[0]<=x[-1] and y[0]<=p[1]<=y[-1] and z[0]<=p[2]<=z[-1]): break
        if np.hypot(p[0],p[1])>r_clip: break
        v1=B_interp(p[0],p[1],p[2])*direction; n1=np.linalg.norm(v1)
        if n1<vmin: break
        pm=p+0.5*h*(v1/n1)
        v2=B_interp(pm[0],pm[1],pm[2])*direction; n2=np.linalg.norm(v2)
        if n2<vmin: break
        p=p+min(h,h_max)*(v2/n2); pts.append(p.copy())
    return np.array(pts)

rng=np.random.default_rng(7)
N_SEEDS=80; R_MIN,R_MAX=0.08,0.26; Z0=0.22; R_CLIP=0.38
seeds=[]
for _ in range(N_SEEDS):
    r=rng.uniform(R_MIN,R_MAX); a=rng.uniform(0,2*np.pi)
    x0,y0=r*np.cos(a), r*np.sin(a)
    seeds.append((x0,y0, Z0,+1)); seeds.append((x0,y0,-Z0,-1))

mlab.options.offscreen=not args.show
fig=mlab.figure(size=(1200,1200), bgcolor=(0.02,0.02,0.08))
torus=np.exp(-((R-1.0)**2)/0.06)*np.exp(-(Z**2)/0.12)
mlab.contour3d(X,Y,Z,torus,contours=[0.18,0.36,0.54],opacity=0.08,colormap="hot")
mlab.points3d(0,0,0,scale_factor=0.5,color=(0,0,0.05),resolution=64)

for (x0,y0,z0,dirn) in seeds:
    pts=trace_line((x0,y0,z0),direction=dirn,r_clip=R_CLIP)
    if len(pts)<4: continue
    col=(1.0,0.55,0.10) if dirn>0 else (0.15,0.55,1.0)
    mlab.plot3d(pts[:,0],pts[:,1],pts[:,2],tube_radius=0.015,color=col,line_width=1.0,opacity=1.0)

# stima r_core
radii=[]
for (x0,y0,z0,dirn) in seeds[:40]:
    pts=trace_line((x0,y0,z0),nsteps=400,direction=dirn,r_clip=R_CLIP)
    if len(pts): radii.append(np.median(np.hypot(pts[:,0],pts[:,1])))
if radii: print(f"[COREJET] r_core_med ~ {np.median(radii):.3f} (clip={R_CLIP})")

mlab.title("Core Jet", size=0.52, color=(0.92,0.92,1.0))
mlab.view(azimuth=40,elevation=68,distance=12,focalpoint=(0,0,0.5))
os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
mlab.savefig(args.outfile, magnification=2); print(f"[OK] Saved {args.outfile}")
if args.show: mlab.show()
else: mlab.close(fig)
