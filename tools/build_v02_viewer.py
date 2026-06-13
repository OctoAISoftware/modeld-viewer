#!/usr/bin/env python3
"""
Autenried V12 transient v02 — comprehensive standalone 3D viewer (GitHub Pages).

Read-only against the calibrated v02 transient MF6 outputs. Decimates the
*active* cell point cloud (idomain>0) to ~8k and writes a single self-contained
Plotly HTML with a property dropdown (head, drawdown, initial head, K continuous
[blue=high], K/Sy/Ss/recharge categorical zones), a 48-SP time slider, gray
terrain, and overlays (GHB black diamonds, CHD orange crosses, 7 AUT wells red).
Full matching plotly.js (v3.3.1) inlined for a version-correct, self-contained page.
"""
import os
import re
import time
import numpy as np
import pandas as pd
import flopy
import plotly.graph_objects as go

MODEL = "/home/suporte/modeld-flopy/autenried/Catchment_Autenried_flopy/V12_TRANSIENT_V02"
WS = os.path.join(MODEL, "mf6_workspace")
SIMOBS = os.path.join(MODEL, "sim_vs_obs.csv")
OUT_HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "autenried_v02_viewer.html")

N_SAMPLE = 8000
N_GHB = 2500
N_CHD = 452
SEED = 42
RNG = np.random.default_rng(SEED)
CAL = dict(mae=0.237, rmse=0.299, bias=0.034, r=0.574, npts=3230, nwells=7)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#17becf", "#bcbd22", "#393b79", "#637939", "#843c39"]


def log(m):
    print(f"[build] {m}", flush=True)


def cat_scale(n):
    pal = [PALETTE[i % len(PALETTE)] for i in range(n)]
    if n <= 1:
        return [[0, pal[0]], [1, pal[0]]]
    sc = []
    for i, c in enumerate(pal):
        sc.append([i / n, c])
        sc.append([(i + 1) / n, c])
    return sorted(sc, key=lambda p: p[0])


def parse_wells(path):
    w = {}
    if os.path.exists(path):
        for ln in open(path):
            t = ln.split()
            if len(t) == 5 and t[1].upper() == "HEAD":
                try:
                    w[t[0]] = (int(t[2]), int(t[3]), int(t[4]))
                except ValueError:
                    pass
    return w


def main():
    t0 = time.time()
    log("loading sim")
    sim = flopy.mf6.MFSimulation.load(sim_ws=WS, verbosity_level=0)
    gwf = sim.get_model()
    mg = gwf.modelgrid
    nlay, nrow, ncol = mg.nlay, mg.nrow, mg.ncol
    idom = mg.idomain
    if idom is None:
        idom = np.ones((nlay, nrow, ncol), dtype=int)

    xc, yc, zc = np.asarray(mg.xcellcenters), np.asarray(mg.ycellcenters), np.asarray(mg.zcellcenters)
    X = np.broadcast_to(xc, (nlay, nrow, ncol))
    Y = np.broadcast_to(yc, (nlay, nrow, ncol))
    Z = zc
    top = np.asarray(mg.top)
    k = np.asarray(gwf.npf.k.array)
    strt = np.asarray(gwf.ic.strt.array)
    sy = np.asarray(gwf.sto.sy.array)
    ss = np.asarray(gwf.sto.ss.array)
    rch = np.asarray(gwf.rcha.recharge.array)        # (nsp,1,nrow,ncol)
    rch0 = rch[0, 0] if rch.ndim == 4 else (rch[0] if rch.ndim == 3 else rch)

    hf = flopy.utils.HeadFile(os.path.join(WS, "gwf_tr_v02.hds"))
    kstpkper = hf.get_kstpkper()
    times = hf.get_times()
    nsp = len(kstpkper)
    last = hf.get_data(kstpkper=kstpkper[-1])

    # STRICT active mask: idomain>0 AND finite head
    valid = (idom > 0) & np.isfinite(last) & (np.abs(last) < 1e29)
    flat = np.flatnonzero(valid.ravel())
    sample = (np.sort(RNG.choice(flat, N_SAMPLE, replace=False))
              if flat.size > N_SAMPLE else flat)
    log(f"active(idomain>0 & valid head) {flat.size:,} -> sample {sample.size:,}")
    li, ri, ci = np.unravel_index(sample, (nlay, nrow, ncol))

    x0, y0 = float(np.nanmin(xc)), float(np.nanmin(yc))
    z0 = float(np.nanmin(Z.ravel()[sample]))
    Xl = np.round(X.ravel()[sample] - x0, 1)
    Yl = np.round(Y.ravel()[sample] - y0, 1)
    z_raw = Z.ravel()[sample] - z0
    xs = float(Xl.max() - Xl.min()) or 1.0
    ys = float(Yl.max() - Yl.min()) or 1.0
    zs = float(z_raw.max() - z_raw.min()) or 1.0
    VE = max(1.0, round((0.45 * max(xs, ys)) / zs))
    Zp = np.round(z_raw * VE, 1)
    log(f"VE x{VE:.0f}")

    Ks = k.ravel()[sample]
    Sys = sy.ravel()[sample]
    Sss = ss.ravel()[sample]
    STRTs = np.round(strt.ravel()[sample], 2)
    Rchs = rch0[ri, ci]

    # head + drawdown per SP at sampled cells
    head_cols, dd_cols = [], []
    for kk in kstpkper:
        h = hf.get_data(kstpkper=kk).ravel()[sample]
        h = np.where(np.abs(h) < 1e29, h, np.nan)
        head_cols.append(np.round(h, 2).astype(np.float32))
        dd_cols.append(np.round(STRTs - h, 3).astype(np.float32))
    hf.close()
    ha = np.array(head_cols)
    hmin, hmax = float(np.nanpercentile(ha, 1)), float(np.nanpercentile(ha, 99))
    da = np.array(dd_cols)
    dmin, dmax = float(np.nanpercentile(da, 1)), float(np.nanpercentile(da, 99))
    smin, smax = float(np.nanmin(STRTs)), float(np.nanmax(STRTs))

    # ---- categorical encoders ----
    def encode(vals, fmt):
        u = sorted(set(np.round(vals[np.isfinite(vals)], 8).tolist()))
        m = {v: i for i, v in enumerate(u)}
        idx = np.array([m.get(round(float(v), 8), -1) for v in vals], dtype=float)
        return idx, u, [fmt(v) for v in u]

    kzi, ku, ktt = encode(np.round(Ks, 4), lambda v: f"{v:g}")
    syi, syu, sytt = encode(Sys, lambda v: f"{v:g}")
    ssi, ssu, sstt = encode(Sss, lambda v: f"{v:.1e}")
    rci, rcu, rctt = encode(Rchs, lambda v: f"{v:.2e}")

    # K continuous (log10), diverging RdBu so BLUE=HIGH (reversescale)
    Kpos = np.where(Ks > 0, Ks, np.nan)
    Klog = np.log10(Kpos)
    klo, khi = float(np.nanmin(Klog)), float(np.nanmax(Klog))
    ktickv = np.linspace(klo, khi, 5)
    ktickt = [f"{10**v:.2g}" for v in ktickv]

    # ---- overlays ----
    def cells_xyz(reclist):
        ax, ay, az = [], [], []
        for rec in reclist:
            l, r, c = rec["cellid"]
            ax.append(float(xc[r, c]) - x0)
            ay.append(float(yc[r, c]) - y0)
            az.append((float(zc[l, r, c]) - z0) * VE)
        return np.round(ax, 1), np.round(ay, 1), np.round(az, 1)

    ghb = gwf.ghb.stress_period_data.get_data(0)
    gx, gy, gz = cells_xyz(ghb)
    if gx.size > N_GHB:
        gi = np.sort(RNG.choice(gx.size, N_GHB, replace=False))
        gx, gy, gz = gx[gi], gy[gi], gz[gi]
    chd = gwf.chd.stress_period_data.get_data(0)
    cx, cy, cz = cells_xyz(chd)
    log(f"GHB {gx.size}  CHD {cx.size}")

    wells = parse_wells(os.path.join(WS, "gwf_tr_v02.obs"))
    resid = {}
    if os.path.exists(SIMOBS):
        df = pd.read_csv(SIMOBS)
        for w, sub in df.groupby("well"):
            resid[w] = float(sub["residual"].mean())
    wx, wy, wz, wtxt, whov = [], [], [], [], []
    for name, (l, r, c) in wells.items():
        l0, r0, c0 = l - 1, r - 1, c - 1
        wx.append(float(xc[r0, c0]) - x0)
        wy.append(float(yc[r0, c0]) - y0)
        wz.append((float(zc[l0, r0, c0]) - z0) * VE)
        disp = "AUT-" + re.sub(r"\D", "", name).zfill(2)
        wtxt.append(disp)
        mr = resid.get(disp)
        whov.append(f"{disp}<br>cell L{l} R{r} C{c}"
                    + (f"<br>mean resid {mr:+.3f} m" if mr is not None else ""))

    cb = lambda title: dict(title=title, len=0.62, x=0.0, thickness=14)

    # ===== base property traces (toggled by dropdown) =====
    fig = go.Figure()
    base = dict(x=Xl, y=Yl, z=Zp, mode="markers")
    # 0 head
    fig.add_trace(go.Scatter3d(**base, name="Head", visible=True,
        marker=dict(size=2.6, color=head_cols[-1], colorscale="Viridis",
                    cmin=hmin, cmax=hmax, opacity=0.85,
                    colorbar=cb(f"Head (m)<br>{hmin:.1f}–{hmax:.1f}")),
        hovertemplate="head %{marker.color:.2f} m<extra></extra>"))
    # 1 drawdown
    fig.add_trace(go.Scatter3d(**base, name="Drawdown", visible=False,
        marker=dict(size=2.6, color=dd_cols[-1], colorscale="Cividis",
                    cmin=dmin, cmax=dmax, opacity=0.85,
                    colorbar=cb(f"Drawdown (m)<br>SS−transient<br>{dmin:.2f}–{dmax:.2f}")),
        hovertemplate="drawdown %{marker.color:.2f} m<extra></extra>"))
    # 2 initial head
    fig.add_trace(go.Scatter3d(**base, name="Initial head", visible=False,
        marker=dict(size=2.6, color=STRTs, colorscale="Viridis",
                    cmin=smin, cmax=smax, opacity=0.85,
                    colorbar=cb(f"Initial head (m)<br>{smin:.1f}–{smax:.1f}")),
        hovertemplate="strt %{marker.color:.2f} m<extra></extra>"))
    # 3 K continuous (blue=high)
    fig.add_trace(go.Scatter3d(**base, name="K (continuous)", visible=False,
        marker=dict(size=2.6, color=Klog, colorscale="RdBu", reversescale=False,
                    cmin=klo, cmax=khi, opacity=0.85,
                    colorbar=dict(title="Kx (m/d)<br>blue=high", len=0.62, x=0.0,
                                  thickness=14, tickvals=ktickv, ticktext=ktickt)),
        customdata=np.round(Ks, 4),
        hovertemplate="Kx %{customdata:.4g} m/d<extra></extra>"))
    # 4 K zones
    fig.add_trace(go.Scatter3d(**base, name="K zones", visible=False,
        marker=dict(size=2.6, color=kzi, colorscale=cat_scale(len(ku)), cmin=0, cmax=len(ku),
                    opacity=0.85, colorbar=dict(title="Kx zones (m/d)", len=0.75, x=0.0,
                    thickness=14, tickvals=[i + .5 for i in range(len(ku))], ticktext=ktt)),
        customdata=np.round(Ks, 4), hovertemplate="Kx %{customdata:.4g} m/d<extra></extra>"))
    # 5 Sy zones
    fig.add_trace(go.Scatter3d(**base, name="Sy zones", visible=False,
        marker=dict(size=2.6, color=syi, colorscale=cat_scale(len(syu)), cmin=0, cmax=len(syu),
                    opacity=0.85, colorbar=dict(title="Sy zones (–)", len=0.75, x=0.0,
                    thickness=14, tickvals=[i + .5 for i in range(len(syu))], ticktext=sytt)),
        customdata=Sys, hovertemplate="Sy %{customdata:.3g}<extra></extra>"))
    # 6 Ss zones
    fig.add_trace(go.Scatter3d(**base, name="Ss zones", visible=False,
        marker=dict(size=2.6, color=ssi, colorscale=cat_scale(len(ssu)), cmin=0, cmax=len(ssu),
                    opacity=0.85, colorbar=dict(title="Ss zones (1/m)", len=0.75, x=0.0,
                    thickness=14, tickvals=[i + .5 for i in range(len(ssu))], ticktext=sstt)),
        customdata=Sss, hovertemplate="Ss %{customdata:.2e} 1/m<extra></extra>"))
    # 7 Recharge zones
    fig.add_trace(go.Scatter3d(**base, name="Recharge zones", visible=False,
        marker=dict(size=2.6, color=rci, colorscale=cat_scale(len(rcu)), cmin=0, cmax=len(rcu),
                    opacity=0.85, colorbar=dict(title="Recharge zones (m/d)", len=0.75, x=0.0,
                    thickness=14, tickvals=[i + .5 for i in range(len(rcu))], ticktext=rctt)),
        customdata=Rchs, hovertemplate="recharge %{customdata:.2e} m/d<extra></extra>"))

    # ===== overlays (always on; toggle via legend) =====
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="markers", name="GHB",
        marker=dict(size=3.0, color="black", symbol="diamond", opacity=0.9),
        hovertemplate="GHB cell<extra></extra>", visible=True))
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="markers", name="CHD",
        marker=dict(size=3.4, color="#ff7f0e", symbol="x", opacity=0.95),
        hovertemplate="CHD cell<extra></extra>", visible=True))
    fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="markers+text", name="AUT wells",
        marker=dict(size=6, color="red", line=dict(color="white", width=1)),
        text=wtxt, textposition="top center", textfont=dict(color="red", size=11),
        customdata=whov, hovertemplate="%{customdata}<extra></extra>", visible=True))
    st = max(1, nrow // 90), max(1, ncol // 90)
    fig.add_trace(go.Surface(
        x=xc[::st[0], ::st[1]] - x0, y=yc[::st[0], ::st[1]] - y0,
        z=(top[::st[0], ::st[1]] - z0) * VE, colorscale="Greys", showscale=False,
        opacity=0.28, hoverinfo="skip", name="Terrain", visible=True))

    N_BASE = 8           # traces 0..7 are toggleable bases
    N_TOTAL = len(fig.data)

    def vis(active):
        v = [i == active for i in range(N_BASE)]
        v += [True] * (N_TOTAL - N_BASE)   # overlays always on
        return v

    labels = ["Hydraulic head", "Drawdown", "Initial head", "K (continuous, blue=high)",
              "K zones", "Sy zones", "Ss zones", "Recharge zones"]
    buttons = [dict(label=lab, method="update", args=[{"visible": vis(i)}])
               for i, lab in enumerate(labels)]

    steps = [dict(method="restyle",
                  args=[{"marker.color": [head_cols[i], dd_cols[i]]}, [0, 1]],
                  label=f"{int(times[i])}d") for i in range(nsp)]

    meta = (f"Autenried V12 Transient v02 (calibrated) — MAE {CAL['mae']} m · "
            f"RMSE {CAL['rmse']} m · bias +{CAL['bias']} m · R {CAL['r']} · "
            f"{CAL['npts']} daily pts / {CAL['nwells']} AUT wells · "
            f"grid {nlay}×{nrow}×{ncol}, {flat.size:,} active cells "
            f"(showing {sample.size:,}) · VE ×{VE:.0f}")

    fig.update_layout(
        title=dict(text=meta, font=dict(size=12)),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)",
                   zaxis_title=f"elevation (m, ×{VE:.0f})", aspectmode="data",
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1)), bgcolor="white"),
        updatemenus=[dict(type="dropdown", direction="down", showactive=True,
                          x=0.0, y=1.10, xanchor="left", buttons=buttons,
                          pad={"r": 6, "t": 4})],
        sliders=[dict(active=nsp - 1, currentvalue={"prefix": "Day "},
                      pad={"t": 36}, steps=steps)],
        margin=dict(l=0, r=0, t=58, b=0), paper_bgcolor="white",
        legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.6)"),
        annotations=[dict(text="Property ▾", x=0.0, y=1.16, xref="paper",
                          yref="paper", showarrow=False, font=dict(size=11))])

    log(f"writing {OUT_HTML}  ({N_TOTAL} traces)")
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True,
                   config={"responsive": True})
    log(f"DONE {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e6:.2f} MB) in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
