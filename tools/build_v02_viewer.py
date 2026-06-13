#!/usr/bin/env python3
"""
Autenried V12 transient v02 — comprehensive standalone 3D viewer (GitHub Pages).

v2 (session 1591) — faithful to Vinicius's 2026-06-13 feedback:
  1. Active-cell mask: ONLY idomain>0 cells are rendered (no phantom inactive grid;
     terrain DEM is also masked to active columns).
  2. Full property dropdown (11 entries): hydraulic head, drawdown, K, Sy, Ss,
     recharge (RCH), porosity, GHB cells, CHD cells, topography (DEM), initial
     conditions (IC).
  3. Value legend: every categorical property carries a discrete colorbar whose
     ticks are the REAL calibrated values (Kx m/d, Sy, Ss 1/m, RCH m/d) read
     straight from the MF6 NPF / STO / RCHA arrays — no invented ranges.
  4. Continuous ramps (Viridis) for head / drawdown / IC; distinct categorical
     colors (one per zone) for K / Sy / Ss / recharge — "categorized" properties
     pop, continuous fields stay smooth.

Read-only against the calibrated v02 transient MF6 outputs. Plotly served from CDN
(GitHub Pages has no CSP restriction — see knowledge 283/284). Porosity is NOT a
MODFLOW-6 flow parameter (knowledge 344: flow model stores Sy/Ss only; porosity
lives in the GWT/MST transport model) — its dropdown entry shows the active
geometry in neutral gray with an explicit note instead of fabricated values.
"""
import os
import re
import json
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
LEGEND_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legend_values.json")

N_SAMPLE = 8000
N_GHB = 2500
SP_STRIDE = 3          # subsample the 48 stress periods for the time slider
SEED = 42
RNG = np.random.default_rng(SEED)
CAL = dict(mae=0.237, rmse=0.299, bias=0.034, r=0.574, npts=3230, nwells=7)
# Distinct categorical palette (Plotly Dark24 + extras) — one clear color per zone
PALETTE = ["#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
           "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
           "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
           "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038"]


def log(m):
    print(f"[build] {m}", flush=True)


def cat_scale(n):
    """A stepwise (banded) colorscale with one distinct color per category."""
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
    rch_pkg = gwf.get_package("rcha_0") or getattr(gwf, "rcha", None)
    rch = np.asarray(rch_pkg.recharge.array)             # (nsp,1,nrow,ncol) or similar
    rch0 = rch[0, 0] if rch.ndim == 4 else (rch[0] if rch.ndim == 3 else rch)

    hf = flopy.utils.HeadFile(os.path.join(WS, "gwf_tr_v02.hds"))
    kstpkper = hf.get_kstpkper()
    times = hf.get_times()
    nsp = len(kstpkper)
    last = hf.get_data(kstpkper=kstpkper[-1])

    # STRICT active mask: idomain>0 AND finite head (kills any phantom/inactive cell)
    valid = (idom > 0) & np.isfinite(last) & (np.abs(last) < 1e29)
    nactive = int(valid.sum())
    flat = np.flatnonzero(valid.ravel())
    sample = (np.sort(RNG.choice(flat, N_SAMPLE, replace=False))
              if flat.size > N_SAMPLE else flat)
    # Guarantee every K zone is represented so the value legend is complete
    # (rare zones can have only a handful of cells and miss a uniform sample).
    kflat = np.round(k.ravel(), 8)
    validflat = valid.ravel()
    kx_universe = sorted(set(kflat[validflat].tolist()))
    NEED = 25
    extra = []
    for v in kx_universe:
        inzone = np.flatnonzero((kflat == v) & validflat)
        have = np.intersect1d(sample, inzone, assume_unique=False).size
        if have < NEED and inzone.size:
            extra.append(RNG.choice(inzone, min(NEED - have, inzone.size), replace=False))
    if extra:
        sample = np.unique(np.concatenate([sample] + extra))
    log(f"active(idomain>0 & valid head) {flat.size:,} -> sample {sample.size:,} "
        f"(all {len(kx_universe)} K zones represented)")
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

    # head + drawdown per (subsampled) SP at sampled cells -> time slider
    sp_idx = list(range(0, nsp, SP_STRIDE))
    if sp_idx[-1] != nsp - 1:
        sp_idx.append(nsp - 1)
    head_cols, dd_cols, sp_times = [], [], []
    for j in sp_idx:
        h = hf.get_data(kstpkper=kstpkper[j]).ravel()[sample]
        h = np.where(np.abs(h) < 1e29, h, np.nan)
        head_cols.append(np.round(h, 2).astype(np.float32))
        dd_cols.append(np.round(STRTs - h, 3).astype(np.float32))
        sp_times.append(float(times[j]))
    hf.close()
    ha = np.array(head_cols)
    hmin, hmax = float(np.nanpercentile(ha, 1)), float(np.nanpercentile(ha, 99))
    da = np.array(dd_cols)
    dmin, dmax = float(np.nanpercentile(da, 1)), float(np.nanpercentile(da, 99))
    smin, smax = float(np.nanmin(STRTs)), float(np.nanmax(STRTs))

    # Full-domain distinct value tables — these become BOTH the on-screen value
    # legend (complete real set) and the legend write-back.
    kx_all = sorted(set(np.round(k[idom > 0], 8).tolist()))
    sy_all = sorted(set(np.round(sy[idom > 0], 8).tolist()))
    ss_all = sorted(set(np.round(ss[idom > 0], 10).tolist()))
    rch_all = sorted(set(np.round(rch0[np.isfinite(rch0)], 10).tolist()))

    # ---- categorical encoders: index against the FULL-domain universe so every
    # real zone value shows in the legend even if a zone has very few cells ----
    def encode(vals, universe, fmt, dec=8):
        u = list(universe)
        m = {round(float(v), dec): i for i, v in enumerate(u)}
        idx = np.array([m.get(round(float(v), dec), -1) for v in vals], dtype=float)
        return idx, u, [fmt(v) for v in u]

    kzi, ku, ktt = encode(Ks, kx_all, lambda v: f"{v:.4g}")
    syi, syu, sytt = encode(Sys, sy_all, lambda v: f"{v:g}")
    ssi, ssu, sstt = encode(Sss, ss_all, lambda v: f"{v:.1e}")
    rci, rcu, rctt = encode(Rchs, rch_all, lambda v: f"{v:.3g}", dec=10)
    legend = dict(
        grid=[int(nlay), int(nrow), int(ncol)], nactive=nactive,
        kx_md=kx_all, sy=sy_all, ss_per_m=ss_all,
        rch_sp0_md=sorted(set(np.round(rch0[np.isfinite(rch0)], 10).tolist())),
        ic_min=smin, ic_max=smax, n_sp=nsp, t_last_d=float(times[-1]),
        cal=CAL)
    json.dump(legend, open(LEGEND_JSON, "w"), indent=2)
    log(f"K zones={len(kx_all)} Sy={len(sy_all)} Ss={len(ss_all)}")

    # ---- overlays geometry ----
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
    n_ghb_total = gx.size
    if gx.size > N_GHB:
        gi = np.sort(RNG.choice(gx.size, N_GHB, replace=False))
        gx, gy, gz = gx[gi], gy[gi], gz[gi]
    chd = gwf.chd.stress_period_data.get_data(0)
    cx, cy, cz = cells_xyz(chd)
    log(f"GHB {gx.size}/{n_ghb_total}  CHD {cx.size}")

    wells = parse_wells(os.path.join(WS, "gwf_tr_v02.obs"))
    resid = {}
    if os.path.exists(SIMOBS):
        df = pd.read_csv(SIMOBS)
        wcol = "well" if "well" in df.columns else df.columns[0]
        rcol = "residual" if "residual" in df.columns else None
        if rcol:
            for w, sub in df.groupby(wcol):
                resid[str(w)] = float(sub[rcol].mean())
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
                    + (f"<br>resid médio {mr:+.3f} m" if mr is not None else ""))

    # masked terrain (active columns only) — kills the phantom inactive slab
    actcol = (idom > 0).any(axis=0)
    st = max(1, nrow // 110), max(1, ncol // 110)
    txx = xc[::st[0], ::st[1]] - x0
    tyy = yc[::st[0], ::st[1]] - y0
    ttop = top[::st[0], ::st[1]].astype(float)
    tmask = actcol[::st[0], ::st[1]]
    ttop = np.where(tmask, ttop, np.nan)
    tzz = (ttop - z0) * VE

    # colorbars parked mid-left (área laranja), well below the dropdown block
    cb = lambda title: dict(title=title, len=0.55, x=0.0, y=0.40, yanchor="middle", thickness=14)

    # ===== traces =====
    fig = go.Figure()
    base = dict(x=Xl, y=Yl, z=Zp, mode="markers")
    MS = 2.6

    # 0 head (continuous Viridis)
    fig.add_trace(go.Scatter3d(**base, name="Carga", visible=True,
        marker=dict(size=MS, color=head_cols[-1], colorscale="Viridis",
                    cmin=hmin, cmax=hmax, opacity=0.85,
                    colorbar=cb(f"Carga (m)<br>{hmin:.1f}–{hmax:.1f}")),
        hovertemplate="carga %{marker.color:.2f} m<extra></extra>"))
    # 1 drawdown (continuous Viridis)
    fig.add_trace(go.Scatter3d(**base, name="Rebaixamento", visible=False,
        marker=dict(size=MS, color=dd_cols[-1], colorscale="Viridis",
                    cmin=dmin, cmax=dmax, opacity=0.85,
                    colorbar=cb(f"Rebaix. (m)<br>SS−transiente<br>{dmin:.2f}–{dmax:.2f}")),
        hovertemplate="rebaixamento %{marker.color:.3f} m<extra></extra>"))
    # 2 IC / initial head (continuous Viridis)
    fig.add_trace(go.Scatter3d(**base, name="IC (carga inicial)", visible=False,
        marker=dict(size=MS, color=STRTs, colorscale="Viridis",
                    cmin=smin, cmax=smax, opacity=0.85,
                    colorbar=cb(f"IC carga (m)<br>{smin:.1f}–{smax:.1f}")),
        hovertemplate="IC %{marker.color:.2f} m<extra></extra>"))
    # Vini 2026-06-13 (session 1594): K, Sy, Ss on a RED->BLUE ramp where BLUE=HIGH
    # ("valores maiores = azul = mais água andando"). Discrete zone values still read
    # as distinct bands on the ramp, so the zones stay evident. Plotly "RdBu" maps
    # cmin->red, cmax->blue (reversescale=False), i.e. high=blue exactly as asked.
    # 3 K (continuous red->blue, blue=high, log10 — spans ~4 orders 0.03..191 m/d)
    Klog = np.log10(np.where(Ks > 0, Ks, np.nan))
    klo, khi = float(np.nanmin(Klog)), float(np.nanmax(Klog))
    ktv = np.linspace(klo, khi, 5)
    fig.add_trace(go.Scatter3d(**base, name="K", visible=False,
        marker=dict(size=MS, color=Klog, colorscale="RdBu", reversescale=False,
                    cmin=klo, cmax=khi, opacity=0.85,
                    colorbar=dict(title="Kx (m/d)<br>azul=alto", len=0.55, x=0.0, y=0.40,
                                  yanchor="middle", thickness=15,
                                  tickvals=ktv, ticktext=[f"{10**v:.2g}" for v in ktv])),
        customdata=np.round(Ks, 4), hovertemplate="Kx %{customdata:.4g} m/d<extra></extra>"))
    # 4 Sy (continuous red->blue, blue=high)
    symin, symax = float(np.nanmin(Sys)), float(np.nanmax(Sys))
    fig.add_trace(go.Scatter3d(**base, name="Sy", visible=False,
        marker=dict(size=MS, color=Sys, colorscale="RdBu", reversescale=False,
                    cmin=symin, cmax=symax, opacity=0.85,
                    colorbar=cb(f"Sy (–)<br>azul=alto<br>{symin:g}–{symax:g}")),
        hovertemplate="Sy %{marker.color:.3g}<extra></extra>"))
    # 5 Ss (continuous red->blue, blue=high, log10)
    Sslog = np.log10(np.where(Sss > 0, Sss, np.nan))
    sslo, sshi = float(np.nanmin(Sslog)), float(np.nanmax(Sslog))
    stv = np.linspace(sslo, sshi, 5)
    fig.add_trace(go.Scatter3d(**base, name="Ss", visible=False,
        marker=dict(size=MS, color=Sslog, colorscale="RdBu", reversescale=False,
                    cmin=sslo, cmax=sshi, opacity=0.85,
                    colorbar=dict(title="Ss (1/m)<br>azul=alto", len=0.55, x=0.0, y=0.40,
                                  yanchor="middle", thickness=15,
                                  tickvals=stv, ticktext=[f"{10**v:.1e}" for v in stv])),
        customdata=Sss, hovertemplate="Ss %{customdata:.2e} 1/m<extra></extra>"))
    # 6 Recarga (continuous Viridis — Vini: variáveis contínuas ficam em rampa contínua)
    rmin, rmax = float(np.nanmin(Rchs)), float(np.nanmax(Rchs))
    fig.add_trace(go.Scatter3d(**base, name="Recarga", visible=False,
        marker=dict(size=MS, color=Rchs, colorscale="Viridis",
                    cmin=rmin, cmax=rmax, opacity=0.85,
                    colorbar=cb(f"Recarga (m/d)<br>{rmin:.1e}–{rmax:.1e}")),
        hovertemplate="recarga %{marker.color:.2e} m/d (SP0)<extra></extra>"))
    # 7 Porosity (NOT a flow parameter — neutral gray geometry + note in title)
    fig.add_trace(go.Scatter3d(**base, name="Porosidade", visible=False,
        marker=dict(size=MS, color="#7f8c8d", opacity=0.5),
        hoverinfo="skip", showlegend=False))
    IDX = dict(head=0, draw=1, ic=2, k=3, sy=4, ss=5, rch=6, poro=7)

    # 8 GHB diamonds
    fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="markers", name="Células GHB",
        marker=dict(size=3.0, color="#111111", symbol="diamond", opacity=0.9),
        hovertemplate="GHB<extra></extra>", visible=True))
    # 9 CHD crosses
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="markers", name="Células CHD",
        marker=dict(size=3.6, color="#ff7f0e", symbol="x", opacity=0.95),
        hovertemplate="CHD<extra></extra>", visible=True))
    # 10 AUT wells
    fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode="markers+text", name="Poços AUT",
        marker=dict(size=6, color="red", line=dict(color="white", width=1)),
        text=wtxt, textposition="top center", textfont=dict(color="#c0392b", size=11),
        customdata=whov, hovertemplate="%{customdata}<extra></extra>", visible=True))
    # 11 terrain (gray backdrop)
    fig.add_trace(go.Surface(x=txx, y=tyy, z=tzz, colorscale="Greys", showscale=False,
        opacity=0.22, hoverinfo="skip", name="Terreno", visible=True))
    # 12 terrain DEM (colored, for the Topografia view)
    fig.add_trace(go.Surface(x=txx, y=tyy, z=tzz, surfacecolor=ttop, colorscale="Earth",
        reversescale=True, opacity=0.95, name="Topografia (DEM)",
        colorbar=cb("Cota DEM (m)"), hovertemplate="DEM %{surfacecolor:.1f} m<extra></extra>",
        visible=False))
    G, C, W, TG, TDEM = 8, 9, 10, 11, 12
    N_TOTAL = len(fig.data)

    POR_NOTE = ("Porosidade NÃO é parâmetro do modelo de fluxo MODFLOW-6 "
                "(armazena apenas Sy/Ss). Vive no modelo de transporte (GWT/MST). "
                "Geometria ativa mostrada em cinza — sem valores fabricados.")

    def vis(scalar=None, ghb=True, chd=True, wells=True, terr=True, dem=False):
        v = [False] * N_TOTAL
        if scalar is not None:
            v[scalar] = True
        v[G] = ghb; v[C] = chd; v[W] = wells; v[TG] = terr; v[TDEM] = dem
        return v

    head_title = f"Autenried v02 — Carga hidráulica (transiente, rampa contínua Viridis)"
    options = [
        ("Carga hidráulica (HDS)", vis(IDX["head"]), head_title),
        ("Rebaixamento (SS−transiente)", vis(IDX["draw"]),
         "Autenried v02 — Rebaixamento (rampa contínua Viridis)"),
        ("K — condutividade (vermelho→azul, azul=alto)", vis(IDX["k"]),
         f"Autenried v02 — Kx calibrado (SS PEST++), rampa vermelho→azul · azul=alto · {len(ku)} valores, m/d"),
        ("Sy — rendimento específico (azul=alto)", vis(IDX["sy"], ghb=False, chd=False),
         f"Autenried v02 — Sy, rampa vermelho→azul · azul=alto · {len(syu)} valores"),
        ("Ss — armazenamento específico (azul=alto)", vis(IDX["ss"], ghb=False, chd=False),
         f"Autenried v02 — Ss, rampa vermelho→azul · azul=alto · {len(ssu)} valores, 1/m"),
        ("Recarga (RCH, contínua)", vis(IDX["rch"], ghb=False, chd=False),
         f"Autenried v02 — Recarga RCH SP0 (contínua, transiente mensal Thornthwaite)"),
        ("Porosidade (ver nota)", vis(IDX["poro"], ghb=False, chd=False),
         "Autenried v02 — " + POR_NOTE),
        ("Células GHB", vis(None, ghb=True, chd=False),
         f"Autenried v02 — Condição de contorno GHB ({n_ghb_total:,} células)"),
        ("Células CHD", vis(None, ghb=False, chd=True),
         f"Autenried v02 — Condição de contorno CHD ({cx.size} células)"),
        ("Topografia (DEM)", vis(None, ghb=False, chd=False, terr=False, dem=True),
         "Autenried v02 — Topografia (DEM, topo do modelo, células ativas)"),
        ("IC — condições iniciais (carga)", vis(IDX["ic"]),
         f"Autenried v02 — Carga inicial (IC = heads SS otimizados): {smin:.1f}–{smax:.1f} m"),
    ]
    buttons = [dict(label=lab, method="update",
                    args=[{"visible": v}, {"title.text": ttl}])
               for lab, v, ttl in options]

    # time slider restyles head(0) + drawdown(1) colors
    steps = [dict(method="restyle",
                  args=[{"marker.color": [head_cols[i], dd_cols[i]]}, [IDX["head"], IDX["draw"]]],
                  label=f"{int(sp_times[i])}d") for i in range(len(sp_idx))]

    meta = (f"Autenried V12 Transiente v02 (calibrado) — MAE {CAL['mae']} m · "
            f"RMSE {CAL['rmse']} m · viés +{CAL['bias']} m · R {CAL['r']} · "
            f"{CAL['npts']} pts diários / {CAL['nwells']} poços AUT · "
            f"malha {nlay}×{nrow}×{ncol}, {flat.size:,} células ativas "
            f"(mostrando {sample.size:,}) · VE ×{VE:.0f} · {nsp} períodos (t→{int(times[-1])}d)")

    fig.update_layout(
        # área azul: plot expandido — margens mínimas, scene ocupa todo o container
        title=dict(text=head_title, font=dict(size=13, color="#33445a"),
                   x=0.01, xanchor="left", y=0.985, yanchor="top"),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)",
                   zaxis_title=f"cota (m, ×{VE:.0f})", aspectmode="data",
                   domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1)), bgcolor="white"),
        # área laranja: título → label → dropdown espaçados; colorbar fica em y=0.40
        updatemenus=[dict(type="dropdown", direction="down", showactive=True,
                          x=0.0, y=0.90, xanchor="left", buttons=buttons,
                          pad={"r": 8, "t": 6},
                          bgcolor="rgba(30,30,40,0.97)", bordercolor="rgba(100,160,255,0.85)",
                          borderwidth=2, font=dict(size=13, color="#e0e8ff", family="Arial"))],
        sliders=[dict(active=len(sp_idx) - 1, currentvalue={"prefix": "Dia "},
                      pad={"t": 18}, steps=steps, x=0.18, len=0.66)],
        margin=dict(l=0, r=0, t=28, b=0), paper_bgcolor="white",
        legend=dict(x=0.0, y=0.0, bgcolor="rgba(255,255,255,0.7)"),
        annotations=[
            dict(text="Propriedade ▾", x=0.0, y=0.95, xref="paper", yref="paper",
                 showarrow=False, font=dict(size=12, color="#33445a"))])
    _META = meta  # carried into the calibration panel (área roxa) instead of a footer

    log(f"writing {OUT_HTML}  ({N_TOTAL} traces, {len(sp_idx)} slider frames)")
    fig.write_html(OUT_HTML, include_plotlyjs="cdn", full_html=True,
                   config={"responsive": True, "displaylogo": False})

    # ---- layout post-process (session 1598, Vini feedback — CSS/HTML only) ----
    active_br = f"{nactive:,}".replace(",", ".")     # pt-BR thousands
    shown_br = f"{int(sample.size):,}".replace(",", ".")
    css = """
<style>
:root{--rox:#7b4fb5;}
html,body{height:100%;margin:0;padding:0;background:#fff;}
/* área azul: plot ocupa a viewport inteira */
.plotly-graph-div{width:100vw !important;height:100vh !important;}
/* área roxa: caixinha de parâmetros de calibração, canto superior direito */
#calib-panel{position:fixed;top:14px;right:14px;width:290px;z-index:1000;
  background:rgba(247,243,252,0.96);border:2px solid var(--rox);border-radius:10px;
  padding:10px 13px 11px;font:12px/1.45 Arial,Helvetica,sans-serif;color:#2c2540;
  box-shadow:0 4px 14px rgba(80,40,130,0.22);}
#calib-panel h4{margin:0 0 7px;font-size:13px;color:#5b2d91;
  border-bottom:1px solid #d8c7ee;padding-bottom:4px;}
#calib-panel .grp{margin:0 0 9px;} #calib-panel .grp:last-child{margin-bottom:0;}
#calib-panel b{color:#5b2d91;display:block;margin-bottom:2px;font-size:11.5px;}
#calib-panel .dim{color:#7a7290;font-size:11px;}
#calib-panel .mono{font-variant-numeric:tabular-nums;}
@media(max-width:760px){#calib-panel{position:static;width:auto;margin:8px;}}
</style>"""
    panel = f"""
<div id="calib-panel">
  <h4>Parâmetros de calibração — v02</h4>
  <div class="grp"><b>Métricas globais</b>
    <span class="mono">MAE 0,237 m · RMSE 0,299 m · R 0,5734</span><br>
    malha 19×403×577 · VE ×12<br>
    {active_br} células ativas ({shown_br} mostradas)<br>
    48 períodos (t = 1096 d)</div>
  <div class="grp"><b>Sy livres — 5 zonas (PEST++ v02B)</b>
    <span class="mono">Z3 0,18 · Z4 0,22 · Z5 0,15</span><br>
    <span class="mono">Z6 0,20 · Z7 0,12</span><br>
    <span class="dim">bounds 0,05–0,35 · Z1/Z2/Z8 fixos</span></div>
  <div class="grp"><b>Base v02 congelada</b>
    K SS — 9 Kxy + 9 Kz (phi 35,95)<br>
    Ss por zona · RCH Thornthwaite (6 zonas)<br>
    Pump schedule fixo · CHD / GHB / OC</div>
</div>"""
    html = open(OUT_HTML, encoding="utf-8").read()
    if not html.lstrip().lower().startswith("<!doctype"):
        html = "<!DOCTYPE html>\n" + html.replace("<html>", '<html lang="pt-BR">', 1)
    html = html.replace("</head>", css + "\n</head>", 1)
    html = html.replace("</body>", panel + "\n</body>", 1)
    open(OUT_HTML, "w", encoding="utf-8").write(html)
    log(f"DONE {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e6:.2f} MB) in {time.time()-t0:.1f}s")
    print("LEGEND:", json.dumps(legend, default=float)[:1200])


if __name__ == "__main__":
    main()
