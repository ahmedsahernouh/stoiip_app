import os
import math
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Helper functions
# ============================================================
def make_gaussian_field(mean, unc_percent, shape, min_val=None, max_val=None, seed=1234):
    """
    Build a 2D Gaussian random field around 'mean' with ±unc_percent.
    If unc_percent == 0, returns a constant field.
    """
    Ny, Nx = shape
    if unc_percent is None or unc_percent <= 0.0:
        return np.full((Ny, Nx), mean, dtype=float)

    rng = np.random.default_rng(seed)
    std = mean * unc_percent / 100.0
    field = rng.normal(mean, std, size=(Ny, Nx))

    if min_val is not None or max_val is not None:
        field = np.clip(
            field,
            min_val if min_val is not None else -np.inf,
            max_val if max_val is not None else np.inf,
        )
    return field


def stoiip_stb(area_acres, thickness_ft, phi, so, ntg, fvf):
    """STOIIP formula in STB."""
    return 7758.0 * area_acres * thickness_ft * phi * so * ntg / fvf


def read_zmap_grid_full(filename: str):
    """
    ZMAP .dat reader following your working logic, but robust to
    stray non-numeric tokens in the data section.

    Returns:
      null, xmin, xmax, ymin, ymax, nrows, ncols, dx, dy, grid
    """
    count = 0
    hdr = []

    # Header parsing
    with open(filename) as h:
        for line in h:
            if line.startswith("!"):
                count += 1
            if line.startswith("@") and not line.startswith("@\n"):
                count += 1
                hdr.append(next(h)); count += 1
                hdr.append(next(h)); count += 1
                count += 1  # blank line

    hdr = [x.strip("\n") for x in ",".join(hdr).split(",")]
    null = np.nan if hdr[1] == " " else float(hdr[1])

    xmin, xmax, ymin, ymax = [float(x) for x in hdr[7:11]]
    nrows, ncols = [int(y) for y in hdr[5:7]]
    dx = (xmax - xmin) / (ncols - 1)
    dy = (ymax - ymin) / (nrows - 1)

    # Data section – skip non-numeric tokens
    values = []
    with open(filename) as f:
        for _ in range(count):
            next(f, None)
        for line in f:
            for word in line.split():
                try:
                    values.append(float(word))
                except ValueError:
                    continue

    values = np.asarray(values, dtype=float)
    if values.size != nrows * ncols:
        raise ValueError(
            f"ZMAP size mismatch: got {values.size}, expected {nrows * ncols}"
        )

    # Same reshape/orientation as your working function
    grid = values.reshape(ncols, nrows).T  # (rows, cols)
    grid[grid == null] = np.nan

    return null, xmin, xmax, ymin, ymax, nrows, ncols, dx, dy, grid


def upscale_grid_mean(grid, dx, dy, target_dx):
    """
    Pure upscaling by block averaging.
    If target_dx <= native dx → keep original resolution.
    Extents (xmin/xmax/ymin/ymax) are handled outside this function.
    """
    if target_dx <= dx:
        return grid.copy(), dx, dy

    factor_x = int(round(target_dx / dx))
    factor_y = int(round(target_dx / dy))

    Ny, Nx = grid.shape
    new_Nx = max(1, Nx // factor_x)
    new_Ny = max(1, Ny // factor_y)

    coarse = np.empty((new_Ny, new_Nx), dtype=float)

    for j in range(new_Ny):
        for i in range(new_Nx):
            block = grid[
                j * factor_y:(j + 1) * factor_y,
                i * factor_x:(i + 1) * factor_x
            ]
            if np.all(np.isnan(block)):
                coarse[j, i] = np.nan
            else:
                coarse[j, i] = np.nanmean(block)

    return coarse, target_dx, target_dx


def load_property_zmap(uploaded_file, target_dx):
    """
    Read and upscale a property ZMAP grid if file is uploaded.
    Returns (grid, dx, dy) or (None, None, None) if not available.
    """
    if uploaded_file is None:
        return None, None, None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    _, xmin, xmax, ymin, ymax, nrows, ncols, dx0, dy0, grid_raw = read_zmap_grid_full(path)
    grid_coarse, _, _ = upscale_grid_mean(grid_raw, dx0, dy0, target_dx)

    Ny, Nx = grid_coarse.shape
    dx_new = (xmax - xmin) / (Nx - 1) if Nx > 1 else dx0
    dy_new = (ymax - ymin) / (Ny - 1) if Ny > 1 else dy0

    return grid_coarse, dx_new, dy_new


def sample_param(base, unc_percent, size, min_val=None, max_val=None, rng=None):
    """Sample parameter around base ±% and clip."""
    if rng is None:
        rng = np.random.default_rng()
    if unc_percent <= 0.0:
        return np.full(size, base, dtype=float)
    std = base * unc_percent / 100.0
    vals = rng.normal(base, std, size)
    if min_val is not None or max_val is not None:
        vals = np.clip(
            vals,
            min_val if min_val is not None else -np.inf,
            max_val if max_val is not None else np.inf
        )
    return vals


# ============================================================
# Streamlit UI
# ============================================================

if st.sidebar.button("Exit App"):
    os._exit(0)

st.title("STOIIP Calculator With Uncertainties and ZMAP Integration")

st.latex(
    r"""
\text{STOIIP (STB)} = 7758 \times A(\text{acres}) \times h(\text{ft}) \times \phi \times S_o \times \frac{\text{NTG}}{B_o}
"""
)

st.sidebar.header("Inputs")

# Area source mode
area_source = st.sidebar.selectbox(
    "Area / Structure Source",
    ["Manual area (no ZMAP)", "From ZMAP structure grid"]
)

# App cell size for visualization and ZMAP upscaling
app_cell_size_m = st.sidebar.number_input(
    "App Cell Size (m) for grid / maps",
    min_value=25.0,
    max_value=2000.0,
    value=200.0,
    step=25.0
)

# Manual area and uncertainty (used only in manual mode)
area_acres_manual = st.sidebar.slider(
    "Area (acres) (manual mode)",
    10.0,
    2_000_000.0,
    10000.0,
    step=500.0
)
area_unc = st.sidebar.slider(
    "Area Uncertainty ±% (manual mode)",
    0.0,
    50.0,
    0.0,
    step=1.0
)

# Reservoir parameters (means and uncertainties)
thickness_manual = st.sidebar.slider("Thickness (ft)", 1.0, 500.0, 100.0, step=1.0)
thick_unc = st.sidebar.slider("Thickness Uncertainty ±%", 0.0, 50.0, 0.0, step=1.0)

porosity_manual = st.sidebar.slider("Porosity Φ", 0.05, 0.40, 0.20, step=0.01)
por_unc = st.sidebar.slider("Porosity Uncertainty ±%", 0.0, 50.0, 0.0, step=1.0)

oil_saturation_manual = st.sidebar.slider("Oil Saturation So", 0.20, 0.95, 0.70, step=0.01)
sat_unc = st.sidebar.slider("Oil Saturation Uncertainty ±%", 0.0, 50.0, 0.0, step=1.0)

fvf = st.sidebar.slider("FVF (Bo)", 1.0, 2.0, 1.20, step=0.05)
fvf_unc = st.sidebar.slider("FVF Uncertainty ±%", 0.0, 50.0, 0.0, step=1.0)

ntg_manual = st.sidebar.slider("NTG", 0.10, 1.0, 0.80, step=0.01)
ntg_unc = st.sidebar.slider("NTG Uncertainty ±%", 0.0, 50.0, 0.0, step=1.0)

iterations = st.sidebar.number_input(
    "Monte Carlo Iterations",
    min_value=100,
    max_value=20000,
    value=1000,
    step=100
)

# OWC / ZMAP-related inputs
owc_value = 0.0
owc_unc = 0.0
invert_structure_z = False

uploaded_structure = None
uploaded_thickness_zmap = None
uploaded_phi_zmap = None
uploaded_so_zmap = None
uploaded_ntg_zmap = None

thickness_source = "Manual"
por_source = "Manual"
so_source = "Manual"
ntg_source = "Manual"

if area_source == "From ZMAP structure grid":
    st.sidebar.subheader("ZMAP Structure & OWC")

    uploaded_structure = st.sidebar.file_uploader("Structure ZMAP (.dat)", type=["dat"])
    owc_value = st.sidebar.number_input(
        "OWC value (same units as structure grid)",
        value=0.0,
        format="%.2f"
    )
    owc_unc = st.sidebar.number_input(
        "OWC Uncertainty ± (same units as grid)",
        value=0.0,
        min_value=0.0,
        step=1.0,
        format="%.2f"
    )
    invert_structure_z = st.sidebar.checkbox(
        "Invert structure Z values (× -1)?",
        value=False,
        help=(
            "If unchecked (normal): active cells have Z ≥ OWC.\n"
            "If checked (inverted): active cells have Z ≤ OWC."
        )
    )

    st.sidebar.subheader("Property Sources (Manual vs ZMAP)")

    thickness_source = st.sidebar.selectbox(
        "Thickness source", ["Manual", "ZMAP grid"], index=0
    )
    if thickness_source == "ZMAP grid":
        uploaded_thickness_zmap = st.sidebar.file_uploader(
            "Thickness ZMAP (ft)", type=["dat"], key="thick_zmap"
        )

    por_source = st.sidebar.selectbox(
        "Porosity source", ["Manual", "ZMAP grid"], index=0
    )
    if por_source == "ZMAP grid":
        uploaded_phi_zmap = st.sidebar.file_uploader(
            "Porosity ZMAP (fraction)", type=["dat"], key="phi_zmap"
        )

    so_source = st.sidebar.selectbox(
        "Oil Saturation source", ["Manual", "ZMAP grid"], index=0
    )
    if so_source == "ZMAP grid":
        uploaded_so_zmap = st.sidebar.file_uploader(
            "Oil Saturation ZMAP (fraction)", type=["dat"], key="so_zmap"
        )

    ntg_source = st.sidebar.selectbox(
        "NTG source", ["Manual", "ZMAP grid"], index=0
    )
    if ntg_source == "ZMAP grid":
        uploaded_ntg_zmap = st.sidebar.file_uploader(
            "NTG ZMAP (fraction)", type=["dat"], key="ntg_zmap"
        )

# ============================================================
# Build grids and deterministic per-cell STOIIP_0
# ============================================================

rng = np.random.default_rng(12345)

structure_grid = None
dx = dy = None
zmap_xmin = zmap_xmax = None
zmap_ymin = zmap_ymax = None
active_mask = None

thickness_grid = None
phi_grid = None
so_grid = None
ntg_grid = None

cell_area_ac = None
cell_stoiip_grid_stb = None
total_stoiip0_stb = None
area_total_acres_0 = None

# ---- ZMAP MODE ----
if area_source == "From ZMAP structure grid" and uploaded_structure is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp:
        tmp.write(uploaded_structure.read())
        path_struct = tmp.name

    (
        _,
        xmin,
        xmax,
        ymin,
        ymax,
        nrows,
        ncols,
        dx_native,
        dy_native,
        grid_raw,
    ) = read_zmap_grid_full(path_struct)

    # Optional inversion (e.g. +depth -> negative TVDSS)
    if invert_structure_z:
        grid_raw = -grid_raw

    # Upscale to app cell size (coarse grid) – structure defines reference grid
    structure_grid, _, _ = upscale_grid_mean(grid_raw, dx_native, dy_native, app_cell_size_m)
    Ny_s, Nx_s = structure_grid.shape

    # Recompute dx, dy from global extents and new size
    dx = (xmax - xmin) / (Nx_s - 1) if Nx_s > 1 else dx_native
    dy = (ymax - ymin) / (Ny_s - 1) if Ny_s > 1 else dy_native

    zmap_xmin, zmap_xmax = xmin, xmax
    zmap_ymin, zmap_ymax = ymin, ymax

    # OWC-based active mask
    if invert_structure_z:
        active_mask = (structure_grid <= owc_value) & ~np.isnan(structure_grid)
    else:
        active_mask = (structure_grid >= owc_value) & ~np.isnan(structure_grid)

    # Property grids – all upscaled using the SAME app_cell_size_m
    if thickness_source == "ZMAP grid" and uploaded_thickness_zmap is not None:
        thickness_grid, _, _ = load_property_zmap(uploaded_thickness_zmap, app_cell_size_m)
    else:
        thickness_grid = np.full_like(structure_grid, thickness_manual, dtype=float)

    if por_source == "ZMAP grid" and uploaded_phi_zmap is not None:
        phi_grid, _, _ = load_property_zmap(uploaded_phi_zmap, app_cell_size_m)
    else:
        phi_grid = np.full_like(structure_grid, porosity_manual, dtype=float)

    if so_source == "ZMAP grid" and uploaded_so_zmap is not None:
        so_grid, _, _ = load_property_zmap(uploaded_so_zmap, app_cell_size_m)
    else:
        so_grid = np.full_like(structure_grid, oil_saturation_manual, dtype=float)

    if ntg_source == "ZMAP grid" and uploaded_ntg_zmap is not None:
        ntg_grid, _, _ = load_property_zmap(uploaded_ntg_zmap, app_cell_size_m)
    else:
        ntg_grid = np.full_like(structure_grid, ntg_manual, dtype=float)

    # --------------------------------------------------------
    # HARD ALIGNMENT STEP: make all grids same shape
    # --------------------------------------------------------
    shapes = [
        structure_grid.shape,
        thickness_grid.shape,
        phi_grid.shape,
        so_grid.shape,
        ntg_grid.shape,
    ]
    Ny_common = min(s[0] for s in shapes)
    Nx_common = min(s[1] for s in shapes)

    structure_grid = structure_grid[:Ny_common, :Nx_common]
    active_mask    = active_mask[:Ny_common, :Nx_common]
    thickness_grid = thickness_grid[:Ny_common, :Nx_common]
    phi_grid       = phi_grid[:Ny_common, :Nx_common]
    so_grid        = so_grid[:Ny_common, :Nx_common]
    ntg_grid       = ntg_grid[:Ny_common, :Nx_common]

    Ny_s, Nx_s = Ny_common, Nx_common  # update size

    # Cell area in acres
    cell_area_m2 = dx * dy
    cell_area_ac = cell_area_m2 / 4046.8564224

    # Deterministic per-cell STOIIP (STB)
    cell_stoiip_grid_stb = (
        7758.0
        * cell_area_ac
        * thickness_grid
        * phi_grid
        * so_grid
        * ntg_grid
        / fvf
    )
    cell_stoiip_grid_stb = np.where(active_mask, cell_stoiip_grid_stb, np.nan)

    total_stoiip0_stb = np.nansum(cell_stoiip_grid_stb)
    area_total_acres_0 = np.sum(active_mask) * cell_area_ac

# ---- MANUAL MODE ----
if area_source == "Manual area (no ZMAP)" or (structure_grid is None):
    area_m2 = area_acres_manual * 4046.8564224
    dx = dy = app_cell_size_m
    cell_area_m2 = dx * dy
    cell_area_ac = cell_area_m2 / 4046.8564224

    Nx_s = max(1, int(math.sqrt(area_m2 / cell_area_m2)))
    Ny_s = Nx_s

    # Synthetic extents
    zmap_xmin, zmap_xmax = 0.0, Nx_s * dx
    zmap_ymin, zmap_ymax = 0.0, Ny_s * dy

    structure_grid = np.full((Ny_s, Nx_s), np.nan)
    active_mask = np.ones((Ny_s, Nx_s), dtype=bool)

    # Synthetic property fields:
    # - Constant if uncertainty = 0
    # - Gaussian random field if uncertainty > 0
    shape = (Ny_s, Nx_s)

    thickness_grid = make_gaussian_field(
        thickness_manual, thick_unc, shape,
        min_val=1.0
    )

    phi_grid = make_gaussian_field(
        porosity_manual, por_unc, shape,
        min_val=0.05, max_val=0.40
    )

    so_grid = make_gaussian_field(
        oil_saturation_manual, sat_unc, shape,
        min_val=0.20, max_val=0.95
    )

    ntg_grid = make_gaussian_field(
        ntg_manual, ntg_unc, shape,
        min_val=0.10, max_val=1.0
    )


    cell_stoiip_grid_stb = (
        7758.0
        * cell_area_ac
        * thickness_grid
        * phi_grid
        * so_grid
        * ntg_grid
        / fvf
    )
    cell_stoiip_grid_stb = np.where(active_mask, cell_stoiip_grid_stb, np.nan)

    total_stoiip0_stb = np.nansum(cell_stoiip_grid_stb)
    area_total_acres_0 = np.sum(active_mask) * cell_area_ac


# ============================================================
# Monte Carlo (global multipliers around deterministic STOIIP_0)
# ============================================================

if total_stoiip0_stb is None or total_stoiip0_stb <= 0.0:
    st.error("Deterministic STOIIP is not positive. Please check inputs / ZMAP / OWC.")
    st.stop()

valid_cells = (
    active_mask
    & ~np.isnan(thickness_grid)
    & ~np.isnan(phi_grid)
    & ~np.isnan(so_grid)
    & ~np.isnan(ntg_grid)
)

if not np.any(valid_cells):
    st.error("No valid active cells after masking. Check OWC, structure, and property grids.")
    st.stop()

thickness_base = np.nanmean(thickness_grid[valid_cells])
phi_base = np.nanmean(phi_grid[valid_cells])
so_base = np.nanmean(so_grid[valid_cells])
ntg_base = np.nanmean(ntg_grid[valid_cells])
fvf_base = fvf
area_base = area_total_acres_0

n_iter = iterations
rng = np.random.default_rng(12345)

# Area factor
if area_source == "Manual area (no ZMAP)":
    area_samples = sample_param(area_base, area_unc, n_iter, min_val=1.0, rng=rng)
    k_area = area_samples / area_base
else:
    flat_struct = structure_grid.flatten()
    valid_struct = ~np.isnan(flat_struct)
    active_count0 = np.sum(active_mask)

    if owc_unc > 0.0:
        owc_samples = rng.normal(owc_value, owc_unc, n_iter)
        k_area = np.empty(n_iter)
        for i in range(n_iter):
            owc_i = owc_samples[i]
            if invert_structure_z:
                mask_i = (flat_struct <= owc_i) & valid_struct
            else:
                mask_i = (flat_struct >= owc_i) & valid_struct
            active_i = np.sum(mask_i)
            if active_i <= 0 or active_count0 <= 0:
                k_area[i] = 0.0
            else:
                k_area[i] = active_i / active_count0
    else:
        k_area = np.ones(n_iter, dtype=float)

# Other factors
thick_samples = sample_param(thickness_base, thick_unc, n_iter, min_val=1.0, rng=rng)
k_h = thick_samples / thickness_base

phi_samples = sample_param(phi_base, por_unc, n_iter, min_val=0.05, max_val=0.40, rng=rng)
k_phi = phi_samples / phi_base

so_samples = sample_param(so_base, sat_unc, n_iter, min_val=0.20, max_val=0.95, rng=rng)
k_so = so_samples / so_base

ntg_samples = sample_param(ntg_base, ntg_unc, n_iter, min_val=0.10, max_val=1.0, rng=rng)
k_ntg = ntg_samples / ntg_base

fvf_samples = sample_param(fvf_base, fvf_unc, n_iter, min_val=1.0, max_val=2.0, rng=rng)
k_fvf = fvf_samples / fvf_base

K_total = k_area * k_h * k_phi * k_so * k_ntg / k_fvf
stoiip_samples_stb = total_stoiip0_stb * K_total
stoiip_samples_bstb = stoiip_samples_stb / 1e9

p10 = np.percentile(stoiip_samples_bstb, 10)
p50 = np.percentile(stoiip_samples_bstb, 50)
p90 = np.percentile(stoiip_samples_bstb, 90)

stoiip0_bstb = total_stoiip0_stb / 1e9

c1, c2, c3, c4 = st.columns(4)
c1.metric("Deterministic STOIIP₀", f"{stoiip0_bstb:,.3f} BSTB")
c2.metric("P10 (Low)", f"{p10:,.3f} BSTB")
c3.metric("P50 (Base)", f"{p50:,.3f} BSTB")
c4.metric("P90 (High)", f"{p90:,.3f} BSTB")

# ============================================================
# Tornado sensitivity
# ============================================================

def tornado_weight_for_factor(k_vector):
    stoiip = total_stoiip0_stb * k_vector
    return np.std(stoiip)

n_torn = min(2000, n_iter)

if area_source == "Manual area (no ZMAP)":
    area_samples_t = sample_param(area_base, area_unc, n_torn, min_val=1.0, rng=rng)
    k_area_t = area_samples_t / area_base
elif owc_unc > 0.0:
    flat_struct = structure_grid.flatten()
    valid_struct = ~np.isnan(flat_struct)
    active_count0 = np.sum(active_mask)
    owc_samples_t = rng.normal(owc_value, owc_unc, n_torn)
    k_area_t = np.empty(n_torn)
    for i in range(n_torn):
        owc_i = owc_samples_t[i]
        if invert_structure_z:
            mask_i = (flat_struct <= owc_i) & valid_struct
        else:
            mask_i = (flat_struct >= owc_i) & valid_struct
        active_i = np.sum(mask_i)
        if active_i <= 0 or active_count0 <= 0:
            k_area_t[i] = 0.0
        else:
            k_area_t[i] = active_i / active_count0
else:
    k_area_t = np.ones(n_torn, dtype=float)

thick_samples_t = sample_param(thickness_base, thick_unc, n_torn, min_val=1.0, rng=rng)
k_h_t = thick_samples_t / thickness_base

phi_samples_t = sample_param(phi_base, por_unc, n_torn, min_val=0.05, max_val=0.40, rng=rng)
k_phi_t = phi_samples_t / phi_base

so_samples_t = sample_param(so_base, sat_unc, n_torn, min_val=0.20, max_val=0.95, rng=rng)
k_so_t = so_samples_t / so_base

ntg_samples_t = sample_param(ntg_base, ntg_unc, n_torn, min_val=0.10, max_val=1.0, rng=rng)
k_ntg_t = ntg_samples_t / ntg_base

fvf_samples_t = sample_param(fvf_base, fvf_unc, n_torn, min_val=1.0, max_val=2.0, rng=rng)
k_fvf_t = fvf_samples_t / fvf_base

weights_raw = {
    "Area/OWC": tornado_weight_for_factor(k_area_t),
    "Thickness": tornado_weight_for_factor(k_h_t),
    "Porosity": tornado_weight_for_factor(k_phi_t),
    "Oil Sat": tornado_weight_for_factor(k_so_t),
    "NTG": tornado_weight_for_factor(k_ntg_t),
    "FVF": tornado_weight_for_factor(1.0 / k_fvf_t),
}

total_w = sum(weights_raw.values())
if total_w > 0:
    weights_norm = {k: v / total_w for k, v in weights_raw.items()}
else:
    weights_norm = {k: 0.0 for k in weights_raw.keys()}

st.subheader("STOIIP Cases (BSTB)")
cases_df = pd.DataFrame({"Case": ["P10", "P50", "P90"], "Volume": [p10, p50, p90]})
cases_chart = (
    alt.Chart(cases_df)
    .mark_bar()
    .encode(x="Case:N", y="Volume:Q", color="Case:N", tooltip=["Case", "Volume"])
)
st.altair_chart(cases_chart, use_container_width=True)

st.subheader("STOIIP Distribution")
dist_df = pd.DataFrame({"STOIIP (BSTB)": stoiip_samples_bstb})
fig_hist = px.histogram(dist_df, x="STOIIP (BSTB)", nbins=50)

# Highlight P10, P50, P90
fig_hist.add_vline(
    x=p10,
    line_dash="dash",
    line_color="blue",
    annotation_text="P10",
    annotation_position="top left",
)
fig_hist.add_vline(
    x=p50,
    line_dash="dash",
    line_color="green",
    annotation_text="P50",
    annotation_position="top",
)
fig_hist.add_vline(
    x=p90,
    line_dash="dash",
    line_color="red",
    annotation_text="P90",
    annotation_position="top right",
)

st.plotly_chart(fig_hist, use_container_width=True)


st.subheader("Tornado Sensitivity")
wdf = (
    pd.DataFrame({"Variable": list(weights_norm.keys()),
                  "Weight": list(weights_norm.values())})
    .sort_values("Weight", ascending=True)
)
fig_torn = go.Figure(
    go.Bar(x=wdf["Weight"], y=wdf["Variable"], orientation="h")
)
fig_torn.update_layout(title="Relative Impact on Total STOIIP")
st.plotly_chart(fig_torn, use_container_width=True)

# ============================================================
# Reservoir Property Visualization (deterministic per-cell)
# ============================================================

st.header("Reservoir Property Visualization (Deterministic Base Case)")

Ny, Nx = structure_grid.shape

# ZMAP mode: same convention as your standalone code
if area_source == "From ZMAP structure grid" and zmap_xmin is not None:
    x_vals = np.linspace(zmap_xmin, zmap_xmax, Nx)
    y_vals = np.linspace(zmap_ymax, zmap_ymin, Ny)  # row 0 = ymax
else:
    x_vals = np.linspace(zmap_xmin, zmap_xmax, Nx)
    y_vals = np.linspace(zmap_ymin, zmap_ymax, Ny)

cell_stoiip_grid_mstb = cell_stoiip_grid_stb / 1e6

prop = st.selectbox(
    "Property to display",
    [
        "Cell STOIIP (MSTB)",
        "Thickness (ft)",
        "Porosity",
        "Oil Saturation",
        "NTG",
        "Structure (active cells)",
    ],
)

# Apply active cell mask to all displayed property grids
if prop == "Cell STOIIP (MSTB)":
    prop_grid = np.where(active_mask, cell_stoiip_grid_mstb, np.nan)

elif prop == "Thickness (ft)":
    prop_grid = np.where(active_mask, thickness_grid, np.nan)

elif prop == "Porosity":
    prop_grid = np.where(active_mask, phi_grid, np.nan)

elif prop == "Oil Saturation":
    prop_grid = np.where(active_mask, so_grid, np.nan)

elif prop == "NTG":
    prop_grid = np.where(active_mask, ntg_grid, np.nan)

elif prop == "Structure (active cells)":
    prop_grid = np.where(active_mask, structure_grid, np.nan)


view_mode = st.radio("View mode", ["Map View", "3D Surface"])
vert_exag = st.slider("Vertical Exaggeration (3D only)", 1.0, 10.0, 4.0)

if view_mode == "Map View":
    fig_map = go.Figure(
        data=go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=prop_grid,
            colorscale="Viridis",
            colorbar=dict(title=prop),
        )
    )
    fig_map.update_layout(
        title=f"Map – {prop}",
        xaxis_title="X",
        yaxis_title="Y",
    )
    st.plotly_chart(fig_map, use_container_width=True)

else:
    if np.all(np.isnan(prop_grid)):
        st.info("No valid values to display in 3D.")
    else:
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = prop_grid.copy()

        if prop == "Structure (active cells)":
            Zplot = Z
        else:
            Zmin = np.nanmin(Z)
            Zmax = np.nanmax(Z)
            if Zmax > Zmin:
                Zplot = (Z - Zmin) / (Zmax - Zmin)
            else:
                Zplot = np.zeros_like(Z)

        fig3d = go.Figure()
        fig3d.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Zplot,
                surfacecolor=Z,
                colorscale="Viridis",
                colorbar=dict(title=prop),
            )
        )
        fig3d.update_layout(
            title=f"3D Surface – {prop}",
            scene=dict(aspectratio=dict(x=1, y=1, z=vert_exag)),
        )
        st.plotly_chart(fig3d, use_container_width=True)
