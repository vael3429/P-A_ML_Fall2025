import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---------- Configuration ----------
TARGET_BANDS = ["g'", "r'", "i'", "z'", "u'", "G"]
MIN_POINTS_TO_FIT = 4
DEFAULT_PLOTFMT = dict(fmt="o", capsize=3, alpha=0.85)


# ---------- Utilities ----------
def _safe_filename(s):
    """Simple sanitization for filenames."""
    s = str(s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-_\.]", "", s)
    return s


# ---------- 1) Load data ----------
def load_data(input_path, sn_file):
    """Load csv or dat file and return (df, metadata_cols, time_col)."""
    path = os.path.join(input_path, sn_file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if sn_file.endswith(".csv"):
        df = pd.read_csv(path)
        metadata = {"sn_name", "norm_time", "MJD", "sn_type"}
        time_col = "MJD"
    elif sn_file.endswith(".dat"):
        df = pd.read_csv(path, sep="\t", engine="python")
        metadata = {"time", "sn_name", "sn_type", "redshift", "norm_time"}
        # Per your choice: use "time" for .dat files
        time_col = "time"
    else:
        raise ValueError(f"Unsupported extension for file: {sn_file}")

    return df, metadata, time_col


# ---------- 2) Models ----------


def model1(t, A, phi, sigma, k):
    t = np.asarray(t, float)
    dt = t - phi
    flux = np.zeros_like(dt, dtype=float)
    mask = dt >= 0
    if np.any(mask):
        x = dt[mask] / sigma
        flux[mask] = A * (x**k) * np.exp(k - x)
    return flux


def model2(t, t0, t1, A, B, Tfall, Trise):
    dt = t - t0
    b = 1 + B * np.square(t - t1)
    b = np.maximum(b, 0)
    numerator = np.exp(-dt / Tfall)
    denominator = 1 + np.exp(-dt / Trise)
    return A * b * numerator / denominator


# ---------- 3) Mag -> Flux ----------
def mag_to_flux_array(mags):
    """
    Convert magnitudes -> linear flux (Pogson).
    Returns (flux, flux_norm, scale)
    """
    mags = np.asarray(mags, dtype=float)
    flux = np.full_like(mags, np.nan, dtype=float)
    finite = np.isfinite(mags)
    if np.any(finite):
        flux[finite] = 10.0 ** (-0.4 * mags[finite])
        scale = np.nanmax(flux)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        flux_norm = np.full_like(flux, np.nan, dtype=float)
        flux_norm[finite] = flux[finite] / scale
    else:
        flux_norm = flux.copy()
        scale = 1.0
    return flux, flux_norm, scale


# ---------- 4) Convert columns ----------
def convert_mag2flux_columns(df, metadata_cols):
    """
    Detect magnitude columns and create these columns for each:
      <band>_flux, <band>_flux_norm, <band>_flux_scale
    Returns (df, mag_cols)
    """
    mag_cols = [
        c
        for c in df.columns
        if c not in metadata_cols and not c.endswith("_err") and not c.endswith("_flux")
    ]
    for band in mag_cols:
        errcol = f"{band}_err"
        if errcol not in df.columns:
            df[errcol] = np.nan
        flux, flux_norm, scale = mag_to_flux_array(df[band].values)
        df[f"{band}_flux"] = flux
        df[f"{band}_flux_norm"] = flux_norm
        df[f"{band}_flux_scale"] = scale
        # df[f"{band}_flux_err"] = magerr_to_fluxerr(flux, df[errcol].values)
    return df, mag_cols


# ---------- 5) Fit band ----------
def fit_band(times, fluxes):
    """
    Fit model1 to (times, fluxes). Returns (popt, pcov, success_flag).
    Times & fluxes are arrays (may contain NaNs) — function filters finite points.
    """
    mask = np.isfinite(times) & np.isfinite(fluxes)
    t = times[mask]
    f = fluxes[mask]

    if len(t) < MIN_POINTS_TO_FIT:
        return None, None, False

    """
    # initial guesses for model 1
    A0 = np.nanmax(f)
    phi0 = t[np.nanargmax(f)]
    sigma0 = max((t.max() - t.min()) / 6.0, 0.1)
    k0 = 2.0
    p0 = [A0, phi0, sigma0, k0]

    # bounds
    lower = [0.0, t.min() - 5.0, 0.1, 0.1]
    upper = [np.inf, t.max() + 5.0, 200.0, 20.0]
    """
    peak_idx = np.argmax(f)
    t_peak = t[peak_idx]

    mask = t >= (t_peak - 15)
    t = t[mask]
    f = f[mask]

    # model 2 values
    t_peak = times[np.argmax(f)]
    amp = np.max(f)
    baseline = np.min(f)
    p0 = [t_peak, t_peak + 10, amp, amp * 0.50, 50, 50]
    lower = [t_peak - 15, t_peak - 15, 0, 0, 0.1, 0.1]
    upper = [t_peak + 50, t_peak + 100, np.inf, np.inf, 100, 100]

    try:
        popt, pcov = curve_fit(model2, t, f, p0=p0, bounds=(lower, upper), maxfev=20000)
        return popt, pcov, True
    except Exception as e:
        # Return initial p0 when fit fails (consistent with original behavior)
        print("  Fit failed:", e)
        return p0, None, False


# ---------- 6) Plotting ----------
def plot_flux_fit(times, flux, popt, band, sn_name, save_path=None, show=True):
    """
    Plot data and fitted model; optionally save.
    times, flux are 1D numpy arrays (can include NaNs).
    """
    finite_mask = np.isfinite(times) & np.isfinite(flux)
    if not np.any(finite_mask):
        print(f"  No finite data to plot for {sn_name} {band}")
        return

    tmin, tmax = np.nanmin(times[finite_mask]), np.nanmax(times[finite_mask])
    tfit = np.linspace(tmin, tmax, 400)

    plt.figure(figsize=(9, 5))
    plt.scatter(times[finite_mask], flux[finite_mask], s=36, alpha=0.9, label="Data")

    # model (if parameters available)
    if popt is not None and np.all(np.isfinite(popt)):
        plt.plot(tfit, model2(tfit, *popt), "r-", lw=2, label="Model fit")
    else:
        plt.plot([], [], "r-", lw=2, label="Model fit (n/a)")

    plt.title(f"{sn_name} — {band}")
    plt.xlabel("Time")
    plt.ylabel("Flux (normalized)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot → {save_path}")

    # if show:
    #     plt.show()
    # else:
    plt.close()


# ---------- 7) Process a single SN file ----------
def process_sn_file(
    input_path,
    group,
    sn_file,
    target_bands=TARGET_BANDS,
    output_dir=None,
    plot_each=True,
):
    """
    Process one file:
      - load data
      - convert mags -> flux
      - for each TARGET band: fit; plot; optionally save png; collect results rows

    Model 1:
    Returns pd.DataFrame of rows for this SN (columns: sn_name, sn_type, band, A, phi, sigma, k)

    Model 2:
    Returns pd.DataFrame of rows for this SN (columns: sn_name, sn_type, band, t0, t1, A, B, Tfall, Trise)


    """
    folder = os.path.join(input_path, group)
    df, metadata, time_col = load_data(folder, sn_file)
    df, mag_cols = convert_mag2flux_columns(df, metadata)

    # keep only the requested bands that exist
    bands = [b for b in mag_cols if b in target_bands]
    if len(bands) == 0:
        print(f"No target bands found in {sn_file}. Available: {mag_cols}")
        return pd.DataFrame(
            # columns=["sn_name", "sn_type", "band", "A", "phi", "sigma", "k"]
            columns=[
                "sn_name",
                "sn_type",
                "band",
                "t0",
                "t1",
                "A",
                "B",
                "Tfall",
                "Trise",
            ]
        )

    # basic SN info
    sn_name = (
        df["sn_name"].iloc[0]
        if "sn_name" in df.columns
        else os.path.splitext(sn_file)[0]
    )
    sn_type = df["sn_type"].iloc[0] if "sn_type" in df.columns else ""

    results = []
    for band in bands:
        print(f"Processing {sn_name}  band={band}")

        # choose times based on file type (CSV -> MJD, DAT -> time)
        times = (
            df[time_col].values
            if time_col in df.columns
            else df.get("MJD", df.get("time")).values
        )
        flux = df[f"{band}_flux_norm"].values

        # skip if not enough valid points
        n_finite = np.sum(np.isfinite(times) & np.isfinite(flux))
        if n_finite < MIN_POINTS_TO_FIT:
            print(
                f"  Skipping {band}: only {n_finite} finite points (need {MIN_POINTS_TO_FIT})"
            )
            continue

        # fit
        popt, pcov, ok = fit_band(times, flux)

        # gather results (only required columns)
        if popt is None:
            row = {
                "sn_name": sn_name,
                "sn_type": sn_type,
                "band": band,
                # "A": np.nan,
                # "phi": np.nan,
                # "sigma": np.nan,
                # "k": np.nan,
                "t0": np.nan,
                "t1": np.nan,
                "A": np.nan,
                "B": np.nan,
                "Tfall": np.nan,
                "Trise": np.nan,
            }
        else:
            row = {
                "sn_name": sn_name,
                "sn_type": sn_type,
                "band": band,
                # "A": popt[0],
                # "phi": popt[1],
                # "sigma": popt[2],
                # "k": popt[3],
                "t0": popt[0],
                "t1": popt[1],
                "A": popt[2],
                "B": popt[3],
                "Tfall": popt[4],
                "Trise": popt[5],
            }
        results.append(row)

        # plotting & saving
        save_png = None
        if output_dir:
            fname = f"{_safe_filename(sn_name)}_{_safe_filename(band)}_flux_fit.png"
            save_png = os.path.join(output_dir, fname)

        # plot (show by default)
        plot_flux_fit(
            times,
            flux,
            popt if ok else None,
            band,
            sn_name,
            save_path=save_png,
            show=plot_each,
        )

    if results:
        return pd.DataFrame(
            results,
            # columns=["sn_name", "sn_type", "band", "A", "phi", "sigma", "k"]
            columns=[
                "sn_name",
                "sn_type",
                "band",
                "t0",
                "t1",
                "A",
                "B",
                "Tfall",
                "Trise",
            ],
        )
    else:
        return pd.DataFrame(
            # columns=["sn_name", "sn_type", "band", "A", "phi", "sigma", "k"]
            columns=[
                "sn_name",
                "sn_type",
                "band",
                "t0",
                "t1",
                "A",
                "B",
                "Tfall",
                "Trise",
            ]
        )


# ---------- 8) Build master table ----------
def build_master_ml_table(
    input_path,
    group,
    output_csv=None,
    target_bands=TARGET_BANDS,
    output_dir=None,
    plot_each=False,
):
    """
    Walk the group directory and process all .csv/.dat files.
    Returns master DataFrame and saves master CSV if output_csv is provided.
    """

    folder = os.path.join(input_path, group)
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)

    files = [
        f
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and f.endswith((".csv", ".dat"))
    ]
    if not files:
        print("No input files found in", folder)
        return pd.DataFrame(
            # columns=["sn_name", "sn_type", "band", "A", "phi", "sigma", "k"]
            columns=[
                "sn_name",
                "sn_type",
                "band",
                "t0",
                "t1",
                "A",
                "B",
                "Tfall",
                "Trise",
            ]
        )

    all_dfs = []
    for fn in files:
        print("----")
        print("File:", fn)
        df_res = process_sn_file(
            input_path,
            group,
            fn,
            target_bands=target_bands,
            output_dir=output_dir,
            plot_each=plot_each,
        )
        if not df_res.empty:
            all_dfs.append(df_res)

    if all_dfs:
        master = pd.concat(all_dfs, ignore_index=True)
    else:
        master = pd.DataFrame(
            # columns=["sn_name", "sn_type", "band", "A", "phi", "sigma", "k"]
            columns=[
                "sn_name",
                "sn_type",
                "band",
                "t0",
                "t1",
                "A",
                "B",
                "Tfall",
                "Trise",
            ]
        )

    if output_csv:
        outdir = os.path.dirname(output_csv)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        master.to_csv(output_csv, index=False)
        print(f"\nSaved master ML table to {output_csv}")

    return master


if __name__ == "__main__":
    # input_path = "/Users/tranghuynh/Developer/PHYS_7895_Fall2025/project/SNclassification_input"
    input_path = "/data/"
    groups = ["Ia", "Ibc", "II", "IIn"]  # subdirectory with SN files
    # output_dir = "/Users/tranghuynh/Developer/PHYS_7895_Fall2025/project/fit_outputs/"  # set to None to skip saving pngs
    output_dir = "/data/fit_outputs/"
    output_files = [
        "model2_ master_ML_table_TypeIa.csv",
        "model2_master_ML_table_TypeIbc.csv",
        "model2_master_ML_table_TypeII.csv",
        "model2_master_ML_table_TypeIIn.csv",
    ]

    for group, output_csv in zip(groups, output_files):
        output_filepath = os.path.join(output_dir, output_csv)
        output_png = os.path.join(output_dir, group)

        if group == "II":
            print(f"\nProcessing group {group}...")
            master_df = build_master_ml_table(
                input_path,
                group,
                output_csv=output_filepath,
                target_bands=TARGET_BANDS,
                output_dir=output_png,
                plot_each=False,
            )

    print(master_df.head(10))
