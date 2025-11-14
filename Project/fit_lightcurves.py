import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json


# Define a light curve model (example: stretched exponential rise + decline)
def light_curve_model(t, t0, A, tau_rise, tau_fall, baseline):
    """
    Simple parametric light curve model
    t0: peak time
    A: amplitude
    tau_rise: rise timescale
    tau_fall: decay timescale
    baseline: background level
    """
    dt = t - t0
    rise = np.exp(dt / tau_rise)
    fall = np.exp(-dt / tau_fall)
    return baseline + A * rise * fall / (rise + fall)


def read_supernova_file(filepath):
    """
    Read a single .dat file containing supernova observations
    Assumes columns: time, name, type, redshift, then band-specific luminosities
    """
    df = pd.read_csv(filepath, sep="\s+", comment="#")
    return df


def extract_lightcurve_by_band(df, band_columns):
    """
    Extract light curves for each observational band
    Returns dict of {band: (times, luminosities)}
    """
    lightcurves = {}
    time_col = df.columns[0]  # Assuming first column is time

    for band_col in band_columns:
        if band_col in df.columns:
            # Filter out NaN values
            mask = ~df[band_col].isna()
            times = df[time_col][mask].values
            lums = df[band_col][mask].values
            if len(times) > 0:
                lightcurves[band_col] = (times, lums)

    return lightcurves


def fit_light_curve(times, luminosities, model_func=light_curve_model):
    """
    Fit light curve to parametric model
    Returns fitted parameters and covariance
    """
    # Initial parameter guesses
    t_peak_guess = times[np.argmax(luminosities)]
    amp_guess = np.max(luminosities) - np.min(luminosities)
    baseline_guess = np.min(luminosities)

    p0 = [t_peak_guess, amp_guess, 10, 30, baseline_guess]

    try:
        popt, pcov = curve_fit(
            model_func,
            times,
            luminosities,
            p0=p0,
            maxfev=5000,
            bounds=([0, 0, 0.1, 0.1, -np.inf], [np.inf, np.inf, 100, 200, np.inf]),
        )
        return popt, pcov, True
    except:
        return p0, None, False


def plot_light_curve(
    times, luminosities, fit_params, band_name, sn_name, save_path=None
):
    """
    Plot observed light curve with fitted model
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(times, luminosities, label="Observations", alpha=0.6)

    # Generate smooth curve for fit
    t_smooth = np.linspace(times.min(), times.max(), 200)
    fit_curve = light_curve_model(t_smooth, *fit_params)
    plt.plot(t_smooth, fit_curve, "r-", label="Fitted Model", linewidth=2)

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Luminosity", fontsize=12)
    plt.title(f"{sn_name} - {band_name} Band", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def process_supernova_dataset(data_dir, output_dir, plot=True):
    """
    Process all .dat files in directory
    Returns DataFrame with fitted parameters for NN input
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if plot:
        (output_path / "plots").mkdir(exist_ok=True)

    results = []

    dat_files = list(data_path.glob("*.dat"))
    print(f"Found {len(dat_files)} .dat files in {data_dir}")

    if len(dat_files) == 0:
        print(f"WARNING: No .dat files found in {data_dir}")
        return pd.DataFrame()

    for dat_file in data_path.glob("*.dat"):
        print(f"Processing {dat_file.name}...")

        # Read data
        df = read_supernova_file(dat_file)
        sn_name = (
            df.iloc[0, 1] if len(df) > 0 else dat_file.stem
        )  # Get name from 2nd column
        sn_type = df.iloc[0, 2] if len(df) > 0 else "Unknown"
        redshift = df.iloc[0, 3] if len(df) > 0 else np.nan

        # Identify band columns (skip time, name, type, redshift, and error columns)
        band_columns = [col for col in df.columns[4:] if "err" not in col.lower()]

        # Extract and fit each band
        lightcurves = extract_lightcurve_by_band(df, band_columns)

        sn_params = {
            "supernova": sn_name,
            "type": sn_type,
            "redshift": redshift,
            "n_bands": len(lightcurves),
        }

        for band, (times, lums) in lightcurves.items():
            if len(times) < 5:  # Skip if too few points
                continue

            # Fit light curve
            params, cov, success = fit_light_curve(times, lums)

            # Store parameters with band prefix
            param_names = ["t_peak", "amplitude", "tau_rise", "tau_fall", "baseline"]
            for i, pname in enumerate(param_names):
                sn_params[f"{band}_{pname}"] = params[i]

            sn_params[f"{band}_fit_success"] = success
            sn_params[f"{band}_n_obs"] = len(times)

            # Plot if requested
            if plot:
                plot_path = output_path / "plots" / f"{sn_name}_{band}.png"
                plot_light_curve(times, lums, params, band, sn_name, plot_path)

        results.append(sn_params)

    # Create DataFrame with all results
    results_df = pd.DataFrame(results)

    # Save to CSV for NN input
    results_df.to_csv(output_path / "supernova_parameters.csv", index=False)

    # Also save as JSON for easier inspection
    results_df.to_json(
        output_path / "supernova_parameters.json", orient="records", indent=2
    )

    print(f"\nProcessed {len(results)} supernovae")
    print(f"Parameters saved to {output_path / 'supernova_parameters.csv'}")

    return results_df


# Example usage:
if __name__ == "__main__":
    # Process all .dat files in 'supernova_data' directory
    results_df = process_supernova_dataset("/data/Ia", output_dir="/data/Ia", plot=True)

    print("\nParameter summary:")
    # print(results_df.describe())

    # Preview the parameter table
    print("\nFirst few entries:")
    print(results_df.head())
