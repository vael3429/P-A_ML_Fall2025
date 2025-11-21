import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json


# Define a light curve model (example: stretched exponential rise + decline)
def LC_model_generic(t, t0, A, tau_rise, tau_fall, baseline):
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


def LC_model_1(t, A, phi, sigma, k, b):
    """Follows Model 1 from paper
    phi: starting time of explosion
    k: determines rise and fall
    sigma: stretch
    b: tailing
    """
    dt = t - phi
    coeff = A * ((dt / sigma) ** k)
    exponential = np.exp(-dt / sigma) * (k ** (-k)) * np.exp(k)


def LC_model_2(t, t0, t1, A, B, Tfall, Trise):
    """Follows the Model 2 from paper
    A and B are both amplitudes here, allows for a second peak"""
    dt = t - t0
    b = 1 + B * ((t - t1) ** 2)
    numerator = np.exp(-dt / Tfall)
    denominator = 1 + np.exp(-dt / Trise)
    return A * b * numerator / denominator


def mag_to_flux(magnitudes):
    # this equation is Norman Pogson formula, provides a relative flux
    # normalize each lightcurve to peak flux
    magnitudes = np.asarray(magnitudes)
    flux = 10 ** (-0.4 * magnitudes)
    flux_normalized = flux / np.max(flux)

    return flux_normalized


def read_supernova_file(filepath):
    """
    Read a single .dat file containing supernova observations
    Assumes columns: time, name, type, redshift, then band-specific luminosities
    """
    df = pd.read_csv(filepath, sep="\s+", comment="#")
    return df


def extract_lightcurve_by_band(df, band_columns, convert_mag_to_flux=True):
    """
    Extract light curves for each observational band

    Parameters:
    -----------
    df : DataFrame
        Supernova observation data
    band_columns : list
        Column names for different observational bands
    convert_mag_to_flux : bool
        If True, converts magnitudes to flux (default: True)

    Returns:
    --------
    lightcurves : dict
        {band: (times, flux_values)}
    """
    lightcurves = {}
    time_col = df.columns[0]

    for band_col in band_columns:
        if band_col in df.columns:
            # Filter out NaN values
            mask = ~df[band_col].isna()
            times = df[time_col][mask].values
            values = df[band_col][mask].values

            if len(times) > 0:
                # Convert magnitudes to flux if needed
                if convert_mag_to_flux:
                    flux = mag_to_flux(values)
                    lightcurves[band_col] = (times, flux)
                else:
                    lightcurves[band_col] = (times, values)

    return lightcurves


def fit_light_curve(times, fluxes, model_func=LC_model_2):
    """
    Fit light curve to parametric model
    Returns fitted parameters and covariance
    """
    t_peak_guess = times[np.argmax(fluxes)]
    amp_guess = np.max(fluxes) - np.min(fluxes)
    baseline_guess = np.min(fluxes)

    # for the generic model: t0, A, tau_rise, tau_fall, B
    # p0 = [t_peak_guess, amp_guess, 10, 30, baseline_guess]

    # model 2: t0, t1, A, B, Tfall, Trise
    p0 = [t_peak_guess, t_peak_guess * 1.01, amp_guess, amp_guess * 0.75, 10, 30]

    try:
        popt, pcov = curve_fit(
            model_func,
            times,
            fluxes,
            p0=p0,
            maxfev=5000,
            bounds=(
                [0, 0, -np.inf, -np.inf, 0.1, 0.1],
                [np.inf, np.inf, np.inf, np.inf, 100, 100],
            ),
            # bounds=([0, 0, 0.1, 0.1, -np.inf], [np.inf, np.inf, 100, 200, np.inf]),
        )
        return popt, pcov, True
    except Exception as e:
        print(f" Fit failed : {e}")
        return p0, None, False


def plot_light_curve(times, fluxes, fit_params, band_name, sn_name, save_path=None):
    """
    Plot observed light curve with fitted model
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(times, fluxes, label="Observations", alpha=0.6, s=50)

    # Generate smooth curve for fit
    t_smooth = np.linspace(times.min(), times.max(), 200)
    fit_curve = LC_model_2(t_smooth, *fit_params)
    plt.plot(t_smooth, fit_curve, "r-", label="Fitted Model", linewidth=2)

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.title(f"{sn_name} - {band_name} Band", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def process_supernova_dataset(data_dir, output_dir, plot=True, convert_magnitudes=True):
    """
    Process all .dat files in directory

    Parameters:
    -----------
    data_dir : str
        Directory containing .dat files
    output_dir : str
        Directory for output files and plots
    plot : bool
        Whether to generate plots
    convert_magnitudes : bool
        Whether to convert magnitudes to flux (default: True)

    Returns:
    --------
    successful_df, unsuccessful_df : tuple of DataFrames
        Fitted parameters for successful and unsuccessful fits
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Create separate directories for successful and unsuccessful fits
    if plot:
        (output_path / "plots_successful").mkdir(exist_ok=True)
        (output_path / "plots_unsuccessful").mkdir(exist_ok=True)

    successful_results = []
    unsuccessful_results = []

    # Track statistics
    total_fits_attempted = 0
    total_fits_successful = 0
    total_fits_unsuccessful = 0

    dat_files = list(data_path.glob("*.dat"))
    print(f"Found {len(dat_files)} .dat files in {data_dir}")

    if len(dat_files) == 0:
        print(f"WARNING: No .dat files found in {data_dir}")
        return pd.DataFrame(), pd.DataFrame()

    for dat_file in data_path.glob("*.dat"):
        print(f"Processing {dat_file.name}...")

        # Read data
        df = read_supernova_file(dat_file)
        sn_name = df.iloc[0, 1] if len(df) > 0 else dat_file.stem
        sn_type = df.iloc[0, 2] if len(df) > 0 else "Unknown"
        redshift = df.iloc[0, 3] if len(df) > 0 else np.nan

        # Identify band columns (skip time, name, type, redshift, and error columns)
        band_columns = [col for col in df.columns[4:] if "err" not in col.lower()]

        # Extract and fit each band (with magnitude conversion)
        lightcurves = extract_lightcurve_by_band(
            df, band_columns, convert_mag_to_flux=convert_magnitudes
        )

        for band, (times, flux) in lightcurves.items():
            if len(times) < 10:  # Skip if too few points
                print(f"  Skipping {band}: only {len(times)} observations")
                continue

            # Fit light curve
            params, cov, success = fit_light_curve(times, flux)
            total_fits_attempted += 1

            # Create parameter dictionary for this fit
            sn_params = {
                "supernova": sn_name,
                "type": sn_type,
                "redshift": redshift,
                "band": band,
                "n_obs": len(times),
            }

            # Store parameters with descriptive names
            param_names = ["t0", "t1", "A", "B", "Tfall", "Trise"]
            for i, pname in enumerate(param_names):
                sn_params[pname] = params[i]

            # Separate successful and unsuccessful fits
            if success:
                total_fits_successful += 1
                successful_results.append(sn_params)

                # Plot successful fit
                if plot:
                    plot_path = (
                        output_path / "plots_successful" / f"{sn_name}_{band}.png"
                    )
                    plot_light_curve(times, flux, params, band, sn_name, plot_path)
            else:
                total_fits_unsuccessful += 1
                unsuccessful_results.append(sn_params)

                # Plot unsuccessful fit
                if plot:
                    plot_path = (
                        output_path / "plots_unsuccessful" / f"{sn_name}_{band}.png"
                    )
                    plot_light_curve(times, flux, params, band, sn_name, plot_path)

    # Create DataFrames
    successful_df = pd.DataFrame(successful_results)
    unsuccessful_df = pd.DataFrame(unsuccessful_results)

    # Save successful fits
    if len(successful_df) > 0:
        successful_df.to_csv(
            output_path / "supernova_parameters_successful.csv", index=False
        )
        successful_df.to_json(
            output_path / "supernova_parameters_successful.json",
            orient="records",
            indent=2,
        )

    # Save unsuccessful fits
    if len(unsuccessful_df) > 0:
        unsuccessful_df.to_csv(
            output_path / "supernova_parameters_unsuccessful.csv", index=False
        )
        unsuccessful_df.to_json(
            output_path / "supernova_parameters_unsuccessful.json",
            orient="records",
            indent=2,
        )

    # Print statistics
    print("\n" + "=" * 60)
    print("FIT STATISTICS")
    print("=" * 60)
    print(f"Total fits attempted:        {total_fits_attempted}")
    print(f"Successful fits:             {total_fits_successful}")
    print(f"Unsuccessful fits:           {total_fits_unsuccessful}")
    if total_fits_attempted > 0:
        success_rate = (total_fits_successful / total_fits_attempted) * 100
        print(f"Success rate:                {success_rate:.2f}%")
    print("=" * 60)

    print(
        f"\nSuccessful parameters saved to: {output_path / 'supernova_parameters_successful.csv'}"
    )
    print(
        f"Unsuccessful parameters saved to: {output_path / 'supernova_parameters_unsuccessful.csv'}"
    )

    if plot:
        print(f"\nSuccessful plots saved to: {output_path / 'plots_successful'}")
        print(f"Unsuccessful plots saved to: {output_path / 'plots_unsuccessful'}")

    return successful_df, unsuccessful_df


# Example usage:
if __name__ == "__main__":
    # Process all .dat files in 'supernova_data' directory
    successful_df, unsuccessful_df = process_supernova_dataset(
        "/data/Ia",
        output_dir="/data/Ia/model2",
        plot=True,
        convert_magnitudes=True,  # Set to False if data is already in flux units
    )

    print("\n" + "=" * 60)
    print("SUCCESSFUL FITS - Parameter Summary:")
    print("=" * 60)
    if len(successful_df) > 0:
        print(successful_df.describe())
        print("\nFirst few successful entries:")
        print(successful_df.head())
    else:
        print("No successful fits")

    print("\n" + "=" * 60)
    print("UNSUCCESSFUL FITS - Parameter Summary:")
    print("=" * 60)
    if len(unsuccessful_df) > 0:
        print(unsuccessful_df.describe())
        print("\nFirst few unsuccessful entries:")
        print(unsuccessful_df.head())
    else:
        print("No unsuccessful fits")
