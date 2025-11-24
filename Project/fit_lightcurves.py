import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import json


# Various light curve models
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


def LC_model_1(t, A, phi, sigma, k):
    """
    Simplified Model 1 from paper

    A : Peak flux
    phi : Explosion start time
    sigma : Stretch parameter (width)
    k : Rise power law index
    """
    t = np.asarray(t)
    dt = t - phi

    flux = np.zeros_like(dt, dtype=float)
    valid_mask = dt >= 0

    if np.any(valid_mask):
        dt_valid = dt[valid_mask]
        x = dt_valid / sigma
        flux[valid_mask] = A * (x**k) * np.exp(k - x)

    return flux


def LC_model_2(t, t0, t1, A, B, Tfall, Trise):
    """Follows the Model 2 from paper
    A and B are both amplitudes here, allows for a second peak"""
    dt = t - t0
    b = 1 + B * ((t - t1) ** 2)
    numerator = np.exp(-dt / Tfall)
    denominator = 1 + np.exp(-dt / Trise)
    return A * b * numerator / denominator


def mag_to_flux(magnitudes):
    """
    Convert astronomical magnitudes to relative flux.
    Uses the Pogson formula: flux ∝ 10^(-0.4 * mag)

    flux_normalized : Relative flux values (normalized so peak = 1)
    flux_scale : The normalization factor (original peak flux)
    """
    magnitudes = np.asarray(magnitudes)

    # Remove any NaN or infinite values before conversion
    if np.any(~np.isfinite(magnitudes)):
        print(f"  Warning: Found non-finite magnitude values, filtering them out")

    # Pogson's formula
    flux = 10 ** (-0.4 * magnitudes)

    # Check for valid flux values
    if not np.any(np.isfinite(flux)) or np.max(flux) <= 0:
        print(f"  Warning: Invalid flux values after conversion")
        return flux, 1.0

    # Normalize to peak flux = 1 for easier fitting
    max_flux = np.max(flux[np.isfinite(flux)])
    flux_normalized = flux / max_flux

    return flux_normalized, max_flux


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
        {band: (times, flux_values, flux_scale)}
    """
    lightcurves = {}
    time_col = df.columns[0]

    for band_col in band_columns:
        if band_col in df.columns:
            # Filter out NaN and infinite values
            mask = np.isfinite(df[band_col])
            times = df[time_col][mask].values
            values = df[band_col][mask].values

            if len(times) > 0 and len(values) > 0:
                # Convert magnitudes to flux if needed
                if convert_mag_to_flux:
                    flux, flux_scale = mag_to_flux(
                        values
                    )  # Unpack the two return values
                    # Double-check normalization worked
                    finite_flux = flux[np.isfinite(flux)]
                    if len(finite_flux) > 0 and np.max(finite_flux) > 10:
                        print(
                            f"  Warning: Flux not properly normalized for {band_col}, max={np.max(finite_flux):.2e}"
                        )
                        # Force renormalization
                        flux = flux / np.max(finite_flux)
                    lightcurves[band_col] = (times, flux, flux_scale)
                else:
                    lightcurves[band_col] = (times, values, 1.0)

    return lightcurves


def fit_light_curve(times, fluxes, model_func):
    """
    Fit light curve to parametric model

    Parameters:
    -----------
    times : array
        Observation times
    fluxes : array
        Flux values (should be normalized, with peak near 1)
    model_func : callable
        Light curve model function

    Returns:
    --------
    popt : Optimal parameters
    pcov : Parameter covariance matrix
    success : Whether fit succeeded
    """

    # these are some guesses for starting the fit
    # these could be adjusted, sort of just an educated guess
    t_peak_guess = times[np.argmax(fluxes)]
    amp_guess = np.max(fluxes) - np.min(fluxes)
    baseline_guess = np.min(fluxes)

    """need to change p0 for different fit models
    and also need to change the bounds below in curve_fit to match"""

    # model 1: A, phi, sigma, k
    # p0 = [amp_guess, t_peak_guess, 30, 20]

    # model 2: t0, t1, A, B, Tfall, Trise
    p0 = [t_peak_guess, t_peak_guess * 1.01, amp_guess, amp_guess * 0.75, 10, 30]

    try:
        popt, pcov = curve_fit(
            model_func,
            times,
            fluxes,
            p0=p0,
            maxfev=5000,
            # model 2 bounds
            bounds=(
                [0, 0, -np.inf, -np.inf, 0.1, 0.1],
                [np.inf, np.inf, np.inf, np.inf, 100, 100],
            ),
            # model 1 bounds
            # bounds=(
            #    [0, 0, 0, 0],
            #    [np.inf, np.inf, 500, 500],
            # ),
        )
        return popt, pcov, True
    except Exception as e:
        print(f"  Fit failed for {len(times)} points: {str(e)[:80]}")
        return p0, None, False


def plot_light_curve(times, fluxes, fit_params, band_name, sn_name, save_path=None):
    """
    Plot observed light curve with fitted model
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(times, fluxes, label="Observations", alpha=0.6, s=50)

    # Generate smooth curve for fit
    t_smooth = np.linspace(times.min(), times.max(), 200)
    fit_curve = LC_model_2(t_smooth, *fit_params)  # change model here
    plt.plot(t_smooth, fit_curve, "r-", label="Fitted Model", linewidth=2)

    plt.xlabel("Time (days)", fontsize=12)
    plt.ylabel("Normalized Flux", fontsize=12)
    plt.title(f"{sn_name} - {band_name} Band", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)

    # Set reasonable y-axis limits (normalized data should be 0-1 range, plus some margin)
    y_data_max = np.max(fluxes[np.isfinite(fluxes)])
    y_data_min = np.min(fluxes[np.isfinite(fluxes)])
    y_margin = (y_data_max - y_data_min) * 0.2
    plt.ylim(y_data_min - y_margin, y_data_max + y_margin)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def process_supernova_dataset(
    data_dir, output_dir, model_func, plot=True, convert_magnitudes=True
):
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

        for band, (times, flux, flux_scale) in lightcurves.items():
            if len(times) < 10:  # Skip if too few points
                print(f"  Skipping {band}: only {len(times)} observations")
                continue

            # Fit light curve on normalized data
            params, cov, success = fit_light_curve(times, flux, model_func)
            total_fits_attempted += 1

            # Create parameter dictionary for this fit
            sn_params = {
                "supernova": sn_name,
                "type": sn_type,
                "redshift": redshift,
                "band": band,
                "n_obs": len(times),
                "flux_scale": flux_scale,  # Store normalization factor
            }

            # Store NORMALIZED parameters (as fitted to normalized data)
            # These are the parameters your ML model will use

            # model 2
            param_names = ["t0", "t1", "A", "B", "Tfall", "Trise"]

            # model 1
            # param_names = ["A", "phi", "sigma", "k"]

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

    # Print statistics: successful vs unsuccessful fits
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
    # change info here for proper directories, fit model, and if you want plots
    successful_df, unsuccessful_df = process_supernova_dataset(
        data_dir="/data/Candidate",
        output_dir="/data/Candidate/model2",
        model_func=LC_model_2,
        plot=True,
        convert_magnitudes=True,
    )

    # will print some basic statistics in the end
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
