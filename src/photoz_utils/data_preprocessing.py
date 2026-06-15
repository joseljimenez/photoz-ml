import numpy as np
from typing import Any, TypeAlias
# from tables_io.types import Tablelike
Tablelike: TypeAlias = Any

def fluxes_to_mags(
    fluxes: np.ndarray, zero_points: float | np.ndarray = 31.4
) -> np.ndarray:
    """Convert fluxes to magnitudes

    Parameters
    ----------
    fluxes:
        Input data
    zero_points:
        Zero-point magnitudes

    Returns
    -------
    Output magnitudes
    """
    return -2.5 * np.log10(fluxes) + zero_points


def make_mag_dict(
    data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Extract magnitudes from a table to a dict of numpy arrays

    Parameters
    ----------
    data:
        Input data
    band_name_template:
        Template to make the column names
    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output dict
    """
    mag_dict = {}
    for band_ in bands:
        col = band_name_template.format(band=band_)
        mags = fluxes_to_mags(data[col])
        mag_dict[band_] = mags
    return mag_dict

def make_color_dict(
    data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Extract colors from a table of fluxes to a dict of numpy arrays

    Parameters
    ----------
    data:
        Input data, with fluxes

    band_name_template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output dict
    """
    colors = {}
    for i, band_ in enumerate(bands[0:-1]):
        col_a = band_name_template.format(band=band_)
        col_b = band_name_template.format(band=bands[i + 1])
        try:
            mag_a = fluxes_to_mags(data[col_a].to_numpy())
            mag_b = fluxes_to_mags(data[col_b].to_numpy())
        except AttributeError:
            mag_a = fluxes_to_mags(data[col_a])
            mag_b = fluxes_to_mags(data[col_b])
        colors[f"{band_}-{bands[i+1]}"] = mag_a - mag_b
    return colors

def mags_correct_reddening(
    mags: dict[str, np.ndarray], 
    ebv: np.ndarray, 
    lsst_def_a_env: dict[str, np.ndarray], 
    band_e_env_name_template:str, 
    bands: list[str],
    ) -> dict[str, np.ndarray]:
    mag_dict_correct = {}
    for band_ in bands:
        lsst_def_a_env_band = band_e_env_name_template.format(band=band_)
        R_band = lsst_def_a_env[lsst_def_a_env_band]
        mags_correct = mags[band_] - R_band*ebv
        mag_dict_correct[band_] = mags_correct
    return mag_dict_correct 

def prepare_features(data : Tablelike,
    band_name_template: str,
    bands: list[str],
    ) -> np.ndarray:
    band_names = [band_name_template.format(band=band_) for band_ in bands]
    fluxes = extract_data_to_2d_array(data, band_names)
    fluxes = np.where(np.isfinite(fluxes), fluxes, 0.0)
    total_fluxes = np.sum(fluxes, axis=1)
    mag_total = fluxes_to_mags(total_fluxes, 31.4)
    mag_total = np.where(np.isfinite(mag_total), mag_total, 25.0)
    mags = fluxes_to_mags(fluxes, 31.4)
    mags = np.where(np.isfinite(mags), mags, 27.0)
    colors = adjacent_band_colors(mags).clip(-2, 2)
    features = np.vstack([mag_total, colors.T]).T
    return (features)

def extract_data_to_2d_array(data: Tablelike, column_names: list[str]) -> np.ndarray:
    """Extract a set of columns from a table to a 2D array

    Parameters
    ----------
    data:
        Input data

    column_names:
        Names of the columns to extract

    Returns
    -------
    Output 2D-Array
    """
    column_data = [data[column_] for column_ in column_names]
    return np.vstack(column_data).T

def mags_to_fluxes(
    mags: np.ndarray, zero_points: float | np.ndarray = 31.4
) -> np.ndarray:
    """Convert magnitudes to fluxes

    Parameters
    ----------
    mags:
        Input data

    zero_points:
        Zero-point magnitudes

    Returns
    -------
    Output fluxes
    """
    return np.power(10, (zero_points - mags) / 2.5)

def adjacent_band_colors(mags: np.ndarray) -> np.ndarray:
    """Return a set of colors using magnitudes in adjacent bands

    I.e., u-g, g-r, r-i, i-z, z-y

    Note that there will be one less color than bands

    Parameters
    ----------
    mags:
        Input data

    Returns
    -------
    Output colors
    """
    n_bands = mags.shape[-1]
    colors = [mags[:, i] - mags[:, i + 1] for i in range(n_bands - 1)]
    return np.vstack(colors).T

def prepare_data_total_mag_and_colors(
    input_data: np.ndarray,
    band_name_template: str,
    bands: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract data for Regression algorithms

    Parameters
    ----------
    input_data:
        Table with input data

    band_name_template:
        Template for the band names

    bands:
        List of the bands

    Returns
    -------
    Tuple with ndarray(N) of target redshift and
    ndarray(N,N_color+1) of summed magntiude and colors
    """
    band_names = [band_name_template.format(band=band_) for band_ in bands]
    mags = extract_data_to_2d_array(input_data, band_names)
    fluxes = mags_to_fluxes(mags, 31.4)
    fluxes = np.where(np.isfinite(fluxes), fluxes, 0.0)
    total_fluxes = np.sum(fluxes, axis=1)
    mag_total = fluxes_to_mags(total_fluxes, 31.4)
    min_mag_total = np.min(mag_total)
    max_mag_total = np.max(mag_total)
    mag_total = np.where(np.isfinite(mag_total), mag_total, 25.0)
    mags = np.where(np.isfinite(mags), mags, 27.0)
    colors = adjacent_band_colors(mags).clip(-2, 2)
    
    try:
        targets = input_data["redshift"]
    except KeyError:
        targets = None
    features = np.vstack([mag_total, colors.T]).T
    return (targets, features)