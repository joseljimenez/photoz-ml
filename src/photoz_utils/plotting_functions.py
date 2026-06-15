import numpy as np
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.stats import sigmaclip
from astropy.stats import biweight_location, biweight_scale
from pathlib import Path

Base_DIR = Path.cwd().parent
PLOT_DIR = Base_DIR/'plots'

def plot_magnitude_histograms(magnitudes : dict[str, np.ndarray], 
  bands:list[str],
  bins=100, 
  figsize=(14, 8)):
    """
    Plot histograms for ugrizy magnitudes in a 2x3 grid.

    Parameters
    ----------
    magnitudes : dict
        Dictionary with keys ['u', 'g', 'r', 'i', 'z', 'y'].
        Each value must be an array-like of magnitudes.
    
    bins : int, optional
        Number of bins for histograms. Default is 50.
    
    figsize : tuple, optional
        Figure size. Default is (14, 8).
    """

    # Verify required bands exist
    missing = [band for band in bands if band not in magnitudes]
    if missing:
        raise ValueError(f"Missing bands in dictionary: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        data = np.asarray(magnitudes[band])

        # Remove NaNs if present
        data = data[~np.isnan(data)]

        ax.hist(data, bins=bins)
        ax.set_title(f'{band}-band')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
    PLOT_DIR / 'magnitude_histograms_psf.png',
    dpi=300,
    bbox_inches='tight'
)
    plt.show()

def plot_color_histograms(colors: dict[str, np.ndarray],
    bands: list[str], 
    bins=50,
    figsize=(14, 8)):
    """
    Plot histograms for ugrizy colors in a 2x3 grid.

    Parameters
    ----------
    colors : dict
        Dictionary with keys:
        ['u-g', 'g-r', 'r-i', 'i-z', 'z-y'].
        Each value must be an array-like of color values.

    bins : int, optional
        Number of bins for histograms. Default is 50.

    figsize : tuple, optional
        Figure size. Default is (14, 8).
    """
    color_keys = [f"{bands[i]}-{bands[i+1]}" for i in range(len(bands)-1)]

    # Verify required colors exist
    missing = [color for color in color_keys if color not in colors]
    if missing:
        raise ValueError(f"Missing color keys in dictionary: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, color in zip(axes, color_keys):
        data = np.asarray(colors[color])

        # Remove NaNs if present
        data = data[~np.isnan(data)]

        ax.hist(data, bins=bins)
        ax.set_title(color)
        ax.set_xlabel('Color')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)

    # Remove unused subplot (6th panel)
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(
    PLOT_DIR / 'colors_histograms_psf.png',
    dpi=300,
    bbox_inches='tight'
)
    plt.show()

def plot_color_color_diagrams(
    colors: dict[str, np.ndarray],
    bands: list[str],
    figsize=(18, 15),
    point_size=1,
    alpha=0.3
):
    """
    Plot all pairwise color-color diagrams.

    Parameters
    ----------
    colors : dict
        Dictionary with keys:
        ['u-g', 'g-r', 'r-i', 'i-z', 'z-y'].
        Values must be array-like.

    figsize : tuple, optional
        Figure size. Default is (18, 10).

    point_size : float, optional
        Scatter marker size. Default is 1.

    alpha : float, optional
        Point transparency. Default is 0.3.
    """

    color_keys = [f"{bands[i]}-{bands[i+1]}" for i in range(len(bands)-1)]

    # Verify required colors exist
    missing = [c for c in color_keys if c not in colors]
    if missing:
        raise ValueError(f"Missing color keys in dictionary: {missing}")

    # Generate all unique color pairs
    color_pairs = list(combinations(color_keys, 2))

    n_plots = len(color_pairs)
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for ax, (x_color, y_color) in zip(axes, color_pairs):

        x = np.asarray(colors[x_color])
        y = np.asarray(colors[y_color])

        # Remove NaNs consistently
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]

        ax.scatter(x, y, s=point_size, alpha=alpha)

        ax.set_xlabel(x_color)
        ax.set_ylabel(y_color)
        ax.set_title(f'{x_color} vs {y_color}')

    # Remove unused axes
    for ax in axes[n_plots:]:
        fig.delaxes(ax)
    plt.savefig(
    PLOT_DIR / 'color_color_diagrams_psf.png',
    dpi=300,
    bbox_inches='tight'
)
    plt.tight_layout()
    plt.show()

def plot_true_predict_fancy(targets: np.ndarray, predictions: np.ndarray) -> Figure:
    """Plot predicted redshift v. true redshift as with nice overlayes

    Parameters
    ----------
    targets:
        Target redshifts [N_objects]

    predictions:
        Predicted redshifts [N_objects]

    Returns
    -------
    Figure with requested plots and nice overlays
    """
    z_min = 0.0
    z_max = 3.0
    figure, axes = plt.subplots(figsize=(7, 6))
    bin_edges = np.linspace(0, 3.0, 301)
    dz = (predictions - targets) / (1 + targets)
    mean, _mean_err, std, outlier_rate = get_biweight_mean_sigma_outlier(dz, nclip=3)
    mean, std, outlier_rate = round(mean, 4), round(std, 4), round(outlier_rate, 4)
    h = axes.hist2d(
        targets,
        predictions,
        bins=(bin_edges, bin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10, z_max + 10],
        "--",
        color="red",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10 - 3 * std, z_max + 10 - 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [z_min - 10, z_max + 10],
        [z_min - 10 + 3 * std, z_max + 10 + 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [],
        [],
        ".",
        alpha=0.0,
        label=rf"$\Delta z = {mean} $"
        + "\n"
        + rf"$\sigma z = {std} $"
        + "\n"
        + f"outlier rate = {outlier_rate}",
    )

    plt.xlabel("True Redshift")
    plt.ylabel("Estimated Redshift")
    cb = figure.colorbar(h[3], ax=axes)
    cb.set_label("Density")

    plt.legend()
    figure.tight_layout()
    return figure

def get_biweight_mean_sigma_outlier(
    subset: np.ndarray, nclip: int = 3
) -> tuple[float, float, float, float]:
    """Return biweight stats with sigma clipping

    Parameters
    ----------
    subset:
        Input data, estimate - reference redshifts

    nclip:
        Value for sigma clipping

    Returns
    -------
    Mean, error on mean, std, outlier_rate
    """

    subset_clip, _, _ = sigmaclip(subset, low=nclip, high=nclip)
    for _j in range(nclip):
        subset_clip, _, _ = sigmaclip(subset_clip, low=nclip, high=nclip)

    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    outlier_rate = np.sum(np.abs(subset) > nclip * biweight_scale(subset_clip)) / len(
        subset
    )

    return mean, std / np.sqrt(len(subset_clip)), std, outlier_rate