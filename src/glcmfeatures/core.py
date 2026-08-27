"""
Core GLCM texture feature calculations.

Implements the 7-feature set identified by Hall-Beyer (2017, IJRS 38:1312-1338)
as carrying most of the discriminating information in the full 21-feature
Haralick/GLCM catalogue, for window sizes in the same range (5-25 px) as
typical forest-inventory plots at sub-metre aerial resolution:

    contrast, correlation, homogeneity, ASM (angular second moment),
    variance, sum average, entropy

Built directly on skimage.feature.graycomatrix / graycoprops. Four of the
seven (contrast, correlation, homogeneity, ASM) are native skimage props;
the other three (variance, sum average, entropy) are computed here directly
from the raw co-occurrence matrix that graycomatrix returns.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Angles 0, 45, 90, 135 degrees, in radians (skimage convention)
DEFAULT_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
DEFAULT_DISTANCES = (1,)

CORE_FEATURES = (
    "contrast",
    "correlation",
    "homogeneity",
    "ASM",
    "variance",
    "sum_average",
    "entropy",
)

_SKIMAGE_NATIVE = {"contrast", "correlation", "homogeneity", "ASM"}


def quantize(image, levels=32, vmin=None, vmax=None):
    """Rescale an image to integer grey levels in [0, levels-1].

    Parameters
    ----------
    image : array_like
        2D array of pixel values (e.g. a single band of a raster window).
    levels : int
        Number of grey levels to quantize to. For small windows (a few
        hundred to a few thousand pixels, as with an 18x18 m window at
        0.5-1 m resolution) 16-64 levels is standard practice; 256 levels
        will usually leave the co-occurrence matrix too sparse to be
        informative.
    vmin, vmax : float, optional
        Fixed intensity range to quantize against. Important when you plan
        to compare features across multiple windows/plots: if you leave
        these as None, quantization uses *this window's own* min/max, which
        makes features non-comparable across windows with different
        brightness ranges. Pass a fixed vmin/vmax (e.g. the min/max of the
        whole orthomosaic, or a known sensor range) whenever you need
        comparability across plots.

    Returns
    -------
    numpy.ndarray of dtype uint8, same shape as ``image``.
    """
    image = np.asarray(image, dtype=float)
    auto_range = vmin is None and vmax is None
    vmin = float(image.min()) if vmin is None else float(vmin)
    vmax = float(image.max()) if vmax is None else float(vmax)
    if vmax <= vmin:
        if auto_range:
            # A perfectly uniform window (e.g. bare ground, water, shadow)
            # has no intensity range of its own - map it to a single grey
            # level rather than erroring.
            return np.zeros(image.shape, dtype=np.uint8)
        raise ValueError("vmax must be greater than vmin")

    scaled = (image - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    return np.floor(scaled * (levels - 1)).astype(np.uint8)


def compute_glcm(image, levels=32, distances=DEFAULT_DISTANCES,
                  angles=DEFAULT_ANGLES, symmetric=True, normed=True,
                  already_quantized=False, vmin=None, vmax=None):
    """Quantize (unless already done) and compute the GLCM.

    Returns
    -------
    glcm : numpy.ndarray, shape (levels, levels, n_distances, n_angles)
        The raw normalized co-occurrence matrix/matrices, exactly as
        returned by skimage.feature.graycomatrix. Kept available so you
        can add further features beyond the core 7 later, using the same
        matrix (see the Hall-Beyer GLCM tutorial for formulas).
    """
    image = np.asarray(image)
    if not already_quantized:
        image = quantize(image, levels=levels, vmin=vmin, vmax=vmax)
    return graycomatrix(
        image, distances=list(distances), angles=list(angles),
        levels=levels, symmetric=symmetric, normed=normed,
    )


def _entropy(P):
    p = P[P > 0]
    return float(-np.sum(p * np.log2(p)))


def _variance(P):
    """GLCM variance (a.k.a. sum of squares), Hall-Beyer convention."""
    levels = P.shape[0]
    i = np.arange(levels)
    marg = P.sum(axis=1)  # p_i, marginal over rows
    mu = np.sum(i * marg)
    return float(np.sum(((i - mu) ** 2) * marg))


def _sum_average(P):
    levels = P.shape[0]
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    k = i + j
    kvals = np.arange(0, 2 * levels - 1)
    p_xplusy = np.array([P[k == kv].sum() for kv in kvals])
    return float(np.sum(kvals * p_xplusy))


_CUSTOM_FEATURES = {
    "entropy": _entropy,
    "variance": _variance,
    "sum_average": _sum_average,
}


def glcm_features(image, levels=32, distances=DEFAULT_DISTANCES,
                   angles=DEFAULT_ANGLES, features=CORE_FEATURES,
                   vmin=None, vmax=None, per_angle=False):
    """Compute the core GLCM texture features for an image window.

    Parameters
    ----------
    image : array_like
        2D array (e.g. a single-band raster window, such as an 18x18 m
        forest-inventory plot clipped from an orthomosaic).
    levels : int
        Number of grey levels to quantize to before building the GLCM.
    distances : sequence of int
    angles : sequence of float
        Offsets/angles (radians) used to build the GLCM. Features are
        averaged across all given angles by default (rotation-invariant),
        matching common practice for vegetation/canopy texture where
        there's no dominant orientation.
    features : sequence of str
        Which features to compute; defaults to the 7-feature core set.
    vmin, vmax : float, optional
        Fixed quantization range - see `quantize`. Strongly recommended
        when comparing features across multiple windows/plots.
    per_angle : bool
        If True, return each feature as an array over (distance, angle)
        instead of averaging across angles.

    Returns
    -------
    dict mapping feature name -> float (or ndarray if per_angle=True)
    """
    glcm = compute_glcm(image, levels=levels, distances=distances,
                         angles=angles, vmin=vmin, vmax=vmax)

    results = {}
    for name in features:
        if name in _SKIMAGE_NATIVE:
            values = graycoprops(glcm, name)  # shape (n_distances, n_angles)
        elif name in _CUSTOM_FEATURES:
            fn = _CUSTOM_FEATURES[name]
            n_d, n_a = glcm.shape[2], glcm.shape[3]
            values = np.array([[fn(glcm[:, :, d, a]) for a in range(n_a)]
                                for d in range(n_d)])
        else:
            raise ValueError(
                f"Unknown feature '{name}'. Available: "
                f"{sorted(_SKIMAGE_NATIVE | set(_CUSTOM_FEATURES))}"
            )
        results[name] = values if per_angle else float(np.mean(values))

    return results
