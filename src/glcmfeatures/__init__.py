"""glcmfeatures: GLCM texture features on top of scikit-image.

Implements the 7-feature core set recommended by Hall-Beyer (2017) for
window sizes in the tens-of-pixels range, plus utilities (quantize,
compute_glcm) for adding further features from the fuller Haralick set
yourself later.
"""

from .core import (
    CORE_FEATURES,
    DEFAULT_ANGLES,
    DEFAULT_DISTANCES,
    compute_glcm,
    glcm_features,
    quantize,
)

__all__ = [
    "CORE_FEATURES",
    "DEFAULT_ANGLES",
    "DEFAULT_DISTANCES",
    "compute_glcm",
    "glcm_features",
    "quantize",
]

__version__ = "0.1.0"
