# glcmfeatures

Core GLCM texture features, built on `scikit-image`.

Computes the 7-feature subset identified by Hall-Beyer (2017, https://doi.org/10.1080/01431161.2016.1278314)
as carrying most of the discriminating power in the full 21-feature Haralick/GLCM
catalogue, validated across window sizes (5-25 px) that overlap typical
forest-inventory plots at sub-metre aerial resolution (e.g. 18x18 m at 0.5-1 m
gives 18-36 px windows):

`contrast`, `correlation`, `homogeneity`, `ASM`, `variance`, `sum_average`, `entropy`

## Install

From the package directory:

```bash
pip install .
```

Or, for local development (edits take effect without reinstalling):

```bash
pip install -e .
```

You can also install it directly from a git repo in any future project, the
same way you'd `remotes::install_github()` in R:

```bash
pip install git+https://github.com/balazsan/glcmfeatures.git
```

## Usage

```python
from glcmfeatures import glcm_features

# window: a 2D numpy array, e.g. a single band clipped to your 18x18m plot
feats = glcm_features(window, levels=32)
# {'contrast': ..., 'correlation': ..., 'homogeneity': ..., 'ASM': ...,
#  'variance': ..., 'sum_average': ..., 'entropy': ...}
```

### Comparing features across multiple windows/plots

Pass a fixed `vmin`/`vmax` so all windows are quantized against the same
intensity range (otherwise each window is scaled to its own min/max, and
features stop being comparable across plots):

```python
feats = glcm_features(window, levels=32, vmin=raster_min, vmax=raster_max)
```

### Adding more features later

`compute_glcm()` returns the raw normalized co-occurrence matrix (shape
`levels x levels x n_distances x n_angles`), so you can implement any of
the other Haralick/GLCM features (e.g. cluster shade, IDMN, max
probability) the same way `variance`/`sum_average`/`entropy` are
implemented in `core.py` — index into `glcm[:, :, d, a]` and apply the
formula from the Hall-Beyer GLCM tutorial.

```python
from glcmfeatures import compute_glcm

glcm = compute_glcm(window, levels=32)
```
