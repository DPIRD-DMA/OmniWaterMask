try:
    from ._version import __version__
except ImportError:
    # Source checkout without a build step (e.g. running tests directly from
    # the repo). setuptools-scm writes _version.py at build time.
    __version__ = "0.0.0+unknown"

from .water_inf_pipeline import make_water_mask, make_water_mask_debug

__all__ = [
    "make_water_mask",
    "make_water_mask_debug",
    "__version__",
]
