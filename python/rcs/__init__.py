"""Robot control stack python bindings."""
from rcs._core import __version__, common
from rcs import camera, envs, hand, sim

# make submodules available
__all__ = ["__doc__", "__version__", "common", "sim", "camera", "envs", "hand"]
