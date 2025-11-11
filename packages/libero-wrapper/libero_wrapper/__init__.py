"""
LIBERO wrapper package that handles automatic installation of LIBERO.
"""
import sys
from pathlib import Path

from .installer import get_libero_install_dir, install_libero

# Add the LIBERO install directory to sys.path so it can be imported
_libero_install_dir = get_libero_install_dir()
if _libero_install_dir.exists() and str(_libero_install_dir) not in sys.path:
    sys.path.insert(0, str(_libero_install_dir))

# Auto-install LIBERO on package import (only happens when --group libero is used)
try:
    # Check if libero is importable
    import libero.libero  # noqa: F401
except (ImportError, ModuleNotFoundError):
    # If not, install it automatically
    install_libero()
    # Add to path after installation
    if _libero_install_dir.exists() and str(_libero_install_dir) not in sys.path:
        sys.path.insert(0, str(_libero_install_dir))

__all__ = ["install_libero"]

