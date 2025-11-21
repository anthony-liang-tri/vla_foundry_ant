"""
Libero installer module that fixes the LIBERO installation on import.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def get_libero_install_dir():
    """Get the directory where LIBERO should be installed (inside libero_wrapper)."""
    # Install LIBERO inside the libero_wrapper package directory
    return Path(__file__).parent / "_libero_install"


def install_libero():
    """Install LIBERO properly by cloning and copying files."""
    install_dir = get_libero_install_dir()
    libero_path = install_dir / "libero"

    # Check if already installed
    if libero_path.exists() and (libero_path / "libero" / "__init__.py").exists():
        # print("✓ LIBERO already installed")
        return True

    print("Installing LIBERO from GitHub...")

    # Install the broken package first to get dependencies
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "git+https://github.com/sedrick-keh-tri/LIBERO.git"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Warning: pip install failed: {result.stderr}")

    # Now fix the installation by copying the actual package
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "LIBERO"

        print("  Cloning LIBERO repository...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/sedrick-keh-tri/LIBERO.git", str(repo_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"✗ Failed to clone LIBERO: {result.stderr}")
            return False

        # Copy the libero package to the wrapper's install directory
        src_libero = repo_path / "libero"

        if src_libero.exists():
            print(f"  Copying libero to {libero_path}...")
            # Ensure install directory exists
            install_dir.mkdir(parents=True, exist_ok=True)
            if libero_path.exists():
                shutil.rmtree(libero_path)
            shutil.copytree(src_libero, libero_path)

            # Patch LIBERO to work with PyTorch 2.7+ (weights_only=True default)
            print("  Patching LIBERO for PyTorch 2.7+ compatibility...")
            patch_libero_torch_load(libero_path)

            print("✓ LIBERO installed successfully!")
            return True
        else:
            print("✗ Could not find libero source directory")
            return False


def patch_libero_torch_load(libero_path: Path):
    """Patch LIBERO's torch.load calls to use weights_only=False for PyTorch 2.6+."""
    benchmark_init = libero_path / "libero" / "benchmark" / "__init__.py"

    if not benchmark_init.exists():
        print("Warning: Could not find benchmark/__init__.py to patch")
        return

    content = benchmark_init.read_text()

    # Patch the torch.load call that's causing issues
    if "torch.load(init_states_path)" in content:
        content = content.replace("torch.load(init_states_path)", "torch.load(init_states_path, weights_only=False)")
        benchmark_init.write_text(content)
        print("  ✓ Patched torch.load in benchmark/__init__.py")


# Note: Auto-install is triggered from __init__.py when the package is imported
