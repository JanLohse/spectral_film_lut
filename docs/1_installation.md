---
icon: lucide/download
tags:
  - Guide
  - Setup
---

# Installation

=== "Windows"
    Download the latest `.exe` from the [releases](https://github.com/JanLohse/spectral_film_lut/releases) page and run it.

    Alternatively, install via Python (see [below](#python-package)).

=== "Linux"
    Download the `.AppImage` from the [releases](https://github.com/JanLohse/spectral_film_lut/releases) page and make it executable:

    ```bash
    chmod +x spectral_film_lut-{version}.AppImage
    ./spectral_film_lut-{version}.AppImage
    ```

    Alternatively, install via Python (see [below](#python-package)).

=== "macOS"
    There is currently no native binary available for macOS.
    Install and run the application using a Python-based method.
    See the [Python Package](#python-package) section below.

## Python Package

Install the application using your preferred Python package manager.

=== "pip"
    Installs the package into the current Python environment:

    ```bash
    pip install spectral_film_lut
    ```

=== "pipx"
    Recommended for installing standalone applications globally in an isolated environment:

    ```bash
    pipx install spectral_film_lut
    ```

=== "uv"
    Install the application as an isolated tool:

    ```bash
    uv tool install spectral_film_lut
    ```

    Alternatively, run it directly:

    ```bash
    uvx spectral_film_lut
    ```

After installation, run the application:

```bash
spectral_film_lut
```

## Legacy CUDA support

CUDA support has been removed in the current versions. The current pipeline and LUT
application is much faster than it was in the past. To keep the code more streamlined
we have thus removed the CUDA support.

A legacy CUDA branch has been added. Importantly it is also far out of date in many
regards. When using it additionally installing [CuPy package](https://cupy.dev) is
necessary to activate the GPU functionality. To disable CUDA on an installation with
CUDA capabilities, use the argument `--no-cuda`.
