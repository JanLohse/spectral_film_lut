# spectral_film_lut.spec

# IMPORTANT:
# Use this file with:
#     pyinstaller spectral_film_lut.spec

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# --- Collect package metadata for version lookups ---
datas = []
datas += copy_metadata("imageio")
datas += copy_metadata("networkx")
datas += copy_metadata("numpy")
datas += copy_metadata("colour-science")

# --- Include your resources ---
datas += collect_data_files("spectral_film_lut", includes=["resources/**"])

# --- rawtoaces ---
# (From colour-science package: dataset folder)
import colour.characterisation.datasets.rawtoaces as rawtoaces
import os
rawtoaces_path = rawtoaces.__path__[0]
datas.append((rawtoaces_path, "colour/characterisation/datasets/rawtoaces"))

# --- Collect hidden imports automatically ---
hiddenimports = []
hiddenimports += collect_submodules("imageio")
hiddenimports += collect_submodules("networkx")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("colour")

# --- Entry point ---
entry_script = "src/spectral_film_lut/gui.py"

# --- PyInstaller objects ---
block_cipher = None

a = Analysis(
    [entry_script],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="SpectralFilmLUT",
    debug=False,
    strip=False,
    UPX=False,
    console=False,
)

# Onefile bundle
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="SpectralFilmLUT",
)
