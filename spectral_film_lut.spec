# spectral_film_lut.spec

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_all
import colour.characterisation.datasets.rawtoaces as rawtoaces
import os

block_cipher = None

# ---- metadata ----
datas = []
datas += copy_metadata("imageio")
datas += copy_metadata("networkx")
datas += copy_metadata("numpy")
datas += copy_metadata("colour-science")

# ---- resources ----
datas_pkg, _, _ += collect_all("spectral_film_lut")
datas += datas_pkg

# rawtoaces dataset
rawtoaces_path = rawtoaces.__path__[0]
datas.append((rawtoaces_path, "colour/characterisation/datasets/rawtoaces"))

# ---- hidden imports ----
hiddenimports = []
hiddenimports += collect_submodules("imageio")
hiddenimports += collect_submodules("networkx")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("colour")

# ---- entry script ----
entry_script = "src/spectral_film_lut/gui.py"

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
    upx=False,
    console=False,
    icon=None,
)
