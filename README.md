# Spectral Film LUT

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://janlohse.github.io/spectral_film_lut/)
[![CI](https://github.com/JanLohse/spectral_film_lut/actions/workflows/python-app.yml/badge.svg)](https://github.com/JanLohse/spectral_film_lut/actions/workflows/python-app.yml)
[![Version](https://img.shields.io/github/v/release/JanLohse/spectral_film_lut)](https://github.com/JanLohse/spectral_film_lut/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/JanLohse/spectral_film_lut?tab=MIT-1-ov-file#readme)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)

Spectral Film LUT is a GUI application made to generate LUT files for film emulation in
video editing. For more details take a look at
the [documentation](https://janlohse.github.io/spectral_film_lut/).

To emulate the look of a film stock its datasheet was digitized and a multistep color
pipeline simulates its reaction to light to the final appearance of the print material.

- A wide variety of negative, print, and slide materials are available.
- Options include still photography, motion picture, color, black and white, Kodak,
  Fuji, contemporary, and vintage.
- The accuracy is limited by the precision of the published data, which can be
  especially poor for long discontinued formats, and simplified assumptions in the color
  pipeline.
- There is especially little data available about the interlayer interaction, e.g., how
  aggressive the color masking couplers in a negative film are.

<img width="100%" alt="Spectral Film LUT main ui" src="https://github.com/user-attachments/assets/b7e31c74-d522-42dd-97e8-9374c9efa8ee" />

## Installation

### Windows

The easiest way to run Spectral Film LUT is to download the latest `.exe` from
the [releases](../../releases) section.

### Linux

Download the `.AppImage` from the [releases](../../releases) section or install the
python package.

### macOS

Install UV using your favorite package manager beforehand.
Git clone to your desired local location

```bash
uv run spectral_film_lut
```

### Python Package

You can also install the program using pip (or pipx):

```bash
pip install git+https://github.com/JanLohse/spectral_film_lut
```

Then run with `spectral_film_lut`.

### CUDA support

For hardware acceleration we make us of CuPy. It might be removed in futur releases though.
There is not a relevant speed-up for generating LUTs, that justifies the added complexity in code.

Once CUDA support is added, a legacy CUDA branch will be added. To use CUDA pull that branch
and install using pip, and additionally install the CuPy package. 
To disable CUDA on an installation with CUDA capabilities, use the argument `--no-cuda`.

## Usage

1. Select a sample image which is in the intended input colorspace.
2. Adjust the preview with the parameters on the sidebar.
3. Once satisfied, export the LUT.

- Hovering over a setting should explain what it does, and double clicking the label resets to the default value.
- When simulating slide film, set the print stock to None or an appropriate reversal print medium.
- You can also export the LUT in a negative and print stage. In your image processing pipeline this gives you the option to add grain to the negative, and use printer lights controls.

### Resolve Node Tree

If you want to use the Grain LUT, the following node tree is recommended. 
It is important to set the correct offset of -43, so that the grain does not alter overall brightness.
Import the grain overlay as a matte from the media pool tab.

It is important that all LUTs where exported with the same ADX d-max setting, and that the grain and print LUTs where made
for that specific negative film stock.
The grain overlay is generated independent of film stock selection, and only affected by the simulated frame size.

To manually alter the grain intensity, change the gain on the unlabeled node between the multiplicative and the additive layer mixers.
Printer light controlls should be done after the additive mixer and before the print LUT. Halation should be added before the negative LUT.

<img width="100%" alt="Resolve Node Tree" src="https://github.com/user-attachments/assets/2fd2acee-675f-430f-8775-22038241c66e" />

### Filmstock Selector

When clicking on the magnifying, glass a window opens to search and browse through the
available film stocks.
<img width="100%" alt="Film stock selection ui" src="https://github.com/user-attachments/assets/5af71ab0-3802-4d22-b9e7-6f9e09efc7c4" />

