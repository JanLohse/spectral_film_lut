---
icon: lucide/film
tags:
  - Guide
---


# Spectral Film LUT

[![PyPI version](https://img.shields.io/pypi/v/spectral-film-lut)](https://pypi.org/project/spectral-film-lut/)
[![GitHub](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/JanLohse/spectral_film_lut)
[![CI](https://github.com/JanLohse/spectral_film_lut/actions/workflows/python-app.yml/badge.svg)](https://github.com/JanLohse/spectral_film_lut/actions/workflows/python-app.yml)
[![Version](https://img.shields.io/github/v/release/JanLohse/spectral_film_lut)](https://github.com/JanLohse/spectral_film_lut/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/JanLohse/spectral_film_lut?tab=MIT-1-ov-file#readme)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)

Spectral Film LUT is a GUI application made to generate LUT files for film emulation in
video editing.

To emulate the look of a film stock, its datasheet was digitized and a multistep color
pipeline simulates its reaction to light to the final appearance of the print material.

- A wide variety of negative, print, and slide materials are available.
- Options include still photography, motion picture, color, black and white, Kodak,
  Fuji, contemporary, and vintage.
- The accuracy is limited by the precision of the published data, which can be
  especially poor for long discontinued formats. There is also undocumented behavior
  like interlayer effects, for which simplified assumptions have been made.
- There is especially little data available about the interlayer interaction, e.g., how
  aggressive the color masking couplers in a negative film are.

<img width="100%" alt="Spectral Film LUT main ui" src="https://github.com/user-attachments/assets/b7e31c74-d522-42dd-97e8-9374c9efa8ee" />
