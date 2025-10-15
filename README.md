# Spectral Film LUT

Spectral Film LUT is a GUI application made to generate LUT files for film emulation in video editing.  

To emulate the look of a film stock its datasheet was digitized and a multi-step color pipeline simulates its reaction to light to the final appearance of the print material.  

- A wide variety of negative, print, and slide materials are available.  
- Options include still photography, motion picture, color, black and white, Kodak, Fuji, contemporary, and vintage.  
- The accuracy is limited by the precision of the published data, which can be especially poor for long discontinued formats, and simplified assumptions in the color pipeline.  
- There is especially little data available about the inter-layer interaction, e.g., how aggressive the color masking couplers in a negative film are.  
<img width="1087" height="837" alt="main gui" src="https://github.com/user-attachments/assets/2eb00673-ac03-4fc3-a877-31e02372211b#gh-light-mode-only" />
<img width="1072" height="823" alt="main gui" src="https://github.com/user-attachments/assets/9a4b7a74-5d87-487a-8bd5-1ea7361384fc#gh-dark-mode-only" />

## Installation

### Windows
The easiest way to run Spectral Film LUT is to download the latest `.exe` from the [releases](../../releases) section.  

### Python Package
You can also install the program using pip:  

```bash
pip install git+https://github.com/JanLohse/spectral_film_lut
```
Then run with `spectral_film_lut`.

This should also work on other operating systems, even if it has not yet been tested.

If CuPy
 has been installed, CUDA is used to display the preview, which results in a more responsive UI.

## Usage

- Select a sample image which is in the intended input colorspace.

- Adjust the preview with the parameters on the sidebar.

- Once satisfied, export the LUT.

Be sure to set the correct LUT size.
You can also export the LUT in a negative and print stage.
In your image editing pipeline this gives you the option to add grain to the negative stage.

### Filmstock Selector
When clicking on the magnifying glass a window opens to search and browse through the available film stocks.
<img width="848" height="597" alt="filmstock selector" src="https://github.com/user-attachments/assets/3886eea0-f21f-43de-a319-f84105daae50#gh-dark-mode-only" />
<img width="833" height="595" alt="filmstock selector" src="https://github.com/user-attachments/assets/54b82d41-9386-49e6-9d3e-b389e8f1585c#gh-light-mode-only" />
