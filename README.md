# Spectral Film LUT

Spectral Film LUT is a GUI application made to generate LUT files for film emulation in video editing.  

To emulate the look of a film stock its datasheet was digitized and a multi-step color pipeline simulates its reaction to light to the final appearance of the print material.  

- A wide variety of negative, print, and slide materials are available.  
- Options include still photography, motion picture, color, black and white, Kodak, Fuji, contemporary, and vintage.  
- The accuracy is limited by the precision of the published data, which can be especially poor for long discontinued formats, and simplified assumptions in the color pipeline.  
- There is especially little data available about the inter-layer interaction, e.g., how aggressive the color masking couplers in a negative film are.  

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

### Main GUI
<img width="1026" height="606" alt="main gui" src="https://github.com/user-attachments/assets/d4117797-6140-45d9-a35b-6337fdd9c403" />

### Filmstock Selector
When clicking on the magnifying glass a window opens to search and browse through the available film stocks.

<img width="866" height="585" alt="filmstock selector" src="https://github.com/user-attachments/assets/1a651ba2-cd53-4484-92ef-ed4cbcc0971b" /> 
