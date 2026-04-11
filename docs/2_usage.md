
# Usage

1. Select a sample image which is in the intended input colorspace.
2. Adjust the preview with the parameters on the sidebar.
3. Once satisfied, export the LUT.

- Hovering over a setting should explain what it does, and double clicking the label resets to the default value.
- When simulating slide film, set the print stock to None or an appropriate reversal print medium.
- You can also export the LUT in a negative and print stage. In your image processing pipeline this gives you the option to add grain to the negative, and use printer lights controls.

## Resolve Node Tree

If you want to use the Grain LUT, the following node tree is recommended.
It is important to set the correct offset of -43, so that the grain does not alter overall brightness.
Import the grain overlay as a matte from the media pool tab.

It is important that all LUTs where exported with the same ADX d-max setting, and that the grain and print LUTs where made
for that specific negative film stock.
The grain overlay is generated independent of film stock selection, and only affected by the simulated frame size.

To manually alter the grain intensity, change the gain on the unlabeled node between the multiplicative and the additive layer mixers.
Printer light controlls should be done after the additive mixer and before the print LUT. Halation should be added before the negative LUT.

<img width="100%" alt="Resolve Node Tree" src="https://github.com/user-attachments/assets/2fd2acee-675f-430f-8775-22038241c66e" />

## Filmstock Selector

When clicking on the magnifying glass a window opens to search and browse through the
available film stocks.
<img width="100%" alt="Film stock selection ui" src="https://github.com/user-attachments/assets/5af71ab0-3802-4d22-b9e7-6f9e09efc7c4" />
