# Conv + Pool Visualizer

A lightweight, front-end only web app that illustrates how a convolutional layer combines with a pooling layer. No frameworks, just HTML/CSS/JS. Students can adjust only the number of kernels and input size (kernel is 3×3, pooling is 2×2 Max with stride 2).

## Quick start

You can open `index.html` directly in your browser, or serve the folder with any static server.

Option A: open the file

1. Double-click `index.html` (or right-click and open with your preferred browser).

Option B: use a simple static server (optional)

1. If you have Python installed:
	 - Python 3: `python3 -m http.server 8000`
	 - Then visit http://localhost:8000

## What you’ll see

- Diagram-only: outline grids focusing on shape changes (no in-canvas dimension labels).
- Input: An RGB input (C=3) drawn as stacked grids (H×W×3).
- Kernels: 3D kernels (3×3×3) drawn as stacked grids per kernel column.
- Outputs: Convolution and pooled outputs are single 2D maps per kernel.

## Controls

- Input size: Size of the input grid.
- Number of kernels: How many convolution kernels to visualize (each creates its own feature map).
- Kernel size: Fixed at 3×3.
- Convolution stride: Fixed at 1.
- Padding: None (valid).
 - Pooling type: Fixed at Max.
 - Pooling size: Fixed at 2×2; stride is fixed at 2.

## Notes

- The app is symbolic and educational: diagram emphasizes sizes and shapes; no numeric values are displayed.
- Everything runs in the browser; no external dependencies.

## Files

- `index.html` — Main page and UI structure.
- `styles.css` — Basic layout and visual styles.
- `script.js` — Logic for generating data, running convolution/pooling, and drawing grids.

## License

MIT