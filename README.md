# Depth-Based Image Perspective Transformation App

This Python application uses depth estimation to apply transformations to an image in a 3D space. It employs Gradio to provide an interactive web interface where users can adjust camera properties such as field of view and rotation angles to see real-time changes in the image perspective.

## Features

- **Depth Estimation**: Automatically estimate the depth map of an image using DepthAnything.
- **3D Transformation**: Adjust field of view, rotation, and translation parameters to warp the image based on depth.
- **Interactive Interface**: Control parameters with Gradio sliders and see real-time transformation results.

## Requirements

- Python 3.8 or higher
- Conda (recommended for environment management)

### Install Dependencies

To install the required packages, create a new Conda environment and install dependencies:

```bash
conda create -n depth_transform python=3.9
conda activate depth_transform
pip install requirements.txt
```

## Usage

### 1. Run the App

Run the application from the terminal, specifying the path to an input image and optionally a depth map. If no depth map is provided, the app will generate one using the `estimate_depth` module.

#### If depthmap is already available
```bash
python app.py "/path/to/image.jpg" "path/to/depthmap.npy"
```

#### If depthmap is not available and needs to be estimated
```bash
python app.py "/path/to/image.jpg" ""
```

### 2. Interface Controls

The Gradio interface provides sliders to control the following parameters:

- **Field of View (FOV)**: Changes the camera’s field of view, allowing you to zoom in or out.
- **Rx, Ry, Rz**: Rotate the image along the x, y, and z axes to adjust the viewing angle.
- **Tx, Ty, Tz**: Translate the image along the x, y, and z axes for fine positioning.

### Example

```bash
python app.py images/shocked.jpg
```

Adjust the sliders in the web interface to explore different views of the transformed image.

## Files Overview

- `app.py`: Main application script.
- `utils.py`: Utility functions for image processing and transformations.
- `estimate_depth.py`: Provides depth estimation functionality.

## Acknowledgments

This project uses PyTorch for depth map handling and image transformations. The interactive GUI is powered by Gradio, making it easy to visualize transformations directly from a web interface.