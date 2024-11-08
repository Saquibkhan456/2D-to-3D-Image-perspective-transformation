import numpy as np
import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
from utils import *
import estimate_depth
from tkinter import Scale, Label, Frame
from PIL import Image, ImageTk
import argparse


# Set up argparse to take the image path from the terminal
parser = argparse.ArgumentParser(description="Run image transformation with depth estimation")
parser.add_argument("image_path", type=str, help="Path to the input image file.")
parser.add_argument("depth", type=str, help="Path to the depthmap.")
args = parser.parse_args()

image = Image.open(args.image_path)
if args.depth == "":
    depth = estimate_depth.estimate_depth(image)
else:
    depth = np.array(Image.open(args.depth))
    depth = depth / depth.max()
depth = torch.from_numpy(depth.astype(np.float32)).cuda()
image = np.array(image)
height, width, c = image.shape
aspect = height / width

image_t = torch.from_numpy(image).to(torch.uint8).cuda()

# Camera parameters
fov_h_source = 50.0
K_source = create_intrinsic_matrix(height, width, fov_h_source, fov_h_source * aspect, device='cuda')

# Define the transformation function
def run_transformation(fov_h_target, rx, ry, rz, tx, ty, tz):
    t = torch.from_numpy(np.array([tx, ty, tz]) + 1e-6).cuda()
    R, _ = cv2.Rodrigues(np.deg2rad(np.array([rx, ry, rz])))
    R = torch.from_numpy(R).to(torch.float16).cuda()
    K_target = create_intrinsic_matrix(height, width, fov_h_target, fov_h_target * aspect, device='cuda')
    mapping = depthmap_to_transformation_mapping(depth, K_source, K_target, R, t)
    inverted_mapping = invert_mapping(mapping, depth)
    inverted_mapping = max_filter(inverted_mapping)
    return warp(image_t, inverted_mapping).to(torch.uint8)[0].cpu().numpy()

class ImageUpdaterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Transformation Tool")

        # Styling options
        self.root.configure(bg="#2e3b4e")
        self.slider_bg = "#40546e"
        self.font = ("Arial", 10, "bold")

        # Header
        header = Label(root, text="Image Transformation Tool", font=("Arial", 14, "bold"), 
                       bg="#2e3b4e", fg="white", pady=10)
        header.pack(fill="x")

        # Set up the main layout frames
        main_frame = Frame(root, bg="#2e3b4e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.image_frame = Frame(main_frame, bg="#2e3b4e")
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.slider_frame = Frame(main_frame, bg=self.slider_bg, padx=10, pady=10)
        self.slider_frame.grid(row=0, column=1, sticky="nsew")

        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Label to display the image
        self.image_label = Label(self.image_frame, bg="#2e3b4e")
        self.image_label.pack(fill="both", expand=True)

        # Initialize slider variables
        self.vals = [tk.DoubleVar() for _ in range(7)]

        # Slider configuration
        slider_configs = [
            {"label": "FoV", "from_": 0, "to": 100, "default": 50},
            {"label": "Translation X", "from_": -0.5, "to": 0.5, "resolution": 0.001, "default": 0},
            {"label": "Translation Y", "from_": -0.5, "to": 0.5, "resolution": 0.001, "default": 0},
            {"label": "Translation Z", "from_": -0.5, "to": 0.5, "resolution": 0.001, "default": 0},
            {"label": "Rotation X", "from_": -50, "to": 50, "default": 0},
            {"label": "Rotation Y", "from_": -50, "to": 50, "default": 0},
            {"label": "Rotation Z", "from_": -50, "to": 50, "default": 0}
        ]

        # Create sliders in the right frame
        for i, config in enumerate(slider_configs):
            scale = Scale(self.slider_frame, from_=config["from_"], to=config["to"], 
                          orient="horizontal", label=config["label"], variable=self.vals[i], 
                          resolution=config.get("resolution", 1), command=self.update_image,
                          font=self.font, bg=self.slider_bg, fg="white", troughcolor="#607a9b",
                          highlightthickness=0)
            scale.set(config["default"])
            scale.pack(fill="x", pady=5)

        # Status bar to show current slider values
        self.status = Label(root, text="", font=self.font, bg="#2e3b4e", fg="white")
        self.status.pack(fill="x")

        # Initial image update
        self.update_image()

    # Method to update the image based on slider values
    def update_image(self, event=None):
        # Get current slider values
        values = [val.get() for val in self.vals]
        self.status.config(text=f"FoV: {values[0]:.2f}, RotX: {values[1]:.2f}, RotY: {values[2]:.2f}, "
                                f"RotZ: {values[3]:.2f}, TransX: {values[4]:.3f}, TransY: {values[5]:.3f}, "
                                f"TransZ: {values[6]:.3f}")

        # Get transformed image as a NumPy array
        array = run_transformation(*values)
        
        # Convert NumPy array to Tkinter-compatible image
        original_image = Image.fromarray(array.astype('uint8'))

        # Resize the image to fit within a max dimension, keeping aspect ratio
        max_width, max_height = 1280, 720
        original_image.thumbnail((max_width, max_height), Image.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(original_image)

        # Update the image display
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

# Run the app
root = tk.Tk()
app = ImageUpdaterApp(root)
root.mainloop()