import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from utils import *
import gradio as gr
import torch
from PIL import Image
import estimate_depth


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
    depth = depth/depth.max()
depth = torch.from_numpy(depth.astype(np.float32)).cuda()
image = np.array(image)
height, width, c = image.shape
aspect = height / width

image_t = torch.from_numpy(image).to(torch.uint8).cuda()

fov_h_source = 50.0
K_source = create_intrinsic_matrix(height, width, fov_h_source, fov_h_source * aspect, device='cuda')

def run_tranformation(fov_h_target, rx, ry, rz, tx, ty, tz):
    t = torch.from_numpy(np.array([tx, ty, tz]) + 1e-6).cuda()
    R, _ = cv2.Rodrigues(np.deg2rad(np.array([rx, ry, rz])))
    R = torch.from_numpy(R).to(torch.float16).cuda()
    K_target = create_intrinsic_matrix(height, width, fov_h_target, fov_h_target * aspect, device='cuda')
    mapping = depthmap_to_transformation_mapping(depth, K_source, K_target, R, t)
    inverted_mapping = invert_mapping(mapping, depth)
    inverted_mapping = max_filter(inverted_mapping)
    return warp(image_t, inverted_mapping).to(torch.uint8)[0].cpu().numpy()

# Create the Gradio interface
iface = gr.Interface(
    fn=run_tranformation,
    inputs=[
        gr.Slider(minimum=1, maximum=180, value=fov_h_source, label="Field of View (FOV)"),
        gr.Slider(minimum=-60, maximum=60, value=0, label="Rx"),
        gr.Slider(minimum=-60, maximum=60, value=0, label="Ry"),
        gr.Slider(minimum=-60, maximum=60, value=0, label="Rz"),
        gr.Slider(minimum=-0.2, maximum=0.2, value=0, label="Tx"),
        gr.Slider(minimum=-0.2, maximum=0.2, value=0, label="Ty"),
        gr.Slider(minimum=-0.2, maximum=0.2, value=0, label="Tz"),
    ],
    outputs=gr.Image(type="numpy", label="Processed Image"),
    live=True
)

# Launch the app
iface.launch()