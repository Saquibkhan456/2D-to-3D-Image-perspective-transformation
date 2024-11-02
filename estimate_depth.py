from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def estimate_depth(image, model_choice = "depth-anything/Depth-Anything-V2-Small-hf"):
    pipe = pipeline(task="depth-estimation", model=model_choice, device='cuda')
    depth = np.array(pipe(image)["depth"])
    depth = depth / depth.max()
    depth = (((1 - depth) * 1.0) + 0.3)
    return depth

if __name__ == "__main__":
    image_path = "images/shocked.jpg"
    image = Image.open(image_path)
    depth = estimate_depth(image)
    plt.imshow(depth)
    plt.show()