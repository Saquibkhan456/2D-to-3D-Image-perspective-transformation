import numpy as np
import matplotlib.pyplot as plt
import cv2
import estimate_depth
import torch
from PIL import Image
from utils import create_intrinsic_matrix, transform_image, save_rgb_image_as_jpg
from video_utils import make_video_from_image_paths


image = Image.open("images/shocked.jpg")
depth = estimate_depth.estimate_depth(image)
image = np.array(image)
save_folder = "image_frames"
video_frames = []

height, width, c = image.shape
aspect = height/width
image_t = torch.from_numpy(image).to(torch.uint8).cuda()
depth = torch.from_numpy(depth).to(torch.float32).cuda()


fov_h_source = 50.0

K_source = create_intrinsic_matrix(height, width, fov_h_source, fov_h_source*aspect, device= 'cuda')

R, _ = cv2.Rodrigues(np.deg2rad(np.array([0.0, 0.0, 0.0])))
R = torch.from_numpy(R).to(torch.float16).cuda()

num_frames = 60
fov_start = 50
fov_end = 68

t_start = 0 
t_end = 0.15

with torch.no_grad():
    fov_h_target = np.arange(fov_start, fov_end, (fov_end-fov_start)/num_frames) 
    t_z = np.arange(t_start, t_end, (t_end-t_start)/num_frames)
    for i in range(60):
        K_target = create_intrinsic_matrix(height, width, fov_h_target[i], fov_h_target[i]*aspect, device= 'cuda')
        t = torch.from_numpy(np.array([0.0, 0.0, -t_z[i]])+1e-6).cuda()
        trasformed = transform_image(image_t, depth, K_source, K_target, R, t)
        image_out = trasformed[0].cpu().numpy()
        # video_frames.append(image_out)
        save_rgb_image_as_jpg(trasformed[0].cpu().numpy(), save_folder, f"{i:03d}.jpg")
        print(i)

