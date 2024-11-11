import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import open3d as o3d
import os
import torch.nn.functional as F

def depthmap_to_transformation_mapping(depth_map, K_source, K_target, R, t):
    height, width = depth_map.shape
    
    Y, X = torch.meshgrid(torch.arange(height, device='cuda'), torch.arange(width, device='cuda'), indexing='ij')

    X = (X - K_source[0, 2]) * depth_map / K_source[0, 0]
    Y = (Y - K_source[1, 2]) * depth_map / K_source[1, 1]
    
    ones = torch.ones_like(X, dtype=torch.float32, device='cuda')
    
    pixel_coords = torch.stack((X.to(torch.float32), Y.to(torch.float32), depth_map.to(torch.float32), ones), dim=2)
    pixel_coords = pixel_coords.reshape(-1, 4).T

    extrinsic_matrix = torch.cat((R.to(torch.float32), t.reshape(-1, 1).to(torch.float32)), dim=1)

    projected_coords = torch.matmul(K_target.to(torch.float32), torch.matmul(extrinsic_matrix, pixel_coords))
    projected_coords = projected_coords.reshape(3, height, width).permute(1, 2, 0)
    projected_coords[..., 0] /= projected_coords[..., 2]
    projected_coords[..., 1] /= projected_coords[..., 2]
    
    mapping = torch.stack((projected_coords[..., 0], projected_coords[..., 1]), dim=-1)

    
    return mapping



def invert_mapping(mapping, depth_map):
    h, w, _ = mapping.shape
    inv_mapping = torch.zeros((h, w, 2), dtype=torch.float32, device='cuda').long() 
    pos_x = torch.where(mapping[:, :, 0]!=0)

    
    sorted_indices = torch.argsort(depth_map[pos_x], descending=True)
    sorted_map_x = mapping[:, :, 0].flatten()[sorted_indices]
    sorted_map_y =  mapping[:, :, 1].flatten()[sorted_indices]

    sorted_map_x = torch.clamp(sorted_map_x, min=0, max=w-1).long()
    sorted_map_y = torch.clamp(sorted_map_y, min=0, max=h-1).long()
    new_pos = (sorted_map_y, sorted_map_x)
    inv_mapping[:,:,0][new_pos] = pos_x[1][sorted_indices]
    inv_mapping[:,:,1][new_pos] = pos_x[0][sorted_indices]
    # mask = (inv_mapping[:,:,0]!=0)*(inv_mapping[:,:,1]!=0)
    return inv_mapping.float()

def warp(image, mapping):
    H, W, C = image.size()
    mapping[..., 0] = 2.0 * mapping[..., 0].clone() / max(W-1, 1) - 1.0
    mapping[..., 1] = 2.0 * mapping[..., 1].clone() / max(H-1, 1) - 1.0
    mapping = torch.unsqueeze(mapping, 0)
    image = torch.unsqueeze(image, 0).to(torch.float32)
    image = image.permute(0,3,1,2)
    image = image/255
    if float(torch.__version__[:3]) >= 1.3:
        output = nn.functional.grid_sample(image, mapping, align_corners=True)
    else:
        output = nn.functional.grid_sample(image, mapping)
    return output.permute(0,2,3,1)*255


def max_filter(image_tensor, kernel_size=2):
    image_tensor = image_tensor.permute(2,0,1)
    padding = kernel_size // 2
    return nn.functional.max_pool2d(image_tensor.unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze(0).permute(1,2,0)

def remove_stray_pixels_morphology(image, n=1):
    image = image.permute(2, 0, 1)  
    processed_channels = []
    
    kernel = torch.ones((1, 1, n, n), dtype=torch.float32, device=image.device)
    for channel in range(2):
        
        binary_channel = (image[channel] != 0).float()
        
        eroded = F.conv2d(binary_channel.unsqueeze(0).unsqueeze(0), kernel, padding=n//2) == (n * n)
        dilated = F.conv2d(eroded.float(), kernel, padding=n//2) > 0
     
        processed_channel = image[channel] * dilated.squeeze().float()
        processed_channels.append(processed_channel)
    
    return torch.stack(processed_channels).permute(1, 2, 0)

def transform_image(image, depth,  K_source, K_target, R, t):
    mapping = depthmap_to_transformation_mapping(depth,   K_source, K_target, R, t)
    inverted_mapping = invert_mapping(mapping, depth)
    inverted_mapping = max_filter(inverted_mapping)
    inverted_mapping = remove_stray_pixels_morphology(image=inverted_mapping, n=1)
    return warp(image, inverted_mapping).to(torch.uint8)

def create_intrinsic_matrix(height, width, fov_x, fov_y, device='cpu'):
    fov_x = torch.tensor(fov_x, device=device) * torch.pi / 180
    fov_y = torch.tensor(fov_y, device=device) * torch.pi / 180

    f_x = width / (2 * torch.tan(fov_x / 2))
    f_y = height / (2 * torch.tan(fov_y / 2))

    c_x = width / 2
    c_y = height / 2

    intrinsic_matrix = torch.zeros((3, 3), device=device, dtype=torch.float32)

    intrinsic_matrix[0, 0] = f_x
    intrinsic_matrix[0, 2] = c_x
    intrinsic_matrix[1, 1] = f_y
    intrinsic_matrix[1, 2] = c_y
    intrinsic_matrix[2, 2] = 1

    return intrinsic_matrix


def depth_to_point_cloud(depth_image, K, scale=1.0):
    
    height, width = depth_image.shape
    fx, fy = K[0, 0], K[1, 1] 
    cx, cy = K[0, 2], K[1, 2]  

    x_indices = np.linspace(0, width - 1, width)
    y_indices = np.linspace(0, height - 1, height)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

    z = depth_image / scale
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    valid_points = points[depth_image.flatten() > 0]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(valid_points)

    return point_cloud


def save_rgb_image_as_jpg(rgb_image, folder_path, filename):
    
    os.makedirs(folder_path, exist_ok=True)

    full_path = os.path.join(folder_path, filename)

    image = Image.fromarray(rgb_image)

    image.save(full_path, format='JPEG')

    return full_path

if __name__ == "__main__":
    image = np.array(Image.open("img_taj.jpg"))
    image = cv2.resize(image, (image.shape[1]//3, image.shape[0]//3))
    height, width, c = image.shape
    depth_t = torch.ones((image.shape[0], image.shape[1]), device= 'cuda', dtype=torch.float32)
    image_t = torch.from_numpy(image).to(torch.uint8).cuda()

    K = create_intrinsic_matrix(height, width, 100.0, 100.0, device= 'cuda')
    R, _ = cv2.Rodrigues(np.deg2rad(np.array([10.0, 3.0, 10.0])))
    R = torch.from_numpy(R).to(torch.float32).cuda()
    t = torch.from_numpy(np.array([0.2, 0, 0.0])).cuda()
    with torch.no_grad():
        for i in range(100):
            s = time.time()
            trasformed = transform_image(image_t, depth_t, K, R, t)
            print(time.time() - s)
    plt.imshow(trasformed[0].cpu())
    plt.show()