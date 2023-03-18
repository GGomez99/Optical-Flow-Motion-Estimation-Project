import glob
import torch
import os
import re
from PIL import Image
import torchvision.transforms.functional as TF

import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, grayscale=False):
    """
    Loads one image into a torch.Tensor
    """

    if grayscale:
        return TF.rgb_to_grayscale(TF.to_tensor(Image.open(image_path)))
    else:
        return TF.to_tensor(Image.open(image_path))
    

def load_images_from_folder(folder_path, with_grayscale=False):
    """
    Loads images found in the given folder.

    Args:
        folder_path (str) : Images root folder path
    """

    exts = ('*.bmp', '*.png')

    images = []

    for ext in exts:
        img_paths = sorted(list(glob.glob(os.path.join(folder_path, ext))), key=lambda s: int(re.search(r'\d+', s).group()))
        for img_file in img_paths:
            images.append(TF.to_tensor(Image.open(img_file)))

    n_channels, h, w = images[0].shape # 3, w, h for RGB

    # Stack everything into one tensor
    output = torch.cat(images).to(DEVICE).reshape(-1, n_channels, h, w)

    if with_grayscale:
        return output, TF.rgb_to_grayscale(output)
    else:
        return output
        

def conv2d(img, kernel):
    """
    Args:
        img (torch.Tensor) : (n_batch, n_channels, w, h).
        kernel (torch.Tensor) : (w_kernel, h_kernel), the same for every images.
    """

    n_images, n_channels, _, _ = img.shape
    return F.conv2d(img.permute(1, 0, 2, 3), kernel.repeat(n_images, n_channels, 1, 1), groups=n_images, padding='same').permute(1, 0, 2, 3)