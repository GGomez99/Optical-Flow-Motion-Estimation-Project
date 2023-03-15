import glob
import torch
import os
from PIL import Image
import torchvision.transforms.functional as TF

import torch.nn.functional as F

if torch.cuda.is_available:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def load_images_from_folder(folder_path):
    """
    Loads images found in the given folder.

    Args:
        folder_path (str) : Images root folder path
    """

    exts = ('*.bmp', '*.png')

    images = []

    for ext in exts:
        for img_file in glob.glob(os.path.join(folder_path, ext)):
            images.append(TF.to_tensor(Image.open(img_file)))

    n_channels, h, w = images[0].shape # 3, w, h for RGB

    # Stack everything into one tensor
    output = torch.cat(images).to(DEVICE).reshape(-1, n_channels, h, w)
    return TF.rgb_to_grayscale(output)

def conv2d(img, kernel):
    """
    Args:
        img (torch.Tensor) : (n_batch, n_channels, w, h).
        kernel (torch.Tensor) : (w_kernel, h_kernel), the same for every images.
    """

    n_images, n_channels, _, _ = img.shape
    return F.conv2d(img.permute(1, 0, 2, 3), kernel.repeat(n_images, n_channels, 1, 1), groups=n_images, padding='same').permute(1, 0, 2, 3)