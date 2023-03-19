import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.image_utils import *

import warnings
warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def estimate_derivatives(img1, img2):
    n_images, n_channels, _, _ = img1.shape

    kernelX = torch.tensor([[-1, -1],[1, 1]], device=DEVICE)*.25  # kernel for computing d/dx
    kernelY = torch.tensor([[-1, 1],[ -1, 1]], device=DEVICE)*.25 # kernel for computing d/dy
    kernelT = torch.ones((2,2), device=DEVICE)*.25


    fx = conv2d(img1, kernelX)
    fy = conv2d(img1, kernelY)
    ft = conv2d(img1 - img2, kernelT) # Convolution to ensure everything is differentiable (smoothing)

    return fx, fy, ft

def horn_schunck(img1, img2, lambda_=0.025, Niter=200):
    n_images, n_channels, _, _ = img1.shape

    kernel_mean = torch.tensor([[1./12,1./6,1./12],[1./6,0,1./6], [1./12,1./6,1./12]], device=DEVICE)
    
    batch_size, n_channels, h, w = img1.shape
    u = torch.zeros((batch_size, 1, h, w), device=DEVICE)
    v = torch.zeros((batch_size, 1, h, w), device=DEVICE)

    fx, fy, ft = estimate_derivatives(img1, img2)

    normalization_term =  1 / (fx**2 + fy**2 + lambda_)
    
    for it in tqdm(range(Niter), desc='Horn-Schunck'):
        u_mean = conv2d(u, kernel_mean)
        v_mean = conv2d(v, kernel_mean)

        central_term = fx * u_mean + fy * v_mean + ft

        u = u_mean - fx * central_term * normalization_term
        v = v_mean - fy * central_term * normalization_term
        
    return torch.stack((v, u), dim=-1)

def compute_flow_seq(images):
    """
    Computes the flow sequentially on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """

    return horn_schunck(images[:-1], images[1:])[:, 0, :, :, :].permute(0, 3, 1, 2)

def compute_flow_direct(images):
    """
    Computes the flow directly on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """
    return horn_schunck(torch.stack([images[0]] * (images.shape[0]-1)).to(DEVICE), images[1:])[:, 0, :, :, :].permute(0, 3, 1, 2)