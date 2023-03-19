# Implements Lucas kanade algorithm but doesn't use tensors, needed for flow methods consistency
import numpy as np
import torch
from numba import njit, prange
from scipy.ndimage import convolve
from skimage import filters
from skimage.color import rgb2gray
from tqdm import tqdm

from utils.image_utils import *

import warnings

warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GAUSSIAN_SIGMA = 1.5

def estimate_derivatives(img1, img2):
    kernelX = np.array([[-1, -1],[1, 1]])  # kernel for computing d/dx
    kernelY = np.array([[-1, 1],[ -1, 1]]) # kernel for computing d/dy
    kernelT = np.ones((2,2))*.25

    fx = convolve(img1+img2/2, kernelX)
    fy = convolve(img1+img2/2, kernelY)
    ft = convolve(img1 - img2, kernelT)
    return fx, fy, ft

@njit(parallel=True)
def lucas_kanade(img1, img2, fx, fy, ft, window_size=21, tau=1e-2):
    w = int(window_size/2)
    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)
    for i in prange(w, img1.shape[0]-w):
        for j in prange(w, img1.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
            b = np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T
            if np.min(np.abs(np.linalg.eigvals(np.dot(A.T, A)))) >= tau:
                nu = np.dot(np.linalg.pinv(A), b)
                u[i, j] = nu[0][0]
                v[i, j] = nu[1][0]
    flow = np.zeros((u.shape[0], u.shape[1], 2))
    flow[:, :, 0], flow[:, :, 1] = -v, -u
    return flow


def compute_flow_seq(images: torch.Tensor):
    """
    Computes the flow sequentially on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """
    images = images.permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()

    previous_image = filters.gaussian(images[0], GAUSSIAN_SIGMA)
    flows = []
    for image in tqdm(images[1:], desc="Lucas-Kanade"):
        current_image = filters.gaussian(image, GAUSSIAN_SIGMA)
        fx, fy, ft = estimate_derivatives(previous_image, current_image)
        flows.append(lucas_kanade(previous_image, current_image, fx, fy, ft))
        previous_image = current_image

    tensor_flows = torch.Tensor(flows)

    return tensor_flows.permute(0, 3, 1, 2)

def compute_flow_direct(images: torch.Tensor):
    """
    Computes the flow directly on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """
    images = images.permute(0, 2, 3, 1)[:, :, :, 0].cpu().numpy()

    first_image = filters.gaussian(images[0], GAUSSIAN_SIGMA)
    flows = []
    for image in tqdm(images[1:], desc="Farneback"):
        current_image = filters.gaussian(image, GAUSSIAN_SIGMA)
        flows.append(lucas_kanade(first_image, current_image))

    tensor_flows = torch.Tensor(flows)

    return tensor_flows.permute(0, 3, 1, 2)
