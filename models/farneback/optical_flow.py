# Implements Farneback algorithm but doesn't use tensors, needed for flow methods consistency
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.image_utils import *

import warnings

warnings.simplefilter('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def farneback(img1, img2):
    flow = np.zeros((img1.shape[0], img1.shape[1], 2))
    img1 = (img1 * 255.).astype(np.uint8)
    img2 = (img2 * 255.).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(img1, img2, flow, pyr_scale=0.5, levels=3, winsize=7, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=0)


def compute_flow_seq(images: torch.Tensor):
    """
    Computes the flow sequentially on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    previous_image = cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)
    flows = []
    for image in tqdm(images[1:], desc="Farneback"):
        current_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        flows.append(farneback(previous_image, current_image))
        previous_image = current_image

    tensor_flows = torch.Tensor(flows)

    return tensor_flows.permute(0, 3, 1, 2)

def compute_flow_direct(images: torch.Tensor):
    """
    Computes the flow sequentially on the given image sequence.

    Args:
        images (torch.Tensor) : (n_images, w, h)
    """
    first_image = images[0].cpu().numpy()
    flows = []
    for image in tqdm(images[1:], desc="Farneback"):
        flows.append(farneback(first_image, image.cpu().numpy()))

    tensor_flows = torch.Tensor(flows)
    print(tensor_flows.shape)

    return tensor_flows
