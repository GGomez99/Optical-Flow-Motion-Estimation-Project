"""
Main file of the project, triggers the segmentation of the input images using the optical flow computations.
"""
from pathlib import Path

import fire
import numpy as np
import torch

import matplotlib.pyplot as plt
from skimage import io
from tqdm import tqdm

from utils.image_utils import load_image
from utils.flow_utils_old import propagate_mask_parallel, flow_concatenation_parallel
from utils.postprocess import process_single_mask

# def main(input_images_folder, first_mask_file, flow_file):
def main(data_path: str, method_name: str, sequence: str):
    """
    Loads flows and first mask for a given sequence, then
    """

    flows = torch.load(data_path + "/flows-outputs/" + method_name + "_"+sequence+"_flow.pt")
    mask = load_image(data_path + "/sequences-train/" + sequence + "-001.png").to(flows.device)

    if method_name.find("seqpost") != -1:
        # not used it sucks
        masks = old_seq_propagate_with_postproc(mask, flows)
    elif method_name.find("seq") != -1:
        masks = old_seq_propagate(mask, flows)
    elif method_name.find("direct") != -1:
        masks = old_direct_propagate(mask, flows)
    else:
        raise "Method " + method_name + " not available !"

    save_masks(masks, data_path + "/mask-outputs/" + method_name + "_" + sequence)

    """
    _, h, w = mask.shape

    # flows = torch.zeros((3, 2, h, w))
    # flows[:, 0, :, :] = 20
    # flows[:, 1, :, :] = 20
    integrated_flow = cumsum_flow(flows)
    output = apply_flow(mask, integrated_flow)

    # plt.imshow(mask.permute(1, 2, 0).cpu().numpy())
    plt.imshow(output.permute(1, 2, 0).cpu().numpy())
    plt.show()"""

def save_masks(masks, output_path):
    """
    Saves all generated masks to corresponding path
    :param masks: masks as numpy arrays
    :param output_path: folder where to save them, while be automatically created if not existing
    :return: None
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    print("Saving all generated masks to", output_path)
    for idx, mask in tqdm(enumerate(masks)):
        io.imsave(output_path + '/%0*d.png' % (3, idx+2), mask)

def old_seq_propagate(first_mask, flows: torch.Tensor):
    """
    Generates segmentation masks based on given flow, using sequential propagation (unoptimal loops)
    :param first_mask: initial segmentation mask
    :param flows: flows from each frame to next frame
    :return: all generated masks
    """
    flows = flows.permute(0, 2, 3, 1).cpu().numpy()
    first_mask = first_mask.cpu().numpy()[0]
    from_ref_flow = np.zeros(flows[0].shape)
    all_masks = []

    print("Propagating all segmentation masks using sequential method")
    for flow in tqdm(flows):
        from_ref_flow = flow_concatenation_parallel(flow, from_ref_flow)
        propagation_mask = propagate_mask_parallel(from_ref_flow, first_mask)
        all_masks.append(propagation_mask)

    return all_masks

def old_seq_propagate_with_postproc(first_mask, flows: torch.Tensor):
    """
        Generates segmentation masks based on given flow, using sequential propagation and post process (unoptimal loops)
        :param first_mask: initial segmentation mask
        :param flows: flows from each frame to next frame
        :return: all generated masks
        """
    flows = flows.permute(0, 2, 3, 1).cpu().numpy()
    all_masks = []
    previous_mask = first_mask.cpu().numpy()[0]

    print("Propagating all segmentation masks using sequential method")
    for flow in tqdm(flows):
        next_mask = propagate_mask_parallel(flow, previous_mask)
        next_mask = process_single_mask(next_mask)
        all_masks.append(next_mask)
        previous_mask = next_mask

    return all_masks

def old_direct_propagate(first_mask, flows: torch.Tensor):
    """
    Generates segmentation masks based on given flow, using direct propagation (unoptimal loops)
    :param first_mask: initial segmentation mask
    :param flows: flows from init from to each frame
    :return: all generated masks
    """
    flows = flows.permute(0, 2, 3, 1).cpu().numpy()
    first_mask = first_mask.cpu().numpy()[0]
    all_masks = []

    print("Propagating all segmentation masks using direct method")
    for flow in tqdm(flows):
        propagation_mask = propagate_mask_parallel(flow, first_mask)
        all_masks.append(propagation_mask)

    return all_masks

def compose(input, coords):
    """
    Composes an input with given coordinate system.
    Essentially, replace the (x, y) pixel in the input by the Cn(x, y) pixel instead.

    Args:
        input (torch.Tensor) : (n_channels, h, w)
        coords(torch.Tensor) : (n_channels, h, w)
    """

    # check if first dim is (x, y) or (y, x). Following is considering (y, x).
    # coords_x = coords[1, :, :].flatten().long() # Taking the x coord, flattening over y axis.
    # coords_y = coords[0, :, :].flatten().long() # Taking the y coord, flattening over x axis.

    # Make sure there is no coordinates overflow
    coords[0] = torch.clamp(coords[0], 0, coords.shape[1]-1)
    coords[1] = torch.clamp(coords[1], 0, coords.shape[2]-1)
    return input[:, coords[0, :, :].long(), coords[1, :, :].long()]

def warp_coordinates(coords, flow, mask):
    """
    Warps coordinates by applying the given flow, masked by the given mask.
    Essentially applies one step of :
        Cn+1 = Cn + flow_n(Cn) * mask_n

    When a coordinate reaches the border, it is clipped to the image size.
    
    Args:
        coords (torch.Tensor) : (2, h, w)
        flow (torch.Tensor) : (2, h, w)
        mask (torch.Tensor) : (1, h, w)
    """

    warped = coords + compose(flow, coords) * mask

    return warped

def warp_mask(coords, flow_previous, flow_current, mask, eps=1e-3):
    """
    One step of :
        Mn+1 = (||flow_n+1|| > eps) - Mn(Cn - flow_n) + Mn
    """

    return torch.clamp( (torch.linalg.norm(flow_current, dim=0, keepdim=True) > eps).int() - compose(mask, coords - flow_previous).int(), 0, 1).bool() + mask

def cumsum_flow(flows, eps=1e-3):
    """
    Computes the cumulative sum of the given flow, forward direction (from 0 to n).
    For backward cumsum, just reverse the batch dimension of the flows tensor.

    Args:
        flows (torch.Tensor) : (n_flows, 2, h, w)
    """

    n_flows, _, h, w = flows.shape

    # Initialization
    grid = torch.cartesian_prod(torch.arange(h), torch.arange(w)).to(flows.device).reshape(h, w, 2).permute(2, 0, 1) # coords[:, x, y] = (x, y)

    coords = torch.cartesian_prod(torch.arange(h), torch.arange(w)).to(flows.device).reshape(h, w, 2).permute(2, 0, 1) # coords[:, x, y] = (x, y)
    mask = torch.linalg.norm(flows[0], dim=0, keepdim=True) > eps

    for i in range(n_flows):
        if i == 0:
            next_mask = next_mask = warp_mask(coords, torch.zeros_like(flows[i]), flows[i], mask, eps)
        else:
            next_mask = warp_mask(coords, flows[i-1], flows[i], mask, eps)

        next_coords = warp_coordinates(coords, flows[i], mask)

        mask, coords = next_mask, next_coords

    # coords(:, x, y) = (integrated x, integrated y), coords of shape (2, h, w).
    # Each point of coords contains the target pixel coordinates after applying the flow at this coordinate.
    # The total flow is thus coords - grid
    
    return coords - grid

def apply_flow(input, flow):
    """
    Applied one flow to the given input.

    Args:
        input (torch.Tensor) : of shape (2, h, w)
        flow (torch.Tensor) : of shape (2, h, w)
    """

    _, h, w = input.shape
    grid = torch.cartesian_prod(torch.arange(h), torch.arange(w)).to(input.device).reshape(h, w, 2).permute(2, 0, 1) # coords[:, x, y] = (x, y)
    
    # (0, 0) having a flow of (2, 1) means that the pixel that was in (0, 0) is now in (2, 1).
    # the compose function does the opposite : it would set the (0, 0) pixel to the (2, 1) value, instead of the (2, 1) pixel to the (0, 0) value.
    # The pixel (x, y) should take the value of the pixel where the flow comes from.

    return compose(input, grid-compose(flow, grid-flow))

if __name__ == '__main__':
    fire.Fire(main)