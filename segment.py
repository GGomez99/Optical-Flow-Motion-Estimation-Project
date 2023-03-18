"""
Main file of the project, triggers the segmentation of the input images using the optical flow computations.
"""

import fire
import torch

import matplotlib.pyplot as plt
from utils.image_utils import load_image


# def main(input_images_folder, first_mask_file, flow_file):
def main(first_mask_file, flow_file):
# def main(flow_file):
    """
    
    """

    flows = torch.load(flow_file)
    mask = load_image(first_mask_file).to(flows.device)
    _, h, w = mask.shape

    # flows = torch.zeros((3, 2, h, w))
    # flows[:, 0, :, :] = 20
    # flows[:, 1, :, :] = 20
    integrated_flow = cumsum_flow(flows)
    output = apply_flow(mask, integrated_flow)

    # plt.imshow(mask.permute(1, 2, 0).cpu().numpy())
    plt.imshow(output.permute(1, 2, 0).cpu().numpy())
    plt.show()


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