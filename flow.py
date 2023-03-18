"""
Triggers the computation of optical flow on the given sequence of images.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import cv2

import fire
from utils.image_utils import load_images_from_folder

from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg
from models.horn_schunck.optical_flow import compute_flow_seq as HS_compute_flow_seq
from models.raft.optical_flow import compute_flow_seq as RAFT_compute_flow_seq

def draw_flow(img, flow, step):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 1, 0))
    for (x1, y1), (x2, y2) in lines:
        if abs(x1-x2)>1 or abs(y1-y2)>1:
            cv2.circle(vis, (x1, y1), 1, (0, 1, 0), -1)
    return vis

def draw_hsv(flow):
    flow = -flow
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb

def visualization(img1, img2, flow, step=6, suffix=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    axs[0].set_title("Flow vector field" + suffix)
    axs[0].axis('off')
    axs[0].imshow(draw_flow(img1, flow, step))

    axs[1].set_title("Flow color field" + suffix)
    axs[1].axis('off')
    axs[1].imshow(draw_hsv(flow))

    plt.show()

def save_flow_imgs(output_path, flows, method_name):
    output_folder = os.path.join(output_path, method_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    flow_imgs = flow_to_image(flows).to('cpu')
    for i, img in enumerate(flow_imgs):
        write_jpeg(img, output_folder + f"/predicted_flow_{i}.jpg")

def compute_flow_and_save(images, output_path, flow_func=RAFT_compute_flow_seq):
    """
    Computes flow using the given flow_func function and saves it in a npy.
    """

    flows = flow_func(images)
    torch.save(flows, output_path)

    return flows

def main(data_folder, sequence, method_name):
    """
    data_folder should be ./data
    """

    images, grayscale_images = load_images_from_folder(data_folder+"/sequences-train", sequence_name=sequence, with_grayscale=True)
    
    # Compute flow
    if method_name == "seq-raft":
        flows = compute_flow_and_save(
            images,
            output_path=os.path.join(data_folder, 'flows-outputs', "seq-raft_" + sequence + '_flow.pt'),
            flow_func=RAFT_compute_flow_seq)
    elif method_name == "seq-HS":
        flows = compute_flow_and_save(
            images,
            output_path=os.path.join(data_folder, 'flows-outputs', "seq-HS_" + sequence + '_flow.pt'),
            flow_func=HS_compute_flow_seq)
    else:
        raise "Method " + method_name + " not available"

    save_flow_imgs(
        output_path=os.path.join(data_folder, 'flows-img-outputs'),
        flows=flows / torch.norm(flows, dim=1, keepdim=True),
        method_name=method_name+"_"+sequence)

    # flow_imgs = flow_to_image(RAFT_flow)
    # plt.imshow(flow_imgs[80].permute(1, 2, 0).detach().cpu().numpy())
    # plt.show()

    # visualization(
    #     images[0].permute(1, 2, 0).detach().cpu().numpy(),
    #     images[1].permute(1, 2, 0).detach().cpu().numpy(),
    #     HS_flow[0].detach().cpu().numpy()
    # )


if __name__ == '__main__':
    fire.Fire(main)