"""
Triggers the computation of optical flow on the given sequence of images.
"""

import matplotlib.pyplot as plt
import numpy as np

import cv2

import fire
from utils.image_utils import load_images_from_folder

from models.horn_schunck.optical_flow import compute_flow_seq as HS_compute_flow_seq

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

def visualization(img1, img2, flow, step, suffix=""):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    axs[0].set_title("Flow vector field" + suffix)
    axs[0].axis('off')
    axs[0].imshow(draw_flow(img1, flow, step))

    axs[1].set_title("Flow color field" + suffix)
    axs[1].axis('off')
    axs[1].imshow(draw_hsv(flow))

    plt.show()

def main(input_folder):
    """
    
    """

    images = load_images_from_folder(input_folder)
    print(images.shape)
    
    HS_flow = HS_compute_flow_seq(images)
    print(HS_flow.shape)

    # visualization(
    #     images[0].detach().cpu().numpy(),
    #     images[1].detach().cpu().permute()
    # )


if __name__ == '__main__':
    fire.Fire(main)