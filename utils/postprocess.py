import glob
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


def erode(img, kernel_size=3, iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                       (kernel_size, kernel_size))
    img = cv2.erode(img, kernel, iterations=iter)
    return img


def dilate(img, kernel_size=3, iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                       (kernel_size, kernel_size))
    img = cv2.dilate(img, kernel, iterations=iter)
    return img

def close(img, kernel_size=3, iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                       (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter)
    return img

def open(img, kernel_size=3, iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kernel_size + 1, 2 * kernel_size + 1),
                                       (kernel_size, kernel_size))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iter)
    return img

def find_and_fill_contour(img):
    contours, hierar = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(img, contours, -1, 255, thickness=cv2.FILLED)

def process_single_mask(mask_img):
    contoured_img = find_and_fill_contour(mask_img)
    img_closed = close(contoured_img, kernel_size=1, iter=2)
    img_opened = open(img_closed, kernel_size=1, iter=2)
    return img_opened

def process_all_masks(sequence_name, method_name, masks_path):
    # create directory to save files
    save_dir = masks_path + method_name + "-post_" + sequence_name
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # get number frames
    files = glob.glob(masks_path + method_name + "_" + sequence_name + "/*.png")
    im_begin = 1
    im_end = len(files)+2
    for im in tqdm(range(im_begin + 1, im_end)):
        mask_raw = cv2.imread(masks_path + method_name + "_" + sequence_name + '/%0*d.png' % (3, im), cv2.IMREAD_GRAYSCALE)
        contoured_img = find_and_fill_contour(mask_raw)
        img_closed = close(contoured_img, kernel_size=1, iter=2)
        img_opened = open(img_closed, kernel_size=1, iter=2)
        cv2.imwrite(save_dir + '/%0*d.png' % (3, im), img_opened)
