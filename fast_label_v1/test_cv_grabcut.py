#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
TODO - Idée pour améliorer la rapidité permise par le programme : plutôt préremplir un point de centre de gravité de PR_FG quand on dessine une bounding box PR_BG (avec à l'extérieur SURE_BG) 
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import os
import keyboard
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from assets.coco_names import *
from path_config import PATHS

@dataclass
class Constants:
    BLUE = [255, 0, 0]  # rectangle color
    RED = [0, 0, 255]  # PR BG
    GREEN = [0, 255, 0]  # PR FG
    BLACK = [0, 0, 0]  # sure BG
    WHITE = [255, 255, 255]  # sure FG
    DRAW_BG = {'color': BLACK, 'val': 0}
    DRAW_FG = {'color': WHITE, 'val': 1}
    DRAW_PR_BG = {'color': RED, 'val': 2}
    DRAW_PR_FG = {'color': GREEN, 'val': 3}
    ASSETS_DIR = PATHS.examples
    IMG_DIR = Path('images')
    MASK_DIR = Path('masks')
    HEADER_CSV = ['img_name', 'value_in_mask', 'label', 'label_id']
cst = Constants()

i_img = 0
nth_in_img = 1
rect = (0, 0, 1, 1)
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
value = cst.DRAW_FG  # drawing initialized to FG
thickness = 3  # brush thickness
typing_label = False
label = "unknown"
label_id = -1
building_csv = []

# setting up flags

def reset_draw():
    global rect, drawing, rectangle, rect_or_mask, rect_over, value
    rect = (0, 0, 1, 1)
    drawing = False
    rectangle = False
    rect_or_mask = 100
    rect_over = False
    value = cst.DRAW_FG

def reset_img_matrices(img_original):
    global img, grabmask, output
    img = img_original.copy()
    grabmask = np.zeros(img.shape[:2], dtype=np.uint8) + cst.DRAW_PR_BG['val']  # mask initialized to PR_BG
    output = np.zeros(img.shape, np.uint8)  # output image to be shown

def init_img(img:np.ndarray):
    savemask = np.zeros(img.shape[:2], dtype=np.uint8)
    img2 = img.copy()  # a copy of original image
    grabmask = np.zeros(img.shape[:2], dtype=np.uint8) + cst.DRAW_PR_BG['val']  # mask initialized to PR_BG
    output = np.zeros(img.shape, np.uint8)  # output image to be shown
    return img2, savemask, grabmask, output

def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, grabmask, rectangle, rect, rect_or_mask, ix, iy, rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), cst.BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img, (ix, iy), (x, y), cst.BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        iterate_grabcut()
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(grabmask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(grabmask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(grabmask, (x, y), thickness, value['val'], -1)
            iterate_grabcut()


def add_grabmask_elmt_to_savemask(grabmask:np.ndarray, savemask:np.ndarray, nth_in_img:int):
    """This function adds the grabmask{0;1;2;3} (SURE_BG, SURE_FG, PROB_BG, PROB_FG)
    of one elmt to the savemask which contains {0;1;...;N} (BG; elmt1; ... ; elmtN)"""
    savemask += nth_in_img*(np.where((grabmask == 1) + (grabmask == 3), 1, 0).astype('uint8'))
    return savemask

def iterate_grabcut(verbose=False):
    global rect_or_mask, img2, grabmask, rect
    print(""" For finer touchups, mark foreground and background after pressing keys 0-3
    and again press 'n' \n""")
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    if rect_or_mask == 0:  # grabcut with rect
        print("[GRABCUT WITH RECT]")
        cv2.grabCut(img2, grabmask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
        rect_or_mask = 1
    elif rect_or_mask == 1:  # grabcut with mask
        print("[GRABCUT WITH MASK]")
        cv2.grabCut(img2, grabmask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

    if verbose: print("grabmaskAFTER", type(grabmask), grabmask.shape, grabmask.dtype, np.unique(grabmask))

def process_label(label_tmp: str):
    if label_tmp.isnumeric():
        # try:
        new_label = COCO_INSTANCE_CATEGORY_NAMES[label_tmp]
        label_id = label_tmp
        # except ValueError:
        #     print('Index from name not found')
    else:
        # try:
        new_label = label_tmp
        label_id = COCO_INSTANCE_CATEGORY_NAMES.index(label_tmp)
        # except ValueError:
        #     print('Name from index not found')
    return new_label, label_id


def on_press(key_event):
    global rect_or_mask, value, grabmask, rect, nth_in_img, savemask, i_img, img_name, typing_label, label, img2, img, output
    k_tmp = key_event.name
    DEBUG = True

    if DEBUG and not typing_label:
        print(f'{key_event}, Actually typing={typing_label}')

    if k_tmp == '0' and not typing_label:  # BG drawing
        print(" mark background regions with left mouse button \n")
        value = cst.DRAW_BG
    elif k_tmp == '1' and not typing_label:  # FG drawing
        print(" mark foreground regions with left mouse button \n")
        value = cst.DRAW_FG
    elif k_tmp == '2' and not typing_label:  # PR_BG drawing
        value = cst.DRAW_PR_BG
    elif k_tmp == '3' and not typing_label:  # PR_FG drawing
        value = cst.DRAW_PR_FG
    elif k_tmp == 's':  # save image
        cv2.imwrite('grabcut_output.png', mask_display)
        # cv2.imwrite(str(PATHS.project / "fast_label" / "new_masks" / img_path.name), mask_display)
        print(" Result saved as image \n")
    elif k_tmp == 'r' and not typing_label:  # reset everything
        reset_draw()
        reset_img_matrices(img2)
    elif k_tmp == 'n' and not typing_label:  # segment the image
        iterate_grabcut()
    elif k_tmp == 'w' and not typing_label:  # next element
        savemask = add_grabmask_elmt_to_savemask(grabmask, savemask, nth_in_img)
        print(f'Saved element "{label}" in image "{img_name}"')

        single_line_csv = [img_name, nth_in_img, label, label_id]
        building_csv.append(single_line_csv)

        nth_in_img += 1
        reset_draw()
        reset_img_matrices(img2)
    elif k_tmp == 'l' and not typing_label:
        typing_label = True
    elif k_tmp == 'enter' and typing_label:
        typing_label = False
    elif k_tmp == 'x' and not typing_label:  # next img
        savemask = add_grabmask_elmt_to_savemask(grabmask, savemask, nth_in_img)
        if DEBUG:
            print(f'Unique values of savemask before valid: {np.unique(savemask)}')
        pd.DataFrame(building_csv).to_csv("annotations.csv", header=cst.HEADER_CSV)
        cv2.imwrite(str(cst.ASSETS_DIR / 'masks' / f'{img_name.stem}.png'), savemask)
        if DEBUG:
            print(f'Number of elements in validated img: {np.unique(savemask).size}')
        nth_in_img = 1
        i_img, img_name, img = get_next_img(PATH_IMAGES, i_img)
        img2, savemask, grabmask, output = init_img(img)
        run_window()

def grabmask_to_displaymask(mask: np.ndarray):
    mask_display = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    return mask_display

def grabmask_to_savemask(mask: np.ndarray):
    savemask = np.where((mask == 1) + (mask == 3), 1, 0).astype('uint8')
    return savemask

def get_next_img(dir_images, i=-1):
    i += 1
    img_name = Path(all_imgs[i_img])
    img = cv2.imread(str(dir_images / img_name))
    return i, img_name, img

# def get_img_i(all_imgs, i_img):
#     img_name = Path(all_imgs[i_img])


def run_window():
    global grabmask, img, output, mask_display
    cv2.imshow('output', output)
    cv2.imshow('input', img)
    k = cv2.waitKey(1)

    # key bindings
    if k == 27:  # esc to exit
        exit()
    if typing_label:
        label_tmp = input("Please enter the label of the element (or its ID number) (enter nothing to take last label):")
        print(f'Label_tmp = {label_tmp}')
        if label_tmp:
            label, label_id = process_label(label_tmp)
        print(f'Label = {label}')

    mask_display = grabmask_to_displaymask(grabmask)

    if k == 32:  # space to save
        import matplotlib.pyplot as plt
        plt.imshow(mask_display)
        print(np.unique(mask_display))
        plt.show(block=False)
        plt.imshow(savemask)
        print(np.unique(savemask))
        plt.show(block=False)
        plt.imshow(grabmask)
        print(np.unique(grabmask))
        plt.show()

if __name__ == '__main__':

    PATH_IMAGES = PATHS.project / "fast_label" / "images"
    mask_display = None
    # print documentation
    print(__doc__)

    all_imgs = os.listdir(PATH_IMAGES)
    all_imgs = [x for x in all_imgs if x.endswith('.jpg') or x.endswith('.png')]
    i_img, img_name, img = get_next_img(PATH_IMAGES)
    keyboard.on_press(on_press)
    # keyboard.on_release(on_release)
    img2, savemask, grabmask, output = init_img(img)
    # input and output windows
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    cv2.setMouseCallback('input', onmouse)
    cv2.moveWindow('input', img.shape[1] + 10, 90)

    print(" Instructions: \n")
    print(" Draw a rectangle around the object using right mouse button \n")

    while 1:  # not typing_label:
        run_window()
        output = cv2.bitwise_and(img2, img2, mask=mask_display)

    cv2.destroyAllWindows()
