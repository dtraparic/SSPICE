from __future__ import print_function, annotations
from typing import Literal
import numpy as np
import cv2
import os
import keyboard
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from keyboard import KeyboardEvent

from assets.coco_names import *
from path_config import PATHS

"""The best for a very efficient and convenient interactive grabcut, would be beginning SURE_BACKGROUND+SURE_FOREGROUND
for the initial mask, then each brush stroke would make a very wide implicit surrounding circle, morphing BG into PR_BG
and PR_FG into FG, but this would be invisible on img_painted, it would only be displayed on grabmask, 
then the user strokes would apply on top of it.

"""

@dataclass
class color_constants:
    CMAP_HOT = [(0, 0, 139), (0, 35, 255), (0, 158, 255), (0, 223, 255)]
    CMAP_AFMHOT = [(0, 0, 126), (0, 63, 190), (0, 125, 252), (86, 213, 255)]
    CMAP_VIRIDIS = [(141, 141, 34), (121, 183, 53), (63, 215, 149), (30, 225, 202)]
    CMAP_VIRI2 = [(141, 96, 51), (139, 149, 31), (113, 190, 66), (60, 217, 155)]
    CMAP_VIRI3 = [(141, 96, 51), (121, 183, 53), (63, 215, 149), (30, 225, 202)]
    BG = {'color_bgr': [141, 96, 51], 'val': 0, 'keyboard': ['0', '&'], 'thickness': 3}
    FG = {'color_bgr': [30, 225, 202], 'val': 1, 'keyboard': ['1', 'é'], 'thickness': 3}
    PR_BG = {'color_bgr': [121, 183, 53], 'val': 2, 'keyboard': ['2', '"'], 'thickness': 7}
    PR_FG = {'color_bgr': [63, 215, 149], 'val': 3, 'keyboard': ['3', "'"], 'thickness': 7}
    ALL_PENS = {'BG': BG, 'FG': FG, 'PR_BG': PR_BG, 'PR_FG': PR_FG}

class StatesGrabCut(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.drawing = False
        self.doing_rectangle = False
        self.rectangle_is_drawn = False
        self.rectangle_done_and_accounted = False
        self.pencil = color_constants.FG
        self.typing_a_label = False


class GrabCut:
    def __init__(self, init_grabcut_with: Literal["rect", "mask"], input_dir: Path, output_dir: Path,
                 initial_mask_dir: Path = None, cmap: Literal["VIRI", "VIRI2", "AFMHOT", "HOT"] = 'VIRI'):
        # je peux clairement faire un autre objet avec les states des images, dont tous les buffers machin
        # genre states_this_img
        self.cmap = cmap
        # self.apply_cmap =
        self.should_exit_now = False
        self.states = StatesGrabCut()
        self.color_cst = color_constants()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.i_img_in_folder = 0
        self.label = "unknown"
        self.label_id = 1
        self.building_csv = []
        self.init_grabcut_with = init_grabcut_with
        self.initial_mask_dir = initial_mask_dir
        self.all_img_paths = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        assert init_grabcut_with == "rect" or (init_grabcut_with == "mask" and initial_mask_dir is not None), \
            f"init_grabcut_with={init_grabcut_with} but initial_mask_dir is None"
        assert len(self.all_img_paths), f'No images png or jpg found in {input_dir}'

        self.rect = (0, 0, 1, 1)
        self.nth_elmt_in_img = 1
        self.img_original, self.img_name = self.init_image(self.all_img_paths, i_img=self.i_img_in_folder)
        all_buffers = self.init_buffers_from_img(self.img_original, self.color_cst)
        self.img_painted, self.savemask, self.grabmask, self.img_display = all_buffers
        if init_grabcut_with == "mask":
            self.initial_mask_as_grabmask()

    def apply_cmap(self):
        pass
        # if self.cmap == "VIRI":
        #     self.BG

    def reset(self):
        """I should use different levels of reset, reset for next element in same class, reset for different class,
        reset for next img"""
        # Reset drawing flags
        self.states.reset()
        self.rect = (0, 0, 1, 1)

        # Reset image matrices
        self.img_painted = self.img_original.copy()
        self.grabmask = (np.zeros(self.img_painted.shape[:2], dtype=np.uint8)
                         + self.color_cst.PR_BG['val'])  # mask initialized to PR_BG
        self.img_display = np.zeros(self.img_painted.shape, np.uint8)  # output image to be shown

    def initial_mask_as_grabmask(self):
        grabmask_human = cv2.imread(str(self.initial_mask_dir / f"{self.img_name}.png"),
                                    cv2.IMREAD_GRAYSCALE)
        print("OLD", grabmask_human.shape, np.unique(grabmask_human))
        assert grabmask_human is not None, f'Could not read {self.initial_mask_dir / f"{self.img_name}_lid{self.label_id}.png"}'
        grabmask_human[grabmask_human < 128] = self.color_cst.ALL_PENS["BG"]["val"]
        grabmask_human[grabmask_human >= 128] = self.color_cst.ALL_PENS["FG"]["val"]
        print("NEW", grabmask_human.shape, np.unique(grabmask_human))
        self.grabmask = grabmask_human

    @staticmethod
    def grabmask_to_displaymask(mask: np.ndarray):
        # print("grabmask_to_displaymask", mask.shape, np.unique(mask))
        mask_display = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        # print("grabmask_to_displaymask", mask_display.shape, np.unique(mask_display))
        return mask_display

    @staticmethod
    def init_image(all_img_paths: list[Path], i_img: int) -> (np.ndarray, str):
        img_original = cv2.imread(str(all_img_paths[i_img]))
        return img_original, all_img_paths[i_img].stem


    @staticmethod
    def probablify_around(x, y, grabmask, actual_pen, ALL_PENS, debug=False) -> np.ndarray:
        """
        Can I have this method with "self" param and then, just for doctesting it, creating a mini-self Namespace() ?
        Does this work ?
        >>> color_cst = color_constants()
        >>> grabmask = np.vstack((np.zeros((50, 100), dtype=np.uint8), np.ones((50, 100), dtype=np.uint8)))
        >>> grabmask[1,1] = 0; grabmask[2,2] = 1; grabmask[3,3] = 2; grabmask[4,4] = 3
        >>> actual_pen = color_cst.ALL_PENS["FG"]
        >>> GrabCut.probablify(x=50, y=50, grabmask=grabmask, actual_pen=actual_pen, ALL_PENS=color_cst.ALL_PENS, debug=True)
        """
        nearby_mask = np.zeros_like(grabmask)
        cv2.circle(nearby_mask, (x, y), 20, 1, -1)

        pr_grabmask = grabmask.copy()  # probablified grabmask, where all "SURE_" values become "PR_" values
        pr_grabmask[pr_grabmask == ALL_PENS["BG"]["val"]] = ALL_PENS["PR_BG"]["val"]
        pr_grabmask[pr_grabmask == ALL_PENS["FG"]["val"]] = ALL_PENS["PR_FG"]["val"]

        new_grabmask = np.where(nearby_mask, pr_grabmask, grabmask)
        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(nearby_mask); plt.show()
            plt.imshow(grabmask); plt.show()
            plt.imshow(pr_grabmask); plt.show()
            plt.imshow(new_grabmask); plt.show()
        return new_grabmask


    def apply_brush_here(self, x, y, beta_test=False, brush_thickness_on_img_painted=1):
        """How to test this function ?? Should I do everything as staticmethod, then have long param list ?
        Should I do a method(self) that then calls the static one/doctestable one ?"""
        if beta_test:
            print("[WARNING] It does not work properly as the nearbyzone erase the user brush as it progress")
            if self.states.pencil in [self.color_cst.ALL_PENS["FG"], self.color_cst.ALL_PENS["BG"]]:
                self.grabmask = self.probablify_around(x, y, self.grabmask, self.states.pencil, self.color_cst.ALL_PENS)
        if brush_thickness_on_img_painted:
            cv2.circle(self.img_painted, (x, y), brush_thickness_on_img_painted, self.states.pencil['color_bgr'], -1)
        else:
            cv2.circle(self.img_painted, (x, y), self.states.pencil['thickness'], self.states.pencil['color_bgr'], -1)
        cv2.circle(self.grabmask, (x, y), self.states.pencil['thickness'], self.states.pencil['val'], -1)


    @staticmethod
    def mouse_out_of_img(x, y, box, verbose=False):
        """Box should be (y1, x1, y2, x2)"""
        out_of_img = x < box[1] or x > box[3] or y < box[0] or y > box[2]
        if out_of_img:
            if verbose: print("Mouse out of img", x, y, box)
        return out_of_img

    def onmouse(self, event, x, y, flags, param):
        """On mouse is called when mouse is inside window"""
        # Draw Rectangle
        if GrabCut.mouse_out_of_img(x, y, [0, 0] + list(self.img_original.shape[:2])):
            y = y % self.img_original.shape[0]

        if event == cv2.EVENT_RBUTTONDOWN:
            if self.states.rectangle_is_drawn:
                print("To restart a initial rectangle, please reset the image with \"r\"")
                return
            else:
                self.states.doing_rectangle = True
                self.states.x1, self.states.y1 = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.states.doing_rectangle:
                # img = self.img.copy()
                self.img_painted = self.img_original.copy()
                cv2.rectangle(self.img_painted,
                              (self.states.x1, self.states.y1),
                              (x, y),
                              self.color_cst.PR_FG['color_bgr'], 2)
                self.rect = (min(self.states.x1, x), min(self.states.y1, y),
                             abs(self.states.x1 - x), abs(self.states.y1 - y))

        elif event == cv2.EVENT_RBUTTONUP and not self.states.rectangle_is_drawn:
            self.states.doing_rectangle = False
            self.states.rectangle_is_drawn = True
            cv2.rectangle(self.img_painted, (self.states.x1, self.states.y1), (x, y), self.color_cst.PR_FG['color_bgr'],
                          2)
            self.rect = (min(self.states.x1, x),
                         min(self.states.y1, y),
                         abs(self.states.x1 - x),
                         abs(self.states.y1 - y))
            self.iterate_grabcut()
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.init_grabcut_with == "rect" and not self.states.rectangle_is_drawn:
                print("first draw rectangle \n")
                return
            else:
                self.states.drawing = True
                self.apply_brush_here(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.states.drawing:
                self.apply_brush_here(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.states.drawing:
                self.states.drawing = False
                self.apply_brush_here(x, y)
                self.iterate_grabcut()

    @staticmethod
    def init_buffers_from_img(or_img: np.ndarray, color_cst: color_constants):
        savemask = np.zeros(or_img.shape[:2], dtype=np.uint8)
        img = or_img.copy()  # a copy of original image
        grabmask = np.zeros(or_img.shape[:2], dtype=np.uint8) + color_cst.PR_BG['val']  # mask initialized to PR_BG
        output = np.zeros(or_img.shape, np.uint8)  # output image to be shown
        return img, savemask, grabmask, output

    @staticmethod
    def add_elmt_to_savemask(grabmask: np.ndarray, savemask: np.ndarray, nth_in_img: int):
        """This function adds the grabmask{0;1;2;3} (SURE_BG, SURE_FG, PROB_BG, PROB_FG)
        of one elmt to the savemask which contains {0;1;...;N} (BG; elmt1; ... ; elmtN)"""
        assert 256 > (256 - nth_in_img) > 0, "Unexpected value for nth_in_img"
        savemask += (256 - nth_in_img) * (np.where((grabmask == 1) + (grabmask == 3), 1, 0).astype('uint8'))
        return savemask

    def iterate_grabcut(self, verbose=False):
        """This modifies inplace img, grabmask and output
        It was to emph on this, that I called them _ptr"""

        print(""" For finer touchups, mark foreground and background after pressing keys 0-3
        and again press 'n' \n""")
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        if self.init_grabcut_with == "rect" and not self.states.rectangle_done_and_accounted:  # grabcut with rect
            print("[GRABCUT WITH RECT]")
            cv2.grabCut(self.img_original, self.grabmask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            self.states.rectangle_done_and_accounted = True
        elif self.init_grabcut_with == "mask" or self.states.rectangle_done_and_accounted:  # grabcut with mask
            print("[GRABCUT WITH MASK]")
            cv2.grabCut(self.img_original, self.grabmask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
        else:
            raise ValueError(f"Unknown init_grabcut_with={self.init_grabcut_with}")

        if verbose: print("grabmaskAFTER", type(self.grabmask),
                          self.grabmask.shape, self.grabmask.dtype, np.unique(self.grabmask))

    @staticmethod
    def update_label_to(label_tmp: str):
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

    def change_pencil(self, color: str):
        ALL_PENS = self.color_cst.ALL_PENS
        assert color in GrabCut.get_all_pencil_keys(ALL_PENS)
        if color in ALL_PENS['BG']['keyboard']:  # BG drawing
            print(" mark background regions with left mouse button \n")
            self.states.pencil = self.color_cst.BG
        elif color in ALL_PENS['FG']['keyboard']:  # FG drawing
            print(" mark foreground regions with left mouse button \n")
            self.states.pencil = self.color_cst.FG
        elif color in ALL_PENS['PR_BG']['keyboard']:  # PR_BG drawing
            print(" mark probable background regions with left mouse button \n")
            self.states.pencil = self.color_cst.PR_BG
        elif color in ALL_PENS['PR_FG']['keyboard']:  # PR_FG drawing
            self.states.pencil = self.color_cst.PR_FG
            print(" mark probable foreground regions with left mouse button \n")

    def compute_display_img(self):
        """
        As a doctest : create mini-imgs then show that it correctly stack them
        """
        mask_display = GrabCut.grabmask_to_displaymask(self.grabmask)

        grabmask_human = np.stack((self.grabmask,) * 3, axis=-1)
        grabmask_human[np.all(grabmask_human == [0, 0, 0], axis=-1)] = self.color_cst.ALL_PENS["BG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [1, 1, 1], axis=-1)] = self.color_cst.ALL_PENS["FG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [2, 2, 2], axis=-1)] = self.color_cst.ALL_PENS["PR_BG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [3, 3, 3], axis=-1)] = self.color_cst.ALL_PENS["PR_FG"]["color_bgr"]
        mask_display_tripliquated = np.stack((mask_display,) * 3, axis=-1)
        masked_img = cv2.bitwise_and(self.img_original, self.img_original, mask=mask_display)

        display_img = np.vstack((self.img_painted, grabmask_human, mask_display_tripliquated, masked_img))
        return display_img

    def save_mask_display(self, name: str, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
        print("SAVE A", np.unique(self.grabmask), self.grabmask.shape)
        mask_display = GrabCut.grabmask_to_displaymask(self.grabmask)
        print("SAVE B", np.unique(mask_display), mask_display.shape)
        cv2.imwrite(str(output_dir / name), mask_display)

    @staticmethod
    def get_all_pencil_keys(ALL_PENS):
        all_keys = []
        for pencil in ALL_PENS.values():
            all_keys += list(pencil['keyboard'])
        return all_keys

    def validate_element(self):
        self.savemask = GrabCut.add_elmt_to_savemask(self.grabmask, self.savemask, self.nth_elmt_in_img)
        print(f'Saved element "{self.label}" in image "{self.img_name}_lid{self.label_id}.png" (value = 256 - '
              f'{self.nth_elmt_in_img})')

        single_line_csv = [self.img_name, self.nth_elmt_in_img, self.label, self.label_id]
        self.building_csv.append(single_line_csv)

        self.nth_elmt_in_img += 1
        # self.reset()

    def validate_annotation(self):
        HEADER_CSV = ['img_name', 'value_in_mask', 'label', 'label_id']
        pd.DataFrame(self.building_csv).to_csv(PATHS.project / "fast_label_2024" / "annotations.csv",
                                               header=HEADER_CSV, index=False)

    def on_press(self, key_event: KeyboardEvent) -> bool | None:
        k_tmp = key_event.name
        DEBUG = True

        if DEBUG: print(f'{key_event}, Actually typing={self.states.typing_a_label}')
        if self.states.typing_a_label:
            if k_tmp == 'enter':
                self.states.typing_a_label = False
            else:
                return None

        if k_tmp in GrabCut.get_all_pencil_keys(self.color_cst.ALL_PENS):
            self.change_pencil(k_tmp)
        elif k_tmp == 's':  # save image
            self.save_mask_display(f'{self.img_name}_lid{self.label_id}.png')
            print(" Result saved as image \n")
        elif k_tmp == 'r':  # reset everything
            self.reset()
        elif k_tmp == 'n':  # grabcut-alg iteration
            self.iterate_grabcut()
        elif k_tmp == 'w':  # next element (of the same class, of the same image)
            self.validate_element()
            self.reset()
        elif k_tmp == 'l':
            self.states.typing_a_label = True
        elif k_tmp == 'a':  # validate annotation for all the images done then quit
            self.validate_annotation()
        elif k_tmp == 'x':  # next img
            self.validate_element()
            if DEBUG: print(f'Unique values of savemask before valid : {np.unique(self.img_display)}')
            #
            #  Le array se buildup et des csv se créent de plus en plus long
            #  Il faudrait que le csv soit créé au lancement du programme avec un timecode + update a chaque "X"
            #
            print("ALLEZ SAVE", )
            self.save_mask_display(f'{self.img_name}_lid{self.label_id}.png')

            if DEBUG: print(f'Number of element in validated img : {np.unique(self.img_display).size}')
            # cv2.imwrite(str(cst.ASSETS_DIR / 'masks' / f'{img_name.stem}_{label}{nth_in_img}.png'), savemask)
            self.nth_elmt_in_img = 1  # because we reset
            self.i_img_in_folder += 1
            print()
            if self.i_img_in_folder >= len(self.all_img_paths):
                self.should_exit_now = True
                exit()  # Exit this listener thread
            self.img_original, self.img_name = self.init_image(self.all_img_paths, self.i_img_in_folder)
            all_buffers = GrabCut.init_buffers_from_img(self.img_original, self.color_cst)
            self.img_painted, self.savemask, self.grabmask, self.img_display = all_buffers
            if self.init_grabcut_with == "mask":
                self.initial_mask_as_grabmask()

    def running(self):
        while not self.should_exit_now:  # not typing_label:
            cv2.imshow('output', self.img_display)
            cv2.setMouseCallback('output', self.onmouse)
            k = cv2.waitKey(1)

            # key bindings
            if k == 27:  # esc to exit
                break
            if self.states.typing_a_label:
                label_tmp = input(
                    "Please enter the label of the element (or its ID number) (enter nothing to take last label):")
                print(f'Label_tmp = {label_tmp}')
                if label_tmp:
                    label, label_id = GrabCut.update_label_to(label_tmp)
                    print(f'Label = {label}')
            self.img_display = self.compute_display_img()
            if k == 32:  # space to save
                print(f'Grabmask = {np.unique(self.grabmask)}')
                display_mask = GrabCut.grabmask_to_displaymask(self.grabmask)
                print(f'Displaymask = {np.unique(display_mask)}')

            #     import matplotlib.pyplot as plt
            #     plt.imshow(mask_display); print(np.unique(mask_display)); plt.show()
            #     plt.imshow(self.savemask); print(np.unique(self.savemask)); plt.show()
            #     plt.imshow(self.grabmask); print(np.unique(self.grabmask)); plt.show()

        cv2.destroyAllWindows()

def run_grabcut_with_rect():
    grabcut = GrabCut(init_grabcut_with="rect", input_dir=PATHS.project / "fast_label_2024" / "images",
                      output_dir=PATHS.project / "fast_label_2024" / "new_masks")
    keyboard.on_press(grabcut.on_press)
    grabcut.running()

def run_grabcut_with_maskfolder():
    grabcut = GrabCut(init_grabcut_with="mask", input_dir=PATHS.project / "fast_label_2024" / "images",
                      output_dir=PATHS.project / "fast_label_2024" / "new_masks",
                      initial_mask_dir=PATHS.project / "fast_label_2024" / "initial_masks")
    keyboard.on_press(grabcut.on_press)
    grabcut.running()


if __name__ == '__main__':
    run_grabcut_with_maskfolder()
