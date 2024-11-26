from __future__ import print_function, annotations
import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
import keyboard
import sys
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
# from inpainting_scripts_2024.demo_total_chain_inpainting_v4 import human_sorted
from keyboard import KeyboardEvent

from assets.coco_names import *
from runnable.scripts.try_ffmpeg_to_get_only_borders_or_blur_frames import concat_vertical_videos

# from path_config import PATHS

"""The best for a very efficient and convenient interactive grabcut, would be beginning SURE_BACKGROUND+SURE_FOREGROUND
for the initial mask, then each brush stroke would make a very wide implicit surrounding circle, morphing BG into PR_BG
and PR_FG into FG, but this would be invisible on img_painted, it would only be displayed on grabmask, 
then the user strokes would apply on top of it.

"""

project_root_fast_label = Path(__file__).parent.parent

def debug_plot_nparray(img):
    plt.imshow(img)
    print(np.unique(img))
    plt.colorbar()
    plt.show()

def alphanum_key(s, backslash_is_a_char=True):
    """ Turn a string into a list of string and number chunks.
    >>> alphanum_key("z23a")
    ['z', 23, 'a']
    >>> alphanum_key("880\frame_000031.png")
    """
    import re
    if backslash_is_a_char:
        s = s.replace('\\', '\\\\')
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def human_sorted(l, backslash_is_a_char=True):
    """ Sort the given list in the way that humans expect.
    https://stackoverflow.com/a/4623518
    >>> arr = ["f_0_1", "f_0_10", "f_0_2", "f_0_23"]
    >>> human_sorted(arr)
    ['f_0_1', 'f_0_2', 'f_0_10', 'f_0_23']
    """
    on_path = False
    if len(l) == 0:
        return l
    if isinstance(l[0], Path):
        l = [str(x) for x in l]
        on_path = True
    try:
        res = sorted(l, key=alphanum_key)
    except Exception as e:
        print("Arr=", l)
        raise e
    if on_path:
        res = [Path(x) for x in res]

    return res

def search_for_images(im_dir: Path, recursive=False) -> [Path]:
    search_method = im_dir.rglob if recursive else im_dir.glob
    return human_sorted(list(search_method("*.jpg")) + list(search_method("*.png")))

@dataclass
class ColorConstants:
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
        self.pencil = ColorConstants.FG
        self.typing_a_label = False


class GrabCut:
    def __init__(self, input_dir: Path, output_dir: Path, init_grabcut_with: Literal["rect", "mask"],
                 initial_mask_dir: Path = None, cmap: Literal["VIRI", "VIRI2", "AFMHOT", "HOT"] = 'VIRI',
                 recursive_img_search=False, downsize: float = False, debug=False, save_in_original_shape=True,
                 vertical_guide_lines=True, blank_out_middle_third=True, begin_at_frame=0, nb_preview_next_frames=0):
        # je peux clairement faire un autre objet avec les states des images, dont tous les buffers machin
        # genre states_this_img
        self.all_corrected_masks = None
        self.cmap = cmap
        self.blank_out_middle_third = blank_out_middle_third
        # self.apply_cmap =
        self.next_frames = []
        self.guide_lines = vertical_guide_lines
        self.should_exit_now = False
        self.states = StatesGrabCut()
        self.color_cst = ColorConstants()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.i_img_in_folder = begin_at_frame
        self.label = "unknown"
        self.label_id = 1
        self.building_csv = []
        self.nb_next_frames = nb_preview_next_frames
        self.init_grabcut_with = init_grabcut_with
        self.initial_mask_dir = initial_mask_dir
        self.downsize = downsize

        self.all_img_paths = search_for_images(input_dir, recursive=recursive_img_search)

        assert init_grabcut_with == "rect" or (init_grabcut_with == "mask" and initial_mask_dir is not None), \
            f"init_grabcut_with={init_grabcut_with} but initial_mask_dir is None"
        assert len(self.all_img_paths), f'No images png or jpg found in {input_dir}'

        self.rect = (0, 0, 1, 1)
        self.nth_elmt_in_img = 1
        self.img_original, self.img_name, self.img_original_shape, self.new_hw = self.init_image(self.all_img_paths, i_img=self.i_img_in_folder, downsize=self.downsize)
        if self.nb_next_frames > 0:
            self.next_frames = GrabCut.get_all_next_frames(self.all_img_paths, self.i_img_in_folder, self.nb_next_frames,
                                                           new_h=self.new_hw[0]//self.nb_next_frames, new_w=self.new_hw[1]//self.nb_next_frames)
        if save_in_original_shape:
            self.save_shape = self.img_original_shape[:2]
        else:
            raise NotImplementedError
        all_buffers = self.init_buffers_from_img(self.img_original, self.color_cst)
        self.img_painted, self.savemask, self.grabmask, self.img_display = all_buffers
        if init_grabcut_with == "mask":
            self.all_initial_masks = search_for_images(self.initial_mask_dir, recursive=recursive_img_search)
            assert len(self.all_initial_masks) == len(self.all_img_paths), \
                f'len(initial_masks)={len(self.all_initial_masks)} but len(all_img)={len(self.all_img_paths)}'
            self.grabmask = GrabCut.initial_mask_as_grabmask(self.color_cst, self.all_initial_masks, self.i_img_in_folder, new_h=self.new_hw[0], new_w=self.new_hw[1])
        if debug:
            debug_plot_nparray(self.img_painted)
            debug_plot_nparray(self.grabmask)


    def apply_cmap(self):
        pass
        # if self.cmap == "VIRI":
        #     self.BG

    @staticmethod
    def load_thumbnail(all_image_path: [Path], i: int, new_h: int, new_w: int, verbose=True):
        if verbose:
            print(all_image_path)
            print(i, new_h, new_w)
        if i < len(all_image_path) and all_image_path[i].exists():
            img = cv2.imread(str(all_image_path[i]))
            img = cv2.resize(img, (new_w, new_h))
        else:
            img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        return img

    @staticmethod
    def get_all_next_frames(all_img: [Path], actual_i_img: int, nb_next_frames: int, new_h: int, new_w: int):

        with ThreadPoolExecutor(max_workers=min(nb_next_frames, 8)) as executor:
            images = list(executor.map(GrabCut.load_thumbnail, all_img*nb_next_frames, range(actual_i_img+1,
                                                                              actual_i_img+nb_next_frames+1),
                                       [new_h]*nb_next_frames, [new_w]*nb_next_frames))

        for idx, img in enumerate(images):
            assert img is not None
        return images

    @staticmethod
    def shift_next_frames(all_img: [Path], next_frames, actual_i_img: int, new_h: int, new_w: int):

        nb_next_frames = len(next_frames)
        new_frame = GrabCut.load_thumbnail(all_img, actual_i_img+nb_next_frames, new_h, new_w)
        next_frames = next_frames[1:]+[new_frame]
        return next_frames

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

    @staticmethod
    def initial_mask_as_grabmask(color_cst: ColorConstants, all_masks_path: [Path], i_img_in_folder: int,
                                 new_h: int = None, new_w: int = None):
        """
        >>> color_cst = ColorConstants()
        >>> all_initial_masks = search_for_images(project_root_fast_label / 'demo_data' / 'initial_masks', recursive=True)
        >>> grabmask = GrabCut.initial_mask_as_grabmask(color_cst, all_masks_path, 0)
        >>> _ = plt.imshow(grabmask); _ = plt.show()
        >>> color_cst = ColorConstants()
        >>> all_initial_masks = search_for_images(project_root_fast_label / 'demo_data' / 'initial_masks', recursive=True)
        >>> grabmask = GrabCut.initial_mask_as_grabmask(color_cst, all_masks_path, 0, downsize=True)
        >>> _ = plt.imshow(grabmask); _ = plt.show()
        """
        grabmask_human = cv2.imread(str(all_masks_path[i_img_in_folder]), cv2.IMREAD_GRAYSCALE)
        if new_h or new_w: grabmask_human = cv2.resize(grabmask_human, (new_w, new_h))
        assert grabmask_human is not None, f'Could not read {all_masks_path[i_img_in_folder]}'
        grabmask_human[(0 < grabmask_human) & (grabmask_human <= 127)] = color_cst.ALL_PENS["PR_BG"]["val"]
        grabmask_human[(128 <= grabmask_human) & (grabmask_human < 230)] = color_cst.ALL_PENS["PR_FG"]["val"]
        grabmask_human[grabmask_human == 0] = color_cst.ALL_PENS["BG"]["val"]
        grabmask_human[230 <= grabmask_human] = color_cst.ALL_PENS["FG"]["val"]
        return grabmask_human

    @staticmethod
    def grabmask_to_displaymask(mask: np.ndarray, verbose=False):
        if verbose: print("grabmask_to_displaymask", mask.shape, np.unique(mask))
        mask_display = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        if verbose: print("grabmask_to_displaymask", mask_display.shape, np.unique(mask_display))
        return mask_display

    @staticmethod
    def init_image(all_img_paths: list[Path], i_img: int, downsize: float = False) -> (np.ndarray, str):
        """resize (h, w)"""
        img_original = cv2.imread(str(all_img_paths[i_img]))
        original_img_shape = img_original.shape
        if downsize:
            (new_h, new_w) = (int(original_img_shape[0] // downsize), int(original_img_shape[1] // downsize))
            img_original = cv2.resize(img_original, (new_w, new_h))
        else:
            (new_h, new_w) = original_img_shape[0:2]
        return img_original, all_img_paths[i_img].stem, original_img_shape, (new_h, new_w)

    @staticmethod
    def probablify_around(x, y, grabmask, actual_pen, ALL_PENS, debug=False) -> np.ndarray:
        """
        Can I have this method with "self" param and then, just for doctesting it, creating a mini-self Namespace() ?
        Does this work ?
        >>> color_cst = ColorConstants()
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
        # if brush_thickness_on_img_painted:
        #     cv2.circle(self.img_painted, (x, y), brush_thickness_on_img_painted, self.states.pencil['color_bgr'], -1)
        # else:
        #     cv2.circle(self.img_painted, (x, y), self.states.pencil['thickness'], self.states.pencil['color_bgr'], -1)
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
            x = x % self.img_original.shape[1]

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
    def init_buffers_from_img(or_img: np.ndarray, color_cst: ColorConstants):
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

    def iterate_grabcut(self, verbose=True):
        """This modifies inplace img, grabmask and output
        It was to emph on this, that I called them _ptr"""

        print(""" For finer touchups, mark foreground and background after pressing keys 0-3
        and again press 'n' \n""")
        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)

        if self.init_grabcut_with == "rect" and not self.states.rectangle_done_and_accounted:  # grabcut with rect
            print("[GRABCUT WITH RECT]")
            if np.any(self.grabmask):
                cv2.grabCut(self.img_original, self.grabmask, self.rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            self.states.rectangle_done_and_accounted = True
        elif self.init_grabcut_with == "mask" or self.states.rectangle_done_and_accounted:  # grabcut with mask
            print("[GRABCUT WITH MASK]")
            if np.any(self.grabmask):
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

    @staticmethod
    def add_vertical_lines(img: np.ndarray, pos_x_of_lines: list[int], color=(255, 0, 0)) -> np.ndarray:
        """
        >>> img = np.zeros((9, 18, 3), dtype=np.uint8)
        >>> arr = GrabCut.add_vertical_lines(img, [5, 12])
        >>> _ = plt.imshow(arr); plt.show()
        """
        img_with_lines = img.copy()
        height = img_with_lines.shape[0]

        for x in pos_x_of_lines:
            if 0 <= x < img_with_lines.shape[1]:  # Ensure x is within image width
                cv2.line(img_with_lines, (x, 0), (x, height - 1), color=color, thickness=1)
            else:
                print(f"Warning: x={x} is out of bounds for the image width {img_with_lines.shape[1]}.")

        return img_with_lines

    @staticmethod
    def add_vertical_lines_at_thirds(img: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
        """
        As we can see, the lines are inside the middle third (so we can see clearly the external thirds, without them
        being hidden by the 1-pixel wide lines)
        >>> img = np.zeros((3, 9, 3), dtype=np.uint8)
        >>> arr = GrabCut.add_vertical_lines_at_thirds(img)
        >>> _ = plt.imshow(arr); plt.show()
        >>> img = np.zeros((300, 900, 3), dtype=np.uint8)
        >>> arr = GrabCut.add_vertical_lines_at_thirds(img)
        >>> _ = plt.imshow(arr); plt.show()
        """
        img_with_lines = img.copy()
        height, width, _ = img_with_lines.shape
        positions = [width // 3, (2 * width // 3) -1]
        img_with_lines = GrabCut.add_vertical_lines(img_with_lines, positions, color=color)
        return img_with_lines

    @staticmethod
    def blank_out_middle_thirds(img: np.ndarray, blank_color=0) -> np.ndarray:
        """
        Put zeroes in the middle third

        >>> img = np.random.randint(0, 256, (3, 9, 3), dtype=np.uint8)
        >>> arr = GrabCut.blank_out_middle_thirds(img)
        >>> _ = plt.imshow(arr); plt.show()
        >>> img = np.random.randint(0, 256, (300, 900, 3), dtype=np.uint8)
        >>> arr = GrabCut.blank_out_middle_thirds(img)
        >>> _ = plt.imshow(arr); plt.show()
        """
        img = img.copy()
        assert img.ndim == 2
        height, width = img.shape
        positions = [(width // 3), (2 * width // 3)]
        img[:, positions[0]:positions[1]] = blank_color
        return img

    @staticmethod
    def compute_display_img(grabmask, img_painted, img_original, color_cst, mode: Literal["4x1col", "2x2"]="2x2",
                            guide_lines=False, verbose=False, previews=None):
        """
        As a doctest : create mini-imgs then show that it correctly stack them
        """
        if verbose: print("[compute_display_img] grabmask", grabmask.shape, np.unique(grabmask))
        mask_display = GrabCut.grabmask_to_displaymask(grabmask)
        if verbose: print("[compute_display_img] displaymask", mask_display.shape, np.unique(mask_display))

        grabmask_human = np.stack((grabmask,) * 3, axis=-1)
        grabmask_human[np.all(grabmask_human == [0, 0, 0], axis=-1)] = color_cst.ALL_PENS["BG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [1, 1, 1], axis=-1)] = color_cst.ALL_PENS["FG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [2, 2, 2], axis=-1)] = color_cst.ALL_PENS["PR_BG"]["color_bgr"]
        grabmask_human[np.all(grabmask_human == [3, 3, 3], axis=-1)] = color_cst.ALL_PENS["PR_FG"]["color_bgr"]
        mask_display_tripliquated = np.stack((mask_display,) * 3, axis=-1)
        if guide_lines:
            mask_display_tripliquated = GrabCut.add_vertical_lines_at_thirds(mask_display_tripliquated, color=(0, 0, 255))
        if verbose: print("[compute_display_img] bitwise and", img_original.shape, mask_display.shape)
        masked_img = cv2.bitwise_and(img_original, img_original, mask=mask_display)

        if mode == "4x1col":
            display_img = np.vstack((img_painted, grabmask_human, mask_display_tripliquated, masked_img))
        elif mode == "2x2":
            display_img = np.vstack((np.hstack((img_painted, mask_display_tripliquated)),
                                     np.hstack((grabmask_human, masked_img))))

        if previews:
            previews_display = np.vstack(previews)
            blank_space = np.zeros((display_img.shape[0] - previews_display.shape[0], previews_display.shape[1], 3), dtype=np.uint8)
            previews_display = np.vstack((previews_display, blank_space))
            display_img = np.hstack((display_img, previews_display))

        return display_img

    @staticmethod
    def save_mask_display(name: str, grabmask: np.ndarray, output_dir: Path, verbose=True,
                          resize_before_save: tuple[int, int] = None, in_thread=True, blank_out_middle_third=False):

        if verbose: print("SAVE A", np.unique(grabmask), grabmask.shape)
        mask_display = GrabCut.grabmask_to_displaymask(grabmask)
        if verbose: print("SAVE B", np.unique(mask_display), mask_display.shape)
        if verbose: print("saving to:", str(output_dir / name))
        output_dir.mkdir(parents=True, exist_ok=True)
        if resize_before_save is not None:
            mask_display = cv2.resize(mask_display, resize_before_save[::-1])  # -1 for flip height and width (torch=h,w, cv2=w,h)
        if blank_out_middle_third:
            mask_display = GrabCut.blank_out_middle_thirds(mask_display)
        if (output_dir / name).exists():
            print(f"WARNING: {output_dir / name} already exists, overwriting it.", file=sys.stderr)
        if in_thread:
            import threading
            t = (threading.Thread(target=cv2.imwrite, args=(str(output_dir / name), mask_display)))
            t.start()
        else:
            cv2.imwrite(str(output_dir / name), mask_display)
        # return output_dir / name
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

    def validate_annotation(self, annotation_csv_path: Path):
        HEADER_CSV = ['img_name', 'value_in_mask', 'label', 'label_id']
        pd.DataFrame(self.building_csv).to_csv(annotation_csv_path,
                                               header=HEADER_CSV, index=False)


    def load_previous_mask(self):
        self.all_corrected_masks = search_for_images(self.output_dir, recursive=True)
        self.grabmask = GrabCut.initial_mask_as_grabmask(self.color_cst, self.all_corrected_masks, self.i_img_in_folder-1, new_h=self.new_hw[0], new_w=self.new_hw[1])

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
            self.save_mask_display(f'frame_{self.i_img_in_folder:06d}_lid{self.label_id}.png',self.grabmask, self.output_dir,
                                   resize_before_save=self.save_shape, blank_out_middle_third=self.blank_out_middle_third)
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
        elif k_tmp == 'p':  # load previous frame mask
            self.load_previous_mask()
        elif k_tmp == 'o':
            self.grabmask = GrabCut.initial_mask_as_grabmask(self.color_cst, self.all_initial_masks,
                                                             self.i_img_in_folder, new_h=self.new_hw[0], new_w=self.new_hw[1])
        elif k_tmp == 'd':
            self.states.pencil['thickness'] = max(1, self.states.pencil['thickness'] - 2)
        elif k_tmp == 'f':
            self.states.pencil['thickness'] = min(100, self.states.pencil['thickness'] + 2)
        elif k_tmp == 'x':  # next img
            self.validate_element()
            if DEBUG: print(f'Unique values of savemask before valid : {np.unique(self.img_display)}')
            #
            #  Le array se buildup et des csv se créent de plus en plus long
            #  Il faudrait que le csv soit créé au lancement du programme avec un timecode + update a chaque "X"
            #
            print(f"ALLEZ SAVE {self.img_name}_lid{self.label_id}.png")
            self.last_corrected_mask = self.save_mask_display(f'frame_{self.i_img_in_folder:06d}_lid{self.label_id}.png',self.grabmask, self.output_dir,
                                   resize_before_save=self.save_shape, blank_out_middle_third=self.blank_out_middle_third)

            if DEBUG: print(f'Number of element in validated img : {np.unique(self.img_display).size}')
            # cv2.imwrite(str(cst.ASSETS_DIR / 'masks' / f'{img_name.stem}_{label}{nth_in_img}.png'), savemask)
            self.nth_elmt_in_img = 1  # because we reset
            self.i_img_in_folder += 1
            print()
            if self.i_img_in_folder >= len(self.all_img_paths):
                self.should_exit_now = True
                exit()  # Exit this listener thread
            self.img_original, self.img_name, self.img_original_shape, self.new_hw = GrabCut.init_image(self.all_img_paths, self.i_img_in_folder, downsize=self.downsize)
            all_buffers = GrabCut.init_buffers_from_img(self.img_original, self.color_cst)
            self.img_painted, self.savemask, self.grabmask, self.img_display = all_buffers
            if self.nb_next_frames > 0:
                self.next_frames = GrabCut.shift_next_frames(self.all_img_paths, self.next_frames, self.i_img_in_folder, self.new_hw[0], self.new_hw[1])
            if self.init_grabcut_with == "mask":
                self.grabmask = GrabCut.initial_mask_as_grabmask(self.color_cst, self.all_initial_masks,
                                                                 self.i_img_in_folder, new_h=self.new_hw[0], new_w=self.new_hw[1])

    def running(self):
        while not self.should_exit_now:  # not typing_label:
            cv2.imshow("Output", self.img_display)
            cv2.setMouseCallback("Output", self.onmouse)
            parent_folder = self.all_img_paths[self.i_img_in_folder].parent
            window_title = f"{parent_folder/self.img_name}, {self.i_img_in_folder}/{len(self.all_img_paths)-1}"
            cv2.setWindowTitle("Output", window_title)
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
            self.img_display = self.compute_display_img(self.grabmask, self.img_painted, self.img_original, self.color_cst, guide_lines=self.guide_lines,
                                                        previews=self.next_frames)
            if k == 32:  # space to save
                print(f'Grabmask = {np.unique(self.grabmask)}')
                display_mask = GrabCut.grabmask_to_displaymask(self.grabmask)
                print(f'Displaymask = {np.unique(display_mask)}')

            #     import matplotlib.pyplot as plt
            #     plt.imshow(mask_display); print(np.unique(mask_display)); plt.show()
            #     plt.imshow(self.savemask); print(np.unique(self.savemask)); plt.show()
            #     plt.imshow(self.grabmask); print(np.unique(self.grabmask)); plt.show()

        cv2.destroyAllWindows()

def run_grabcut_with_rect(input_dir, output_dir):
    grabcut = GrabCut(input_dir, output_dir, init_grabcut_with="rect")
    keyboard.on_press(grabcut.on_press)
    grabcut.running()

def run_grabcut_with_maskfolder(input_dir, output_dir, initial_mask_dir, recursive_img_search=False,
                                downsize: float = False, begin_at_frame : int =0):
    grabcut = GrabCut(input_dir, output_dir, init_grabcut_with="mask", initial_mask_dir=initial_mask_dir,
                      recursive_img_search=recursive_img_search, downsize=downsize, begin_at_frame=begin_at_frame)
    keyboard.on_press(grabcut.on_press)
    grabcut.running()

def run_demo_BDS():
    demo_DBS_path = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper")
    run_grabcut_with_maskfolder(demo_DBS_path / "images",
                                demo_DBS_path / "corrected_masks",
                                demo_DBS_path / "initial_masks")

def framefolder_to_video(video_path, framefolder_path):
    import subprocess
    subprocess.run(f'ffmpeg -framerate 24 -i "{framefolder_path}/frame_%06d.png" -c:v '
                   f'libx264rgb -crf 0 "{video_path}"', shell=True)

def video_to_framefolder(vid_path, framefolder_path):
    import subprocess
    framefolder_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(f'ffmpeg -i "{vid_path}" "{framefolder_path}/%06d.png"', shell=True)

def run_threads_video_to_framefolder(video_dir, frames_dir, BAs: [str], suffix: str = "", verbose=True):
    """
    >>> video_dir = Path("E:/DATA fast label retrain biref/video_ltrt_maskfilttemp_BIREF")
    >>> frames_dir = Path("E:/DATA fast label retrain biref/frames")
    >>> BAs = ["Dune_0", "Moana_0", "Dune2_0", "TopGun_0", "Furiosa_0"]
    >>> suffix = "_BIREF_gk7gi2px160mxt5"
    >>> run_threads_video_to_framefolder(video_dir, frames_dir, BAs, suffix)
    """
    threads = {}
    import threading
    for BA in BAs:
        threads[BA] = (threading.Thread(target=video_to_framefolder, args=(video_dir / f"{BA}{suffix}.mp4", frames_dir / f"{BA}{suffix}")))
    for BA in BAs:
        threads[BA].start()
        if verbose: print(f"thread {BA} started")
    for BA in BAs:
        threads[BA].join()
        if verbose: print(f"thread {BA} finished")

def concat_vertically_mask_video_and_video(video_dir, mask_dir, output_dir, suffix: str = ""):
    """C'était déjà fait faut juste que je fasse un alias vers la methode qui a déjà été faite"""
    concat_vertical_videos()

def truc():
    pass
    # all_frames = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
    # all_masks = list(initial_mask_dir.rglob("*.jpg")) + list(initial_mask_dir.rglob("*.png"))
    #
    # all_frames = [str(f).split('_0_')[-1] for f in all_frames]
    # all_masks = [str(f).split('_0_')[-1] for f in all_masks]
    # #
    # # #replace bugged backslach by double underscore
    # # all_frames = [f.replace('\\', '__') for f in all_frames]
    # # all_masks = [f.replace('\\', '__') for f in all_masks]
    # #
    # # # all_frames = [f for f in all_frames if "927" in f]
    # # # all_masks = [f for f in all_masks if "927" in f]
    # # all_frames = [f for f in all_frames if "908" in f]
    # # all_masks = [f for f in all_masks if "908" in f]
    # # all_frames = [f for f in all_frames if "880" in f]
    # # all_masks = [f for f in all_masks if "880" in f]
    #
    # all_frames = sorted(all_frames)
    # all_masks = sorted(all_masks)
    #
    # # Now create sets of these two lists
    # set_frames = set(all_frames)
    # set_masks = set(all_masks)
    #
    # # Now print the frames that are not in the masks
    # print(len(set_frames - set_masks))
    # frames = []
    # for frame in (set_frames - set_masks):
    #     frames.append(frame)
    # for frame in human_sorted(frames):
    #     print(frame)
    #
    # assert len(all_frames) == len(all_masks), f"{len(all_frames)} != {len(all_masks)}"
    # exit()

if __name__ == '__main__':
    # vid_dir = Path("E:/ICE_CUBED_RESULTS/ltrt_maskfilttemp_BIREF")
    # frames_dir = vid_dir
    # run_threads_video_to_framefolder(vid_dir, vid_dir, ["Dune_0", "Moana_0", "Dune2_0", "TopGun_0", "Furiosa_0", 'ROYAUME_0'],
    #                                  suffix="_BIREFmxt3")
    # vid_path = Path("E:/ICE_CUBED_RESULTS/ltrt_mask_BIREF/ROYAUME_0_BIREF/ROYAUME_0_BIREF.mp4")
    # framefolder_path = Path("E:/ICE_CUBED_RESULTS/ltrt_mask_BIREF/ROYAUME_0_BIREF")
    # video_to_framefolder(vid_path, framefolder_path)


    # input_dir = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper/images")
    # initial_mask_dir = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper/initial_masks")
    # output_dir = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper/corrected_masks")
    #
    # begin_at_frame = 0
    # run_grabcut_with_maskfolder(input_dir, output_dir, initial_mask_dir, recursive_img_search=True, downsize=1, begin_at_frame=begin_at_frame)
    # exit()
    #
    run_demo_BDS()
    exit()

    video = ["Dune_0", "Moana_0", "Dune2_0", "TopGun_0", "Furiosa_0", 'ROYAUME_0']
    for v in video:
        input_dir = Path("E:/ICE_CUBED_RESULTS/frames/" + v)
        suffix = "_BIREFmxt3"
        initial_mask_dir = Path(f"E:/ICE_CUBED_RESULTS/ltrt_maskfilttemp_BIREF/{v}{suffix}")
        output_dir = Path(f"E:/DATA fast label retrain biref/corrected_masks/{v}{suffix}")

        begin_at_frame = 0
        run_grabcut_with_maskfolder(input_dir, output_dir, initial_mask_dir, recursive_img_search=True, downsize=3, begin_at_frame=begin_at_frame)
