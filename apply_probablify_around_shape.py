from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fast_label_v2.test_cv_grabcut_refacto_v2 import GrabCut

demo_object_mask = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper/initial_masks/frame_000001.png")

def contours_filter(img: np.ndarray) -> np.ndarray:
    """
    >>> img = cv2.imread(str(demo_object_mask), cv2.IMREAD_GRAYSCALE)
    >>> img = contours_filter(img)
    >>> plt.imshow(img, cmap="gray"); plt.show()
    """
    canny_filtered_im = cv2.Canny(img, 30, 200)
    return canny_filtered_im

def grow_shape_with_PR_BG_and_PR_FG(mask_uint8x1: np.ndarray, grown_contour_mask: np.ndarray, verbose=True):
    """
    >>> object_mask = cv2.imread(str(demo_object_mask), cv2.IMREAD_GRAYSCALE)
    >>> contour_mask = contours_filter(mask_uint8x1)
    >>> print(np.unique(mask_uint8x1))
    >>> _ = plt.imshow(mask_uint8x1, cmap="gray"); plt.show()
    >>> print(np.unique(contour_mask))
    >>> _ = plt.imshow(contour_mask, cmap="gray"); plt.show()
    >>> grown_contour_mask = cv2.dilate(contour_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    >>> _ = plt.imshow(grown_contour_mask, cmap="gray"); plt.show()
    >>> C, D = grow_shape_with_PR_BG_and_PR_FG(mask_uint8x1, grown_contour_mask)
    >>> print(np.unique(C))
    >>> _ = plt.imshow(C, cmap="gray"); plt.show()
    >>> print(np.unique(D))
    >>> _ = plt.imshow(D, cmap="gray"); plt.show()
    """
    probablified_BG = (mask_uint8x1 < 127.5) & (grown_contour_mask > 127.5)
    probablified_BG = probablified_BG.astype(np.uint8) * 255
    probablified_FG = (mask_uint8x1 > 127.5) & (grown_contour_mask > 127.5)
    probablified_FG = probablified_FG.astype(np.uint8) * 255
    return probablified_BG, probablified_FG

def grabmask_to_binarymask(mask: np.ndarray):
    binarymask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    return binarymask


def apply_probablify_pen_around_shape(grabmask: np.ndarray, external_probablify=True, internal_probablify=False,
                                      value_BG=0, value_FG=1, value_PR_BG=2, value_PR_FG=3, debug=False, ksize=15):
    """
    display mask is a 8bit mask [0->255]. grabmask is a 2bit mask [0->3]
    >>> mask8bit = cv2.imread(str(demo_object_mask), cv2.IMREAD_GRAYSCALE)
    >>> grabmask = GrabCut.mask8bit_to_grabmask(mask8bit)
    >>> bgrgrabmask = GrabCut.grabmask_to_rgbgrabmask(grabmask, [141, 96, 51], [30, 225, 202], [121, 183, 53], [63, 215, 149])
    >>> rgbgrabmask = cv2.cvtColor(bgrgrabmask, cv2.COLOR_BGR2RGB)
    >>> plt.figure(1); _ = plt.imshow(rgbgrabmask);
    >>> grabmask = apply_probablify_pen_around_shape(grabmask)
    >>> bgrgrabmask = GrabCut.grabmask_to_rgbgrabmask(grabmask, [141, 96, 51], [30, 225, 202], [121, 183, 53], [63, 215, 149])
    >>> rgbgrabmask = cv2.cvtColor(bgrgrabmask, cv2.COLOR_BGR2RGB)
    >>> plt.figure(2); _ = plt.imshow(rgbgrabmask); plt.show()
    """
    assert len(np.unique(grabmask)) <= 4
    binarymask = grabmask_to_binarymask(grabmask)
    contour_mask = contours_filter(binarymask)
    grown_contour_mask = cv2.dilate(contour_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)))
    change_to_PRBG, change_to_PRFG = grow_shape_with_PR_BG_and_PR_FG(binarymask, grown_contour_mask)
    if external_probablify: grabmask[change_to_PRBG == 255] = value_PR_BG
    if internal_probablify: grabmask[change_to_PRFG == 255] = value_PR_FG
    return grabmask


if __name__ == '__main__':
    pass

    # img_path = Path("C:/Users/David Traparic/Documents/prog/fast_label/demo_data/DragonBallSuper/initial_masks/frame_000001.png")
    # img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # img = contours_filter(img)
    # growing_filter = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    # dilated = cv2.dilate(img, growing_filter)
    #
    # plt.imshow(dilated, cmap="gray"); plt.show()



