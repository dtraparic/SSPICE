"""
Change a folder from :
frames/
    -> shot_1/
        -> frame_1.png
        -> frame_2.png
    -> shot_2/
        -> frame_1.png
        -> frame_2.png
        -> frame_3.png
to
frames/
    -> frame_1.png
    -> frame_2.png
    -> frame_3.png
    -> frame_4.png
    -> frame_5.png
"""

from pathlib import Path

import cv2
from path_config import PATHS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def from_multiframefolder_to_framefolder(multiframefolder_root: Path, framefolder: Path):
    from fast_label_v2.test_cv_grabcut_refacto_v2 import search_for_images
    from tqdm import tqdm
    all_img_paths = search_for_images(multiframefolder_root, recursive=True, humanly_sort=True)
    for i, img_path in enumerate(tqdm(all_img_paths)):
        data = img_path.read_bytes()
        new_name = f"frame_{i:06d}.png"
        new_path = framefolder / new_name
        new_path.write_bytes(data)

def remove_lid_from_framefolder(framefolder: Path):
    from tqdm import tqdm
    framefolder = Path(framefolder)
    all_img_paths = list(framefolder.glob("frame_*.png"))
    for i, img_path in enumerate(tqdm(all_img_paths)):
        new_name  = img_path.name.split("_lid")[0]
        new_path = framefolder / f"{new_name}.png"
        img_path.rename(new_path)


if __name__ == '__main__':
    multiframefolder_path = Path("E:/ICE_CUBED_RESULTS/frames/TopGun_0")
    Path("E:/DATA fast label retrain biref/images/TopGun_0").mkdir(exist_ok=True)
    framefolder_path = Path("E:/DATA fast label retrain biref/images/TopGun_0")
    from_multiframefolder_to_framefolder(multiframefolder_path, framefolder_path)

    # truc = Path("E:/DICE Dataset/DICE_VD/gt")
    # remove_lid_from_framefolder(truc)