from pathlib import Path
from path_config import PATHS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from global_path_root import project_root

def add_two_grabmasks(grabmask1: np.ndarray, grabmask2: np.ndarray):
    """
    >>> grabmask1 = cv2.imread(str(project_root / 'demo_data' / 'Dune' / 'initialmask_000704.png'), cv2.IMREAD_GRAYSCALE)
    >>> grabmask2 = cv2.imread(str(project_root / 'demo_data' / 'Dune' / 'correctedmask_000704.png'), cv2.IMREAD_GRAYSCALE)
    >>> grabmask3 = add_two_grabmasks(grabmask1, grabmask2)
    >>> _ = plt.subplot(311); _ = plt.imshow(grabmask1);
    >>> _ = plt.subplot(312); _ = plt.imshow(grabmask2);
    >>> _ = plt.subplot(313); _ = plt.imshow(grabmask3); plt.show()
    """
    assert isinstance(grabmask1, np.ndarray)
    assert isinstance(grabmask2, np.ndarray)
    return grabmask1 | grabmask2

if __name__ == '__main__':
    grabmask1 = cv2.imread(str(project_root / 'demo_data' / 'Dune' / 'initialmask_000704.png'), cv2.IMREAD_GRAYSCALE)
