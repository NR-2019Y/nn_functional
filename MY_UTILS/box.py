import numpy as np
import cv2
from MY_UTILS.config import ROOT

def letter_box(image: np.ndarray,
               new_shape=(640, 640),  # (height, width)
               color=114):
    ori_height, ori_width = image.shape[:2]
    new_height, new_width = new_shape
    out_image = np.full((new_height, new_width, 3), color, dtype=np.uint8)
    ratio = min(new_height / ori_height, new_width / ori_width)

    unpad_height = round(ratio * ori_height)
    unpad_width = round(ratio * ori_width)
    image_unpad = cv2.resize(image, (unpad_width, unpad_height))

    top = (new_height - unpad_height) // 2
    left = (new_width - unpad_width) // 2
    out_image[top:top + unpad_height, left:left + unpad_width] = image_unpad
    return out_image, ratio, (left, top)
