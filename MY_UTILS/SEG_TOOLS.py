import numpy as np
import cv2
from typing import Tuple


# reference: utils/segment/general.py


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def crop_mask(masks: np.ndarray, boxes: np.ndarray):
    """
    Args:
        masks: [n, h, w]
        boxes: [n, 4]
    """
    x1 = boxes[:, 0, None, None]
    y1 = boxes[:, 1, None, None]
    x2 = boxes[:, 2, None, None]
    y2 = boxes[:, 3, None, None]
    _, h, w = masks.shape
    r = np.arange(h)[None, :, None]  # [1, h, 1]
    c = np.arange(w)[None, None, :]  # [1, 1, c]
    return masks * ((c >= x1) * (c < x2) * (r >= y1) * (r < y2))


def process_mask(protos: np.ndarray, masks_in: np.ndarray, bboxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        protos: [mask_dim, mask_h, mask_w]
        masks_in: [n, mask_dim] (n: number of masks after nms)
        bboxes: [n, 4]
        shape: input_image_shape, (h, w)

    Returns:
        numpy array, [h, w, n], np.int32
    """
    _, mh, mw = protos.shape
    masks = sigmoid(np.tensordot(masks_in, protos, (-1, 0)))  # [n, mask_h, mask_w]
    ih, iw = shape
    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, [0, 2]] *= mw / iw
    downsampled_bboxes[:, [1, 3]] *= mh / ih
    masks = crop_mask(masks, downsampled_bboxes)

    masks_upsample = []
    for mask in masks:
        masks_upsample.append(cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_LINEAR))
    return np.int32(np.stack(masks_upsample) > 0.5)
