import cv2
import numpy as np


def xyxy2xywh(xyxy: np.ndarray):
    assert xyxy.ndim == 2
    return np.concatenate((
        xyxy[:, 0:2],
        xyxy[:, 2:4] - xyxy[:, 0:2]
    ), axis=1)


def NMS(pred: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
        agnostic: bool = False, nm: int = 0, top_k: int = 1000):
    # pred: [N, 5 + num_classes + nm]
    # N: number of bboxes
    # 5 = (center_x, center_y, width, height, obj_conf)

    # agnostic: if True: do class-independent nms, else different classes do nms respectively

    # output: (x1, y1, x2, y2, conf, cls_id, masks if nm != 0)
    assert pred.ndim == 2
    obj_conf = pred[:, 4]
    num_bboxes = len(pred)
    cls_slice = slice(5, None) if nm == 0 else slice(5, -nm)
    cls_id = np.argmax(pred[:, cls_slice], axis=1)
    cls_conf = pred[range(num_bboxes), 5 + cls_id]

    scores = obj_conf * cls_conf

    masks = None
    if nm != 0:
        masks = pred[:, -nm:]

    bbox_cxcy = pred[:, 0:2]  # [N, 2]
    bbox_half_wh = 0.5 * pred[:, 2:4]  # [N, 2]
    bbox = np.concatenate((
        bbox_cxcy - bbox_half_wh,
        bbox_cxcy + bbox_half_wh
    ), axis=1)  # [N, 4]
    bbox_xywh = xyxy2xywh(bbox).astype(np.int32)

    max_wh = 4096
    if not agnostic:
        bbox_xywh[:, 0:2] += cls_id[:, None] * max_wh
    keep_indices = cv2.dnn.NMSBoxes(bbox_xywh, scores, conf_threshold, iou_threshold, top_k=top_k)
    if not len(keep_indices):
        return np.zeros((0, 6), dtype=pred.dtype)
    if nm == 0:
        output = np.concatenate((
            bbox[keep_indices], scores[keep_indices, None], cls_id[keep_indices, None]
        ), axis=1, dtype=np.float32)
    else:
        output = np.concatenate((
            bbox[keep_indices], scores[keep_indices, None], cls_id[keep_indices, None], masks[keep_indices]
        ), axis=1, dtype=np.float32)
    return output
