import torch
import torchvision


# from yolov6/core/inferer.py
def NMS(pred: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45,
        agnostic: bool = False) -> torch.Tensor:
    # pred [N, 5 + num_classes],
    # N: number of bboxes, 5 = (center_x. center_y, width, height, obj_conf)

    # agnostic True: do class-independent nms
    # agnostic False: different classes do nms respectively

    # output: (x1, y1, x2, y2, conf, cls_id)
    assert pred.ndim == 2
    # num_classes = pred.shape[-1] - 5
    obj_conf = pred[:, 4]
    cls_conf, cls_id = pred[:, 5:].max(1)
    conf = obj_conf * cls_conf
    mask = conf > conf_threshold

    bbox_cxcy = pred[:, :2]  # [N, 2]
    bbox_half_wh = 0.5 * pred[:, 2:4]  # [N, 2]
    xyxy_conf_cls = torch.cat((
        bbox_cxcy - bbox_half_wh,
        bbox_cxcy + bbox_half_wh,
        conf[:, None], cls_id[:, None]
    ), dim=1)  # [N, 6]
    xyxy_conf_cls_filtered = xyxy_conf_cls[mask]
    if not xyxy_conf_cls_filtered.size(0):
        return xyxy_conf_cls_filtered

    # NMS
    if agnostic:
        keep_indices = torchvision.ops.nms(xyxy_conf_cls_filtered[:, :4],
                                           xyxy_conf_cls_filtered[:, 4],
                                           iou_threshold)
    else:
        keep_indices = torchvision.ops.batched_nms(xyxy_conf_cls_filtered[:, :4],
                                                   xyxy_conf_cls_filtered[:, 4],
                                                   xyxy_conf_cls_filtered[:, 5].long(),
                                                   iou_threshold)
    return xyxy_conf_cls_filtered[keep_indices]
