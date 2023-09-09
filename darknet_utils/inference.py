import time
import numpy as np
import cupy as cp
from darknet_utils.yolo import YOLO


def main():
    import cv2
    import glob
    import os
    from MY_UTILS.NMS_NUMPY import NMS
    from MY_UTILS.coco import COCO_CLASS_NAMES as CLASS_NAMES

    # IMAGE_DIR = '/mnt/d/GIT_REPO/datasets/coco128/images/train2017'
    IMAGE_DIR = '/mnt/d/GIT_REPO/datasets/VOC2007/images'
    IMAGE_PATHS = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    # IMAGE_PATHS = ['/home/a/PROJ/SEG/darknet-to-pytorch/src/dog-cycle-car.png']
    NUM_CLASSES = 80
    COLORS = np.random.randint(0, 256, (NUM_CLASSES, 3)).tolist()
    CONF_THRESHOLD = 0.45
    IOU_THRESHOLD = 0.4
    MAX_DETS = 100

    PREFIX = 'yolov4'
    cfg_file = f'/home/a/PROJ/AlexeyAB/darknet/cfg/{PREFIX}.cfg'
    weight_file = f'/home/a/PROJ/AlexeyAB/darknet/weights/{PREFIX}.weights'
    yolo = YOLO(cfg_file, weight_file, input_height=None, input_width=None)
    HEIGHT, WIDTH = yolo.input_height, yolo.input_width

    # from utils.darknet import Darknet
    # yolo = Darknet(cfg_file, weights_file)

    cv2.namedWindow('W', cv2.WINDOW_NORMAL)
    for img_path in IMAGE_PATHS:
        tic = time.time()
        ori_image = cv2.imread(img_path)
        ori_height, ori_width = ori_image.shape[:2]
        ratio = min(WIDTH / ori_width, HEIGHT / ori_height)
        new_width = round(ori_width * ratio)
        new_height = round(ori_height * ratio)
        padw = WIDTH - new_width
        padh = HEIGHT - new_height
        left = padw // 2
        top = padh // 2
        img_pad = np.pad(cv2.resize(ori_image, (new_width, new_height)),
                         pad_width=[[top, padh - top], [left, padw - left], [0, 0]],
                         mode='constant', constant_values=0)
        input_data = np.transpose(np.float32(img_pad) / 255., (2, 0, 1))[None]
        input_array = cp.asarray(input_data)

        output = yolo(input_array)
        output = cp.asnumpy(output)
        obj_dets = NMS(output[0], CONF_THRESHOLD, IOU_THRESHOLD, top_k=MAX_DETS, agnostic=True)
        obj_dets[:, [0, 2]] -= left
        obj_dets[:, [1, 3]] -= top
        obj_dets[:, :4] /= ratio
        print(f"time: {(time.time() - tic) * 1000:.4f} ms")
        im_draw = ori_image.copy()
        for *xyxy, conf, cls in obj_dets:
            x1, y1, x2, y2 = map(round, xyxy)
            cls = int(cls)
            text = f'{CLASS_NAMES[cls]}:{conf:.4f}'
            # text = f'{conf:.4f}'
            FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
            FONT_SCALE = 0.8
            FONT_THICKNESS = 2
            (fw, fh), fb = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(im_draw, (x1, y1 - fh - fb), (x1 + fw, y1), COLORS[cls], -1)
            cv2.rectangle(im_draw, (x1, y1), (x2, y2), COLORS[cls], 2)
            cv2.putText(im_draw, text, (x1, y1 - fb), FONT_FACE, FONT_SCALE, [0, 0, 0], FONT_THICKNESS)
        cv2.imshow('W', im_draw)
        if cv2.waitKey(0) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
