from darknet_utils import layers
from darknet_utils.layers import DarknetLayer
from darknet_utils.parse_config import parse_model_cfg
from typing import List, Optional, Dict
from collections import OrderedDict
import cupy as xp


def create_conv(cfg_dic: dict, in_ch: int, fp):
    use_bn = cfg_dic['batch_normalize']
    out_ch = cfg_dic['filters']
    ksize = cfg_dic['size']
    stride = cfg_dic['stride']
    pad = bool(cfg_dic['pad'])
    act_type = cfg_dic['activation']
    act = layers.ACTIVATION_FUNCTION[act_type]
    conv = layers.Convolutional(in_ch=in_ch, out_ch=out_ch, ksize=ksize, stride=stride, pad=pad, use_bn=use_bn, act=act)
    conv.set_weights_from_bytes(fp)
    return conv


def create_model(model_cfg_dic: List[dict],
                 weight_file: str,
                 input_height: Optional[int] = None,
                 input_width: Optional[int] = None):
    modules = OrderedDict()
    with open(weight_file, 'rb') as fp:
        xp.fromfile(fp, dtype=xp.int32, count=5)

        cfg_dic_iter = iter(model_cfg_dic)
        init_cfg = next(cfg_dic_iter)
        assert init_cfg['type'] == 'net'
        prev_ch = init_cfg['channels']

        if input_height is None:
            input_height = init_cfg['height']
        if input_width is None:
            input_width = init_cfg['width']

        save = set()
        input_indices = dict()
        chs = []

        for i, curr_cfg in enumerate(cfg_dic_iter):
            block_type = curr_cfg['type']
            if block_type == 'convolutional':
                modules[f'm{i}_conv'] = create_conv(curr_cfg, prev_ch, fp)
                prev_ch = curr_cfg['filters']
            elif block_type == 'upsample':
                scale_factor = float(curr_cfg['stride'])
                modules[f'm{i}_upsample'] = layers.UpsampleBilinear(scale_factor=scale_factor)
            elif block_type == 'maxpool':
                stride = curr_cfg['stride']
                ksize = curr_cfg['size']
                # https://github.com/eriklindernoren/PyTorch-YOLOv3.git
                # pytorchyolo/models.py
                modules[f'm{i}_maxpool'] = layers.MaxPool2D(ksize, stride)
            elif block_type == 'route':
                modules[f'm{i}_route'] = layers.Route()
                iidxs = [idx if idx >= 0 else i + idx for idx in curr_cfg['layers']]
                save |= set(idx for idx in iidxs if idx != i - 1)
                input_indices[i] = iidxs
                prev_ch = sum(chs[idx] for idx in iidxs)
            elif block_type == 'shortcut':
                act = layers.ACTIVATION_FUNCTION[curr_cfg['activation']]
                modules[f'm{i}_shortcut'] = layers.ShortCut(act)
                iidxs = [idx if idx >= 0 else i + idx for idx in curr_cfg['from']]
                assert i - 1 not in iidxs
                save |= set(iidxs)
                iidxs.append(i - 1)
                input_indices[i] = iidxs
            elif block_type == 'yolo':
                mask = curr_cfg['mask']
                anchor = curr_cfg['anchors'][mask].tolist()
                scale_x_y = float(curr_cfg.get('scale_x_y', 1))
                new_coords = curr_cfg.get('new_coords', 0)
                modules[f'm{i}_yolodetect'] = layers.YoloDetect(anchor, input_height, input_width, scale_x_y,
                                                                new_coords)
            else:
                raise RuntimeError('bad block type')

            chs.append(prev_ch)
    return modules, save, input_indices, (input_height, input_width)


class YOLO(DarknetLayer):
    def __init__(self, cfg_file: str, weight_file: str,
                 input_height: Optional[int] = None, input_width: Optional[int] = None):
        super(YOLO, self).__init__()
        model_cfg_dic = parse_model_cfg(cfg_file)
        self._modules, self.save, self.input_indices, (self.input_height, self.input_width) = \
            create_model(model_cfg_dic, weight_file, input_height, input_width)

    def __call__(self, x: xp.ndarray):
        saved_array: Dict[int, xp.ndarray] = dict()
        yolo_results: List[xp.ndarray] = []
        for i, module in enumerate(self._modules.values()):
            if i not in self.input_indices:
                x = module(x)
            else:
                input_arrays = []
                for idx in self.input_indices[i]:
                    assert idx >= 0
                    if idx == i - 1:
                        input_arrays.append(x)
                    else:
                        input_arrays.append(saved_array[idx])
                x = module(input_arrays)
            if i in self.save:
                saved_array[i] = x
            if isinstance(module, layers.YoloDetect):
                yolo_results.append(x)
        return xp.concatenate(yolo_results, axis=1)


def _check():
    import time
    cfg_file = '/home/a/PROJ/AlexeyAB/darknet/cfg/yolov3.cfg'
    weight_file = '/home/a/PROJ/AlexeyAB/darknet/weights/yolov3.weights'
    yolo = YOLO(cfg_file=cfg_file, weight_file=weight_file)
    HIEGHT, WIDTH = yolo.input_height, yolo.input_width
    x = xp.random.rand(1, 3, HIEGHT, WIDTH).astype(xp.float32)
    # print(f'height = {HIEGHT}, width = {WIDTH}')

    for _ in range(10):
        tic = time.time()
        outputs = yolo(x)
        print(f'{(time.time() - tic) * 1000:.4f} ms')


if __name__ == '__main__':
    _check()
