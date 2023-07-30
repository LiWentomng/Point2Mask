import torch
from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector
from mmdet.utils.contextmanagers import concurrent
from easymd.apis.test import _id2color, get_pan_edge
from mmdet.datasets import build_dataset
import cv2
from mmcv import Config
import numpy as np
    
def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--config-file', 
        default='./configs/point2mask/voc/point2mask_voc_wsup_r50.py',
        help='Config file')

    parser.add_argument(
        '--weights', 
        default=None, 
        help='Checkpoint file')

    parser.add_argument(
        '--input', 
        default=None, 
        help='Input image')

    parser.add_argument(
        '--out-file',
        type=str,
        default='output.png',
        help='Output file of images or prediction results.')

    args = parser.parse_args()
    model = init_detector(args.config_file, args.weights, device='cuda:0')
    result = inference_detector(model, args.input)
    pan_res = result['pan_results']

    cfg = Config.fromfile(args.config_file)
    dataset = build_dataset(cfg.data.test)
    pan = _id2color(dataset, pan_res).astype(np.uint8)

    pan_edge = get_pan_edge(pan)
    img_show = cv2.imread(args.input)
    
    pan = cv2.addWeighted(img_show, 0.4, pan, 0.6, 0)
    edge = (255, 250, 240)
    pan[pan_edge] = edge
    cv2.imwrite(args.out_file, pan)


if __name__ == '__main__':
    main()