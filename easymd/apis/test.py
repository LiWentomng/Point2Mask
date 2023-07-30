import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from collections import defaultdict
import time
import numpy as np
from easymd.datasets.panopticapi.utils import get_traceback, IdGenerator, id2rgb, rgb2id, save_json
from easymd.datasets.coco_panoptic import id_and_category_maps as coco_categories_dict
import os
import PIL.Image as Image
import json
import cv2

INSTANCE_OFFSET = 1000
def _id2color(dataset, id_map):
    idgenerator = IdGenerator(dataset.categories)
    label2cat = dict((v, k) for k, v in dataset.cat2label.items())

    if isinstance(id_map, np.ndarray):
        unique_ids = np.unique(id_map)
        color_lookup = np.zeros((max(unique_ids) + 1, 3), np.uint8)
        for i in unique_ids:
            try:
                L = label2cat[i % INSTANCE_OFFSET]
            except KeyError:
                # VOID color
                color_lookup[i] = (225, 225, 196)
                continue
            color_lookup[i] = idgenerator.get_color(L)
        return color_lookup[id_map.ravel()].reshape(*id_map.shape, 3)
    else:
        return idgenerator.get_color(label2cat[id_map % INSTANCE_OFFSET])

def get_pan_edge(pan):
    assert isinstance(pan, np.ndarray), type(pan)
    assert pan.dtype == np.uint8, pan.dtype
    assert pan.ndim == 3 and pan.shape[-1] == 3, pan.shape

    edges = []
    for c in range(3):
        x = pan[..., c]
        edge = cv2.Canny(x, 1, 2)
        edges.append(edge)
    #edges = np.abs(np.array(edges)).max(0) > 0.01
    edges = np.array(edges).max(0)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges > 0

def multi_gpu_test_vis(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                pan_res = result[i]['pan_results']
                pan = _id2color(dataset, pan_res).astype(np.uint8)
                pan_edge = get_pan_edge(pan)
                pan = cv2.addWeighted(img_show, 0.4, pan, 0.6, 0)
                edge = (255, 250, 240)
                pan[pan_edge] = edge
                if not osp.exists(out_dir):
                    os.mkdir(out_dir)
                cv2.imwrite(os.path.join(out_dir,img_meta['ori_filename']), pan)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

    