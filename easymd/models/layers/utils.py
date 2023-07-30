import torch
import numpy as np
import random
import copy
from types import GeneratorType

import os
import cv2
from panopticapi.utils import IdGenerator
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET

from mmcv.runner.hooks import Hook, HOOKS
# training-time visualizer


def as_list(x):
    return x if isinstance(x, list) else [x]

def _get_rgb_image(image, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
        assert image.ndim == 3 and image.shape[2] == 3, image.shape
        return image

    assert image.ndim == 3 and image.shape[0] == 3, image.shape
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    if image.max() > 1.5:
        image = np.clip(image, a_min=0, a_max=255)
    else:
        image = np.clip(image, a_min=0, a_max=1) * 255
    image = image.astype(np.uint8)
    return image.copy()


def _get_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

def visualize_raw(image):
    if isinstance(image, torch.Tensor):
        image = image.data.cpu().numpy()
    assert image.ndim  == 3, image.shape

    if image.shape[0] == 3 and image.shape[2] != 3:
        image = image.transpose(1, 2, 0)
    assert image.shape[2] == 3, image.shape

    if image.dtype in [np.uint8]:
        return image
    if image.max() < 1.001:
        image = (np.clip(image, a_min=0, a_max=1) * 255).astype(np.uint8)
    else:
        image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)
    return image

def visualize_panoptic(image, pan_result, id2rgb, edge=(255, 255, 255)):
    """
    Args:
        image: [tensor | None], if Tensor then of Shape [3, h, w], mean-std removed
        pan_result: tensor, of Shape [h, w]
    Returns:
        pan: np.NDArray, of Shape [h, w, 3], dtype np.uint8
    """
    assert pan_result.ndim == 2, pan_result.shape
    if isinstance(pan_result, torch.Tensor):
        pan_result = pan_result.data.cpu().numpy()
    pan = id2rgb(pan_result).astype(np.uint8) 
    if edge is not None:
        pan_edge = get_pan_edge(pan)

    if image is not None:
        image = _get_rgb_image(image)
        # print(image.shape)
        # print(pan.shape)
        # import pdb
        # pdb.set_trace()
        assert image.shape == pan.shape, (image.shape, pan.shape)
        pan = cv2.addWeighted(image, 0.2, pan, 0.8, 0)

    if edge is not None:
        pan[pan_edge] = edge
    return pan

def get_pan_edge(pan):
    assert isinstance(pan, np.ndarray), type(pan)
    assert pan.dtype == np.uint8, pan.dtype
    assert pan.ndim == 3 and pan.shape[-1] == 3, pan.shape

    edges = []
    for c in range(3):
        x = pan[..., c]
        #edge = cv2.Sobel(x, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
        edge = cv2.Canny(x, 1, 2)
        edges.append(edge)
    #edges = np.abs(np.array(edges)).max(0) > 0.01
    edges = np.array(edges).max(0)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return edges > 0

def visualize_bbox(image, bboxes, labels, id2rgb):
    """
    Args:
        image: tensor of Shape [3, h, w]
        bboxes: tensor of shape [N, 4] or [N, 5], each of coordinates 
              (tl_x, tl_y, br_x, br_y) or (tl_x, tl_y, br_x, br_y, score)
        labels: [ tensor | None ], if tensor, then of shape [N,]
    Returns:
        out: np.NDArray, of Shape [h, w, 3], dtype np.uint8
    """

    assert image is not None
    image = _get_rgb_image(image)
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.data.cpu().numpy()
    boxes = bboxes[:, :4].astype(np.int64)
    if labels is not None:
        labels = labels.data.cpu().numpy()

    out = image.copy()
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        c = _get_random_color() if labels is None else id2rgb(int(labels[i]))
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 5)

    return out


def visualize_heatmap(image, heatmap, normalize=True):
    assert heatmap.ndim == 2, heatmap.shape
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.data.cpu().numpy()

    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
    heatmap = np.clip(heatmap, a_min=0, a_max=1)
    heatmap = (heatmap * 255.99).astype(np.uint8)
    out = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)[..., ::-1]

    if image is not None:
        image = _get_rgb_image(image)
        assert image.shape == out.shape, (image.shape, out.shape)
        out = cv2.addWeighted(image, 0.2, out, 0.8, 0)
    return out

def imvstack(images, width=None, space=3):
    assert isinstance(images, (list, tuple)), type(images)
    assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
    images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
    assert all([x.ndim ==3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]

    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    
    W = max(x.shape[1] for x in images) if width is None else width
    for i in range(len(images)):
        h, w = images[i].shape[:2]
        H = h * W // w
        images[i] = cv2.resize(images[i], (W, H))
    
    if space > 0:
        margin = np.full((space, W, 3), 255, np.uint8)
        images = sum([[img, margin] for img in images], [])[:-1]
    out = np.vstack(images)
    return out

def imhstack(images, height=None, space=3):
    assert isinstance(images, (list, tuple)), type(images)
    assert all([isinstance(x, np.ndarray) for x in images]), [type(x) for x in images]
    images = [x[..., None].repeat(3, -1) if x.ndim == 2 else x for x in images]
    assert all([x.ndim == 3 and x.shape[2] == 3 for x in images]), [x.shape for x in images]
    out = imvstack([x.transpose(1, 0, 2) for x in images], width=height, space=space)
    if out is not None:
        out = out.transpose(1, 0, 2)
    return out


@HOOKS.register_module()
class VisualizationHook(Hook):
    def __init__(self, dataset, interval):
        self.interval = interval
        self.cache = []

        if dataset is not None:
            self.idgenerator = IdGenerator(dataset.categories)
            self.dataset = dataset
        else:
            raise NotImplementedError

    def _id2color(self, id_map):
        idgenerator = copy.deepcopy(self.idgenerator)
        label2cat = dict((v, k) for k, v in self.dataset.cat2label.items())

        if isinstance(id_map, np.ndarray):
            unique_ids = np.unique(id_map)
            color_lookup = np.zeros((max(unique_ids)+1, 3), np.uint8)
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

    def _dump_image(self, image, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, image[..., ::-1])

    def _should_visualize(self, runner):
        if runner.rank != 0:
            return False
        return self.every_n_iters(runner, self.interval)

    def _get_visualization(self, result, with_image=True):
        image = _get_rgb_image(result['image'])
        out = [image] if with_image else []
        if 'pan_results' in result:
            for pan_results in as_list(result['pan_results']):
                out.append(visualize_panoptic(image, pan_results, self._id2color))
        if 'bboxes' in result:
            labels = result.get('labels', None)
            if labels is not None:
                labels = as_list(labels)
            for i, bboxes in enumerate(as_list(result['bboxes'])):
                label = None if labels is None else labels[i]
                out.append(visualize_bbox(image, bboxes, label, self._id2color))
        if 'heatmaps' in result:
            for heatmaps in as_list(result['heatmaps']):
                out.append(visualize_heatmap(image, heatmaps))
        if 'raw' in result:
            for raw_image in as_list(result['raw']):
                out.append(visualize_raw(raw_image))
        if 'boundary' in result:
            for bound in as_list(result['boundary']):
                out.append((bound*255).int().data.cpu().numpy())

        assert len(out) > 0, result.keys()
        return out

    def after_train_iter(self, runner):
        if not self._should_visualize(runner):
            return

        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.get_visualization() #raw

        if isinstance(results, dict):
            results = [results]

        out = []
        for i, result in enumerate(results):
            assert isinstance(result, dict), type(result)
            out += self._get_visualization(result, i == 0)
        assert len(out) > 0, [x.keys() for x in results]
        out = imhstack(out, 320)
        self.cache.append(out)
        self._dump_image(out, os.path.join(runner.work_dir, 'visualization', 'last.jpg'))

    def after_train_epoch(self, runner):
        if runner.rank != 0:
            return

        if len(self.cache) == 0:
            return

        out = imvstack(self.cache)
        self._dump_image(out, os.path.join(runner.work_dir, 'visualization', f'epoch{runner.epoch}.jpg'))
        self.cache = []

@HOOKS.register_module()
class VisualizationHook2(Hook):
    def __init__(self, dataset):
        self.interval = 1
        assert dataset is not None
        self.idgenerator = IdGenerator(dataset.categories)
        self.dataset = dataset

    def _id2color(self, id_map):
        idgenerator = copy.deepcopy(self.idgenerator)
        label2cat = dict((v, k) for k, v in self.dataset.cat2label.items())

        if isinstance(id_map, np.ndarray):
            unique_ids = np.unique(id_map)
            color_lookup = np.zeros((max(unique_ids)+1, 3), np.uint8)
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

    def _dump_image(self, image, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, image[..., ::-1])

    def after_train_iter(self, runner):
        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.bbox_head.get_result_visualization()

        for res in results:
            name = res['name']
            image = _get_rgb_image(res['image'])

            # points = as_list(res['points'])

            pan_results = as_list(res['pan_results'])
            pan_results = [visualize_panoptic(image, x, self._id2color) for x in pan_results]

            # b_a_results = as_list(res['b_a_res'])
            # b_a_res = [visualize_panoptic(image, x, self._id2color) for x in b_a_results]
            #
            # for point in points:
            #     b_a_res[0] = cv2.circle(b_a_res[0], point, 1, (0, 0, 255), 4)
            #     b_a_res[1] = cv2.circle(b_a_res[1], point, 1, (0, 0, 255), 4)

            sem_results = as_list(res['sem_results'])
            sem_results = [visualize_panoptic(image, x, self._id2color, None) for x in sem_results]

            heatmaps = as_list(res['heatmaps'])
            heatmaps = [visualize_heatmap(image, x**2) for x in heatmaps]

            features = visualize_raw(res['feature_proj'])

            boundarys = []
            all_boundarys = as_list(res['boundary'])
            for boundary in all_boundarys:
                boundary = (boundary*255).data.cpu().numpy()
                kernel = np.ones((3,3),np.uint8)
                boundary = cv2.dilate(boundary,kernel,iterations=1)
                boundarys.append(boundary)

            # demo = [image] + pan_results + sem_results + [features] + boundarys +heatmaps
            demo = [image] + boundarys
            demo = imhstack(demo, space=0)

            self._dump_image(demo, os.path.join(runner.work_dir, 'result_visualization', name+'.jpg'))

    def after_test_iter(self, runner):
        model = runner.model
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module
        results = model.bbox_head.get_test_visualization()

        for res in results:
            name = res['name']
            image = _get_rgb_image(res['image'])
            pan_results = visualize_panoptic(image, res['test_pan_results'], self._id2color)
            # import pdb
            # pdb.set_trace()

            demo = imhstack([image, pan_results], 0)
            self._dump_image(demo, os.path.join(runner.work_dir, 'test_visualization', name+'.jpg'))

