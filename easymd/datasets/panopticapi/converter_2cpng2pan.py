#!/usr/bin/env python
'''
This script converts panoptic segmentation predictions stored in 2 channels
panoptic format to COCO panoptic format.

2 channels format is described in the panoptic segmentation paper
(https://arxiv.org/pdf/1801.00868.pdf). Two labels are assigned to each pixel of
a segment:
- semantic class label;
- instance ID (nonnegative integer).
PNG format is used to store the data. The first channel stores semantic class
of a pixel and the second one stores instance ID.
For stuff categories instance ID is redundant and is 0 for all pixels
corresponding to stuff segments.

Panoptic COCO format is described fully in http://cocodataset.org/#format-data.
It is used for the Panoptic COCO challenge evaluation.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing
import itertools

import PIL.Image as Image

from .utils import get_traceback, IdGenerator, save_json

OFFSET = 1000

@get_traceback
def convert_single_core(proc_id, image_set, categories,  segmentations_folder, VOID=0):
    annotations = []
    for working_idx, image_result in enumerate(image_set):
        panoptic_result, file_name,shape= image_result
        try:
            original_format = panoptic_result
        except IOError:
            raise KeyError('no prediction png file for id')
        id_and_category_maps = OFFSET * original_format[:, :, 0] + original_format[:, :, 1]
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        id_generator = IdGenerator(categories)
        l = np.unique(id_and_category_maps)
        segm_info = []
        for el in l:
            sem = el // OFFSET
            if sem == VOID:
                continue
            if sem not in categories:
                raise KeyError('Unknown semantic label {}'.format(sem))
            mask = id_and_category_maps == el
            segment_id, color = id_generator.get_id_and_color(sem)
            pan_format[mask] = color
            segm_info.append({"id": segment_id,
                              "category_id": int(sem)
                              })
        suffix = '.png'
        annotations.append({'image_id': int(file_name),
                            'file_name':file_name+suffix,
                            "segments_info": segm_info
                            })

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name+suffix))
    return annotations



def converter_memory(images_json_file,panoptic_result,cat_id, 
              segmentations_folder, predictions_json_file,
              VOID=0):
    start_time = time.time()

    with open(images_json_file, 'r') as f:
        d_coco = json.load(f)

    categories = {el['id']: el for el in cat_id}

    if segmentations_folder is None:
        segmentations_folder = predictions_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("2 channels panoptic format:")
    print("TO")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(predictions_json_file))
    print('\n')
    cpu_num = multiprocessing.cpu_count()
    images_per_cpu = (len(panoptic_result)+cpu_num)//cpu_num
    print("Number of cores: {}, images per core: {}".format(cpu_num,images_per_cpu))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, image_set_begin in enumerate(range(cpu_num)):
        image_set = panoptic_result[image_set_begin*images_per_cpu:(image_set_begin+1)*images_per_cpu]
        p = workers.apply_async(convert_single_core,
                                (proc_id, image_set, categories, segmentations_folder, VOID))
        processes.append(p)
    annotations = []
    for p in processes:
        annotations.extend(p.get())

    print("Writing final JSON in {}".format(predictions_json_file))
    d_coco['annotations'] = annotations
    save_json(d_coco, predictions_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))





@get_traceback
def convert_single_core_naive(proc_id, image_set, categories, source_folder, segmentations_folder, VOID=0):
    annotations = []
    for working_idx, image_info in enumerate(image_set):

        file_name = '{}.png'.format(image_info['file_name'].rsplit('.')[0])
        try:
            original_format = np.array(Image.open(os.path.join(source_folder, file_name)), dtype=np.uint32)
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(image_info['id']))
        
        id_and_category_maps = OFFSET * original_format[:, :, 2] + original_format[:, :, 1]
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        id_generator = IdGenerator(categories)
        l = np.unique(id_and_category_maps)
        segm_info = []
        for el in l:
            sem = el // OFFSET
            if sem == VOID:
                continue
            if sem not in categories:
                raise KeyError('Unknown semantic label {}'.format(sem))
            mask = id_and_category_maps == el
            segment_id, color = id_generator.get_id_and_color(sem)
            pan_format[mask] = color
            segm_info.append({"id": segment_id,
                              "category_id": int(sem)
                              })
        
        annotations.append({'image_id': image_info['id'],
                            'file_name':file_name,
                            "segments_info": segm_info
                            })

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))

    return annotations


def converter(source_folder, images_json_file, categories_json_file,
              segmentations_folder, predictions_json_file,
              VOID=0):
    start_time = time.time()

    print("Reading image set information from {}".format(images_json_file))

    with open(images_json_file, 'r') as f:
        d_coco = json.load(f)
    images = d_coco['images']

    with open(categories_json_file, 'r') as f:
        categories_coco = json.load(f)
    categories = {el['id']: el for el in categories_coco}

    if segmentations_folder is None:
        segmentations_folder = predictions_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(segmentations_folder))
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("2 channels panoptic format:")
    print("\tSource folder: {}".format(source_folder))
    print("TO")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(predictions_json_file))
    print('\n')
    cpu_num = multiprocessing.cpu_count()
    images_split = np.array_split(images, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(images_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, image_set in enumerate(images_split):
        p = workers.apply_async(convert_single_core_naive,
                                (proc_id, image_set, categories, source_folder, segmentations_folder, VOID))
        processes.append(p)
    annotations = []
    for p in processes:
        annotations.extend(p.get())

    print("Writing final JSON in {}".format(predictions_json_file))
    d_coco['annotations'] = annotations
    save_json(d_coco, predictions_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script converts panoptic segmentation predictions \
        stored in 2 channels panoptic format to COCO panoptic format. See this \
        file's head for more information."
    )
    parser.add_argument('--source_folder', type=str,
                        help="folder that contains predictions in 2 channels PNG format")
    parser.add_argument('--images_json_file', type=str,
                        help="JSON file with correponding image set information")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if input_json_file is \
         X.json"
    )
    parser.add_argument('--predictions_json_file', type=str,
                        help="JSON file with resulting COCO format prediction")
    parser.add_argument('-v', '--void', type=int, default=0,
                        help="semantic id that corresponds to VOID region in two channels PNG format")
    args = parser.parse_args()
    converter(args.source_folder, args.images_json_file, args.categories_json_file,
              args.segmentations_folder, args.predictions_json_file,
              args.void)
