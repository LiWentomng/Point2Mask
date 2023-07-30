from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.core import eval_recalls
from .coco_instance import COCOInstanceDataset

import mmcv
import numpy as np
import panopticapi
from panopticapi.evaluation import pq_compute_multi_core
from panopticapi.utils import id2rgb, rgb2id
VOID = 0

from collections import defaultdict
import os
import itertools
from mmcv.utils import print_log
from terminaltables import AsciiTable

import multiprocessing as mp
import PIL.Image as Image


__all__ = ["COCOPanopticDataset"]


INSTANCE_OFFSET = 1000 


_id_and_category_maps =[
    { "supercategory": "person",
        "color": [
            220, 20, 60 ],
        "isthing": 1,
        "id": 1,
        "name": "person" },
    {
        "supercategory": "vehicle",
        "color": [
            119,
            11,
            32
        ],
        "isthing": 1,
        "id": 2,
        "name": "bicycle"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            142
        ],
        "isthing": 1,
        "id": 3,
        "name": "car"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            230
        ],
        "isthing": 1,
        "id": 4,
        "name": "motorcycle"
    },
    {
        "supercategory": "vehicle",
        "color": [
            106,
            0,
            228
        ],
        "isthing": 1,
        "id": 5,
        "name": "airplane"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            60,
            100
        ],
        "isthing": 1,
        "id": 6,
        "name": "bus"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            80,
            100
        ],
        "isthing": 1,
        "id": 7,
        "name": "train"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            70
        ],
        "isthing": 1,
        "id": 8,
        "name": "truck"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            192
        ],
        "isthing": 1,
        "id": 9,
        "name": "boat"
    },
    {
        "supercategory": "outdoor",
        "color": [
            250,
            170,
            30
        ],
        "isthing": 1,
        "id": 10,
        "name": "traffic light"
    },
    {
        "supercategory": "outdoor",
        "color": [
            100,
            170,
            30
        ],
        "isthing": 1,
        "id": 11,
        "name": "fire hydrant"
    },
    {
        "supercategory": "outdoor",
        "color": [
            220,
            220,
            0
        ],
        "isthing": 1,
        "id": 13,
        "name": "stop sign"
    },
    {
        "supercategory": "outdoor",
        "color": [
            175,
            116,
            175
        ],
        "isthing": 1,
        "id": 14,
        "name": "parking meter"
    },
    {
        "supercategory": "outdoor",
        "color": [
            250,
            0,
            30
        ],
        "isthing": 1,
        "id": 15,
        "name": "bench"
    },
    {
        "supercategory": "animal",
        "color": [
            165,
            42,
            42
        ],
        "isthing": 1,
        "id": 16,
        "name": "bird"
    },
    {
        "supercategory": "animal",
        "color": [
            255,
            77,
            255
        ],
        "isthing": 1,
        "id": 17,
        "name": "cat"
    },
    {
        "supercategory": "animal",
        "color": [
            0,
            226,
            252
        ],
        "isthing": 1,
        "id": 18,
        "name": "dog"
    },
    {
        "supercategory": "animal",
        "color": [
            182,
            182,
            255
        ],
        "isthing": 1,
        "id": 19,
        "name": "horse"
    },
    {
        "supercategory": "animal",
        "color": [
            0,
            82,
            0
        ],
        "isthing": 1,
        "id": 20,
        "name": "sheep"
    },
    {
        "supercategory": "animal",
        "color": [
            120,
            166,
            157
        ],
        "isthing": 1,
        "id": 21,
        "name": "cow"
    },
    {
        "supercategory": "animal",
        "color": [
            110,
            76,
            0
        ],
        "isthing": 1,
        "id": 22,
        "name": "elephant"
    },
    {
        "supercategory": "animal",
        "color": [
            174,
            57,
            255
        ],
        "isthing": 1,
        "id": 23,
        "name": "bear"
    },
    {
        "supercategory": "animal",
        "color": [
            199,
            100,
            0
        ],
        "isthing": 1,
        "id": 24,
        "name": "zebra"
    },
    {
        "supercategory": "animal",
        "color": [
            72,
            0,
            118
        ],
        "isthing": 1,
        "id": 25,
        "name": "giraffe"
    },
    {
        "supercategory": "accessory",
        "color": [
            255,
            179,
            240
        ],
        "isthing": 1,
        "id": 27,
        "name": "backpack"
    },
    {
        "supercategory": "accessory",
        "color": [
            0,
            125,
            92
        ],
        "isthing": 1,
        "id": 28,
        "name": "umbrella"
    },
    {
        "supercategory": "accessory",
        "color": [
            209,
            0,
            151
        ],
        "isthing": 1,
        "id": 31,
        "name": "handbag"
    },
    {
        "supercategory": "accessory",
        "color": [
            188,
            208,
            182
        ],
        "isthing": 1,
        "id": 32,
        "name": "tie"
    },
    {
        "supercategory": "accessory",
        "color": [
            0,
            220,
            176
        ],
        "isthing": 1,
        "id": 33,
        "name": "suitcase"
    },
    {
        "supercategory": "sports",
        "color": [
            255,
            99,
            164
        ],
        "isthing": 1,
        "id": 34,
        "name": "frisbee"
    },
    {
        "supercategory": "sports",
        "color": [
            92,
            0,
            73
        ],
        "isthing": 1,
        "id": 35,
        "name": "skis"
    },
    {
        "supercategory": "sports",
        "color": [
            133,
            129,
            255
        ],
        "isthing": 1,
        "id": 36,
        "name": "snowboard"
    },
    {
        "supercategory": "sports",
        "color": [
            78,
            180,
            255
        ],
        "isthing": 1,
        "id": 37,
        "name": "sports ball"
    },
    {
        "supercategory": "sports",
        "color": [
            0,
            228,
            0
        ],
        "isthing": 1,
        "id": 38,
        "name": "kite"
    },
    {
        "supercategory": "sports",
        "color": [
            174,
            255,
            243
        ],
        "isthing": 1,
        "id": 39,
        "name": "baseball bat"
    },
    {
        "supercategory": "sports",
        "color": [
            45,
            89,
            255
        ],
        "isthing": 1,
        "id": 40,
        "name": "baseball glove"
    },
    {
        "supercategory": "sports",
        "color": [
            134,
            134,
            103
        ],
        "isthing": 1,
        "id": 41,
        "name": "skateboard"
    },
    {
        "supercategory": "sports",
        "color": [
            145,
            148,
            174
        ],
        "isthing": 1,
        "id": 42,
        "name": "surfboard"
    },
    {
        "supercategory": "sports",
        "color": [
            255,
            208,
            186
        ],
        "isthing": 1,
        "id": 43,
        "name": "tennis racket"
    },
    {
        "supercategory": "kitchen",
        "color": [
            197,
            226,
            255
        ],
        "isthing": 1,
        "id": 44,
        "name": "bottle"
    },
    {
        "supercategory": "kitchen",
        "color": [
            171,
            134,
            1
        ],
        "isthing": 1,
        "id": 46,
        "name": "wine glass"
    },
    {
        "supercategory": "kitchen",
        "color": [
            109,
            63,
            54
        ],
        "isthing": 1,
        "id": 47,
        "name": "cup"
    },
    {
        "supercategory": "kitchen",
        "color": [
            207,
            138,
            255
        ],
        "isthing": 1,
        "id": 48,
        "name": "fork"
    },
    {
        "supercategory": "kitchen",
        "color": [
            151,
            0,
            95
        ],
        "isthing": 1,
        "id": 49,
        "name": "knife"
    },
    {
        "supercategory": "kitchen",
        "color": [
            9,
            80,
            61
        ],
        "isthing": 1,
        "id": 50,
        "name": "spoon"
    },
    {
        "supercategory": "kitchen",
        "color": [
            84,
            105,
            51
        ],
        "isthing": 1,
        "id": 51,
        "name": "bowl"
    },
    {
        "supercategory": "food",
        "color": [
            74,
            65,
            105
        ],
        "isthing": 1,
        "id": 52,
        "name": "banana"
    },
    {
        "supercategory": "food",
        "color": [
            166,
            196,
            102
        ],
        "isthing": 1,
        "id": 53,
        "name": "apple"
    },
    {
        "supercategory": "food",
        "color": [
            208,
            195,
            210
        ],
        "isthing": 1,
        "id": 54,
        "name": "sandwich"
    },
    {
        "supercategory": "food",
        "color": [
            255,
            109,
            65
        ],
        "isthing": 1,
        "id": 55,
        "name": "orange"
    },
    {
        "supercategory": "food",
        "color": [
            0,
            143,
            149
        ],
        "isthing": 1,
        "id": 56,
        "name": "broccoli"
    },
    {
        "supercategory": "food",
        "color": [
            179,
            0,
            194
        ],
        "isthing": 1,
        "id": 57,
        "name": "carrot"
    },
    {
        "supercategory": "food",
        "color": [
            209,
            99,
            106
        ],
        "isthing": 1,
        "id": 58,
        "name": "hot dog"
    },
    {
        "supercategory": "food",
        "color": [
            5,
            121,
            0
        ],
        "isthing": 1,
        "id": 59,
        "name": "pizza"
    },
    {
        "supercategory": "food",
        "color": [
            227,
            255,
            205
        ],
        "isthing": 1,
        "id": 60,
        "name": "donut"
    },
    {
        "supercategory": "food",
        "color": [
            147,
            186,
            208
        ],
        "isthing": 1,
        "id": 61,
        "name": "cake"
    },
    {
        "supercategory": "furniture",
        "color": [
            153,
            69,
            1
        ],
        "isthing": 1,
        "id": 62,
        "name": "chair"
    },
    {
        "supercategory": "furniture",
        "color": [
            3,
            95,
            161
        ],
        "isthing": 1,
        "id": 63,
        "name": "couch"
    },
    {
        "supercategory": "furniture",
        "color": [
            163,
            255,
            0
        ],
        "isthing": 1,
        "id": 64,
        "name": "potted plant"
    },
    {
        "supercategory": "furniture",
        "color": [
            119,
            0,
            170
        ],
        "isthing": 1,
        "id": 65,
        "name": "bed"
    },
    {
        "supercategory": "furniture",
        "color": [
            0,
            182,
            199
        ],
        "isthing": 1,
        "id": 67,
        "name": "dining table"
    },
    {
        "supercategory": "furniture",
        "color": [
            0,
            165,
            120
        ],
        "isthing": 1,
        "id": 70,
        "name": "toilet"
    },
    {
        "supercategory": "electronic",
        "color": [
            183,
            130,
            88
        ],
        "isthing": 1,
        "id": 72,
        "name": "tv"
    },
    {
        "supercategory": "electronic",
        "color": [
            95,
            32,
            0
        ],
        "isthing": 1,
        "id": 73,
        "name": "laptop"
    },
    {
        "supercategory": "electronic",
        "color": [
            130,
            114,
            135
        ],
        "isthing": 1,
        "id": 74,
        "name": "mouse"
    },
    {
        "supercategory": "electronic",
        "color": [
            110,
            129,
            133
        ],
        "isthing": 1,
        "id": 75,
        "name": "remote"
    },
    {
        "supercategory": "electronic",
        "color": [
            166,
            74,
            118
        ],
        "isthing": 1,
        "id": 76,
        "name": "keyboard"
    },
    {
        "supercategory": "electronic",
        "color": [
            219,
            142,
            185
        ],
        "isthing": 1,
        "id": 77,
        "name": "cell phone"
    },
    {
        "supercategory": "appliance",
        "color": [
            79,
            210,
            114
        ],
        "isthing": 1,
        "id": 78,
        "name": "microwave"
    },
    {
        "supercategory": "appliance",
        "color": [
            178,
            90,
            62
        ],
        "isthing": 1,
        "id": 79,
        "name": "oven"
    },
    {
        "supercategory": "appliance",
        "color": [
            65,
            70,
            15
        ],
        "isthing": 1,
        "id": 80,
        "name": "toaster"
    },
    {
        "supercategory": "appliance",
        "color": [
            127,
            167,
            115
        ],
        "isthing": 1,
        "id": 81,
        "name": "sink"
    },
    {
        "supercategory": "appliance",
        "color": [
            59,
            105,
            106
        ],
        "isthing": 1,
        "id": 82,
        "name": "refrigerator"
    },
    {
        "supercategory": "indoor",
        "color": [
            142,
            108,
            45
        ],
        "isthing": 1,
        "id": 84,
        "name": "book"
    },
    {
        "supercategory": "indoor",
        "color": [
            196,
            172,
            0
        ],
        "isthing": 1,
        "id": 85,
        "name": "clock"
    },
    {
        "supercategory": "indoor",
        "color": [
            95,
            54,
            80
        ],
        "isthing": 1,
        "id": 86,
        "name": "vase"
    },
    {
        "supercategory": "indoor",
        "color": [
            128,
            76,
            255
        ],
        "isthing": 1,
        "id": 87,
        "name": "scissors"
    },
    {
        "supercategory": "indoor",
        "color": [
            201,
            57,
            1
        ],
        "isthing": 1,
        "id": 88,
        "name": "teddy bear"
    },
    {
        "supercategory": "indoor",
        "color": [
            246,
            0,
            122
        ],
        "isthing": 1,
        "id": 89,
        "name": "hair drier"
    },
    {
        "supercategory": "indoor",
        "color": [
            191,
            162,
            208
        ],
        "isthing": 1,
        "id": 90,
        "name": "toothbrush"
    },
    {
        "supercategory": "textile",
        "color": [
            255,
            255,
            128
        ],
        "isthing": 0,
        "id": 92,
        "name": "banner"
    },
    {
        "supercategory": "textile",
        "color": [
            147,
            211,
            203
        ],
        "isthing": 0,
        "id": 93,
        "name": "blanket"
    },
    {
        "supercategory": "building",
        "color": [
            150,
            100,
            100
        ],
        "isthing": 0,
        "id": 95,
        "name": "bridge"
    },
    {
        "supercategory": "raw-material",
        "color": [
            168,
            171,
            172
        ],
        "isthing": 0,
        "id": 100,
        "name": "cardboard"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            146,
            112,
            198
        ],
        "isthing": 0,
        "id": 107,
        "name": "counter"
    },
    {
        "supercategory": "textile",
        "color": [
            210,
            170,
            100
        ],
        "isthing": 0,
        "id": 109,
        "name": "curtain"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            92,
            136,
            89
        ],
        "isthing": 0,
        "id": 112,
        "name": "door-stuff"
    },
    {
        "supercategory": "floor",
        "color": [
            218,
            88,
            184
        ],
        "isthing": 0,
        "id": 118,
        "name": "floor-wood"
    },
    {
        "supercategory": "plant",
        "color": [
            241,
            129,
            0
        ],
        "isthing": 0,
        "id": 119,
        "name": "flower"
    },
    {
        "supercategory": "food-stuff",
        "color": [
            217,
            17,
            255
        ],
        "isthing": 0,
        "id": 122,
        "name": "fruit"
    },
    {
        "supercategory": "ground",
        "color": [
            124,
            74,
            181
        ],
        "isthing": 0,
        "id": 125,
        "name": "gravel"
    },
    {
        "supercategory": "building",
        "color": [
            70,
            70,
            70
        ],
        "isthing": 0,
        "id": 128,
        "name": "house"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            255,
            228,
            255
        ],
        "isthing": 0,
        "id": 130,
        "name": "light"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            154,
            208,
            0
        ],
        "isthing": 0,
        "id": 133,
        "name": "mirror-stuff"
    },
    {
        "supercategory": "structural",
        "color": [
            193,
            0,
            92
        ],
        "isthing": 0,
        "id": 138,
        "name": "net"
    },
    {
        "supercategory": "textile",
        "color": [
            76,
            91,
            113
        ],
        "isthing": 0,
        "id": 141,
        "name": "pillow"
    },
    {
        "supercategory": "ground",
        "color": [
            255,
            180,
            195
        ],
        "isthing": 0,
        "id": 144,
        "name": "platform"
    },
    {
        "supercategory": "ground",
        "color": [
            106,
            154,
            176
        ],
        "isthing": 0,
        "id": 145,
        "name": "playingfield"
    },
    {
        "supercategory": "ground",
        "color": [
            230,
            150,
            140
        ],
        "isthing": 0,
        "id": 147,
        "name": "railroad"
    },
    {
        "supercategory": "water",
        "color": [
            60,
            143,
            255
        ],
        "isthing": 0,
        "id": 148,
        "name": "river"
    },
    {
        "supercategory": "ground",
        "color": [
            128,
            64,
            128
        ],
        "isthing": 0,
        "id": 149,
        "name": "road"
    },
    {
        "supercategory": "building",
        "color": [
            92,
            82,
            55
        ],
        "isthing": 0,
        "id": 151,
        "name": "roof"
    },
    {
        "supercategory": "ground",
        "color": [
            254,
            212,
            124
        ],
        "isthing": 0,
        "id": 154,
        "name": "sand"
    },
    {
        "supercategory": "water",
        "color": [
            73,
            77,
            174
        ],
        "isthing": 0,
        "id": 155,
        "name": "sea"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            255,
            160,
            98
        ],
        "isthing": 0,
        "id": 156,
        "name": "shelf"
    },
    {
        "supercategory": "ground",
        "color": [
            255,
            255,
            255
        ],
        "isthing": 0,
        "id": 159,
        "name": "snow"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            104,
            84,
            109
        ],
        "isthing": 0,
        "id": 161,
        "name": "stairs"
    },
    {
        "supercategory": "building",
        "color": [
            169,
            164,
            131
        ],
        "isthing": 0,
        "id": 166,
        "name": "tent"
    },
    {
        "supercategory": "textile",
        "color": [
            225,
            199,
            255
        ],
        "isthing": 0,
        "id": 168,
        "name": "towel"
    },
    {
        "supercategory": "wall",
        "color": [
            137,
            54,
            74
        ],
        "isthing": 0,
        "id": 171,
        "name": "wall-brick"
    },
    {
        "supercategory": "wall",
        "color": [
            135,
            158,
            223
        ],
        "isthing": 0,
        "id": 175,
        "name": "wall-stone"
    },
    {
        "supercategory": "wall",
        "color": [
            7,
            246,
            231
        ],
        "isthing": 0,
        "id": 176,
        "name": "wall-tile"
    },
    {
        "supercategory": "wall",
        "color": [
            107,
            255,
            200
        ],
        "isthing": 0,
        "id": 177,
        "name": "wall-wood"
    },
    {
        "supercategory": "water",
        "color": [
            58,
            41,
            149
        ],
        "isthing": 0,
        "id": 178,
        "name": "water-other"
    },
    {
        "supercategory": "window",
        "color": [
            183,
            121,
            142
        ],
        "isthing": 0,
        "id": 180,
        "name": "window-blind"
    },
    {
        "supercategory": "window",
        "color": [
            255,
            73,
            97
        ],
        "isthing": 0,
        "id": 181,
        "name": "window-other"
    },
    {
        "supercategory": "plant",
        "color": [
            107,
            142,
            35
        ],
        "isthing": 0,
        "id": 184,
        "name": "tree-merged"
    },
    {
        "supercategory": "structural",
        "color": [
            190,
            153,
            153
        ],
        "isthing": 0,
        "id": 185,
        "name": "fence-merged"
    },
    {
        "supercategory": "ceiling",
        "color": [
            146,
            139,
            141
        ],
        "isthing": 0,
        "id": 186,
        "name": "ceiling-merged"
    },
    {
        "supercategory": "sky",
        "color": [
            70,
            130,
            180
        ],
        "isthing": 0,
        "id": 187,
        "name": "sky-other-merged"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            134,
            199,
            156
        ],
        "isthing": 0,
        "id": 188,
        "name": "cabinet-merged"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            209,
            226,
            140
        ],
        "isthing": 0,
        "id": 189,
        "name": "table-merged"
    },
    {
        "supercategory": "floor",
        "color": [
            96,
            36,
            108
        ],
        "isthing": 0,
        "id": 190,
        "name": "floor-other-merged"
    },
    {
        "supercategory": "ground",
        "color": [
            96,
            96,
            96
        ],
        "isthing": 0,
        "id": 191,
        "name": "pavement-merged"
    },
    {
        "supercategory": "solid",
        "color": [
            64,
            170,
            64
        ],
        "isthing": 0,
        "id": 192,
        "name": "mountain-merged"
    },
    {
        "supercategory": "plant",
        "color": [
            152,
            251,
            152
        ],
        "isthing": 0,
        "id": 193,
        "name": "grass-merged"
    },
    {
        "supercategory": "ground",
        "color": [
            208,
            229,
            228
        ],
        "isthing": 0,
        "id": 194,
        "name": "dirt-merged"
    },
    {
        "supercategory": "raw-material",
        "color": [
            206,
            186,
            171
        ],
        "isthing": 0,
        "id": 195,
        "name": "paper-merged"
    },
    {
        "supercategory": "food-stuff",
        "color": [
            152,
            161,
            64
        ],
        "isthing": 0,
        "id": 196,
        "name": "food-other-merged"
    },
    {
        "supercategory": "building",
        "color": [
            116,
            112,
            0
        ],
        "isthing": 0,
        "id": 197,
        "name": "building-other-merged"
    },
    {
        "supercategory": "solid",
        "color": [
            0,
            114,
            143
        ],
        "isthing": 0,
        "id": 198,
        "name": "rock-merged"
    },
    {
        "supercategory": "wall",
        "color": [
            102,
            102,
            156
        ],
        "isthing": 0,
        "id": 199,
        "name": "wall-other-merged"
    },
    {
        "supercategory": "textile",
        "color": [
            250,
            141,
            255
        ],
        "isthing": 0,
        "id": 200,
        "name": "rug-merged"
    }
]

class COCOPanoptic(COCO):
    def __init__(self, annotation_file=None):
        super(COCOPanoptic, self).__init__(annotation_file)
    
    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for i, cat in enumerate(self.dataset['categories']):
                assert cat['id'] == _id_and_category_maps[i]['id'], (cat, _id_and_category_maps[i])
                assert cat['name'] == _id_and_category_maps[i]['name'], (cat, _id_and_category_maps[i])
                cat['color'] = _id_and_category_maps[i]['color']
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids=[]):
        """Load anns with the specified ids.

        self.anns is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (int array): integer ids specifying anns

        Returns:
            anns (object array): loaded ann objects
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]


@DATASETS.register_module()
class COCOPanopticDataset(COCOInstanceDataset):
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        ' truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    THING_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    STUFF_CLASSES = [
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]

    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCOPanoptic(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.categories = self.coco.cats
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        # filter out unmatched images
        ann_info = [i for i in ann_info if i['image_id'] == img_id]
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []

        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                if not is_crowd:
                    gt_bboxes.append(bbox)
                    gt_labels.append(contiguous_cat_id)
                else:
                    gt_bboxes_ignore.append(bbox)
                    is_thing = False

            mask_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            gt_mask_infos.append(mask_info)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=img_info['segm_file'])

        return ann

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        ids_with_ann = []
        # check whether images have legal thing annotations.
        for lists in self.coco.anns.values():
            for item in lists:
                category_id = item['category_id']
                is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
                if not is_thing:
                    continue
                ids_with_ann.append(item['image_id'])
        ids_with_ann = set(ids_with_ann)

        valid_inds = []
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _pan2json(self, results, outfile_prefix, need_segmantics=False):
        """Convert panoptic results to COCO panoptic json style."""
        label2cat = dict((v, k) for (k, v) in self.cat2label.items())
        pred_annotations = []
        outdir = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        if need_segmantics:
            pred_semantics = []
            sem_outdir = os.path.join(os.path.dirname(outfile_prefix), 'semantic')

        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            segm_file = self.data_infos[idx]['segm_file']
            pan = results[idx]

            if need_segmantics:
                sem = pan % INSTANCE_OFFSET
                sem[sem == len(self.CLASSES)] = len(self.CLASSES) - 1
                mmcv.imwrite(sem.astype(np.uint8), os.path.join(sem_outdir, segm_file))
                pred_semantics.append({'image_id': img_id, 'file_name': segm_file})

            pan_labels = np.unique(pan)
            segm_info = []
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                # We reserve the length of self.CLASSES for VOID label
                if sem_label == len(self.CLASSES):
                    continue
                # convert sem_label to json label
                cat_id = label2cat[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                mask = pan == pan_label
                area = mask.sum()
                segm_info.append({
                    'id': int(pan_label),
                    'category_id': cat_id,
                    'isthing': is_thing,
                    'area': int(area)
                })
            # evaluation script uses 0 for VOID label.
            pan[pan % INSTANCE_OFFSET == len(self.CLASSES)] = VOID
            pan = id2rgb(pan).astype(np.uint8)
            mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
            record = {
                'image_id': img_id,
                'segments_info': segm_info,
                'file_name': segm_file
            }
            pred_annotations.append(record)

        pan_json_results = dict(annotations=pred_annotations)

        if need_segmantics:
            sem_json_results = dict(annotations=pred_semantics)
        else:
            sem_json_results = None

        return pan_json_results, sem_json_results

    def _sem2json(self, results, outfile_prefix):
        outdir = os.path.join(os.path.dirname(outfile_prefix), 'semantic')
        pred_semantics = []

        for idx in range(len(self)):
            segm_file = self.data_infos[idx]['segm_file']
            segm = results[idx]
            segm[segm == len(self.CLASSES)] = len(self.CLASSES) - 1
            mmcv.imwrite(segm.astype(np.uint8), os.path.join(outdir, segm_file))

            img_id = self.img_ids[idx]
            record = {
                    'image_id': img_id,
                    'file_name': segm_file }
            pred_semantics.append(record)
        pred_json_semantics = dict(annotations=pred_semantics)
        return pred_json_semantics

    def results2json(self, results, outfile_prefix):
        """Dump the panoptic results to a COCO panoptic style json file.

        Args:
            results (dict): Testing results of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.panoptic.json"

        Returns:
            dict[str: str]: The key is 'panoptic' and the value is
                corresponding filename.
        """
        result_files = dict()

        if 'semantic' in results[0]:
            sem_results = [result['semantic'] for result in results]
            sem_json_results = self._sem2json(sem_results, outfile_prefix)
            need_segmantics = False
        else:
            need_segmantics = True

        pan_results = [result['pan_results'] for result in results]
        pan_json_results, sem_json_results_ = self._pan2json(pan_results, outfile_prefix, need_segmantics)
        if need_segmantics:
            sem_json_results = sem_json_results_

        result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
        mmcv.dump(pan_json_results, result_files['panoptic'])

        result_files['semantic'] = f'{outfile_prefix}.semantic.json'
        mmcv.dump(sem_json_results, result_files['semantic'])

        return result_files

    def evaluate_pan_json(self,
                          result_files,
                          outfile_prefix,
                          logger=None,
                          classwise=False):
        """Evaluate PQ according to the panoptic results json file."""
        imgs = self.coco.imgs
        gt_json = self.coco.img_ann_map  # image to annotations
        gt_json = [{
            'image_id': k,
            'segments_info': v,
            'file_name': imgs[k]['segm_file']
        } for k, v in gt_json.items()]
        pred_json = mmcv.load(result_files['panoptic'])
        pred_json = dict(
            (el['image_id'], el) for el in pred_json['annotations'])

        # match the gt_anns and pred_anns in the same image
        matched_annotations_list = []
        for gt_ann in gt_json:
            img_id = gt_ann['image_id']
            if img_id not in pred_json.keys():
                raise Exception('no prediction for the image'
                                ' with id: {}'.format(img_id))
            matched_annotations_list.append((gt_ann, pred_json[img_id]))

        gt_folder = self.seg_prefix
        pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder,
                                        pred_folder, self.categories)

        metrics = [('All', None), ('Things', True), ('Stuff', False)]
        pq_results = {}

        for name, isthing in metrics:
            pq_results[name], classwise_results = pq_stat.pq_average(
                self.categories, isthing=isthing)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        classwise_results = None
        if classwise:
            classwise_results = {
                k: v
                for k, v in zip(self.CLASSES, pq_results['classwise'].values())
            }
        print_panoptic_table(pq_results, classwise_results, logger=logger)

        return parse_pq_results(pq_results)

    def evaluate_sem_json(self,
            result_files,
            outfile_prefix,
            logger=None,
            classwise=False):
        imgs = self.coco.imgs
        gt_json = self.coco.img_ann_map
        gt_json = [{
            'image_id': k,
            'segments_info': v,
            'file_name': imgs[k]['segm_file']
        } for k, v in gt_json.items()]

        pred_json = mmcv.load(result_files['semantic'])
        pred_json = dict(
            (el['image_id'], el) for el in pred_json['annotations'])
        # match the gt_anns and pred_anns in the same image
        matched_annotations_list = []
        for gt_ann in gt_json:
            img_id = gt_ann['image_id']
            if img_id not in pred_json.keys():
                raise Exception('no prediction for the image'
                                ' with id: {}'.format(img_id))
            matched_annotations_list.append((gt_ann, pred_json[img_id]))

        gt_folder = self.seg_prefix
        pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'semantic')

        confmat = iou_compute_multi_core(matched_annotations_list, gt_folder, pred_folder,
                self.categories, self.cat2label)
        num_classes = len(self.cat2label)
        confmat = confmat.astype(np.float64).reshape(num_classes, num_classes)
        iou = np.diag(confmat) / (confmat.sum(0) + confmat.sum(1) - np.diag(confmat) + 1e-12)
        label2catname = {v: self.categories[k]['name'] for k, v in self.cat2label.items()}
        print_semantic_table(iou, label2catname, logger=logger)

    def evaluate(self,
                 results,
                 metric='PQ',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 **kwargs):
        """Evaluation in COCO Panoptic protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Only
                support 'PQ' at present. 'pq' will be regarded as 'PQ.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to print classwise evaluation results.
                Default: False.

        Returns:
            dict[str, float]: COCO Panoptic style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        # Compatible with lowercase 'pq'
        #metrics = ['PQ' if metric == 'pq' else metric for metric in metrics]
        metrics = [metric.upper() for metric in metrics]
        allowed_metrics = ['PQ', 'IOU']  # todo: support other metrics like 'bbox'
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}

        outfile_prefix = os.path.join(tmp_dir.name, 'results') \
            if tmp_dir is not None else jsonfile_prefix
        if 'PQ' in metrics:
            eval_pan_results = self.evaluate_pan_json(result_files,
                                                      outfile_prefix, logger,
                                                      classwise)
            eval_results.update(eval_pan_results)

        if 'IOU' in metrics:
            self.evaluate_sem_json(result_files,
                    outfile_prefix, logger, classwise)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


def parse_pq_results(pq_results):
    """Parse the Panoptic Quality results."""
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_th'] = 100 * pq_results['Things']['pq']
    result['SQ_th'] = 100 * pq_results['Things']['sq']
    result['RQ_th'] = 100 * pq_results['Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']
    return result


def print_panoptic_table(pq_results, classwise_results=None, logger=None):
    """Print the panoptic evaluation results table.

    Args:
        pq_results(dict): The Panoptic Quality results.
        classwise_results(dict | None): The classwise Panoptic Quality results.
            The keys are class names and the values are metrics.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
    """

    headers = ['', 'PQ', 'SQ', 'RQ', 'categories']
    data = [headers]
    for name in ['All', 'Things', 'Stuff']:
        numbers = [
            f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq']
        ]
        row = [name] + numbers + [pq_results[name]['n']]
        data.append(row)
    table = AsciiTable(data)
    print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)

    if classwise_results is not None:
        class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                          for k in ['pq', 'sq', 'rq'])
                         for name, metrics in classwise_results.items()]
        num_columns = min(8, len(class_metrics) * 4)
        results_flatten = list(itertools.chain(*class_metrics))
        headers = ['category', 'PQ', 'SQ', 'RQ'] * (num_columns // 4)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        data = [headers]
        data += [result for result in results_2d]
        table = AsciiTable(data)
        print_log(
            'Classwise Panoptic Evaluation Results:\n' + table.table,
            logger=logger)

def print_semantic_table(iou, label2catname, logger=None):
    data = [['mean', float(iou.mean())]]
    for i, x in enumerate(iou):
        data.append([label2catname[i], float(x)])
    data = [[name, f'{x * 100:.3f}'] for name, x in data]
    num_cols = 6

    num_pad = num_cols - len(data) % num_cols
    if num_pad > 0 and num_pad < num_cols:
        data = data + [['', '']] * num_pad

    data = list(zip(*[data[i::num_cols] for i in range(num_cols)]))
    data = sum([list(zip(*group)) for group in data], [])
    table = AsciiTable(data)
    print_log('Semantic Evaluation Results:\n' + table.table, logger=logger)


def iou_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, cat2label):
    num_threads = mp.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, num_threads)
    workers = mp.Pool(num_threads)
    processes = []
    for pid, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(iou_compute, (
            pid, annotation_set, gt_folder, pred_folder, categories, cat2label
            ))
        processes.append(p)
    confmat = sum(p.get() for p in processes)
    workers.close()
    return confmat

def iou_compute(pid, annotation_set, gt_folder, pred_folder, categories, cat2label):
    num_classes = len(cat2label)
    confmat = np.zeros((num_classes**2,), np.int64)

    for gt_ann, pred_ann in annotation_set:
        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.int64)
        assert pan_gt.ndim == 3, pan_gt.shape
        pan_gt = rgb2id(pan_gt)
        pan_ids = np.unique(pan_gt)
        json_pan_ids = [seg_info['id'] for seg_info in gt_ann['segments_info']]
        assert (set(pan_ids) - set([0])) == set(json_pan_ids), f"gt from PNG and JSON does not match: {gt_ann}, {pan_ids}"

        # make semantic gt
        seg_gt = np.full_like(pan_gt, 255)
        for seg_info in gt_ann['segments_info']:
            label = cat2label[seg_info['category_id']]
            seg_gt[pan_gt == seg_info['id']] = label

        # get semantic pred
        seg_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.int64)

        # compute confusion matrix
        valid = seg_gt < 255
        seg_gt = seg_gt[valid]
        seg_pred = seg_pred[valid]
        confmat += np.bincount(seg_gt * num_classes + seg_pred, minlength=num_classes**2)
    return confmat

if __name__ == '__main__':
    label2catname = {i: f'name-{i}' for i in range(21)}
    iou = np.random.rand(21)
    print_semantic_table(iou, label2catname, logger=None)
