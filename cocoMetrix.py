"""COCO-style evaluation metrics.
download COCO API: github.com/cocodataset/cocoapi/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from utils.utils import *
from models import *
from utils.datasets import *

import json
import os
from absl import flags
from absl import logging
from pathlib import Path

import numpy as np
import copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


FLAGS = flags.FLAGS

class EvaluationMetric(object):
    """COCO evaluation metric class."""
    def __init__(self, cocoGt_PATH=None, dataloader = None):
        self.cocoGt_PATH = cocoGt_PATH # JSON file name
        self.coco_gt = COCO(cocoGt_PATH) if cocoGt_PATH is not None else None
        assert self.coco_gt is not None, "need reload ground truth"
        self.result = None
        # self.cocoDt = self.coco_gt.loadRes()
        self.metric_names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'ARmax1',
                            'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
        
        self.output_of_NMS_pred = [] # output of NMS prediction (x1, y1, x2, y2, conf, cls)
        self.ann_json_list = []
        self.annotation_id = 1
        self.category_ids = []
        self.dataloader = dataloader
        self.batch_size = 1
        self.dataloader = dataloader
        self.ann_in_detections = None
        self._init(self.dataloader)

    def _init(self, dataloader):
        self.img_list_from_det = self.dataloader.dataset.img_files
        self.result = 'result.json'
        self.imgIds = [int(''.join(list(filter(str.isdigit, Path(x).stem)))) \
                 for x in self.img_list_from_det]

    def update_op(self,
                imgs = None, 
                output = None,
                shapes = None,
                batch_size = 1,
                batch_i = 0):
        self.batch_size = batch_size
        output_of_NMS_pred = np.array(output)
        for id_of_img_processing, detection in enumerate(output_of_NMS_pred):
            image_id = int([''.join(list(filter(str.isdigit, Path(x).stem))) \
                     for x in self.img_list_from_det[batch_i * batch_size: (batch_i + 1) * batch_size]][id_of_img_processing])
            box = detection[:, :4].clone()  # xyxy
            scale_coords(imgs[id_of_img_processing].shape[1:], box, shapes[id_of_img_processing][0], shapes[id_of_img_processing][1])  # to original shape
            box = xyxy2xywh(box)  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for det, box in zip(detection.tolist(), box.tolist()):
                self.ann_json_list.append({'image_id': image_id,
                                     'category_id': int(det[5]),
                                     'bbox': [round(x, 3) for x in box],
                                     'score': round(det[4], 5)})
   
    def estimate_metric(self):
        """output_of_NMS_pred: nx6 (x1, y1, x2, y2, conf, cls)
        groundtruth_data: representing [y1, x1, y2, x2, is_crowd, area, class]
        """
        logging.info("eval COCO mAP with pycocotools...")
        cocoEval = COCOeval(self.coco_gt, self._produce_result_json(), 'bbox')
        cocoEval.params.imgIds = self.imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize() 
        return cocoEval
    
    def _save_json(self, path_of_result_json):
        assert type(path_of_result_json) == str, "type of result json file path is str"
        with open(path_of_result_json, 'w') as f:
            json.dump(self.ann_json_list, f)

    def _produce_result_json(self):
        """return loadRes
        """
        self._save_json(self.result)
        _tmp_result = COCO()
        _tmp_result.dataset['images'] = [img for img in self.coco_gt.dataset['images']]
        self.ann_in_detections = json.load(open(self.result))
        assert type(self.ann_in_detections) == list, 'results in not an array of objects'
        annsImgId_list = [ann['image_id'] for ann in self.ann_in_detections]
        assert set(annsImgId_list) == (set(annsImgId_list) & set(self.coco_gt.getImgIds())), \
                "Results do not correspond to current coco set"
        if 'bbox' in self.ann_in_detections[0] and not self.ann_in_detections[0]['bbox'] == []:
            _tmp_result.dataset['categories'] = copy.deepcopy(self.coco_gt.dataset['categories'])
            for id, ann in enumerate(self.ann_in_detections):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
        _tmp_result.dataset['annotations'] = self.ann_in_detections
        _tmp_result.createIndex()
        return _tmp_result
        

    




        