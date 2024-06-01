import os
import sys
import cv2
import numpy
import torch
import argparse
from pathlib import Path

sys.path.insert(0, '/workspace/lib/yolo/yolov7')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.datasets import letterbox
from utils.dataset_classes import get_class

class YOLOv7:
    model = None
    result = None
    conf_thresh = 0.0
    nms_thresh = 0.0
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, cfg, device):
        self.model_name  = cfg["model_name"]
        self.weight_path = cfg["model_path"]
        self.dataset     = cfg["dataset"]
        self.conf_thresh = cfg["conf_thresh"]
        self.nms_thresh  = cfg["nms_thresh"]
        self.image_size  = cfg["image_size"]
        self.is_batch    = cfg["is_batch"]
        if self.is_batch:
            self.batch_size = cfg["batch_size"]
        else:
            self.batch_size = 1
        __, self.class_names = get_class(self.dataset)

        self.device = torch.device("cuda:" + device)
        self.model = attempt_load(self.weight_path, map_location=self.device)  # load FP32 model
        self.model.eval()
        self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def inference_image(self, image):
        """
        :param image: input image(np array)
        :return: dict format bounding box(x1, y1, x2, y2), scor, class, class index
            - format:
                [{"label": [{"description": cls, "score": score, "class_idx": cls_idx}],
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...]
        """
        origin_image_shape = image.copy().shape
        augment = False
        stride = int(self.model.stride.max())
        image = letterbox(image, self.image_size, stride=stride)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = numpy.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        pred = self.model(image, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.conf_thresh, iou_thres=self.nms_thresh, labels=None,
                                   multi_label=True)
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(image.shape[2:], det[:, :4], origin_image_shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if conf > self.conf_thresh:
                        score = float(conf)
                        x = float(xyxy[0])
                        y = float(xyxy[1])
                        w = float(xyxy[2]) - x
                        h = float(xyxy[3]) - y
                        str_class = self.class_names[int(cls)]
                        results.append({
                            'label': [{
                                'description': str_class,
                                'score': score,
                                'class_idx': self.class_names.index(str_class)
                            }],
                            'position': {
                                'x': x,
                                'y': y,
                                'w': w,
                                'h': h
                            }
                        })
        self.results = results

        return results


    def inference_image_batch(self, images):
        """
        :param image: input images(list in dict: [np array])
        :return: detection results(bounding box(x1, y1, x2, y2), score, class, class index) of each images
            - format:
                [[{"label": [{"description": cls, "score": score, "class_idx": cls_idx}],
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...], ...]
        """
        results = []
        tensor_images = []
        shapes = []
        stride = int(self.model.stride.max())
        for image in images:
            shapes.append([[image.shape[0], image.shape[1]], [[0.3333333333333333, 0.3333333333333333], [16.0, 12.0]]])
            image = letterbox(image, self.image_size, stride=stride)[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = numpy.ascontiguousarray(image)
            tensor_images.append(torch.from_numpy(image))

        targets =  torch.zeros((0, 6))
        image = torch.stack(tensor_images, 0)
        image = image.to(self.device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(self.device)
        nb, _, height, width = image.shape  # batch size, channels, height, width

        with torch.no_grad():
            out, __ = self.model(image, augment=False)
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(self.device)  # to pixels
            labels = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            out = non_max_suppression(out, conf_thres=self.conf_thresh, iou_thres=self.nms_thresh, labels=labels, multi_label=True)

        for si, det in enumerate(out):
            result = []
            if len(det):
                detn = det.clone()
                detn[:, :4] = scale_coords(image[si].shape[1:], detn[:, :4], tuple(shapes[si][0])).round()
                for *xyxy, conf, cls in reversed(detn.tolist()):
                    if conf > self.conf_thresh:
                        score = float(conf)
                        x = float(xyxy[0])
                        y = float(xyxy[1])
                        w = float(xyxy[2]) - x
                        h = float(xyxy[3]) - y
                        str_class = self.class_names[int(cls)]
                        if w > 10 and h > 10:
                            result.append({
                                'label': [{
                                    'description': str_class,
                                    'score': score,
                                    'class_idx': self.class_names.index(str_class)
                                }],
                                'position': {
                                    'x': x,
                                    'y': y,
                                    'w': w,
                                    'h': h
                                }
                            })
            results.append(result)
        self.results = results

        return results

    def inference(self, image, is_batch=False):
        if is_batch :
            return self.inference_image_batch(image)
        else:
            return self.inference_image(image)