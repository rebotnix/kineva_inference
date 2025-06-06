"""
Kineva Inference Engine
Copyright (C) Rebotnix

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
For license details, see: https://www.gnu.org/licenses/agpl-3.0.html
Project website: https://rebotnix.com
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from kineva.utils.trt import *
from kineva.utils.processing import *
import numpy as np
import torch
import cv2
import json
import sys
import os
sys.path.append(os.path.abspath('./core'))
from .core.detect import Detector

class KINEVA():

    def __init__(self, model: Union[str, Path], img_size = 640, classes="./data/coco_classes_kineva.json", threshold=0.35):
        """
        Initialize an KINEVA model.

        This constructor initializes a detetction model. It needs a model path as well as an image of a perfect product.

        Args:
            model (str | Path): path to model file, i.e. 'metauas_256.trt', 'metauas_512.trt'.
            reference (str | Path): path to reference image file, i.e. reference.

        Examples:
            >>> from kineva import KINEVA
            >>> model = KINEVA("models/kineva_coco.trt")
        """
        with open(classes, 'r') as f:
            self.classes = json.load(f)
        path = Path(model if isinstance(model, (str, Path)) else "")
        self.img_size = img_size
        self.path = str(path)
        if str(path).endswith("trt"):
            self.input = "trt"
            engine = load_engine(path)
            self.context = engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine, mode="kineva")
        
        elif str(path).endswith("pth"):
            self.input = "pth"
            self.model = Detector(
            weight_file=path,
            conf_thres=threshold,
            nms_thres=0.7,
            input_size=[self.img_size, self.img_size],
            fuse=True,
            fp16=False,
            use_decoder=False
            )
            self.classes = self.model.class_names
        else:
            print("Sorry. We only support .trt or .pth files.")

    def export(self):
        from kineva.utils.export import convert_to_trt
        if self.input == "pth":
            convert_to_trt(self.path, self.path.replace("pth", "trt"), self.img_size, "kineva")
        else:
            print("Sorry. You already have a trt model.")

    def detect(self, image, threshold=0.35):
        self.original_image = cv2.imread(image)
        if self.input == "trt":
            input_image, ratios = preprocess(image, img_size=(self.img_size,self.img_size ), mode="kineva")
            np.copyto(self.inputs[0]['host'], input_image.ravel())
            outputs_data = infer(self.context, self.inputs, self.outputs, self.stream, mode="kineva")[0]
            outputs_data = torch.from_numpy(outputs_data).float()
            final_boxes, final_scores, final_labels = postprocess_kineva(outputs_data, ratios, len(self.classes), conf_thre=threshold)
            #final_boxes, final_scores, final_labels = agnostic_nms_with_nested_removal(outputs_data, self.original_image, conf_threshold=threshold, iou_threshold=0.7, max_detections=300)
            return final_boxes, final_scores, final_labels
        else:
            self.model.conf_thres = threshold
            results = self.model([self.original_image], False)
            detections = results[0]  # Get the actual tensor from the list
            final_boxes = detections[:, 0:4]     # x1, y1, x2, y2
            final_scores = detections[:, 4]      # confidence
            final_labels = detections[:, 6].long()  # class_id as integers
            return final_boxes, final_scores, final_labels

    def draw(self, final_boxes, final_scores, final_labels, output_path="output.jpg"):
        # Draw detections
        for ind, det in enumerate(final_boxes):
            x1, y1, x2, y2 = det
            conf = final_scores[ind]
            cls_id = int(final_labels[ind])
            # Draw
            cv2.rectangle(self.original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{self.classes[int(cls_id)]}: {conf:.2f}"
            cv2.putText(self.original_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, self.original_image)