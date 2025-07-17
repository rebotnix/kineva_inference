"""
Kineva Inference Engine
Copyright (C) Rebotnix

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
For license details, see: https://www.gnu.org/licenses/agpl-3.0.html
Project website: https://rebotnix.com
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
#from kineva.utils.trt import *
#from kineva.utils.processing import *
from kineva.utils.general import *
import numpy as np
import torch
import cv2
import json
import sys
import os
sys.path.append(os.path.abspath('./core'))
from .core.detect import Detector, TRTDetector
from .core import EdgeYOLO
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
        #with open(classes, 'r') as f:
        #    self.classes = json.load(f)
        #path = Path(model if isinstance(model, (str, Path)) else "")
        self.img_size = img_size
        self.path = str(model)
        if str(self.path).endswith("pt") or str(self.path).endswith("trt"):
            self.input = "trt"
            self.model = TRTDetector(
            weight_file=self.path,
            conf_thres=threshold,
            nms_thres=0.55,
            input_size=[self.img_size, self.img_size],
            fuse=True,
            fp16=False,
            use_decoder=False
            )
            self.classes = self.model.class_names
            print("model loaded")
            #engine = load_engine(path)
            #self.context = engine.create_execution_context()
            #self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine, mode="kineva")
        elif str(self.path).endswith("pth"):
            self.input = "pth"
            self.model = Detector(
            weight_file=self.path,
            conf_thres=threshold,
            nms_thres=0.55,
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
            #convert_to_trt(self.path, self.path.replace("pth", "trt"), self.img_size, "kineva")
            exp=EdgeYOLO(weights=self.path)
            model = exp.model
            model.fuse()
            model.eval()
            import tensorrt as trt
            from .core.export import torch2onnx2trt
            x = np.ones([1, 3, *[self.img_size, self.img_size]], dtype=np.float32)
            x = torch.from_numpy(x)  # .cuda()
            model(x)  # warm and init
            input_names = ["input_0"]
            output_names = ["output_0"]
            model_trt = torch2onnx2trt(
            model,
            [x],
            fp16_mode=not False,
            int8_mode=False,
            int8_calib_dataset=None,
            log_level=trt.Logger.INFO,
            max_workspace_size=(int((1 << 30) * 8)),
            max_batch_size=1,
            use_onnx=True,
            onnx_opset=11,
            input_names=input_names,
            output_names=output_names,
            simplify=not False,
            save_onnx=None,
            save_trt=True
            )
            data_save = {
            "names": self.model.class_names,
            "img_size": [self.img_size, self.img_size],
            "batch_size": 1,
            "pixel_range": exp.ckpt.get("pixel_range") or 255,  # input image pixel value range: 0-1 or 0-255
            "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
            "input_name": "input_0",
            "output_name": "output_0",
            "dtype": "float"
            }
            data_save["model"] = model_trt.state_dict()
            torch.save(data_save, self.path.replace("pth", "pt"))

        else:
            print("Sorry. You already have a trt model.")

    def detect(self, image, threshold=0.35):
        if isinstance(image, str):
          image = cv2.imread(image)
        self.original_image = image
        if self.input == "trt":
            self.model.conf_thres = threshold
            results = self.model([self.original_image], False)
            detections = results[0]  # Get the actual tensor from the list
            #print(detections)
            final_boxes = detections[:, 0:4]     # x1, y1, x2, y2
            final_scores = detections[:, 4]      # confidence
            final_labels = detections[:, 6].long()  # class_id as integers
            #input_image, ratios = preprocess(image, img_size=(self.img_size,self.img_size ), mode="kineva")
            #np.copyto(self.inputs[0]['host'], input_image.ravel())
            #outputs_data = infer(self.context, self.inputs, self.outputs, self.stream, mode="kineva")[0]
            #outputs_data = torch.from_numpy(outputs_data).float()
            #final_boxes, final_scores, final_labels = postprocess_kineva(outputs_data, ratios, len(self.classes), conf_thre=threshold)
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
        color_palette = generate_color_palette(self.classes)
        # Draw detections
        for ind, det in enumerate(final_boxes):
            x1, y1, x2, y2 = det
            conf = final_scores[ind]
            cls_id = int(final_labels[ind])
            # Draw
            #cv2.rectangle(self.original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            self.original_image = draw_dashed_rectangle(self.original_image, (int(x1), int(y1)), (int(x2), int(y2)), color=color_palette[self.classes[cls_id]])
            label = f"{self.classes[int(cls_id)]}: {conf:.2f}"
            cv2.putText(self.original_image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, self.original_image)
