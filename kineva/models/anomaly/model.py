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
from kineva.utils.export import *
import numpy as np
from kineva.models.anomaly.core import *

class ANOMALY():

    def __init__(self, model: Union[str, Path], img_size = 256):
        """
        Initialize an ANOMALY model.

        This constructor initializes an anomaly detetction model. It needs a model path as well as an image of a perfect product.

        Args:
            model (str | Path): path to model file, i.e. 'metauas_256.trt', 'metauas_512.trt'.
            reference (str | Path): path to reference image file, i.e. reference.

        Examples:
            >>> from kineva import ANOMALY
            >>> model = ANOMALY("metauas_256.trt", "reference.jpg")
        """

        path = Path(model if isinstance(model, (str, Path)) else "")
        self.path = str(path)
        self.img_size = int(model.split("_")[1].split(".")[0])
        
        if str(path).endswith("trt"):
            self.input = "trt"
            engine = load_engine(path)
            self.context = engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(engine, mode="anomaly")

        elif str(path).endswith("pth"):
            self.input = "pth"
            # load model
            encoder = 'efficientnet-b4' 
            decoder = 'unet' 
            encoder_depth = 5
            decoder_depth = 5
            num_crossfa_layers = 3
            alignment_type =  'sa' 
            fusion_policy = 'cat'
            self.model = ANOMALY_PTH(encoder, 
                decoder, 
                encoder_depth, 
                decoder_depth, 
                num_crossfa_layers, 
                alignment_type, 
                fusion_policy
            ) 
            self.model = safely_load_state_dict(self.model, path)
            self.model.cuda()
            self.model.eval()
            
        else:
            print("Sorry. We only support .trt or .pth files.")

    def export(self):
        if self.input == "pth":
            convert_to_trt(self.model, self.path.replace("pth", "trt"), self.img_size, "anomaly")
        else:
            print("Sorry. You already have a trt model.")
            
    def set_reference(self, reference):
        if self.input == "trt":
            self.reference = preprocess(reference, self.img_size)
            self.reference = self.reference.reshape(self.inputs[0]['shape'])
            # Copy to pagelocked host memory
            np.copyto(self.inputs[1]['host'], self.reference)
        else:
            self.reference = read_image_as_tensor(reference)

    def detect(self, image):
        if self.input == "trt":
            self.prompt = preprocess(image, img_size=self.img_size, mode="anomaly")
            #print(self.prompt.shape)
            np.copyto(self.inputs[0]['host'], self.prompt)
            trt_outputs = infer(self.context, self.inputs, self.outputs, self.stream, mode="anomaly")
            pred_mask = trt_outputs[0]
            anomaly_score = pred_mask[:].max()
            return pred_mask,anomaly_score
        else:
            self.prompt = read_image_as_tensor(image)
            if self.prompt.shape[1] != self.img_size:
                resize_trans = K.augmentation.Resize([self.img_size, self.img_size], return_transform=True)
                self.prompt = resize_trans(self.prompt)[0]
                self.reference = resize_trans(self.reference)[0]
            test_data = {
                "query_image": self.prompt.cuda(),
                "prompt_image": self.reference.cuda(),
            }
            # forward
            pred_mask = self.model(test_data)
            anomaly_score = pred_mask[:].max().item()
            return pred_mask,anomaly_score
            
    
    def draw(self, pred_mask, output_path="", save=True):
        if self.input == "trt":
            pred = (1 - pred_mask.squeeze())[:, :, None].repeat(3, axis=2)
            query_img = (self.prompt[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            norm_pred = normalize(pred)
            scoremap_overlay = apply_ad_scoremap(query_img, norm_pred)
            if save:
                cv2.imwrite(output_path, scoremap_overlay)
            else:
                return scoremap_overlay
        else:
            # visualization
            query_img = self.prompt.cuda()[0] * 255
            query_img = query_img.permute(1,2,0)
            pred = (1-pred_mask.squeeze().detach())[:, :, None].cpu().numpy().repeat(3, 2)
            # normalize just for analysis
            scoremap_overlay = apply_ad_scoremap(query_img.cpu(), normalize(pred))
            if save:
                cv2.imwrite(output_path, scoremap_overlay)
            else:
                return scoremap_overlay