"""
Kineva Inference Engine
Copyright (C) Rebotnix

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
For license details, see: https://www.gnu.org/licenses/agpl-3.0.html
Project website: https://rebotnix.com
"""

import torch
import os
from ..models.kineva.core import EdgeYOLO
import numpy as np

class ANOMALY_wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, query_image, prompt_image):
        return self.model({
            "query_image": query_image,
            "prompt_image": prompt_image
        })

def convert_to_trt(model, output_path, img_size=640, mode="anomaly"):
    if mode == "anomaly":
        dummy_query = torch.randn(1, 3, img_size,img_size).cuda()
        dummy_prompt = torch.randn(1, 3, img_size, img_size).cuda()
        onnx_model = ANOMALY_wrapper(model)
        dummy_path = "./dummy.onnx"
        torch.onnx.export(
            onnx_model,
            (dummy_query, dummy_prompt),
            dummy_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['query_image', 'prompt_image'],
            output_names=['output'],
            dynamic_axes={
                'query_image': {0: 'batch_size'},
                'prompt_image': {0: 'batch_size'},
                'output': {0: 'batch_size'}
                }
            )
        #create trt
        os.system("trtexec --onnx="+dummy_path+" --saveEngine="+output_path+" --fp16")
        #remove onnx
        os.remove(dummy_path)

    elif mode == "rfdetr":
        dummy_dir=".dummy_dir"
        model.export(output_dir=dummy_dir)
        dummy_path = os.path.join(dummy_dir,"inference_model.onnx")
        #create trt
        os.system("trtexec --onnx="+dummy_path+" --saveEngine="+output_path+" --fp16")
        #remove onnx
        os.remove(dummy_path)
        os.rmdir(dummy_dir)
        
    elif mode == "kineva":
        exp = EdgeYOLO(weights=model)
        modelc = exp.model
        modelc.fuse()
        modelc.eval()
        x = np.ones([1, 3, img_size, img_size], dtype=np.float32)
        x = torch.from_numpy(x)  # .cuda()
        modelc(x)
        input_names = ["input_0"]
        output_names = ["output_0"]
        dummy_path = "./dummy.onnx"
        torch.onnx.export(modelc,
            x,
            dummy_path,
            verbose=False,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None
        )
        #create trt
        os.system("trtexec --onnx="+dummy_path+" --saveEngine="+output_path+" --fp16")
        #remove onnx
        os.remove(dummy_path)