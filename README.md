# KINEVA Inference

**KINEVA Inference** is a high-performance, GPU-accelerated solution designed for deploying deep learning models on **NVIDIA Jetson** devices. Built for speed and efficiency, KINEVA enables lightning-fast inference at the edge, with seamless integration into real-time systems.

Whether you're running object detection, or anomaly detection, KINEVA provides everything needed to convert, optimize, and deploy models for ultra-fast execution on Jetson hardware.

---

## 🚀 Features

- Optimized for real-time inference on NVIDIA Jetson devices
- Modular design supporting multiple model architectures
- Built-in conversion tools for high-speed TensorRT deployment

---

## 📦 Requirements

Before installing the SDK, ensure the following prerequisites are installed **with GPU support**:

(If you run on jetpack 6.1 you can follow the steps in the Installation section)

- [PyTorch](https://github.com/pytorch/pytorch) (compatible with your CUDA version)
- [torchvision](https://github.com/pytorch/vision)
- [TensorRT](https://github.com/NVIDIA/TensorRT)

> ✅ **Important:**  
> The `trtexec` tool from TensorRT must be installed and accessible in your terminal (i.e., it should be in your system's `PATH`).
>
> You can verify with:
> ```bash
> trtexec --help
> ```
> If not found, you may need to add it:
> ```bash
> export PATH=$PATH:/usr/src/tensorrt/bin  # adjust path if needed
> ```

All additional Python dependencies are listed in `requirements.txt`.

---

## 📥 Installation

Clone the repository:

```bash
#download repository
git clone https://github.com/rebotnix/kineva_inference.git
cd kineva_inference

#create virtual env
virtualenv -p python3.10 venv

#load venv
source venv/bin/activate
```

Install steps for Pytorch + Torchvision + Tensorrt on **jetpack 6.1**:

```bash
#install torch
wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/6ef/f643c0a7acda9/torch-2.7.0-cp310-cp310-linux_aarch64.whl#sha256=6eff643c0a7acda92734cc798338f733ff35c7df1a4434576f5ff7c66fc97319
pip install torch-2.7.0-cp310-cp310-linux_aarch64.whl

#install torchvision
wget https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/daa/bff3a07259968/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl#sha256=daabff3a0725996886b92e4b5dd143f5750ef4b181b5c7d01371a9185e8f0402
pip install torchvision-0.22.0-cp310-cp310-linux_aarch64.whl

# add tensorrt to venv (if present in system)
cp -r /usr/lib/python3.10/dist-packages/tensorrt venv/lib/python3.10/site-packages/
```

Now install requirements:

```bash
pip install -r requirements.txt
```

## 🧠 Model Support: RFDetr
KINEVA Inference includes built-in support for the RFDetr object detection model—an efficient, transformer-based detector designed for high accuracy with edge deployment in mind.

🔗 Download Pretrained RFDetr Models
Pretrained RFDetr models, optimized for NVIDIA Jetson, are available via our official Hugging Face repository:

👉 [https://huggingface.co/rebotnix](https://huggingface.co/rebotnix)

Example to download the rb_coco model. Please insert your own token after getting access to repo.
```bash
mkdir models
cd models
curl -L -o rb_coco.pth -H "Authorization: Bearer YOUR_HF_TOKEN" "https://huggingface.co/rebotnix/rb_coco/resolve/main/rb_coco.pth?download=true"
cd ..
```


## ⚙️ Integration with KINEVA
Once downloaded, RFDetr models can be:

Converted to TensorRT using KINEVA's model conversion tools

Deployed for ultra-fast inference on Jetson devices

Used with built-in postprocessing for bounding boxes and class predictions

The model works seamlessly with KINEVA’s modular interface—just specify the model path and config, and you're ready to run inference.

## Export + Inference
Run the export script to create a .trt file.
```bash
PYTHONPATH=$(pwd) python examples/export_rfdetr.py
```

Edit the file examples/export_rfdetr.py if you want to change the pth file:
```python
from kineva import RFDETR

#initialize model
model = RFDETR(model="models/rb_coco.pth")

#export model to trt
model.export()
```

Now you run an inference with:

```bash
PYTHONPATH=$(pwd) python examples/test_rfdetr.py
```

Edit the file examples/test_rfdetr.py if you want to change the trt model file:

```python
from kineva import RFDETR

myclasses = ['trafficsign']

#initialize model
model = RFDETR(model="models/rb_trafficsign.trt", classes="data/rb_trafficsign.json")

#run inference on image
final_boxes, final_scores, final_labels = model.detect("images/bus.jpg", threshold=0.5)

#draw detection
model.draw(final_boxes, final_scores, final_labels, output_path="./outputs/output_rfdetr.jpg")
```

## Contact

📫 For commercial use or or other questions please contact us here:

✉️ Email: [communicate@rebotnix.com](mailto:communicate@rebotnix.com)

🌐 Website: [https://rebotnix.com](https://rebotnix.com)


