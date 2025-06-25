# KINEVA Inference

**KINEVA Inference** is a high-performance, GPU-accelerated solution designed for deploying deep learning models on **NVIDIA Jetson** devices. Built for speed and efficiency, KINEVA enables lightning-fast inference at the edge, with seamless integration into real-time systems.

Whether you're running object detection, or anomaly detection, KINEVA provides everything needed to convert, optimize, and deploy models for ultra-fast execution on Jetson hardware.

---

## 🚀 Features

- Optimized for real-time inference on NVIDIA Jetson devices
- Modular design supporting multiple model architectures
- Built-in conversion tools for high-speed TensorRT deployment

---

## 🧠 PREBUILT MODELS

KINEVA ships with a growing collection of **ready-to-use, production-grade AI models** — all fine-tuned for real-world edge applications. Whether you're working in robotics, smart cities, industrial automation, or retail analytics, we've got you covered.

| Model Name             | Description                                                                                       | Example Output                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [rb_coco](https://huggingface.co/rebotnix/rb_coco)              | General-purpose object detection based on COCO classes (people, vehicles, etc.)                   | ![rb\_coco](assets/rb_coco.jpg)                           |
| [rb_trafficsign](https://huggingface.co/rebotnix/rb_trafficsign)         | Detection and classification of traffic signs                                                     | ![rb\_trafficsign](assets/rb_trafficsign.jpg)             |
| [rb_productInspection](https://huggingface.co/rebotnix/rb_productInspection) | Visual quality control and anomaly detection in product assembly lines                            | ![rb\_productInspection](assets/rb_productInspection.jpg) |
| [rb_licenseplate](https://huggingface.co/rebotnix/rb_licenseplate)      | License plate detection and OCR-ready localization                                                | ![rb\_licenseplate](assets/rb_licenseplate.jpg)           |
| *...and many more*     | More models coming soon ... |                                                             |


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

---

## 📥 INSTALLATION

Install first neccessary packages if not installed yet (**CMake 3.22 or higher is required.**):
```bash
sudo apt install libopenblas-base libopenblas-dev cmake;
```

Clone the repository:

```bash
#download repository
git clone https://github.com/rebotnix/kineva_inference.git
cd kineva_inference

#create virtual env
virtualenv -p python3 venv

#load venv
source venv/bin/activate
```

Install steps for Pytorch + Torchvision + Tensorrt on **jetpack 5.1 or 5.2**:

```bash
#install torch
wget https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/4c1/d7a5d0ba92527/torch-2.2.0-cp38-cp38-linux_aarch64.whl#sha256=4c1d7a5d0ba92527c163ce9da74a2bdccce47541ef09a14d186e413a47337385
pip install torch-2.2.0-cp38-cp38-linux_aarch64.whl

#install torchvision
wget https://pypi.jetson-ai-lab.dev/jp5/cu114/+f/12c/2173bcd5255bd/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl#sha256=12c2173bcd5255bddad13047c573de24e0ce2ea47374c48ee8fb88466e021d2a
pip install torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl

# add tensorrt to venv (if present in system)
cp -r /usr/lib/python3.8/dist-packages/tensorrt venv/lib/python3.8/site-packages/
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

To run the SDK the following prerequisites need to be installed **with GPU support**:

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

---


## ⚙️ INTEGRATION
Once downloaded, RFDetr models can be:

Converted to TensorRT using KINEVA's model conversion tools

Deployed for ultra-fast inference on Jetson devices

Used with built-in postprocessing for bounding boxes and class predictions

The model works seamlessly with KINEVA’s modular interface—just specify the model path and config, and you're ready to run inference.

---

## EXAMPLES

### Export RFDETR model to TRT
Run the export script to create a .trt file.
```bash
PYTHONPATH=$(pwd) python examples/export_rfdetr.py
```
Edit the file examples/export_rfdetr.py if you want to change the pth file.

### Export KINEVA model to TRT
Run the export script to create a .trt file.
```bash
PYTHONPATH=$(pwd) python examples/export_kineva.py
```
Edit the file examples/export_kineva.py if you want to change the pth file.


### Run inference with RFDETR TRT

```bash
PYTHONPATH=$(pwd) python examples/test_rfdetr.py
```

Edit the file examples/test_rfdetr.py if you want to change the trt model file:


### Run inference with KINEVA TRT

```bash
PYTHONPATH=$(pwd) python examples/test_kineva.py
```


## CONTACT

📫 For commercial use or or other questions please contact us here:

✉️ Email: [communicate@rebotnix.com](mailto:communicate@rebotnix.com)

🌐 Website: [https://rebotnix.com](https://rebotnix.com)
