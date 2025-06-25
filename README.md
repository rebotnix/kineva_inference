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

KINEVA offers a curated suite of **8 high-performance, production-ready AI models**, purpose-built for edge applications. These models are available under a **commercial licensing model**, giving you the flexibility to integrate best-in-class AI into your product without the burden of training from scratch.

From **object detection** and **defect inspection** to **traffic sign recognition** and **license plate localization**, each model is fully optimized for real-time inference on **NVIDIA Jetson** hardware.

| Model Name             | Model Type | Detection Type | Description                                                                                       | Example Output                                              |
| ---------------------- | --- | --- |------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [rb_coco](https://huggingface.co/rebotnix/rb_coco)        |  RF-DETR  |  Object Detection     | General-purpose object detection based on COCO classes (people, vehicles, etc.)                   | ![rb\_coco](assets/rb_coco.jpg)                           |
| [rb_trafficsign](https://huggingface.co/rebotnix/rb_trafficsign)      |  RF-DETR  |  Object Detection   | Detection and classification of traffic signs                                                     | ![rb\_trafficsign](assets/rb_trafficsign.jpg)             |
| [rb_productInspection](https://huggingface.co/rebotnix/rb_productInspection) |  RF-DETR  |  Object Detection | Visual quality control and anomaly detection in product assembly lines                            | ![rb\_productInspection](assets/rb_productInspection.jpg) |
| [rb_licenseplate](https://huggingface.co/rebotnix/rb_licenseplate)   |  RF-DETR  |  Object Detection   | License plate detection and OCR-ready localization                                                | ![rb\_licenseplate](assets/rb_licenseplate.jpg)           |
| [rb_graffiti](https://huggingface.co/rebotnix/rb_graffiti)  |  RF-DETR  |  Object Detection    | Detection of graffiti and visual vandalism in urban environments                       | ![rb\_licenseplate](assets/rb_graffiti.jpg)   
| [rb_aircraft](https://huggingface.co/rebotnix/rb_aircraft)   |  RF-DETR  |  Object Detection   | Aircraft detection and classification for aviation or surveillance use cases on aerial images                      | ![rb\_licenseplate](assets/rb_aircraft.jpg)   
| [rb_vehicle](https://huggingface.co/rebotnix/rb_vehicle)  |  RF-DETR  |  Object Detection    | Specialized vehicle detection in traffic or parking scenarios on aerial images                      | ![rb\_licenseplate](assets/rb_vehicle.jpg)    
| [rb_ship](https://huggingface.co/rebotnix/rb_ship)  |  RF-DETR  |  Object Detection    | Ship and vessel detection in ports, harbors, or maritime surveillance on aerial images                      | ![rb\_licenseplate](assets/rb_ship.jpg)    

💡 Need a custom model? Contact us to train, optimize, and deploy your AI models through the KINEVA pipeline.

🔗 Download Pretrained RFDetr Models
Pretrained RFDetr models, optimized for NVIDIA Jetson, are available via our official Hugging Face repository:

👉 [https://huggingface.co/rebotnix](https://huggingface.co/rebotnix)

---

## 📥 INSTALLATION

Install first neccessary packages if not installed yet:

```bash
sudo apt install libopenblas-base libopenblas-dev cmake curl
```

On Jetpack 5.1 and 5.2 we reccomend **CMake 3.22.**

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

Install steps for **Jetpack 5.1 or 5.2**:

```bash
#install torch
wget https://docs.rebotnix.com/downloads/wheels/torch-2.2.0-cp38-cp38-linux_aarch64.whl
pip install torch-2.2.0-cp38-cp38-linux_aarch64.whl

#install torchvision
wget https://docs.rebotnix.com/downloads/wheels/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
pip install torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl

# install tensorrt
mkdir -p venv/lib/python3.8/site-packages/tensorrt
wget -P venv/lib/python3.8/site-packages/tensorrt https://docs.rebotnix.com/downloads/wheels/tensorrt_38/__init__.py
wget -P venv/lib/python3.8/site-packages/tensorrt https://docs.rebotnix.com/downloads/wheels/tensorrt_38/tensorrt.so
```

Install steps for **Jetpack 6.1**:

```bash
#install torch
wget https://docs.rebotnix.com/downloads/wheels/torch-2.7.0-cp310-cp310-linux_aarch64.whl
pip install torch-2.7.0-cp310-cp310-linux_aarch64.whl

#install torchvision
wget https://docs.rebotnix.com/downloads/wheels/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.22.0-cp310-cp310-linux_aarch64.whl

# install tensorrt
mkdir -p venv/lib/python3.10/site-packages/tensorrt
wget -P venv/lib/python3.10/site-packages/tensorrt https://docs.rebotnix.com/downloads/wheels/tensorrt_310/__init__.py
wget -P venv/lib/python3.10/site-packages/tensorrt https://docs.rebotnix.com/downloads/wheels/tensorrt_310/tensorrt.so
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

### Download a RFDETR model for export
```bash
mkdir models
cd models
curl -L -o rb_coco.pth -H "Authorization: Bearer YOUR_HF_TOKEN" "https://huggingface.co/rebotnix/rb_coco/resolve/main/rb_coco.pth?download=true"
cd ..
```

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
