# KINEVA Inference

**KINEVA Inference** is a high-performance, GPU-accelerated solution designed for deploying deep learning models on **NVIDIA Jetson** devices. Built for speed and efficiency, KINEVA enables lightning-fast inference at the edge, with seamless integration into real-time systems.

Whether you're running object detection, or anomaly detection, KINEVA provides everything needed to convert, optimize, and deploy models for ultra-fast execution on Jetson hardware.

---

## 🚀 Features

- Optimized for real-time inference on NVIDIA Jetson devices
- Modular design supporting multiple model architectures
- Built-in conversion tools for high-speed TensorRT deployment
-  
---

## 📦 Requirements

Before installing the SDK, ensure the following prerequisites are installed **with GPU support**:

- [PyTorch](https://pytorch.org/) (compatible with your CUDA version)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [TensorRT](https://developer.nvidia.com/tensorrt)

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
git clone https://github.com/rebotnix/kineva_inference.git
cd kineva_inference
```

## 🧠 Model Support: RFDetr
KINEVA Inference includes built-in support for the RFDetr object detection model—an efficient, transformer-based detector designed for high accuracy with edge deployment in mind.

🔗 Download Pretrained RFDetr Models
Pretrained RFDetr models, optimized for NVIDIA Jetson, are available via our official Hugging Face repository:

👉 [https://huggingface.co/rebotnix](https://huggingface.co/rebotnix)

## ⚙️ Integration with KINEVA
Once downloaded, RFDetr models can be:

Converted to TensorRT using KINEVA's model conversion tools

Deployed for ultra-fast inference on Jetson devices

Used with built-in postprocessing for bounding boxes and class predictions

The model works seamlessly with KINEVA’s modular interface—just specify the model path and config, and you're ready to run inference.
