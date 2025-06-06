# KINEVA Inference

The **KINEVA Inference** provides a modular, GPU-accelerated framework for deploying deep learning models using PyTorch, TensorRT, and torchvision. It supports multiple architectures (e.g., YOLO-based, transformer-based, anomaly detection) and includes utilities for model conversion, inference, and result postprocessing.

---

## 🚀 Features

- Unified interface for multiple model types
- GPU-accelerated inference with TensorRT
- Postprocessing utilities including NMS, rescaling, and filtering
- Built-in support for batch inference and custom thresholds
- Easy integration with Python projects

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
