# YOLO-12

Implementation of pretrained YOLOv12 Nano for object detection

Sources:

- [Github Repository](https://github.com/sunsmarterjie/yolov12)
- [YOLOv12 Paper](https://arxiv.org/abs/2502.12524)
- [Ultralytics](https://docs.ultralytics.com/models/yolo12/)

## How to Use

### Install

Create a virtual environment. In the venv, run the following commands:

```
pip install git+https://github.com/sunsmarterjie/yolov12.git
pip install huggingface_hub
```

For exporting to TFLite:

```
pip install tensorflow
pip install tf-keras
pip install sng4onnx
pip install onnx-graphsurgeon
pip install onnx
pip install onnxslim
pip install onnxruntime
pip install onnx2tf --no-deps
pip install ai-edge-litert
pip install onnxscript
pip install tflite-support
```

### Run

> python yolo12.py

## Object Detection

### On images

Initialize the directory containing the images.

> img_path = r""

Output images are saved on a **yolo_output** directory in the source's parent directory.

### On video

Initialize the video path.

vid_path = r""

### On live video

Uses the webcam of the computer.

## Testing

The model is tested on the COCO 2017 dataset accessed using the **coco.yaml** file.

> model.val(data="coco.yaml", save_json=True)

Fix of an error during model.val(): np.trapz to np.trapezoid in ultralytics/utils/metrics.py

## Export

To not encounter errors, use Google Colab and set the runtime version to 2025.07 (Python 3.11.13).
