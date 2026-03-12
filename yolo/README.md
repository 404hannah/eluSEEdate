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

Note: When using Google Colab, this is not necessary.

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

## YOLO Information

You Only Look Once

### How does YOLO work?

YOLO creates a feature map for the image and then, it uses Feature Pyramid Network or FPN. In FPN, the input or picture is processed, producing feature maps of ranging resolutions in order to detect objects at different scales. In low resolutions, large objects are detected while in high resolutions, small objects are detected. The feature maps and outputs (also feature maps) of the different scales are combined.

Each resolution is divided into a grid of cells. CNN is utilized to determine all the class probabilities in one cell. Aside from the class probability, the cell also determines the center coordinates, the width and height, and the confidence score of each object's bounding box. In the cells and scales level, multiple bounding boxes could be made for one object to determine the best bounding box.

To make the model accurate during training, the intersection over union of the predicted object and the ground truth is used to calculate the loss.

### Batch size

The parameter batch size in the yolo function predict() expedites the latency for each image. However, this parameter is recommended to be low for edge devices. Batch size is increased to utilize unused computing resources of the GPU.

### Post-training quantization

Quantization changes the precision or reduces the digits of parameters which enables the model to process faster and makes the size smaller. The downside of this is a small cut on the accuracy.

- Float-16 Quantization - weights represented as float occupy 16 bits
- Dynamic Range Quantization - weights represented as float occupy 8 bits. Faster than float-16 but slower than INT8
- Integer Quantization (INT8) - weights, activation outputs, and more use 8 bits
  Conversion to this type of model needs a representative dataset. Fastest among the three and ties with dynamic range in having the smallest size

### References

- https://youtu.be/svn9-xV7wjk?list=LL
- https://learnopencv.com/tensorflow-lite-model-optimization-for-on-device-machine-learning/
- https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_quantization
