from ultralytics import YOLO
import os

def detect_images(model):
    # Inference on images
    img_path = r"C:\Users\User\Videos\Master\yolo_images"
    images = [os.path.join(img_path, img) for img in os.listdir(img_path) if img.endswith(('.png', '.jpg'))]
    results = model(images)

    dir = os.path.dirname(img_path)
    output_dir = os.path.join(dir,'yolo_output')
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bounding box outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename=os.path.join(output_dir, f"{os.path.basename(images[i])}"))
        print(f"{os.path.basename(images[i])} saved in {output_dir}.")

def detect_video(model):
    # Inference on video
    vid_path = r"C:\Users\User\Videos\yolo_videos"
    result = model(vid_path, show=True, save=True)

def detect_cam(model):
    # Inference on camera in real-time
    model.predict(source="0", show=True)

def test_coco(model):
    """Test on COCO dataset"""
    # Test 1: Interrupted by insufficient storage space
    # Test 2: Conf and max_det are not set to default
    # model.val(data="coco.yaml", save_json=True, conf=0.6, max_det=10)
    
    # Test 3
    model.val(data="coco.yaml", save_json=True)

def export(model):
    # Exporting the model
    # model.export(format="onnx")
    model.export(
        format="tflite",
        imgsz=128,
        int8=True,
        data="coco.yaml",
        fraction=0.1
    )

def main():
    # Load pretrained YOLO version 12 nano
    model = YOLO('yolov12n.pt')
    export(model)

if __name__ == '__main__':
    main()