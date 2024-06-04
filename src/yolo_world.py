import os
import numpy as np
import cv2
import supervision as sv
import argparse

from typing import List
from inference.models.yolo_world.yolo_world import YOLOWorld

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO Image Annotator')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output', type=str, help='Path to save the annotated image')
    
    args = parser.parse_args()
    
    if not args.output:
        output_dir = 'runs/yolo_world'
        os.makedirs(output_dir, exist_ok=True)
        input_filename = os.path.basename(args.input)
        input_name, input_ext = os.path.splitext(input_filename)
        default_output = f'{input_name}_annotated.jpg'
        args.output = os.path.join(output_dir, default_output)
    
    return args

def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    return image


class YOLOAnnotator:
    def __init__(self, model_id: str, classes: List[str], device: str):
        self.model_id = model_id
        self.classes = classes
        self.model = YOLOWorld(model_id=model_id)
        
        self.model.set_classes(classes)
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

    def get_bbox(self, detections: sv.Detections) -> np.ndarray:
        if len(detections.xyxy) == 0 or detections.xyxy[0] is None or len(detections.xyxy[0]) == 0:
            raise ValueError('No bounding box detected')
        x, y, x2, y2 = detections.xyxy[0]
        bbox = np.array([x, y, x2, y2 - y])
        return bbox

    def get_bounding_box(self, image):
        results = self.model.infer(image, confidence=0.003)
        detections = sv.Detections.from_inference(results)
        bbox = self.get_bbox(detections)
        return bbox, detections
    
    def annotate_image(self, image):
        _, detections = self.get_bounding_box(image)
        annotated_image = image.copy()
        annotated_image = self.box_annotator.annotate(annotated_image, detections[0])
        annotated_image = self.label_annotator.annotate(annotated_image, detections[0])
        return annotated_image
    
    
if __name__ == '__main__':
    #Load image
    
    args = parse_arguments()
    source_image_path = args.input
    output_image_path = args.output
    image = load_image(source_image_path)
    model = YOLOAnnotator(model_id='yolo_world/v2-l', classes=['pyramid'], device="cuda:3")
    annotated_image = model.annotate_image(image)
    print(f'Saving annotated image to {output_image_path}')
    cv2.imwrite(output_image_path, annotated_image)