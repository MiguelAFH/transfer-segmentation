import torch
import supervision as sv
import numpy as np

from segment_anything import sam_model_registry, SamPredictor
from torchvision.models import vgg19, VGG19_Weights
from transfer import *
from utils import log


class StyleTransfer:
    
    def __init__(self, device):
        self.model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
    def run(self, content_img, style_img):
        log("Running style transfer")
        input_img = content_img.clone()
        return run_style_transfer(self.model, self.cnn_normalization_mean, self.cnn_normalization_std,
                            content_img, style_img, input_img)


class SAMAnnotator:
    
    def __init__(self, model: str, checkpoint: str, device: str):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.sam = sam_model_registry[model](checkpoint=checkpoint).to(device=device)
        self.mask_predictor = SamPredictor(self.sam)
        self.box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
        self.mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)
    
    
    def predict(self, image: np.array, box: np.array):
        self.mask_predictor.set_image(image)
        masks, scores, logits = self.mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        return masks
    
    def annotate(self, image: np.array, box: np.array):
        masks = self.predict(image, box)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]
        segmented_image = self.mask_annotator.annotate(scene=image.copy(), detections=detections)
        return masks, segmented_image