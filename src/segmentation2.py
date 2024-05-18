import argparse
import torch
import os
import cv2
import supervision as sv
import base64
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.models import vgg19, VGG19_Weights
from transfer import *


DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
MODEL_TYPE = "vit_h"
HOME = "./"
DATA_PATH = os.path.join(HOME, "data")
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

def encode_image(filepath):
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        encoded = str(base64.b64encode(image_bytes), 'utf-8')
        return "data:image/jpg;base64,"+encoded
    

def save_masks(masks, output_path):
    for i, mask in enumerate(masks):
        mask_filename = os.path.join(output_path, f'mask_{i}.jpg')
        # Ensure mask is in 0-255 range and uint8 type for saving
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_filename, mask_uint8)
        print(f'Saved mask {i} as {mask_filename}')


def parse_arguments():
    """
    Parses command-line arguments for content and style image paths.
    
    Returns:
        argparse.Namespace: Parsed arguments with content and style image paths.
    """
    parser = argparse.ArgumentParser(description="Process content and style image paths.")
    parser.add_argument(
        "--content", 
        type=str, 
        required=True, 
        help="The path to the content image"
    )
    parser.add_argument(
        "--style", 
        type=str, 
        required=True, 
        help="The path to the style image"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

def get_run_name(content_path, style_path):
    content_name = os.path.basename(content_path)  # Get the base name
    content_file_name, _ = os.path.splitext(content_name)  # Split the name and extension
    style_name = os.path.basename(style_path)  # Get the base name
    style_file_name, _ = os.path.splitext(style_name)  # Split the name and extension
    return f"{content_file_name}_{style_file_name}"

def get_bounding_box(box):
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])
    return box

def log(*msg):
    print("="*50)
    print(*msg)

if __name__ == "__main__":

    args = parse_arguments()

    content_image_path = args.content
    style_image_path = args.style
    assert os.path.exists(content_image_path), f"Content image path {content_image_path} does not exist."
    assert os.path.exists(style_image_path), f"Style image path {style_image_path} does not exist."
    out_path = os.path.join("runs", get_run_name(content_image_path, style_image_path))
    os.makedirs(out_path, exist_ok=True)
    
    stylized_image_path = os.path.join(out_path, "stylized.jpg")
    masked_image_path = os.path.join(out_path, "masked.jpg")
    out_image_path = os.path.join(out_path, "out.jpg")

    imsize = 224
    default_box = {'x': 1, 'y': 64, 'width': 219, 'height': 137, 'label': ''}
    box = get_bounding_box(default_box)
    
    loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    style_img = image_loader(loader, style_image_path, DEVICE)
    content_img = image_loader(loader, content_image_path, DEVICE)
    log("Style image size: ", style_img.shape)
    log("Content image size: ", content_img.shape)
    
    style_img = F.interpolate(style_img, size=content_img.shape[-2:])

    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(content_image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    sam_result = mask_generator.generate(image_rgb)

    mask_predictor = SamPredictor(sam)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    
    if args.debug:
        log("Saving masks")
        save_masks(masks, out_path)

    box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    plt.figure()
    imshow(style_img, title='Style Image')
    plt.clf()

    plt.figure()
    imshow(content_img, title='Content Image')
    plt.clf()

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    input_img = content_img.clone()
    plt.figure()
    imshow(input_img, title='Input Image')
    plt.clf()

    log("Running style transfer")
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

    log("Saving stylized image to ", stylized_image_path)
    plt.figure()
    imshow(output, title='Stylized Image')
    plt.savefig(stylized_image_path)
    plt.clf()

    out = output.cpu().detach().numpy()[0]
    # print("Stylized image shape: ", out.shape)

    log("Saving masked image to ", masked_image_path)
    mask = masks[0]
    out = out[:,:mask.shape[0], :mask.shape[1]]
    # print("Mask shape: ", mask.shape)
    out = out[:] * mask
    plt.imshow(out.transpose(1, 2, 0))
    plt.savefig(masked_image_path)

