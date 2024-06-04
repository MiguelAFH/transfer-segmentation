import argparse
import torch
import os
import cv2
import gc
import base64
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


from transfer import *
from lpips_2imgs import find_similar_images
from models import StyleTransfer, SAMAnnotator

from utils import log, ImageLoader
from yolo_world import YOLOAnnotator


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
torch.cuda.set_device(DEVICE)
MODEL_TYPE = "vit_h"
HOME = "./"
DATA_PATH = os.path.join(HOME, "data")
BASELINE_DIR = os.path.join(DATA_PATH, "images", "style", "baseline")
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
IMSIZE = 224


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
        "--classes", 
        type=str,
        default="pyramid",
        help="List of classes to detect in the image"
    )
    
    parser.add_argument(
        "-a", "--artist",
        type=str,
        default="Vincent_van_Gogh",
        help="The artist to use for style transfer baseline"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top similar images to return"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    assert os.path.exists(args.content), f"Content image path {args.content} does not exist."
    assert os.path.exists(args.style), f"Style image path {args.style} does not exist."
    
    return args

def get_run_name(content_path, style_path, classes):
    content_name = os.path.basename(content_path)  # Get the base name
    content_file_name, _ = os.path.splitext(content_name)  # Split the name and extension
    style_name = os.path.basename(style_path)  # Get the base name
    style_file_name, _ = os.path.splitext(style_name)  # Split the name and extension
    return f"{classes}_{content_file_name}_{style_file_name}"
        
def print_mem():
    allocated_memory = torch.cuda.memory_allocated(DEVICE)
    allocated_memory_gib = allocated_memory / (1024 ** 3)  # Convert from bytes to GiB
    print(f"CUDA memory allocated: {allocated_memory_gib:.2f} GiB")


if __name__ == "__main__":
    # Setup
    args = parse_arguments()
    content_image_path = args.content
    style_image_path = args.style
    out_path = os.path.join("runs", get_run_name(content_image_path, style_image_path, args.classes))
    os.makedirs(out_path, exist_ok=True)
    
    # Image files
    stylized_image_path = os.path.join(out_path, "stylized.jpg")
    masked_image_path = os.path.join(out_path, "masked.jpg")
    out_image_path = os.path.join(out_path, "out.jpg")
    annotated_image_path = os.path.join(out_path, "annotated.jpg")
    
    # Find similar style images
    ARTIST_DIR = os.path.join(BASELINE_DIR, args.artist)
    style_image_name = os.path.basename(style_image_path).split('.')[0]
    similar_images_txt = os.path.join(DATA_PATH, f"similar_{style_image_name}_{args.artist}.txt")
    print(f"Similar images txt: {similar_images_txt}")
    if os.path.exists(similar_images_txt):
        similar_images = []
        with open(similar_images_txt, 'r') as f:
            for line in f:
                float_value, text_value = line.strip().split(', ')
                similar_images.append([float(float_value), text_value])
        similar_images = similar_images[:args.top]
    else: 
        similar_images = find_similar_images(style_image_path, ARTIST_DIR)
        with open(os.path.join(DATA_PATH, f"similar_{args.artist}".txt), 'w') as f:
            for item in similar_images:
                f.write(f'{item[0]}, {item[1]}\n')
    
    # Segmentation images
    image_bgr = cv2.imread(content_image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Object detector
    log("Running object detection for classes: ", args.classes)
    yolo_annotator = YOLOAnnotator(model_id='yolo_world/v2-l', classes=[args.classes], device=DEVICE)
    box, _ = yolo_annotator.get_bounding_box(image_bgr)
    annotated_image = yolo_annotator.annotate_image(image_bgr)
    log(f'Saving annotated image to {annotated_image_path}')
    cv2.imwrite(annotated_image_path, annotated_image)

    # Style images
    loader = ImageLoader(DEVICE, imsize=image_bgr.shape[0])
    style_img = loader.load(style_image_path)
    content_img = loader.load(content_image_path)    
    style_img = F.interpolate(style_img, size=content_img.shape[-2:])
    log("Style image size: ", style_img.shape)
    log("Content image size: ", content_img.shape)
    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
        
    # Object segmentator
    log("Segmenting image...")
    sam_annotator = SAMAnnotator(model=MODEL_TYPE, checkpoint=CHECKPOINT_PATH, device=DEVICE)
    masks, segmented_image = sam_annotator.annotate(image_rgb, box)
    
    # Style transfer
    log("Running style transfer using generative style image...")
    style_transfer = StyleTransfer(device=DEVICE)
    stylized_image = style_transfer.run(content_img, style_img)
    
    # Style transfer using baseline style images
    log("Running style transfer using baseline style images...")
    baseline_style_images = []
    for dist, similar_image_filename in similar_images:
        similar_image_path = os.path.join(ARTIST_DIR, similar_image_filename)
        similar_image = loader.load(similar_image_path)
        similar_image = F.interpolate(similar_image, size=content_img.shape[-2:])
        assert similar_image.size() == content_img.size(), \
        "we need to import style and content images of the same size"
        
        stylized_baseline_image = style_transfer.run(content_img, similar_image)
        stylized_baseline_image = stylized_baseline_image.cpu().detach().numpy()[0].transpose(1, 2, 0)
        stylized_baseline_image = np.array(255 * stylized_baseline_image, dtype=np.uint8)
        baseline_style_images.append([similar_image_filename, stylized_baseline_image])
        cv2.imwrite(os.path.join(out_path, f"stylized_{similar_image_filename}"), cv2.cvtColor(stylized_baseline_image, cv2.COLOR_RGB2BGR))
            
    # Save stylized images
    log("Saving stylized image to ", stylized_image_path)
    stylized = stylized_image.cpu().detach().numpy()[0].transpose(1, 2, 0)
    stylized = np.array(255 * stylized, dtype=np.uint8)
    cv2.imwrite(stylized_image_path, cv2.cvtColor(stylized, cv2.COLOR_RGB2BGR))

    # Save masked image
    log("Saving masked image to ", masked_image_path)
    mask = masks[0]
    stylized = stylized[:mask.shape[0], :mask.shape[1],:]
    stylized = stylized * mask[:, :, np.newaxis]
    cv2.imwrite(masked_image_path, cv2.cvtColor(stylized, cv2.COLOR_RGB2BGR))
    
    for similar_image_filename, stylized_baseline_image in baseline_style_images:
        stylized_baseline_image = stylized_baseline_image[:mask.shape[0], :mask.shape[1],:]
        stylized_baseline_image = stylized_baseline_image * mask[:, :, np.newaxis]
        cv2.imwrite(os.path.join(out_path, f"masked_{similar_image_filename}"), cv2.cvtColor(stylized_baseline_image, cv2.COLOR_RGB2BGR))
        # Create the combined image
        combined_image = np.zeros_like(stylized_baseline_image)
        combined_image[mask] = stylized_baseline_image[mask]
        combined_image[~mask] = image_rgb[~mask]

        # Convert the combined image back to BGR for saving with cv2
        combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        # log("Saving combined image to ", out_image_path)
        cv2.imwrite(os.path.join(out_path, f"out_{similar_image_filename}"), combined_image_bgr)
        
    # Create the combined image
    combined_image = np.zeros_like(stylized)
    combined_image[mask] = stylized[mask]
    combined_image[~mask] = image_rgb[~mask]

    # Convert the combined image back to BGR for saving with cv2
    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    log("Saving combined image to ", out_image_path)
    cv2.imwrite(out_image_path, combined_image_bgr)
    


# Runs:
# Pyramid of Giza
# python3 src/run.py --content data/images/content/Great_Pyramid_of_Giza/4.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png
# Necropolis
# python3 src/run.py --content data/images/content/Theban_Necropolis/4.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png --classes mountain
# Sphinx
# python3 src/run.py --content data/images/content/Sphinx_of_Memphis/0.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png --classes sphinx