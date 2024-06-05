import argparse
import torch
import os
import cv2
import gc
import base64
import numpy as np
import torch.nn.functional as F


from transfer import *
from lpips_2imgs import find_similar_images, compute_similaty
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
GENERATED_DIR = os.path.join(DATA_PATH, "images", "style", "generated")
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

def load_similar_file(artist, top, artist_baseline_dir, artist_generated_dir):
    similar_images_txt = os.path.join(DATA_PATH, f"similar_{artist}.txt")
    if os.path.exists(similar_images_txt):
        similar_images = []
        with open(similar_images_txt, 'r') as f:
            for line in f:
                float_value, text_value, text_Value2 = line.strip().split(', ')
                similar_images.append([float(float_value), text_value, text_Value2])
    else:
        similar_images = []
        for file in os.listdir(artist_generated_dir):
            similar_images.extend(find_similar_images(os.path.join(artist_generated_dir, file), artist_baseline_dir))
            
        similar_images = sorted(similar_images, key=lambda x: x[0])
        with open(os.path.join(DATA_PATH, f"similar_{artist}.txt"), 'w') as f:
            for item in similar_images:
                f.write(f'{item[0]}, {item[1]}, {item[2]}\n')
    
    similar_images = similar_images[:top]
    log("Similar images: ")
    print(similar_images)
    return similar_images

def load_similar_images(similar_images, content_img, loader, artist_baseline_dir, artist_generated_dir):
    def load_image(image_path):
        image = loader.load(image_path)
        image = F.interpolate(image, size=content_img.shape[-2:])
        assert image.size() == content_img.size(), \
        "we need to import style and content images of the same size"
        return image
    
    style_images = []
    for dist, baseline_image_name, generated_image_name in similar_images:
        baseline_image = load_image(os.path.join(artist_baseline_dir, baseline_image_name))
        generated_image = load_image(os.path.join(artist_generated_dir, generated_image_name))
        style_images.append([dist, baseline_image_name, generated_image_name, baseline_image, generated_image])
        
    return style_images

def get_stylized_images(style_images, content_img):
    style_transfer = StyleTransfer(device=DEVICE)
    def stylize_image(style_image):
        stylized_image = style_transfer.run(content_img, style_image)
        stylized_image = stylized_image.cpu().detach().numpy()[0].transpose(1, 2, 0)
        stylized_image = np.array(255 * stylized_image, dtype=np.uint8)
        return stylized_image
    
    stylized_images = []
    for dist, baseline_image_name, generated_image_name, baseline_image, generated_image in style_images:
        stylized_baseline_image = stylize_image(baseline_image)
        stylized_generated_image = stylize_image(generated_image)
        stylized_images.append([baseline_image_name, generated_image_name, stylized_baseline_image, stylized_generated_image])
        cv2.imwrite(os.path.join(out_path, f"stylized_{baseline_image_name}"), cv2.cvtColor(stylized_baseline_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_path, f"stylized_{generated_image_name}"), cv2.cvtColor(stylized_generated_image, cv2.COLOR_RGB2BGR))
    return stylized_images

def combine_images(stylized_images, mask, image_rgb):
    
    def combine(image1, image2):
        combined_image = np.zeros_like(image1)
        combined_image[mask] = image1[mask]
        combined_image[~mask] = image2[~mask]
        combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        return combined_image_bgr
    
    def apply_mask(image):
        image = image[:mask.shape[0], :mask.shape[1],:]
        image = image * mask[:, :, np.newaxis]
        return image
    combined_images = []
    for baseline_image_name, generated_image_name, stylized_baseline_image, stylized_generated_image in stylized_images:
        stylized_baseline_image = apply_mask(stylized_baseline_image)
        cv2.imwrite(os.path.join(out_path, f"masked_{baseline_image_name}"), cv2.cvtColor(stylized_baseline_image, cv2.COLOR_RGB2BGR))
        combined_image_bgr = combine(stylized_baseline_image, image_rgb)
        combined_baseline_path = os.path.join(out_path, f"out_{baseline_image_name}")
        cv2.imwrite(combined_baseline_path, combined_image_bgr)
        
        stylized_generated_image = apply_mask(stylized_generated_image)        
        cv2.imwrite(os.path.join(out_path, f"masked_{generated_image_name}"), cv2.cvtColor(stylized_generated_image, cv2.COLOR_RGB2BGR))
        combined_image_bgr = combine(stylized_generated_image, image_rgb)
        combined_generated_path = os.path.join(out_path, f"out_{generated_image_name}")
        cv2.imwrite(combined_generated_path, combined_image_bgr)
        
        combined_images.append([combined_baseline_path, combined_generated_path])
    
    return combined_images

if __name__ == "__main__":
    # Setup
    args = parse_arguments()
    content_image_path = args.content

    out_path = os.path.join("runs", f"{args.artist}_{args.classes}")
    os.makedirs(out_path, exist_ok=True)
    
    # Image files
    annotated_image_path = os.path.join(out_path, "annotated.jpg")
    
    # Find similar style images
    ARTIST_BASELINE_DIR = os.path.join(BASELINE_DIR, args.artist)
    ARTIST_GENERATED_DIR = os.path.join(GENERATED_DIR, args.artist)
    similar_images = load_similar_file(args.artist, args.top, ARTIST_BASELINE_DIR, ARTIST_GENERATED_DIR)
    
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
    content_img = loader.load(content_image_path)    
        
    # Object segmentator
    log("Segmenting image...")
    sam_annotator = SAMAnnotator(model=MODEL_TYPE, checkpoint=CHECKPOINT_PATH, device=DEVICE)
    masks, segmented_image = sam_annotator.annotate(image_rgb, box)
    
    # Style transfer using baseline style images
    log("Running style transfer using baseline style images...")
    style_images = load_similar_images(similar_images, content_img, loader, ARTIST_BASELINE_DIR, ARTIST_GENERATED_DIR)
    stylized_images = get_stylized_images(style_images, content_img)

    
    log("Creating combined images...")
    mask = masks[0]
    combined_images = combine_images(stylized_images, mask, image_rgb)
    
    combined_images_distance = []
    for i, (baseline_image_path, generated_image_path) in enumerate(combined_images):
        original_distance = similar_images[i][0]
        combined_distance = compute_similaty(content_image_path, baseline_image_path)
        combined_images_distance.append([original_distance, combined_distance, baseline_image_path, generated_image_path])
    
    # save combined_images_distance in a txt file
    combined_images_distance_txt = os.path.join(out_path, "combined_images_distance.txt")
    with open(combined_images_distance_txt, 'w') as f:
        f.write(f'Original distance, Combined distance, Baseline image path, Generated image path\n')
        for item in combined_images_distance:
            f.write(f'{item[0]}, {item[1]}, {item[2]}, {item[3]}\n')


# Runs:
# Pyramid of Giza
# python3 src/run.py --content data/images/content/Great_Pyramid_of_Giza/4.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png
# Necropolis
# python3 src/run.py --content data/images/content/Theban_Necropolis/4.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png --classes mountain
# Sphinx
# python3 src/run.py --content data/images/content/Sphinx_of_Memphis/0.jpg --style data/images/style/generated/Vincent\ van\ Gogh\ style.png --classes sphinx