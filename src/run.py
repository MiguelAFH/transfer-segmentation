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
from models import StyleTransfer, SAMAnnotator

from utils import log, ImageLoader
from yolo_world import YOLOAnnotator


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
torch.cuda.set_device(DEVICE)
MODEL_TYPE = "vit_h"
HOME = "./"
DATA_PATH = os.path.join(HOME, "data")
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
        nargs="+",
        default=["pyramid"],
        help="List of classes to detect in the image"
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

def get_run_name(content_path, style_path):
    content_name = os.path.basename(content_path)  # Get the base name
    content_file_name, _ = os.path.splitext(content_name)  # Split the name and extension
    style_name = os.path.basename(style_path)  # Get the base name
    style_file_name, _ = os.path.splitext(style_name)  # Split the name and extension
    return f"{content_file_name}_{style_file_name}"


def save_cv2_image(image_tensor, output_file):
    numpy_array = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file, opencv_image)
        
def print_mem():
    allocated_memory = torch.cuda.memory_allocated(DEVICE)
    allocated_memory_gib = allocated_memory / (1024 ** 3)  # Convert from bytes to GiB
    print(f"CUDA memory allocated: {allocated_memory_gib:.2f} GiB")


if __name__ == "__main__":
    # Setup
    args = parse_arguments()
    content_image_path = args.content
    style_image_path = args.style
    out_path = os.path.join("runs", get_run_name(content_image_path, style_image_path))
    os.makedirs(out_path, exist_ok=True)
    
    # Image files
    stylized_image_path = os.path.join(out_path, "stylized.jpg")
    masked_image_path = os.path.join(out_path, "masked.jpg")
    out_image_path = os.path.join(out_path, "out.jpg")
    
    # Segmentation images
    image_bgr = cv2.imread(content_image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Object detector
    print("Classes", args.classes)
    yolo_annotator = YOLOAnnotator(model_id='yolo_world/v2-l', classes=args.classes, device=DEVICE)
    box, _ = yolo_annotator.get_bounding_box(image_bgr)

    # Style images
    loader = ImageLoader(DEVICE, imsize=image_rgb.shape[0])
    style_img = loader.load(style_image_path)
    content_img = loader.load(content_image_path)    
    style_img = F.interpolate(style_img, size=content_img.shape[-2:])
    log("Style image size: ", style_img.shape)
    log("Content image size: ", content_img.shape)
    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

    # Object segmentator
    sam_annotator = SAMAnnotator(model=MODEL_TYPE, checkpoint=CHECKPOINT_PATH, device=DEVICE)
    masks, segmented_image = sam_annotator.annotate(image_rgb, box)
    style_transfer = StyleTransfer(device=DEVICE)
    stylized_image = style_transfer.run(content_img, style_img)
    
    # Save stylized images
    log("Saving stylized image to ", stylized_image_path)
    plt.figure()
    imshow(stylized_image, title='Stylized Image')
    plt.savefig(stylized_image_path)
    plt.clf()

    # Save masked image
    log("Saving masked image to ", masked_image_path)
    stylized = stylized_image.cpu().detach().numpy()[0]
    mask = masks[0]
    stylized = stylized[:,:mask.shape[0], :mask.shape[1]]
    stylized = stylized[:] * mask
    stylized = 255 * stylized.transpose(1, 2, 0)
    plt.imshow(stylized)
    plt.savefig(masked_image_path)
    
    # Create the combined image
    stylized = np.array(stylized, dtype=np.uint8)
    combined_image = np.zeros_like(stylized)
    combined_image[mask] = stylized[mask]
    combined_image[~mask] = image_rgb[~mask]

    # Convert the combined image back to BGR for saving with cv2
    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    log("Saving combined image to ", out_image_path)
    cv2.imwrite(out_image_path, combined_image_bgr)

