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
from jupyter_bbox_widget import BBoxWidget


def encode_image(filepath):
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        encoded = str(base64.b64encode(image_bytes), 'utf-8')
        return "data:image/jpg;base64,"+encoded
    

def save_masks_as_png(masks):
    for i, mask in enumerate(masks):
        mask_filename = os.path.join('data', f'mask_{i}.jpg')
        # Ensure mask is in 0-255 range and uint8 type for saving
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_filename, mask_uint8)
        print(f'Saved mask {i} as {mask_filename}')

if __name__ == "__main__":


    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(DEVICE)
    MODEL_TYPE = "vit_h"
    HOME = "./"
    DATA_PATH = os.path.join(HOME, "data")
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    CONTENT_IMAGE_PATH = os.path.join(DATA_PATH, "29.jpg")
    STYLE_IMAGE_PATH = os.path.join(DATA_PATH, "van_gogh_diffusion.jpg")
    MASKED_IMAGE_PATH = os.path.join(DATA_PATH, "mask.jpg")
    OUTPUT_IMAGE_PATH = os.path.join(DATA_PATH, "out2.jpg")

    # imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
    imsize = 224
    default_box = {'x': 1, 'y': 64, 'width': 219, 'height': 137, 'label': ''}
    
    loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    style_img = image_loader(loader, STYLE_IMAGE_PATH, DEVICE)
    content_img = image_loader(loader, CONTENT_IMAGE_PATH, DEVICE)
    print("Style image size: ", style_img.shape)
    print("Content image size: ", content_img.shape)

    # Resize style image to match content image
    style_img = F.interpolate(style_img, size=content_img.shape[-2:])

    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(CONTENT_IMAGE_PATH)
    # print(image_bgr.dtype)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # image_rgb = content_img.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
    # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # print(image_bgr.dtype)
    sam_result = mask_generator.generate(image_rgb)
    #sam_result keys: dict_keys(['segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'])


    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    detections = sv.Detections.from_sam(sam_result=sam_result)
    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image_bgr, annotated_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    masks = [
        mask['segmentation']
        for mask
        in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]

    # sv.plot_images_grid(
    #     images=masks,
    #     grid_size=(11, int(len(masks) / 8)),
    #     size=(16, 16)
    # )

    mask_predictor = SamPredictor(sam)


    widget = BBoxWidget()
    widget.image = encode_image(CONTENT_IMAGE_PATH)
    widget

    widget.bboxes

    

    box = widget.bboxes[0] if widget.bboxes else default_box
    box = np.array([
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ])

    # image_bgr = cv2.imread(CONTENT_IMAGE_PATH)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    print("Saving masks")
    save_masks_as_png(masks)

    box_annotator = sv.BoxAnnotator(color=sv.Color.RED)
    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]

    source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    sv.plot_images_grid(
        images=[source_image, segmented_image],
        grid_size=(1, 2),
        titles=['source image', 'segmented image']
    )

    sv.plot_images_grid(
        images=masks,
        grid_size=(1, 4),
        size=(16, 4)
    )

    assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

    unloader = transforms.ToPILImage()  # reconvert into PIL image

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

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

    plt.figure()
    imshow(output, title='Output Image')
    plt.savefig(OUTPUT_IMAGE_PATH)
    plt.clf()

    out = output.cpu().detach().numpy()[0]
    print("Stylised image shape: ", out.shape)

    mask = masks[0]
    out = out[:,:mask.shape[0], :mask.shape[1]]
    print("Mask shape: ", mask.shape)
    out = out[:] * mask
    plt.imshow(out.transpose(1, 2, 0))
    plt.savefig(MASKED_IMAGE_PATH)

