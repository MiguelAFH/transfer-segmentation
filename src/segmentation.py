from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gc


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks, save_path=None):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  if save_path:
    plt.savefig(save_path)
  del mask
  gc.collect()
  


if __name__ == "__main__":
    generator = pipeline("mask-generation", model="facebook/sam-vit-large", device=0)
    # Local Image
    image = Image.open("./data/segmentation/dancing.jpg").convert("RGB")

    print("Running image segmentation on the image...")
    outputs = generator(image, points_per_batch=256)

    masks = outputs["masks"]
    out = "./data/segmentation/out.png"
    show_masks_on_image(image, masks, "./data/segmentation/out.png")
    print(f"Successfully saved segmented image under: {out}")
