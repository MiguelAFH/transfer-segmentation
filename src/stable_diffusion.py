import argparse
import os
import torch

from diffusers import DiffusionPipeline


MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
OUT_DIR = "./data/images/style/generated/"

def parse_arguments():
    """
    Generate command-line arguments for stable diffusion.
    """
    parser = argparse.ArgumentParser(description="Generate images with stable diffusion.")
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--prompts", 
        type=str, 
        help="The path to the prompts file"
    )

    group.add_argument(
        "--prompt", 
        type=str, 
        help="A single prompt for image generation"
    )
    return parser.parse_args()

def get_base():
    """
    Load the base model.
    """
    base = DiffusionPipeline.from_pretrained(
        MODEL_BASE, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    base.to("cuda")
    return base

def get_refiner():
    """
    Load the refiner model.
    """
    refiner = DiffusionPipeline.from_pretrained(
        MODEL_REFINER,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")
    return refiner


def read_prompt(prompt_path):
    """
    Read the prompt from a file.
    """
    # Read the prompts from a file. Each prompt is separated by new line
    
    with open(prompt_path, "r") as f:
        prompts = f.readlines()
    return prompts

def log(*msg):
    print("#"*50)
    print(*msg)

if __name__ == "__main__":
    args = parse_arguments()
    base = get_base()
    refiner = get_refiner()
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = read_prompt(args.prompts)
    
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 80
    high_noise_frac = 0.8

    for prompt in prompts:
        prompt = prompt.strip()
        log("Generating image for prompt:", prompt)
        image = base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images
        image[0].save(os.path.join(OUT_DIR, f"{prompt}.png"))
