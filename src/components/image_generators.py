import os
import torch
from abc import ABC, abstractmethod
from diffusers import DiffusionPipeline, AutoPipelineForText2Image

def _prepare_seed_folder(output_dir: str) -> str:
    """
    Prepares the output subfolder ('train') within the given directory.
    
    Args:
        output_dir: The base directory where images are to be saved.
    
    Returns:
        The path to the 'train' subfolder.
    """
    folder = os.path.join(output_dir, "train")
    os.makedirs(folder, exist_ok=True)
    return folder


class ImageGenerator(ABC):
    """
    Abstract base class for image generation components.
    """
    @abstractmethod
    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        """
        Generates an image based on a prompt.
        
        Args:
            prompt: The prompt (as a dictionary) to use for image generation.
            output_dir: The directory where the generated image should be saved.
            image_name_prefix: The prefix to use for the image file name.
        
        Returns:
            The file path of the generated image.
        """
        pass


class StableDiffusionImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses Stable Diffusion v1.5.
    """
    def __init__(self, device: str = "cuda", guidance_scale: float = 5.0,
                 num_inference_steps: int = 20, seeds: int = 42) -> None:
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seeds = seeds

        print("Loading Stable Diffusion pipeline (v1.5)...")
        base_model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        ).to(device)
        # Optional: Disable safety checker if needed
        # self.pipe.safety_checker = None

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        seed_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(seed_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        generator = torch.Generator(self.device).manual_seed(self.seeds)
        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator
        ).images[0]
        image.save(image_path)
        return image_path


class StableDiffusion21ImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses Stable Diffusion 2.1.
    """
    def __init__(self, device: str = "cuda", guidance_scale: float = 5.0,
                 num_inference_steps: int = 30, seeds: int = 42) -> None:
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seeds = seeds

        print("Loading Stable Diffusion pipeline (2.1)...")
        base_model_id = "stabilityai/stable-diffusion-2-1"
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        ).to(device)
        # Optional: Disable safety checker if needed
        # self.pipe.safety_checker = None

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        seed_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(seed_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        generator = torch.Generator(self.device).manual_seed(self.seeds)
        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator
        ).images[0]
        image.save(image_path)
        return image_path


class Tuned_SD15_LoRA_ImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses Stable Diffusion v1.5 with LoRA fine-tuning.
    """
    def __init__(self, lora: str, device: str = "cuda", guidance_scale: float = 5.0,
                 num_inference_steps: int = 20, seeds: int = 42) -> None:
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seeds = seeds
        self.lora_path = lora

        print("Loading Stable Diffusion pipeline (v1.5) with LoRA...")
        base_model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.load_lora_weights(self.lora_path)
        # Optional: Disable safety checker if needed
        # self.pipe.safety_checker = None

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        seed_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(seed_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        generator = torch.Generator(self.device).manual_seed(self.seeds)
        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator
        ).images[0]
        image.save(image_path)
        return image_path


class SDXLTurboImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses SDXL-Turbo for fast image generation.
    """
    def __init__(self, device: str = "cuda", guidance_scale: float = 0.0,
                 num_inference_steps: int = 1) -> None:
        self.device = device
        self.guidance_scale = guidance_scale  # Typically 0.0 for SDXL-Turbo
        self.num_inference_steps = num_inference_steps  # Usually 1 step for real-time generation

        print("Loading SDXL-Turbo pipeline...")
        model_id = "stabilityai/sdxl-turbo"
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        output_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(output_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale
        ).images[0]
        image.save(image_path)
        return image_path


class StableDiffusionXLImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses Stable Diffusion XL (base version).
    """
    def __init__(self, device: str = "cuda", guidance_scale: float = 5.0,
                 num_inference_steps: int = 20, seeds: int = 42) -> None:
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seeds = seeds

        print("Loading Stable Diffusion XL pipeline (base version)...")
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        ).to(device)
        # Optional: Disable safety checker if needed
        # self.pipe.safety_checker = None

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        seed_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(seed_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        generator = torch.Generator(self.device).manual_seed(self.seeds)
        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator
        ).images[0]
        image.save(image_path)
        return image_path


class Tuned_StableDiffusionXLImageGenerator(ImageGenerator):
    """
    ImageGenerator that uses Stable Diffusion XL with optional LoRA fine-tuning.
    """
    def __init__(self, device: str = "cuda", lora: str = None, guidance_scale: float = 5.0,
                 num_inference_steps: int = 20, seeds: int = 42) -> None:
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.seeds = seeds

        print("Loading Stable Diffusion XL pipeline (base version) with LoRA if provided...")
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16
        ).to(device)
        if lora is not None:
            self.pipe.load_lora_weights(lora)
        # Optional: Disable safety checker if needed
        # self.pipe.safety_checker = None

    def generate(self, prompt: dict, output_dir: str, image_name_prefix: str) -> str:
        prompt_text = prompt["text"]
        seed_folder = _prepare_seed_folder(output_dir)
        image_path = os.path.join(seed_folder, f"{image_name_prefix}.png")

        if os.path.exists(image_path):
            print(f"Image already exists: {image_path}")
            return image_path

        generator = torch.Generator(self.device).manual_seed(self.seeds)
        image = self.pipe(
            prompt=prompt_text,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator
        ).images[0]
        image.save(image_path)
        return image_path
