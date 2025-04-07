#!/usr/bin/env python3
import os
import json
import argparse
import multiprocessing
from datetime import datetime
from abc import ABC, abstractmethod

# Import required components
from components.prompt_generators import PromptGenerator, JsonFilePromptGenerator, ObjectBasedPromptGenerator
from components.image_generators import ImageGenerator, StableDiffusionImageGenerator, Tuned_SD15_LoRA_ImageGenerator, SDXLTurboImageGenerator, StableDiffusionXLImageGenerator, Tuned_StableDiffusionXLImageGenerator, StableDiffusion21ImageGenerator
from components.image_filters import ImageFilter, GPT4VImageFilter
from components.image_recaptioners import ImageRecaptioner, GPT4VImageRecaptioner, HybridImageRecaptioner, OriginalImageRecaptioner

class ImageGenerationPipeline:
    """
    Pipeline for prompt generation, image generation, filtering, and recaptioning.
    """
    def __init__(self, experiment_dir: str,
                 prompt_generator: PromptGenerator, 
                 image_generator: ImageGenerator,
                 image_filter: ImageFilter,
                 image_recaptioner: ImageRecaptioner,
                 num_images_per_prompt: int = 1,
                 args=None):
        self.experiment_dir = experiment_dir
        self.prompt_generator = prompt_generator
        self.image_generator = image_generator
        self.image_filter = image_filter
        self.image_recaptioner = image_recaptioner
        self.num_images_per_prompt = num_images_per_prompt
        self.args = args

        # Create necessary directories
        self.prompts_dir = os.path.join(self.experiment_dir, "prompts")
        self.images_dir = os.path.join(self.experiment_dir, "images")
        self.cache_dir = os.path.join(self.experiment_dir, "cache")
        self.train_dir = os.path.join(self.images_dir, "train")
        self.filtered_folder = os.path.join(self.images_dir, "filtered_out")

        for directory in [self.prompts_dir, self.images_dir, self.cache_dir, self.train_dir, self.filtered_folder]:
            os.makedirs(directory, exist_ok=True)

    def generate_prompts(self):
        """
        Generate prompts using the prompt generator and save to a JSON file.
        """
        prompts_raw = self.prompt_generator.generate()
        print(f"Generated {len(prompts_raw)} prompts.")
        # Format prompts as dictionaries with text and the desired number of images per prompt
        prompts = [{"text": prompt, "num_images_per_prompt": self.num_images_per_prompt} for prompt in prompts_raw]
        
        # Save prompts to the experiment folder and a cache folder
        prompts_file = os.path.join(self.prompts_dir, f"prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(prompts_file, "w") as f:
            json.dump(prompts, f, indent=4)
        unfiltered_file = os.path.join(self.cache_dir, f"unfiltered_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(unfiltered_file, "w") as f:
            json.dump(prompts, f, indent=4)
        return prompts

    def generate_images(self, prompts):
        """
        Generate images for each prompt.
        Returns a list of (image_path, prompt) tuples.
        """
        image_prompt_mapping = []
        # Define a fixed list of seeds (or replace with a dynamic generator)
        randomseed_list = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        # Process only the first N prompts if --gen is specified
        num_prompts = len(prompts)
        if self.args.gen > 0:
            num_prompts = min(num_prompts, self.args.gen)
        
        for i in range(num_prompts):
            prompt = prompts[i]
            for j in range(prompt['num_images_per_prompt']):
                image_name_prefix = f"{i}_{j}"
                # Cycle through the seed list
                seed = randomseed_list[j % len(randomseed_list)]
                self.image_generator.seeds = seed
                image_path = self.image_generator.generate(prompt, self.images_dir, image_name_prefix)
                image_prompt_mapping.append((image_path, prompt))
        return image_prompt_mapping

    @staticmethod
    def filter_worker(image_prompt_list, image_filter, filtered_folder):
        """
        Worker function to filter images.
        Returns a tuple: (kept_items, filtered_items)
        """
        kept = []
        filtered = []
        for image_path, prompt in image_prompt_list:
            if image_filter.filter(image_path, prompt):
                kept.append((image_path, prompt))
            else:
                # Move the filtered-out image to the filtered folder
                try:
                    os.rename(image_path, os.path.join(filtered_folder, os.path.basename(image_path)))
                except Exception as e:
                    print(f"Error moving file {image_path}: {e}")
                filtered.append({"file_name": os.path.basename(image_path), "text": prompt['text']})
                print(f"Filtered out {image_path}")
        return kept, filtered

    def filter_images(self, image_prompt_mapping):
        """
        Filter images using multiprocessing.
        Returns the list of image_prompt pairs that passed the filter and writes filtered data to a JSONL file.
        """
        num_workers = min(100, len(image_prompt_mapping)) if image_prompt_mapping else 1
        chunk_size = max(1, len(image_prompt_mapping) // num_workers)
        chunks = [image_prompt_mapping[i:i + chunk_size] for i in range(0, len(image_prompt_mapping), chunk_size)]

        pool = multiprocessing.Pool(processes=num_workers)
        results = [pool.apply_async(self.filter_worker, args=(chunk, self.image_filter, self.filtered_folder))
                   for chunk in chunks]
        pool.close()
        pool.join()

        kept_items = []
        filtered_data_all = []
        for r in results:
            kept, filtered = r.get()
            kept_items.extend(kept)
            filtered_data_all.extend(filtered)
        
        # Write all filtered data to a JSONL file
        filtered_prompts_file = os.path.join(self.filtered_folder, "filtered_prompts.jsonl")
        with open(filtered_prompts_file, "a") as f:
            for item in filtered_data_all:
                f.write(json.dumps(item) + "\n")
        return kept_items

    @staticmethod
    def recaption_worker(image_prompt_list, image_recaptioner):
        """
        Worker function for recaptioning images.
        Returns a list of metadata dictionaries.
        """
        metadata_list = []
        for image_path, prompt in image_prompt_list:
            new_caption = image_recaptioner.recaption(image_path, prompt)
            metadata = {
                "file_name": os.path.basename(image_path),
                "text": new_caption
            }
            print(f"Processed {image_path}: {new_caption}")
            metadata_list.append(metadata)
        return metadata_list

    def recaption_images(self, filtered_image_prompt_mapping):
        """
        Recaption images using multiprocessing and write metadata to a JSONL file.
        """
        metadata_file = os.path.join(self.train_dir, "metadata.jsonl")
        if self.args.no_processing:
            # If no processing is requested, use the original prompt text as the caption
            with open(metadata_file, "w") as f:
                for image_path, prompt in filtered_image_prompt_mapping:
                    metadata = {
                        "file_name": os.path.basename(image_path),
                        "text": prompt['text']
                    }
                    f.write(json.dumps(metadata) + "\n")
                    print(f"Processed {image_path} with original prompt")
            return

        num_workers = min(100, len(filtered_image_prompt_mapping)) if filtered_image_prompt_mapping else 1
        chunk_size = max(1, len(filtered_image_prompt_mapping) // num_workers)
        chunks = [filtered_image_prompt_mapping[i:i + chunk_size] for i in range(0, len(filtered_image_prompt_mapping), chunk_size)]
        
        pool = multiprocessing.Pool(processes=num_workers)
        results = [pool.apply_async(self.recaption_worker, args=(chunk, self.image_recaptioner))
                   for chunk in chunks]
        pool.close()
        pool.join()

        all_metadata = []
        for r in results:
            all_metadata.extend(r.get())

        with open(metadata_file, "a") as f:
            for metadata in all_metadata:
                f.write(json.dumps(metadata) + "\n")

    def check_metadata_consistency(self, image_prompt_mapping):
        """
        Check if all images in the train folder have corresponding metadata.
        Moves any mismatched images to the filtered folder and logs the discrepancy.
        """
        metadata_file = os.path.join(self.train_dir, "metadata.jsonl")
        metadata_files = set()
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        metadata_files.add(record['file_name'])
                    except json.JSONDecodeError:
                        continue

        image_files = os.listdir(self.train_dir)
        for image_file in image_files:
            if image_file not in metadata_files and image_file != "metadata.jsonl":
                print(f"Warning: {image_file} is not in the metadata. Moving to filtered folder.")
                src = os.path.join(self.train_dir, image_file)
                dst = os.path.join(self.filtered_folder, image_file)
                try:
                    os.rename(src, dst)
                except Exception as e:
                    print(f"Error moving file {src}: {e}")
                # Log the missing prompt if found in the original mapping
                for img_path, prompt in image_prompt_mapping:
                    if os.path.basename(img_path) == image_file:
                        filtered_prompts_file = os.path.join(self.filtered_folder, "filtered_prompts.jsonl")
                        with open(filtered_prompts_file, "a") as f:
                            f.write(json.dumps({"file_name": image_file, "text": prompt['text']}) + "\n")
                        break

    def run(self):
        """
        Execute the full pipeline.
        """
        # 1. Generate prompts
        prompts = self.generate_prompts()

        # 2. Generate images
        image_prompt_mapping = self.generate_images(prompts)
        if self.args.gen > 0:
            print("Image generation complete. Exiting as per --gen argument.")
            return

        # 3. Filter images
        filtered_mapping = self.filter_images(image_prompt_mapping)

        # 4. Recaption images and save metadata
        self.recaption_images(filtered_mapping)

        # 5. Check metadata consistency
        self.check_metadata_consistency(image_prompt_mapping)

        print("Pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the image generation pipeline.")
    parser.add_argument("--experiment_folder", type=str, default="experiments/experiment_v0",
                        help="Folder to save experiment results.")
    parser.add_argument("--prompts_filepath", type=str, default="prompt_data/image_prompts.json",
                        help="Path to the prompts file.")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for the GPT-4 Vision model.")
    parser.add_argument("--no_processing", action="store_true",
                        help="If set, the pipeline will not process the images (recaptioning step is skipped).")
    parser.add_argument("--image_generator", type=str, default="StableDiffusionImageGenerator",
                        help="The image generator to use.")
    parser.add_argument("--image_filter", type=str, default="GPT4VImageFilter",
                        help="The image filter to use.")
    parser.add_argument("--image_recaptioner", type=str, default="GPT4VImageRecaptioner",
                        help="The image recaptioner to use.")
    parser.add_argument("--prompt_generator", type=str, default="JsonFilePromptGenerator",
                        help="The prompt generator to use.")
    parser.add_argument("--lora", type=str, default="path/to/lora_weights.safetensors",
                        help="Path to the LoRA weights.")
    parser.add_argument("--gen", type=int, default=0,
                        help="If > 0, process only the first N prompts (dry-run mode for image generation).")
    args = parser.parse_args()

    # Instantiate prompt generator
    if args.prompt_generator == "JsonFilePromptGenerator":
        prompt_generator = JsonFilePromptGenerator(args.prompts_filepath)
    elif args.prompt_generator == "ObjectBasedPromptGenerator":
        prompt_generator = ObjectBasedPromptGenerator(args.experiment_folder, api_key=args.api_key)
    else:
        raise ValueError(f"Unsupported prompt generator: {args.prompt_generator}")

    # Instantiate image generator
    if args.image_generator == "StableDiffusionImageGenerator":
        image_generator = StableDiffusionImageGenerator()
    elif args.image_generator == "Tuned_SD15_LoRA_ImageGenerator":
        image_generator = Tuned_SD15_LoRA_ImageGenerator(lora=args.lora, device="cuda",
                                                          guidance_scale=5.0, num_inference_steps=30, seeds=42)
    elif args.image_generator == "SDXLTurboImageGenerator":
        image_generator = SDXLTurboImageGenerator()
    elif args.image_generator == "StableDiffusionXLImageGenerator":
        image_generator = StableDiffusionXLImageGenerator()
    elif args.image_generator == "Tuned_StableDiffusionXLImageGenerator":
        image_generator = Tuned_StableDiffusionXLImageGenerator(lora=args.lora, device="cuda",
                                                                guidance_scale=5.0, num_inference_steps=30, seeds=42)
    elif args.image_generator == "StableDiffusion21ImageGenerator":
        image_generator = StableDiffusion21ImageGenerator()
    else:
        raise ValueError(f"Unsupported image generator: {args.image_generator}")

    # Instantiate image filter
    if args.image_filter == "GPT4VImageFilter":
        image_filter = GPT4VImageFilter(api_key=args.api_key)
    else:
        raise ValueError(f"Unsupported image filter: {args.image_filter}")

    # Instantiate image recaptioner
    if args.image_recaptioner == "GPT4VImageRecaptioner":
        image_recaptioner = GPT4VImageRecaptioner(api_key=args.api_key)
    elif args.image_recaptioner == "HybridImageRecaptioner":
        image_recaptioner = HybridImageRecaptioner(api_key=args.api_key, gpt4v_ratio=0.5, retry=2)
    elif args.image_recaptioner == "OriginalRecaptioner":
        image_recaptioner = OriginalImageRecaptioner()
    else:
        raise ValueError(f"Unsupported image recaptioner: {args.image_recaptioner}")

    pipeline = ImageGenerationPipeline(
        experiment_dir=args.experiment_folder,
        prompt_generator=prompt_generator,
        image_generator=image_generator,
        image_filter=image_filter,
        image_recaptioner=image_recaptioner,
        num_images_per_prompt=7,
        args=args
    )

    pipeline.run()
