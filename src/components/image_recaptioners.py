import os
import json
import base64
import time
import openai
import random
from abc import ABC, abstractmethod
from typing import Dict
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image


class ImageRecaptioner(ABC):
    """
    Abstract base class for image recaptioning components.
    """

    @abstractmethod
    def recaption(self, image_path: str, prompt: Dict) -> str:
        """
        Generates a new caption for an image.

        Args:
            image_path: The path to the image.
            prompt: The original prompt associated with the image (expects a "text" key).

        Returns:
            The new caption as a string.
        """
        pass


class GPT4VImageRecaptioner(ImageRecaptioner):
    """
    ImageRecaptioner that uses the GPT-4V API to generate a new caption for an image.
    """

    def __init__(self, api_key: str, retry: int = 2) -> None:
        """
        Initializes the GPT4VImageRecaptioner.

        Args:
            api_key: Your OpenAI API key.
            retry: Number of retry attempts for the API call.
        """
        self.api_key = api_key
        self.retry = retry

    def encode_image(self, image_path: str) -> str:
        """
        Encodes an image to base64.

        Args:
            image_path: The path to the image file.

        Returns:
            The base64-encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_recaption_prompt(self, original_prompt: str) -> str:
        """
        Creates a structured prompt for recaptioning.

        Args:
            original_prompt: The original prompt text.

        Returns:
            A refined recaption prompt.
        """
        return (
            f"The original prompt for the image is: '{original_prompt}'. "
            "Please refine the prompt by specifying an absent object if it is not already mentioned, "
            "but avoid redundant descriptions of emptiness. Ensure the refined prompt naturally integrates "
            "the missing object without repeating words like 'empty' or 'vacant'. For example: "
            "'An empty table.' → 'A table without any bottles on it.', "
            "'A deserted park.' → 'A park without any people.' If the original prompt is already sufficiently detailed, "
            "return it as is."
        )

    def query_gpt4v(self, image_path: str, prompt: str) -> str:
        """
        Queries the GPT-4V API with an image and prompt.

        Args:
            image_path: The path to the image.
            prompt: The structured prompt text.

        Returns:
            The API response stripped of whitespace or an error message on failure.
        """
        openai.api_key = self.api_key
        base64_image = self.encode_image(image_path)
        messages = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"},
            },
        ]

        for attempt in range(1, self.retry + 1):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": messages}],
                    max_tokens=1024,
                    n=1,
                    temperature=0,
                )
                result = response.choices[0].message.content.strip()
                return result
            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                time.sleep(1)
        return "Failed: Query GPT-4V Error"

    def recaption(self, image_path: str, prompt: Dict) -> str:
        """
        Generates a new caption for an image using the GPT-4V API.

        Args:
            image_path: The path to the image.
            prompt: The original prompt dictionary (expects "text").

        Returns:
            The new caption if successful; otherwise, returns the original prompt.
        """
        structured_prompt = self.create_recaption_prompt(prompt["text"])
        gpt_response = self.query_gpt4v(image_path, structured_prompt)

        if gpt_response.startswith("Failed"):
            print(f"Error querying GPT-4V for image {image_path}: {gpt_response}")
            return prompt["text"]

        return gpt_response


class HybridImageRecaptioner(ImageRecaptioner):
    """
    ImageRecaptioner that uses a hybrid approach:
    - A portion (gpt4v_ratio) of images are processed with GPT-4V.
    - The remaining images use a text-only GPT-4 query focusing on object description.
    """

    def __init__(self, api_key: str, gpt4v_ratio: float = 0.5, retry: int = 2) -> None:
        """
        Initializes the HybridImageRecaptioner.

        Args:
            api_key: Your OpenAI API key.
            gpt4v_ratio: Proportion of images to process with GPT-4V (between 0.0 and 1.0).
            retry: Number of retry attempts for API calls.
        """
        if not 0.0 <= gpt4v_ratio <= 1.0:
            raise ValueError("gpt4v_ratio must be between 0.0 and 1.0")
        self.api_key = api_key
        self.gpt4v_ratio = gpt4v_ratio
        self.retry = retry
        self.gpt4v_recaptioner = GPT4VImageRecaptioner(api_key, retry)

    def create_gpt4_prompt(self, original_prompt: str, image_description: str) -> str:
        """
        Creates a prompt for GPT-4 (text-only) recaptioning.

        Args:
            original_prompt: The original prompt text.
            image_description: The image description provided by GPT-4V.

        Returns:
            A concise prompt for GPT-4 recaptioning.
        """
        return (
            f"Given the image description: '{image_description}', "
            "create a concise prompt that describes only the main object without mentioning its state or any absent objects."
        )

    def query_gpt4(self, prompt: str) -> str:
        """
        Queries GPT-4 (text-only) with a given prompt.

        Args:
            prompt: The text prompt for GPT-4.

        Returns:
            The GPT-4 response stripped of whitespace, or an error message on failure.
        """
        openai.api_key = self.api_key
        for attempt in range(1, self.retry + 1):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    n=1,
                    temperature=0,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error on GPT-4 attempt {attempt}: {e}")
                time.sleep(1)
        return "Failed: Query GPT-4 Error"

    def recaption(self, image_path: str, prompt: Dict) -> str:
        """
        Generates a new caption using a hybrid approach.

        Args:
            image_path: The path to the image.
            prompt: The original prompt dictionary (expects "text").

        Returns:
            The new caption generated by either GPT-4V or GPT-4.
        """
        if random.random() < self.gpt4v_ratio:
            # Use GPT-4V recaptioning.
            return self.gpt4v_recaptioner.recaption(image_path, prompt)
        else:
            # Use GPT-4 (text-only) recaptioning.
            image_description = self.gpt4v_recaptioner.query_gpt4v(
                image_path, self.gpt4v_recaptioner.create_recaption_prompt(prompt["text"])
            )
            if image_description.startswith("Failed"):
                print(f"Failed to describe {image_path}, using original prompt.")
                return prompt["text"]
            structured_prompt = self.create_gpt4_prompt(prompt["text"], image_description)
            gpt4_response = self.query_gpt4(structured_prompt)
            if gpt4_response.startswith("Failed"):
                print(f"Error querying GPT-4 for image {image_path}: {gpt4_response}")
                return prompt["text"]
            return gpt4_response


class OpenSourceVLMRecaptioner(ImageRecaptioner):
    """
    ImageRecaptioner that uses an open-source Vision Language Model (VLM) to generate captions.
    """

    def __init__(self, model_name: str, device: str = "cuda", retry: int = 2) -> None:
        """
        Initializes the OpenSourceVLMRecaptioner.

        Args:
            model_name: The name of the Hugging Face model to use (e.g., "Qwen/Qwen-VL-Chat").
            device: The device to run the model on ("cuda" or "cpu").
            retry: Number of retry attempts.
        """
        self.model_name = model_name
        self.device = device
        self.retry = retry
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
            trust_remote_code=True,
        ).eval()

    def create_recaption_prompt(self, original_prompt: str) -> str:
        """
        Creates the recaptioning prompt for the VLM.

        Args:
            original_prompt: The original prompt text.

        Returns:
            A structured prompt for recaptioning.
        """
        return (
            f"The original prompt for the image is: '{original_prompt}'. "
            "Please provide a refined prompt that adds a detail about an absent object if it's not already present. "
            "For example: 'A deserted park.' → 'A deserted park without any people.' "
            "If the original prompt is already sufficiently detailed, return it as is."
        )

    def query_vlm(self, image_path: str, prompt: str) -> str:
        """
        Queries the open-source VLM with an image and prompt.

        Args:
            image_path: The path to the image.
            prompt: The structured prompt text.

        Returns:
            The VLM response or an error message on failure.
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return "Failed: Unable to open image"

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            formatted_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt").to(self.device)
        except Exception as e:
            print(f"Error processing input for image {image_path}: {e}")
            return "Failed: Input processing error"

        for attempt in range(1, self.retry + 1):
            try:
                with torch.no_grad():
                    generate_ids = self.model.generate(**inputs, max_new_tokens=100)
                    response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

                    lower_model = self.model_name.lower()
                    if "mistral" in lower_model:
                        response = response.split("[/INST]")[1].strip()
                    elif "vicuna" in lower_model or "llava" in lower_model:
                        response = response.split("ASSISTANT:")[1].strip()
                    elif "34b" in lower_model or "72b" in lower_model:
                        response = response.split("<|im_start|>assistant\n")[1].strip()
                    elif "llama3" in lower_model or "next" in lower_model:
                        response = response.split("assistant\n\n")[1].strip()
                    elif "qwen" in lower_model:
                        response = response.split("<|im_end|>")[1].strip()
                        if " " in response:
                            response = response.split(" ", 1)[1].strip()
                    return response
            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                if attempt < self.retry:
                    time.sleep(1)
        return "Failed: Query VLM Error"

    def recaption(self, image_path: str, prompt: Dict) -> str:
        """
        Generates a new caption using the open-source VLM.

        Args:
            image_path: The path to the image.
            prompt: The original prompt dictionary (expects "text").

        Returns:
            The new caption if successful; otherwise, the original prompt.
        """
        structured_prompt = self.create_recaption_prompt(prompt["text"])
        vlm_response = self.query_vlm(image_path, structured_prompt)
        if vlm_response.startswith("Failed"):
            print(f"Error querying VLM for image {image_path}: {vlm_response}")
            return prompt["text"]
        return vlm_response


class OriginalRecaptioner(ImageRecaptioner):
    """
    ImageRecaptioner that simply returns the original prompt.
    """

    def recaption(self, image_path: str, prompt: Dict) -> str:
        """
        Returns the original prompt for an image.

        Args:
            image_path: The path to the image.
            prompt: The original prompt dictionary (expects "text").

        Returns:
            The original prompt text.
        """
        return prompt["text"]
