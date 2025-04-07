import os
import json
import base64
import time
import openai
from abc import ABC, abstractmethod
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
from typing import Any


class ImageFilter(ABC):
    """Abstract base class for image filtering components."""

    @abstractmethod
    def filter(self, image_path: str, prompt: dict) -> bool:
        """
        Determines whether an image passes the filter based on a given prompt.

        Args:
            image_path: The file path to the image.
            prompt: A dictionary containing the prompt (expects a "text" key).

        Returns:
            True if the image passes the filter; False otherwise.
        """
        pass


class GPT4VImageFilter(ImageFilter):
    """
    Image filter that uses the GPT‑4V API to decide if an image accurately represents 
    an "empty state" as described in the prompt.
    """

    def __init__(self, api_key: str, retry: int = 10):
        """
        Initializes the GPT4VImageFilter.

        Args:
            api_key: Your OpenAI API key.
            retry: Number of retry attempts for the API call.
        """
        self.api_key = api_key
        self.retry = retry

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file in base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def create_filter_prompt(self, original_prompt: str) -> str:
        """
        Creates a structured prompt for filtering based on the original prompt text.
        """
        return (
            f"You are an assistant that evaluates whether an image correctly represents the 'empty state' "
            f"of an object as described in the caption. The caption is: {original_prompt}. Specifically, check if "
            "the main object appears empty or unoccupied and confirm that the described absent object is not present in the image. "
            "Does the image accurately reflect both conditions? Return 'yes' or 'no'."
        )

    def query_gpt4v(self, image_path: str, prompt: str) -> str:
        """
        Queries the GPT‑4V API using the provided image and prompt.

        Args:
            image_path: The path to the image.
            prompt: The prompt text.

        Returns:
            The stripped response from the API, or an error message if all retries fail.
        """
        openai.api_key = self.api_key
        base64_image = self.encode_image(image_path)
        messages = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "low"}}
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
                print(f"Attempt {attempt}: {result}")
                return result
            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                time.sleep(1)
        return "Failed: Query GPT-4V Error"

    def filter(self, image_path: str, prompt: dict) -> bool:
        """
        Filters an image by comparing the GPT‑4V API response to the expected condition.

        Args:
            image_path: The path to the image.
            prompt: A dictionary with at least the "text" key containing the prompt.

        Returns:
            True if the image passes the filter (i.e. GPT‑4V returns 'yes'); False otherwise.
        """
        structured_prompt = self.create_filter_prompt(prompt["text"])
        gpt_response = self.query_gpt4v(image_path, structured_prompt)

        if gpt_response.startswith("Failed"):
            print(f"Error querying GPT for image {image_path}: {gpt_response}")
            return False

        return "yes" in gpt_response.lower()


class OpenSourceVLMImageFilter(ImageFilter):
    """
    Image filter that leverages an open-source Vision Language Model (VLM) to verify 
    if an image matches a given prompt.
    """

    def __init__(self, model_name: str, device: str = "cuda", retry: int = 2):
        """
        Initializes the OpenSourceVLMImageFilter.

        Args:
            model_name: Name of the Hugging Face model (e.g., "Qwen/Qwen-VL-Chat").
            device: Device to run the model on ("cuda" or "cpu").
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
            trust_remote_code=True
        ).eval()

    def create_filter_prompt(self, original_prompt: str) -> str:
        """
        Creates a structured prompt for the VLM based on the original prompt text.
        """
        return (
            f"The image corresponds to the prompt: '{original_prompt}'. Verify if the image accurately matches the description. "
            "If it matches, respond with 'yes'. If it doesn't match, respond with 'no'."
        )

    def query_vlm(self, image_path: str, prompt: str) -> str:
        """
        Queries the open-source VLM with the provided image and prompt.

        Args:
            image_path: The path to the image.
            prompt: The prompt text.

        Returns:
            The VLM's response or an error message if the query fails.
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
                    generate_ids = self.model.generate(**inputs, max_new_tokens=10)
                    response = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

                    # Post-process the response based on the model type
                    lower_model_name = self.model_name.lower()
                    if "mistral" in lower_model_name:
                        response = response.split("[/INST]")[1].strip()
                    elif "vicuna" in lower_model_name or "llava" in lower_model_name:
                        response = response.split("ASSISTANT:")[1].strip()
                    elif "34b" in lower_model_name or "72b" in lower_model_name:
                        response = response.split("<|im_start|>assistant\n")[1].strip()
                    elif "llama3" in lower_model_name or "next" in lower_model_name:
                        response = response.split("assistant\n\n")[1].strip()
                    elif "qwen" in lower_model_name:
                        response = response.split("<|im_end|>")[1].strip()
                        if " " in response:
                            response = response.split(" ", 1)[1].strip()
                    # Default: assume the model returns only the assistant's answer

                    print(f"Attempt {attempt}: {response}")
                    return response
            except Exception as e:
                print(f"Error on attempt {attempt}: {e}")
                if attempt < self.retry:
                    time.sleep(1)
        return "Failed: Query VLM Error"

    def filter(self, image_path: str, prompt: dict) -> bool:
        """
        Filters an image using the open-source VLM.

        Args:
            image_path: The path to the image.
            prompt: A dictionary with at least the "text" key containing the prompt.

        Returns:
            True if the VLM returns a response containing 'yes'; False otherwise.
        """
        structured_prompt = self.create_filter_prompt(prompt["text"])
        vlm_response = self.query_vlm(image_path, structured_prompt)

        if vlm_response.startswith("Failed"):
            print(f"Error querying VLM for image {image_path}: {vlm_response}")
            return False

        return "yes" in vlm_response.lower()
