import json
import os
import time
import openai
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import List, Dict


class PromptGenerator(ABC):
    """Abstract base class for prompt generation components."""

    @abstractmethod
    def generate(self) -> List[Dict[str, str]]:
        """
        Generates a list of prompts.

        Returns:
            A list of prompt dictionaries, each with a "text" key.
        """
        pass


class JsonFilePromptGenerator(PromptGenerator):
    """
    PromptGenerator that reads prompts from a JSON or JSONL file.
    """

    def __init__(self, prompts_filepath: str) -> None:
        """
        Initializes the JsonFilePromptGenerator.

        Args:
            prompts_filepath: The path to the JSON/JSONL file containing the prompts.
        """
        if not os.path.exists(prompts_filepath):
            raise FileNotFoundError(f"Prompts file not found: {prompts_filepath}")
        self.prompts_filepath = prompts_filepath

    def generate(self) -> List[Dict[str, str]]:
        """
        Generates a list of prompts by reading them from the file.
        Always returns a list of dictionaries with a "text" key.

        Returns:
            A list of prompt dictionaries.
        """
        if self.prompts_filepath.endswith('.json'):
            with open(self.prompts_filepath, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                # If the file contains a list of strings, convert them.
                if all(isinstance(item, str) for item in data):
                    return [{"text": prompt} for prompt in data]
                # Otherwise, assume a list of dictionaries with a "text" key.
                elif all(isinstance(item, dict) and "text" in item for item in data):
                    return data
            raise ValueError("Invalid JSON format: expected a list of strings or dictionaries with a 'text' key.")
        elif self.prompts_filepath.endswith('.jsonl'):
            prompts = []
            with open(self.prompts_filepath, "r") as f:
                for line in f:
                    record = json.loads(line)
                    # Ensure each record is returned as a dict with a "text" key.
                    if "text" in record:
                        prompts.append({"text": record["text"]})
                    else:
                        prompts.append({"text": str(record)})
            return prompts
        else:
            raise ValueError("Unsupported file format for prompts.")


class ObjectBasedPromptGenerator(PromptGenerator):
    """
    PromptGenerator that first generates object nouns and then uses them to create image prompts.
    """

    def __init__(self, experiment_dir: str, api_key: str) -> None:
        """
        Initializes the ObjectBasedPromptGenerator.

        Args:
            experiment_dir: Directory for storing object nouns and prompts.
            api_key: Your OpenAI API key.
        """
        self.experiment_dir = experiment_dir
        self.api_key = api_key
        self.num_objects = 200
        self.objects_filepath = os.path.join(self.experiment_dir, "object_nouns.json")
        self.prompts_filepath = os.path.join(self.experiment_dir, "image_prompts.json")
        os.makedirs(self.experiment_dir, exist_ok=True)

    def generate_objects(self) -> None:
        """
        Generates a list of object nouns using the GPT-4 API and saves them to a JSON file.
        """
        openai.api_key = self.api_key
        batch_size = 100
        total_batches = (self.num_objects + batch_size - 1) // batch_size
        object_nouns = []

        print(f"Generating {self.num_objects} object nouns using GPT-4...")
        for _ in tqdm(range(total_batches)):
            num_to_generate = min(batch_size, self.num_objects - len(object_nouns))
            prompt = f"List {num_to_generate} common object nouns, separated by commas."
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    max_tokens=1000,
                    n=1,
                    temperature=0.2,
                )
                text = response['choices'][0]['message']['content']
                nouns = [noun.strip() for noun in text.strip().split(',') if noun.strip()]
                object_nouns.extend(nouns)
            except Exception as e:
                print(f"Error generating object nouns: {e}")
                continue
            time.sleep(0.1)  # Respect API rate limits

        object_nouns = object_nouns[:self.num_objects]
        with open(self.objects_filepath, 'w') as f:
            json.dump({'objects': object_nouns}, f, indent=2)
        print(f"Generated {len(object_nouns)} object nouns. Saved to {self.objects_filepath}")

    def generate_prompts_from_objects(self) -> List[Dict[str, str]]:
        """
        Generates image prompts from the list of object nouns using GPT-4.
        
        Returns:
            A list of prompt dictionaries with a "text" key.
        """
        openai.api_key = self.api_key
        with open(self.objects_filepath, 'r') as f:
            data = json.load(f)
            object_nouns = data.get('objects', [])

        prompts = []
        print("Generating image prompts...")
        for obj in tqdm(object_nouns):
            system_prompt = "You are an assistant that generates image prompts for image generation."
            user_prompt = (
                f"Create an image generation prompt describing an empty state of the object '{obj}'. "
                "For example, if the object is 'table', the prompt could be 'An empty table.'. "
                "If the object doesn't have an empty state, return 'NULL'."
            )
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    max_tokens=60,
                    n=1,
                    temperature=0.0,
                )
                generated_prompt = response['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Error generating prompt for '{obj}': {e}")
                generated_prompt = f"An empty {obj}"  # Fallback prompt

            prompts.append({"text": generated_prompt})
            print(f"Object: {obj} | Prompt: {generated_prompt}")

        with open(self.prompts_filepath, 'w') as f:
            json.dump({'prompts': prompts}, f, indent=2)
        print(f"Generated {len(prompts)} image prompts. Saved to {self.prompts_filepath}")
        return prompts

    def generate(self) -> List[Dict[str, str]]:
        """
        Generates prompts using a two-step object-based method.

        Returns:
            A list of prompt dictionaries with a "text" key.
        """
        if not os.path.exists(self.objects_filepath):
            self.generate_objects()
        prompts = self.generate_prompts_from_objects()
        return prompts
