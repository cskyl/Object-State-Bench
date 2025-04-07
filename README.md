# Object State Representation Pipeline

This repository provides a baseline pipeline to test and improve physical object state representation in text-to-image generative systems. The current implementation demonstrates how to generate images from a text prompt while addressing issues like generating objects with incorrect physical states (for example, producing a "kitchen counter without any food" that instead shows a counter full of food).

---

## Overview

Text-to-image generative models often struggle with accurately depicting the intended physical state of objects. This pipeline offers a starting point for addressing these challenges by:

- Allowing you to specify prompts, random seeds, inference steps, and guidance scales.
- Supporting model selection (e.g., switching between Stable Diffusion 1.5 and SDXL).

This repository is intended to evolve. Future components will include synthetic data generation, fine-tuning scripts, and evaluation modules.

---

## Installation

Simply install the provided environment using the supplied file.

```bash
conda env create -f environment.yml
conda activate object-state-pipeline
