# Synthetic Vision Dataset Engine

A lightweight data generation engine for creating synthetic computer vision datasets, compatible with 🤗 Datasets. This project explores scalable methods to produce diverse, high-quality training data for machine learning models.

## Key Features
*   Generates synthetic images with programmatically controlled attributes and annotations.
*   Seamlessly integrates with the Hugging Face `datasets` library for easy loading and sharing.
*   Designed for extensibility, allowing custom object types and scene compositions.
*   Outputs standard vision dataset formats (e.g., COCO-style annotations).

## Tech Stack
Python, Hugging Face Datasets, OpenCV, NumPy

## Getting Started
```bash
git clone https://github.com/zoreanuj/synthetic-vision-dataset-engine.git
cd synthetic-vision-dataset-engine
pip install -r requirements.txt
python generate_dataset.py --config configs/basic.yaml
```