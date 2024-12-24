# DBLP-ACM Feature Extraction

This project implements a feature extraction system for academic paper matching between DBLP and ACM databases.

## Features

- Extracts semantic features from paper titles and venues
- Uses LLM for intelligent feature extraction
- Caches results in MySQL database
- Supports batch processing of multiple datasets

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the values in `.env` with your configurations

3. Prepare your datasets:
- Place your dataset files in the project root
- Supported files: train.txt, test.txt, valid.txt

## Usage

Run the feature extraction:
```bash
python feature_extraction_DA.py
```

## Output

- Features are stored in the MySQL database
- Logs are generated in `feature_extraction_YYYYMMDD.log` 