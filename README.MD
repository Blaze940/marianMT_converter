# MarianMT Converter

A tool for converting MarianMT models to TFLite format for efficient machine translation.

## Description

This project provides utilities to:
- Convert MarianMT models to TFLite format
- Handle tokenization and translation
- Support multiple language pairs
- Optimize for mobile and edge devices

## Requirements

- Python 3.10.12
- Virtual environment (recommended)

## Step-by-Step Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies 
```bash
pip install -r requirements.txt
```

3. Run the converter 
```bash
python marianMT_converter.py
```

4. Run the test with tokenizer
```bash
python test_translation.py
```

