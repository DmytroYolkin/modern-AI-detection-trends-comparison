# Modern AI Detection Trends Comparison

This repository provides a comprehensive comparison of modern AI-driven approaches for detecting and analyzing trends. It includes multiple feature extractors and fusion strategies to combine their outputs for classification tasks.

## Installation

To get started, install the required Python packages individually:

```bash
pip install pandas numpy nltk nela_features
pip install sentence-transformers scikit-learn python-Levenshtein
pip install torch transformers
pip install ollama
```

Additionally, pull the required models for the `ollama` framework:

```bash
ollama pull qwen2
ollama pull llama3
ollama pull mistral
ollama pull gemma
ollama pull phi3
```

## Project Structure

- `extractors/`
  - Contains feature extraction modules such as `NELA`, `StyleDecipher`, and `TRACE`.
- `fusion/`
  - Implements various fusion strategies to combine features from multiple extractors.
- `README.md`
  - Project documentation.

## Usage

1. Run the feature extraction showcase:

```bash
python extractors/features_representation.py
```

2. Test the fusion strategies:

```bash
python fusion/combination_all.py
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.