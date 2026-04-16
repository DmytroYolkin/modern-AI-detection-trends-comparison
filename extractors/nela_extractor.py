# ===== INSTALL FIRST =====
# pip install pandas numpy nela_features

import pandas as pd
import nltk
from nela_features.nela_features import NELAFeatureExtractor


def ensure_nltk_tokenizers() -> None:
	"""Download required NLTK tokenizers if they are not available."""
	required = [
		("tokenizers/punkt", "punkt"),
		("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng/", "averaged_perceptron_tagger_eng"),
        ("chunkers/maxent_ne_chunker_tab/english_ace_multiclass/", "maxent_ne_chunker_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/words", "words"),
	]
	for resource_path, package_name in required:
		try:
			nltk.data.find(resource_path)
		except LookupError:
			nltk.download(package_name, quiet=True)
if __name__ == "__main__":
    with open("text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # ===== SAMPLE INPUT =====
    # ===== INITIALIZE EXTRACTOR =====
    ensure_nltk_tokenizers()
    extractor = NELAFeatureExtractor()

    # ===== EXTRACT FEATURES =====
    features, names = extractor.extract_all(text)

    # ===== CONVERT TO TABLE (same as repo logic) =====
    df = pd.DataFrame([features], columns=names)

    # ===== OUTPUT =====
    # Save results to JSON (sorted for human readability, append mode, no sort_keys)
    output_path = "nela_features_output.json"
    df.to_json(output_path, orient="records", lines=True, indent=4)
    print(f"Results appended to {output_path}")