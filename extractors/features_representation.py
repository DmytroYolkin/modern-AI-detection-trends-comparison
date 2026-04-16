import json
import numpy as np

# Import the extractors
from nela_extractor import ensure_nltk_tokenizers, NELAFeatureExtractor
from styledecipher_extractor import styledecipher_features
from trace_user_profiler import TRACEUserProfileEmbedder

def main():
    print("=== Extractors Showcase ===\n")

    # 1. Sample text for representation
    sample_text = (
        "Artificial intelligence is rapidly transforming various industries across the globe. "
        "From healthcare to finance, machine learning models are becoming increasingly sophisticated. "
        "However, this rapid advancement also brings significant ethical and societal challenges that "
        "must be carefully addressed by researchers and policymakers."
    )

    print("Sample Text:")
    print(f"\"{sample_text}\"\n")

    # 2. NELA Features
    print("--- 1. NELA Feature Extractor ---")
    try:
        ensure_nltk_tokenizers()
        nela_extractor = NELAFeatureExtractor()
        nela_features, nela_names = nela_extractor.extract_all(sample_text)
        
        # Displaying a subset of features for readability
        print(f"Total NELA features extracted: {len(nela_features)}")
        print("First 5 features:")
        for name, val in list(zip(nela_names, nela_features))[:5]:
            print(f"  {name}: {val}")
    except Exception as e:
        print(f"Error running NELA extractor: {e}")
    print()

    # 3. StyleDecipher Features
    print("--- 2. StyleDecipher Extractor ---")
    try:
        # StyleDecipher expects rewrites to compare against
        sample_rewrites = [
            "AI is quickly changing many global industries. Machine learning is getting more advanced in fields like medicine and banking. Yet, this fast progress causes ethical problems that experts must fix.",
            "Across the world, various sectors are being transformed by artificial intelligence. Sophisticated machine learning models are used in finance and healthcare. But we must address the ethical challenges brought by this rapid growth."
        ]
        
        style_features = styledecipher_features(sample_text, sample_rewrites)
        print(f"Total StyleDecipher features extracted: {len(style_features)} (Mean and STD of similarities)")
        print(f"Feature vector shape: {style_features.shape}")
        print("Feature values:")
        for idx, val in enumerate(style_features):
            print(f"  F{idx+1}: {val:.4f}")
    except Exception as e:
        print(f"Error running StyleDecipher extractor: {e}")
    print()

    # 4. TRACE User Profiler Features
    print("--- 3. TRACE User Profiler Extractor ---")
    try:
        # Initialize embedder on CPU for compatibility
        trace_embedder = TRACEUserProfileEmbedder(device='cpu')
        
        # TRACE expects a list of texts from the author
        trace_embedding = trace_embedder.get_author_embedding([sample_text])
        print(f"TRACE embedding shape: {trace_embedding.shape}")
        
        print("First 10 dimensions of the embedding:")
        for idx, val in enumerate(trace_embedding[:10]):
            print(f"  Dim {idx+1}: {val:.4f}")
    except Exception as e:
        print(f"Error running TRACE extractor: {e}")
    print("\n=== Showcase Complete ===")

if __name__ == "__main__":
    main()
