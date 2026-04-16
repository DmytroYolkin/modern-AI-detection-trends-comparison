# pip install sentence-transformers numpy scikit-learn python-Levenshtein

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Levenshtein import distance as levenshtein_distance
import ollama

# ===== INIT MODEL ONCE =====
model = SentenceTransformer('all-mpnet-base-v2')

MODELS = ["llama3", "mistral", "gemma", "phi3", "qwen2"]

PROMPT_TEMPLATE = """
Rewrite the following text.

Constraints:
- Preserve the exact meaning
- Change sentence structure significantly
- Use different vocabulary
- Avoid copying phrases
- Keep it natural and fluent

Text:
{text}
"""

def generate_rewrites_multi_llm(text, models=MODELS, max_per_model=1):
    """
    Generates rewrites using multiple LLMs (StyleDecipher-style).

    Args:
        text (str)
        models (list[str])
        max_per_model (int): number of rewrites per model

    Returns:
        list[str]
    """
    rewrites = []

    for model in models:
        for _ in range(max_per_model):
            try:
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a rewriting assistant."},
                        {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
                    ],
                    options={
                        "temperature": 1.0,
                        "top_p": 0.95
                    }
                )

                out = response["message"]["content"].strip()

                # basic filtering
                if len(out) > 20 and out != text:
                    rewrites.append(out)

            except Exception as e:
                print(f"[{model}] failed:", e)

    # remove duplicates
    return list(set(rewrites))

# ===== N-GRAM OVERLAP =====
def get_ngrams(text, n):
    tokens = text.lower().split()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def ngram_overlap(t1, t2, n):
    n1 = get_ngrams(t1, n)
    n2 = get_ngrams(t2, n)

    if not n1 and not n2:
        return 1.0
    if not n1 or not n2:
        return 0.0

    return len(n1 & n2) / len(n1 | n2)


# ===== EDIT SIMILARITY =====
def edit_similarity(t1, t2):
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0
    return 1.0 - (levenshtein_distance(t1, t2) / max_len)


# ===== EMBEDDING SIMILARITY =====
def embedding_similarity(t1, t2):
    e1 = model.encode(t1)
    e2 = model.encode(t2)
    return cosine_similarity([e1], [e2])[0][0]


# ===== MAIN FEATURE FUNCTION =====
def styledecipher_features(original_text) -> np.ndarray:
    """
    Args:
        original_text (str)

    Returns:
        np.ndarray (10 features)
    """

    rewritten_texts = generate_rewrites_multi_llm(original_text)

    features = []

    for rew in rewritten_texts:
        f = [
            ngram_overlap(original_text, rew, 1),
            ngram_overlap(original_text, rew, 2),
            ngram_overlap(original_text, rew, 3),
            edit_similarity(original_text, rew),
            embedding_similarity(original_text, rew)
        ]
        features.append(f)

    features = np.array(features)

    # ===== AGGREGATION =====
    mean_features = np.mean(features, axis=0)
    std_features  = np.std(features, axis=0)

    return np.concatenate([mean_features, std_features])