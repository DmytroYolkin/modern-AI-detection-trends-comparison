
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ProjectionHead(torch.nn.Module):
    """Non-linear projection head inspired by SimCLR to enhance discriminative power."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class TRACEUserProfileEmbedder:
    """TRansformer-based Attribution using Contrastive Embeddings (TRACE) for user profiling.

    This class implements the core components of the TRACE framework for generating
    author style fingerprints from text, mapping author text to embedding representations.
    It incorporates SBERT-based encoding, a non-linear projection head, and a mechanism
    for principal sentence extraction.

    Theoretical Pitfalls and Considerations:
    1.  **Training Data**: The original TRACE framework relies on supervised contrastive
        learning (NT-Xent Loss) with source information as labels. This implementation
        assumes a pre-trained SBERT model and focuses on the forward pass for embedding
        generation. A full reproduction would require a large, labeled dataset of author
        texts for training the projection head and fine-tuning SBERT.
    2.  **Principal Sentence Extraction (TF-IDF)**: While TF-IDF is used here to identify
        "principal sentences" as suggested in the paper to focus on representative stylistic
        markers, the optimal `WINDOW_SIZE` (number of sentences to extract) is dataset-dependent
        and typically determined via ablation studies. Suboptimal selection can lead to
        loss of crucial stylistic information or inclusion of noise.
    3.  **Generalization**: The effectiveness of the generated embeddings as "author style
        fingerprints" is highly dependent on the diversity and nature of the training data
        used for the underlying SBERT model and, ideally, the contrastive learning phase.
        Embeddings might capture semantic content more strongly than pure stylistic nuances
        without explicit stylistic supervision.
    4.  **Computational Cost**: Training the full TRACE model with contrastive learning
        on large datasets is computationally intensive. This implementation focuses on
        inference, which is less demanding but still requires loading a transformer model.
    5.  **Interpretability**: While embeddings provide a numerical representation, directly
        interpreting what specific stylistic features are captured can be challenging.
    6.  **Dynamic Profiling**: The current implementation generates static embeddings. For
        dynamic user profiling (where author style might evolve), mechanisms for updating
        or incrementally learning embeddings would be necessary.
    """
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2',
                 projection_hidden_dim=256, projection_output_dim=128,
                 device='cuda'):
        """
        Initializes the TRACEUserProfileEmbedder.

        Args:
            model_name (str): Name of the pre-trained SBERT model to use.
            projection_hidden_dim (int): Hidden dimension for the projection head.
            projection_output_dim (int): Output dimension for the projection head (the final embedding size).
            device (str): Device to run the model on (
            'cpu' or 'cuda').
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

        # SBERT models typically output a 768-dimensional embedding
        sbert_output_dim = self.model.config.hidden_size
        self.projection_head = ProjectionHead(sbert_output_dim, projection_hidden_dim, projection_output_dim).to(device)

        # In a real TRACE setup, the projection_head would be trained with contrastive loss.
        # For this demonstration, it's initialized randomly or could be loaded from a checkpoint.
        # For a faithful reproduction, this would involve a training loop.

    def _mean_pooling(self, model_output, attention_mask):
        """Applies mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _extract_principal_sentences(self, texts, top_n_sentences=5):
        """Extracts principal sentences from texts using TF-IDF.

        This is a simplified approach. The original paper suggests TF-IDF for identifying
        significant sentences within documents, often selecting 10-20% of sentences.
        """
        if not texts:
            return []

        # Combine all texts to fit TF-IDF, then process each text individually
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        principal_sentences_per_text = []
        for i, text in enumerate(texts):
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                principal_sentences_per_text.append("")
                continue

            sentence_scores = []
            for sentence in sentences:
                sentence_vector = vectorizer.transform([sentence])
                # Calculate a simple score for the sentence based on its TF-IDF values
                score = sentence_vector.sum()
                sentence_scores.append((score, sentence))

            # Sort by score and take the top N sentences
            sentence_scores.sort(key=lambda x: x[0], reverse=True)
            selected_sentences = [s for score, s in sentence_scores[:top_n_sentences]]
            principal_sentences_per_text.append(" ".join(selected_sentences))

        return principal_sentences_per_text

    def get_author_embedding(self, author_texts: list[str], top_n_sentences: int = 5) -> np.ndarray:
        """
        Generates a single embedding (style fingerprint) for an author given multiple texts.

        Args:
            author_texts (list[str]): A list of text documents written by the author.
            top_n_sentences (int): Number of principal sentences to extract from each text.

        Returns:
            np.ndarray: A 1D numpy array representing the author's style fingerprint embedding.
                        Returns an empty array if no valid texts are provided.
        """
        if not author_texts:
            print("Warning: No author texts provided. Returning empty embedding.")
            return np.array([])

        # 1. Extract principal sentences from each text
        principal_sentences = self._extract_principal_sentences(author_texts, top_n_sentences)
        if not any(principal_sentences):
            print("Warning: No principal sentences extracted. Returning empty embedding.")
            return np.array([])

        # 2. Encode principal sentences using SBERT
        encoded_input = self.tokenizer(principal_sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling to get sentence embeddings
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # 3. Pass through the projection head
        projected_embeddings = self.projection_head(sentence_embeddings)

        # 4. Aggregate embeddings to form a single author embedding
        # A simple mean pooling of sentence embeddings is used here. More sophisticated
        # aggregation (e.g., weighted average, clustering) could be explored.
        author_embedding = torch.mean(projected_embeddings, dim=0)

        return author_embedding.cpu().detach().numpy()

# Example Usage:
if __name__ == "__main__":
    # Initialize the embedder
    # Using a smaller model for demonstration purposes to reduce memory usage and speed.
    # For higher quality embeddings, consider 'sentence-transformers/all-mpnet-base-v2'
    embedder = TRACEUserProfileEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda')

    # Sample author texts
    author1_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic sentence for testing typography.",
        "Programming in Python is fun and efficient. I enjoy using data structures and algorithms.",
        "Artificial intelligence is transforming many industries. Machine learning models are becoming more powerful."
    ]

    author2_texts = [
        "In a galaxy far, far away, a new hope emerged. The force is strong with this one.",
        "The epic saga continued with thrilling battles and unexpected twists. May the force be with you.",
        "Science fiction narratives often explore themes of technology and humanity's future. Space travel is fascinating."
    ]

    author3_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic sentence for testing typography.",
        "Natural language processing is a subfield of AI. It deals with the interaction between computers and human language.",
        "Data science involves extracting insights from data. Statistical analysis is a key component."
    ]

    print("Generating embeddings for Author 1...")
    embedding1 = embedder.get_author_embedding(author1_texts)
    print(f"Author 1 Embedding Shape: {embedding1.shape}")

    print("Generating embeddings for Author 2...")
    embedding2 = embedder.get_author_embedding(author2_texts)
    print(f"Author 2 Embedding Shape: {embedding2.shape}")

    print("Generating embeddings for Author 3...")
    embedding3 = embedder.get_author_embedding(author3_texts)
    print(f"Author 3 Embedding Shape: {embedding3.shape}")

    # Calculate similarity to demonstrate 

    # Calculate similarity to demonstrate
    sim_1_2 = cosine_similarity([embedding1], [embedding2])[0][0]
    sim_1_3 = cosine_similarity([embedding1], [embedding3])[0][0]

    print(f"\nSimilarity between Author 1 and Author 2 (different topics): {sim_1_2:.4f}")
    print(f"Similarity between Author 1 and Author 3 (similar topics): {sim_1_3:.4f}")
