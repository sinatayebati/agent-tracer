"""
Uncertainty Metrics for SAUP Framework

This module implements uncertainty quantification metrics for the SAUP
(Situation-Awareness Uncertainty Propagation) framework on Tau-2 benchmark.

Key metrics:
- Normalized Entropy (single-step uncertainty)
- Token-level uncertainties
- Response statistics
- Semantic Distance Metrics:
  - Inquiry Drift (Da): measures deviation from initial goal
  - Inference Gap (Do): measures coherence between action and observation
- SAUP-D Aggregation: weighted RMS combining all metrics into trajectory score

Reference: SAUP paper, Equation 3 - Normalized Entropy
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

# Vertex AI imports for embeddings
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("Vertex AI not available. Semantic distance metrics will be disabled.")


class TokenUncertainty(BaseModel):
    """Token-level uncertainty information."""

    token: str
    logprob: float
    neg_log_likelihood: float
    probability: float


class UncertaintyStats(BaseModel):
    """Comprehensive uncertainty statistics for a response."""

    normalized_entropy: float
    total_entropy: float
    token_count: int
    min_uncertainty: float
    max_uncertainty: float
    std_uncertainty: float
    mean_probability: float


# Stop words and structural tokens to filter from entropy calculation
STOP_WORDS = {
    # Common stop words
    "the", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "or", "and", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her",
    "am", "can", "will", "would", "could", "should", "may",
    "have", "has", "had", "do", "does", "did",
    # Filler phrases (split into tokens)
    "please", "sorry", "wait", "help", "need", "want",
    # Structural/formatting tokens
    "[tool_call]", "\\n", "\n", " ", "", ".", ",", "!", "?",
    ":", ";", "-", "(", ")", "[", "]", "{", "}", "'", '"',
}


def calculate_normalized_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the Normalized Entropy (single-step uncertainty) for a response.

    This metric is defined as the average token-level negative log-likelihood
    of the generated response, focusing on CONTENT-BEARING tokens only.
    
    Filtering Strategy (fixes entropy dilution):
    1. Filters high-probability tokens (P > 95%) - likely structural
    2. Filters stop words and filler phrases - masks true uncertainty
    3. Measures uncertainty only on entities, actions, and domain-specific terms

    Mathematically:
        U_i = (1/|R_i|) * Σ(-log P(token_j)) for content tokens only

    Where:
        - |R_i| is the number of CONTENT tokens in the response
        - -log P(token_j) is the negative log-likelihood of token j

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.
                        Expected structure:
                        {
                            'content': [
                                {'token': str, 'logprob': float, ...},
                                ...
                            ]
                        }
                        Can be None if logprobs are not available.

    Returns:
        float: The normalized entropy (average negative log-likelihood).
               Returns 1e-4 if:
               - logprobs_object is None
               - No tokens are found
               - The content array is empty
               - All tokens were filtered (response is only filler)

    Example:
        >>> logprobs = {
        ...     'content': [
        ...         {'token': 'I', 'logprob': -0.01},  # Stop word, filtered
        ...         {'token': ' am', 'logprob': -0.01},  # Stop word, filtered
        ...         {'token': ' booking', 'logprob': -1.2},  # Content token
        ...         {'token': ' flight', 'logprob': -0.8},  # Content token
        ...     ]
        ... }
        >>> calculate_normalized_entropy(logprobs)
        1.0  # (1.2 + 0.8) / 2 - only content tokens measured
    """
    # Handle None or missing logprobs
    if logprobs_object is None:
        return 1e-4

    # Extract the content array
    content = logprobs_object.get("content", [])

    if not content or len(content) == 0:
        return 1e-4

    # Initialize counters
    total_neg_log_likelihood = 0.0
    token_count = 0

    # Iterate through each token in the sequence
    for token_data in content:
        # Extract the logprob value
        logprob = token_data.get("logprob")
        token = token_data.get("token", "")

        if logprob is not None:
            # Filter 1: High-probability structural tokens (P > 95%)
            # logprob > -0.05 means P(token) > exp(-0.05) ≈ 0.951
            if logprob > -0.05:
                continue  # Skip structural tokens like newlines, spaces, etc.
            
            # Filter 2: Stop words and filler phrases
            # Clean token: strip whitespace, lowercase, remove punctuation
            cleaned_token = token.strip().lower()
            # Remove leading/trailing punctuation
            cleaned_token = cleaned_token.strip('.,!?;:()[]{}"\'-')
            
            # Skip if empty after cleaning
            if not cleaned_token:
                continue
            
            # Skip if it's a stop word
            if cleaned_token in STOP_WORDS:
                continue
            
            # Skip if it's purely numeric (dates, numbers, etc.)
            if cleaned_token.replace('.', '').replace('-', '').isdigit():
                continue
            
            # This is a content-bearing token - accumulate its uncertainty
            neg_log_likelihood = -logprob
            total_neg_log_likelihood += neg_log_likelihood
            token_count += 1

    # Avoid division by zero - if all tokens were filtered, return small epsilon
    # This indicates the response is 100% filler (a failure signal itself)
    if token_count == 0:
        return 1e-4

    # Calculate and return the normalized entropy (average)
    normalized_entropy = total_neg_log_likelihood / token_count

    return normalized_entropy


def calculate_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the standard entropy (unnormalized) for a response.

    This is the total negative log-likelihood across all tokens in the response,
    without normalization by token count.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        float: The total entropy (sum of negative log-likelihoods).
    """
    if logprobs_object is None:
        return 0.0

    content = logprobs_object.get("content", [])

    if not content or len(content) == 0:
        return 0.0

    total_neg_log_likelihood = 0.0

    for token_data in content:
        logprob = token_data.get("logprob")
        if logprob is not None:
            total_neg_log_likelihood += -logprob

    return total_neg_log_likelihood


def calculate_token_uncertainties(
    logprobs_object: Optional[dict],
) -> list[TokenUncertainty]:
    """
    Calculate token-level uncertainties for detailed analysis.

    Returns uncertainty information for each individual token in the response.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        list[TokenUncertainty]: List of token uncertainty objects.
    """
    if logprobs_object is None:
        return []

    content = logprobs_object.get("content", [])

    if not content:
        return []

    token_uncertainties = []

    for token_data in content:
        token = token_data.get("token", "")
        logprob = token_data.get("logprob")

        if logprob is not None:
            token_uncertainties.append(
                TokenUncertainty(
                    token=token,
                    logprob=logprob,
                    neg_log_likelihood=-logprob,
                    probability=np.exp(logprob),
                )
            )

    return token_uncertainties


def get_uncertainty_stats(logprobs_object: Optional[dict]) -> UncertaintyStats:
    """
    Get comprehensive statistics about the response uncertainty.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        UncertaintyStats: Comprehensive uncertainty statistics.
    """
    if logprobs_object is None:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    content = logprobs_object.get("content", [])

    if not content:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    uncertainties = []
    probabilities = []

    for token_data in content:
        logprob = token_data.get("logprob")
        if logprob is not None:
            uncertainties.append(-logprob)
            probabilities.append(np.exp(logprob))

    if not uncertainties:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    return UncertaintyStats(
        normalized_entropy=float(np.mean(uncertainties)),
        total_entropy=float(np.sum(uncertainties)),
        token_count=len(uncertainties),
        min_uncertainty=float(np.min(uncertainties)),
        max_uncertainty=float(np.max(uncertainties)),
        std_uncertainty=float(np.std(uncertainties)),
        mean_probability=float(np.mean(probabilities)),
    )


# ============================================================================
# Semantic Distance Metrics (SAUP Situational Awareness Layer)
# ============================================================================


class EmbeddingService:
    """
    Singleton service for managing Vertex AI text embeddings.
    
    Loads the embedding model once and reuses it for all semantic distance
    calculations during simulation.
    """
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional['TextEmbeddingModel'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service (only once)."""
        if not self._initialized and VERTEX_AI_AVAILABLE:
            try:
                # Initialize Vertex AI (auto-detects project from environment)
                # User should have GOOGLE_APPLICATION_CREDENTIALS or gcloud auth set up
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                if project_id:
                    vertexai.init(project=project_id)
                else:
                    # Try default initialization
                    vertexai.init()
                
                # Load the embedding model
                # Using text-embedding-004 (latest, 768 dimensions)
                self._model = TextEmbeddingModel.from_pretrained("text-embedding-004")
                self._initialized = True
                logger.info("✓ Vertex AI embedding service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI embeddings: {e}")
                self._model = None
                self._initialized = False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array, or None if service unavailable
        """
        if not self._initialized or self._model is None:
            return None
        
        if not text or not text.strip():
            return None
        
        try:
            # Truncate text if too long (model limit is ~20k tokens)
            # Keep approximately 15k tokens worth of text
            max_chars = 60000
            if len(text) > max_chars:
                text = text[-max_chars:]  # Keep most recent content
                logger.debug(f"Truncated text from {len(text)} to {max_chars} chars")
            
            # Get embeddings
            embeddings = self._model.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                # Convert to numpy array
                return np.array(embeddings[0].values)
            return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        return self._initialized and self._model is not None


def calculate_semantic_distance(text_a: str, text_b: str) -> float:
    """
    Calculate semantic distance between two texts using cosine similarity.
    
    The semantic distance is defined as:
        distance = 1 - cosine_similarity
    
    Where cosine_similarity is the cosine of the angle between two embedding
    vectors. A distance of 0 means the texts are semantically identical,
    while higher values indicate greater semantic divergence.
    
    Args:
        text_a: First text string
        text_b: Second text string
        
    Returns:
        float: Semantic distance in range [0, 2] where:
               - 0.0 = identical meaning
               - ~0.5 = moderately related
               - ~1.0 = unrelated
               - >1.0 = opposite meaning (rare)
               Returns 0.0 if embeddings cannot be generated
    
    Example:
        >>> distance = calculate_semantic_distance(
        ...     "Book a flight to Paris",
        ...     "Reserve tickets to France"
        ... )
        >>> # Should return a low value (similar meaning)
    """
    # Handle edge cases
    if not text_a or not text_b:
        return 0.0
    
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    # Get embeddings
    service = EmbeddingService()
    if not service.is_available():
        logger.debug("Embedding service not available, returning 0.0")
        return 0.0
    
    embedding_a = service.get_embedding(text_a)
    embedding_b = service.get_embedding(text_b)
    
    if embedding_a is None or embedding_b is None:
        logger.debug("Failed to generate embeddings, returning 0.0")
        return 0.0
    
    # Calculate cosine similarity
    try:
        # Normalize vectors
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Cosine similarity = dot product / (norm_a * norm_b)
        cosine_sim = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
        
        # Distance = 1 - similarity
        distance = 1.0 - float(cosine_sim)
        
        # Clamp to [0, 2] range (theoretical maximum)
        distance = max(0.0, min(2.0, distance))
        
        return distance
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def calculate_semantic_flux(
    current_embedding: np.ndarray,
    previous_embedding: np.ndarray
) -> float:
    """
    Calculate Semantic Flux (Φ) - the semantic displacement between consecutive turns.
    
    Measures how much "ground" the agent is covering semantically. High flux indicates
    progress and exploration; low flux indicates stagnation.
    
    Formula:
        Φ_i = ||E_i - E_{i-1}||_2 (Euclidean distance between embeddings)
    
    Args:
        current_embedding: Embedding vector for current turn
        previous_embedding: Embedding vector for previous turn
        
    Returns:
        float: Semantic flux (Euclidean distance), >= 0
               Returns 0.0 if embeddings are invalid
    
    Example:
        >>> # High flux - agent making progress
        >>> flux = calculate_semantic_flux(emb_new_state, emb_old_state)
        >>> # flux ≈ 0.8 (significant semantic movement)
        
        >>> # Low flux - agent stuck in loop
        >>> flux = calculate_semantic_flux(emb_repeat, emb_original)
        >>> # flux ≈ 0.02 (minimal semantic movement)
    """
    if current_embedding is None or previous_embedding is None:
        return 0.0
    
    if len(current_embedding) == 0 or len(previous_embedding) == 0:
        return 0.0
    
    try:
        # Euclidean distance (L2 norm of difference)
        flux = float(np.linalg.norm(current_embedding - previous_embedding))
        return max(0.0, flux)
    except Exception as e:
        logger.error(f"Error calculating semantic flux: {e}")
        return 0.0


def calculate_grounding_ratio(
    semantic_flux: float,
    normalized_entropy: float,
    epsilon: float = 0.01
) -> float:
    """
    Calculate Grounding Ratio (G) - the core "Signal Separator" for failure detection.
    
    This metric identifies the "Confident Refusal Trap" where agents confidently
    produce responses (low entropy) but make zero semantic progress (low flux).
    
    Formula:
        G_i = Φ_i / (U_i + ε)
    
    Interpretation:
        - High G (>1.0): Efficient movement (low uncertainty, high progress) → SUCCESS
        - Low G (0.1-1.0): Stagnant confusion (high uncertainty, low progress) → CONFUSION
        - Near-Zero G (<0.1): Confident refusal/looping (near-zero uncertainty, zero flux) → FAILURE
    
    Args:
        semantic_flux: Semantic displacement (Φ_i) between consecutive turns
        normalized_entropy: Token-level uncertainty (U_i) of current turn
        epsilon: Small constant to prevent division by zero (default 0.01)
                 Small epsilon allows confident failures to drive G → 0
        
    Returns:
        float: Grounding ratio, >= 0
               Higher values indicate efficient progress
               Values near 0 indicate confident failure (the trap)
    
    Example (Confident Refusal - FAILURE SIGNAL):
        >>> G = calculate_grounding_ratio(flux=0.02, entropy=0.05, epsilon=0.01)
        >>> # G ≈ 0.02 / 0.06 ≈ 0.33 (near-zero, confident trap)
    
    Example (Successful Search - PROGRESS):
        >>> G = calculate_grounding_ratio(flux=0.8, entropy=0.15, epsilon=0.01)
        >>> # G ≈ 0.8 / 0.16 = 5.0 (high efficiency)
    """
    # Prevent negative values
    flux = max(0.0, semantic_flux)
    entropy = max(0.0, normalized_entropy)
    
    # Small epsilon allows near-zero entropy to create near-zero grounding
    # (the confident refusal trap)
    denominator = entropy + epsilon
    
    grounding = flux / denominator
    
    return float(grounding)


def calculate_innovation_score(
    current_embedding: np.ndarray,
    history_embeddings: list[np.ndarray],
    window_size: int = 5
) -> float:
    """
    Calculate Information Innovation (I) - universal epistemic change detector.
    
    Uses subspace residual analysis to detect when the agent is introducing
    new information, regardless of domain. This replaces domain-specific
    regex patterns with a universal geometric method.
    
    **The Core Insight:**
    When an agent just repeats templates, the current embedding lies almost
    entirely within the subspace spanned by recent turns (high cosine similarity).
    When an agent introduces NEW information (new flight IDs, novel observations),
    the embedding moves into a NEW dimension (low similarity to all past turns).
    
    **Algorithm (Cosine Max-Pooling for Numerical Stability):**
    1. Take the last `window_size` turns (default 5) as context
    2. Calculate cosine similarity between current and each context turn
    3. Innovation = 1 - max(similarities)
       - If max_similarity ≈ 1.0 → Innovation ≈ 0 (pure repetition)
       - If max_similarity < 0.7 → Innovation > 0.3 (new information)
    
    Formula:
        I_i = 1 - max_j { cos_sim(E_i, E_j) } for j in last K turns
    
    Args:
        current_embedding: Embedding for current turn
        history_embeddings: List of embeddings from previous turns
        window_size: Number of recent turns to use as context (default 5)
        
    Returns:
        float: Innovation score in range [0, 1] where:
               - 0.0 = pure repetition (embedding in subspace)
               - ~0.3 = moderate novelty
               - ~0.7+ = high innovation (new dimension)
               Returns 1.0 if history is empty (first turn is always novel)
    
    Example (Repetition - Low Innovation):
        >>> # Agent repeats "I'm sorry, I can't help"
        >>> current = embed("I apologize, I cannot assist")
        >>> history = [embed("I'm sorry, I can't help"), embed("I apologize...")]
        >>> innovation = calculate_innovation_score(current, history)
        >>> # Returns ≈ 0.05 (max_similarity ≈ 0.95)
    
    Example (New Information - High Innovation):
        >>> # Agent discovers new flight ID
        >>> current = embed("Found flight AA123 at 3pm")
        >>> history = [embed("Searching..."), embed("Found flight BB456...")]
        >>> innovation = calculate_innovation_score(current, history)
        >>> # Returns ≈ 0.6 (max_similarity ≈ 0.4, different entities)
    """
    # Edge case: Empty history (first turn is always novel)
    if not history_embeddings or current_embedding is None:
        return 1.0
    
    if len(current_embedding) == 0:
        return 1.0
    
    try:
        # Take last K turns as context subspace
        context_window = history_embeddings[-window_size:] if len(history_embeddings) > window_size else history_embeddings
        
        if not context_window:
            return 1.0
        
        # Normalize current embedding
        current_norm = np.linalg.norm(current_embedding)
        if current_norm == 0:
            return 0.0
        current_normalized = current_embedding / current_norm
        
        # Calculate cosine similarity with each context turn
        max_similarity = 0.0
        for past_emb in context_window:
            if past_emb is None or len(past_emb) == 0:
                continue
            
            # Normalize past embedding
            past_norm = np.linalg.norm(past_emb)
            if past_norm == 0:
                continue
            past_normalized = past_emb / past_norm
            
            # Cosine similarity = dot product of normalized vectors
            cosine_sim = float(np.dot(current_normalized, past_normalized))
            
            # Clamp to [0, 1] (should already be in [-1, 1], but clamp for safety)
            cosine_sim = max(0.0, min(1.0, cosine_sim))
            
            # Track maximum similarity (closest match in subspace)
            max_similarity = max(max_similarity, cosine_sim)
        
        # Innovation = 1 - max_similarity
        # High similarity to past → Low innovation (repetition)
        # Low similarity to past → High innovation (new information)
        innovation = 1.0 - max_similarity
        
        logger.debug(
            f"Innovation calculation: max_sim={max_similarity:.3f} → "
            f"innovation={innovation:.3f} (window={len(context_window)} turns)"
        )
        
        return float(innovation)
    
    except Exception as e:
        logger.error(f"Error calculating innovation score: {e}")
        return 0.0


def calculate_state_density(
    current_embedding: np.ndarray,
    history_embeddings: list[np.ndarray],
    current_turn: int,
    history_turns: list[int],
    decay_lambda: float = 0.1
) -> float:
    """
    Calculate State Density (D) - measures manifold overlap (revisiting old states).
    
    Uses temporal decay so that revisiting a state from 20 turns ago is penalized
    less than revisiting a state from 2 turns ago.
    
    Formula:
        D_i = max_j { exp(-λ × |i - j|) × (1 - ||E_i - E_j||_2) }
    
    Where:
        - λ (decay_lambda): Controls temporal decay rate (default 0.1)
        - Exponential decay favors recent history
        - (1 - distance) converts distance to similarity (high when states overlap)
    
    Args:
        current_embedding: Embedding for current turn
        history_embeddings: List of embeddings from previous turns
        current_turn: Turn index of current state
        history_turns: List of turn indices for history embeddings
        decay_lambda: Temporal decay parameter (default 0.1)
        
    Returns:
        float: State density in range [0, 1] where:
               - 0.0 = completely novel state (no overlap)
               - ~0.5 = moderate overlap with decayed history
               - 1.0 = exact repetition of recent state (failure signal)
    
    Example (Novel State - Good):
        >>> density = calculate_state_density(emb_new, [emb1, emb2], 10, [8, 9], 0.1)
        >>> # density ≈ 0.1 (new semantic territory)
    
    Example (Recent Loop - Bad):
        >>> density = calculate_state_density(emb_repeat, [emb_orig], 5, [3], 0.1)
        >>> # density ≈ 0.9 (revisiting recent state with high similarity)
    """
    if current_embedding is None or not history_embeddings:
        return 0.0
    
    if len(current_embedding) == 0:
        return 0.0
    
    if len(history_embeddings) != len(history_turns):
        logger.warning(
            f"Mismatch in history lengths: {len(history_embeddings)} embeddings, "
            f"{len(history_turns)} turns"
        )
        return 0.0
    
    max_density = 0.0
    
    try:
        for past_emb, past_turn in zip(history_embeddings, history_turns):
            if past_emb is None or len(past_emb) == 0:
                continue
            
            # Temporal decay: exp(-λ × |i - j|)
            time_diff = abs(current_turn - past_turn)
            temporal_weight = np.exp(-decay_lambda * time_diff)
            
            # Semantic similarity: 1 - distance
            # Distance is normalized by embedding dimension for stability
            distance = float(np.linalg.norm(current_embedding - past_emb))
            similarity = 1.0 - min(distance, 1.0)  # Clamp distance to [0, 1]
            
            # Weighted density
            density = temporal_weight * similarity
            
            # Track maximum (worst-case overlap)
            max_density = max(max_density, density)
        
        return float(max_density)
    except Exception as e:
        logger.error(f"Error calculating state density: {e}")
        return 0.0


def calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate Jaccard similarity between two texts (lexical overlap).
    
    DEPRECATED: This function is kept for backward compatibility but is not
    used in the SAUP-Flux framework.
    
    Measures the intersection-over-union of content tokens, filtering out
    stop words and punctuation.
    
    Args:
        text_a: First text string
        text_b: Second text string
        
    Returns:
        float: Jaccard similarity in range [0, 1]
    """
    # Handle edge cases
    if not text_a or not text_b:
        return 0.0
    
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    # Tokenize: split by whitespace
    tokens_a_raw = text_a.lower().split()
    tokens_b_raw = text_b.lower().split()
    
    # Clean tokens: remove stop words and punctuation
    def clean_token(token: str) -> str:
        """Remove leading/trailing punctuation from token."""
        return token.strip('.,!?;:()[]{}"\'-')
    
    # Filter tokens
    tokens_a = set()
    for token in tokens_a_raw:
        cleaned = clean_token(token)
        # Skip if empty, stop word, or purely numeric
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace('.', '').replace('-', '').isdigit():
            continue
        tokens_a.add(cleaned)
    
    tokens_b = set()
    for token in tokens_b_raw:
        cleaned = clean_token(token)
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace('.', '').replace('-', '').isdigit():
            continue
        tokens_b.add(cleaned)
    
    # Handle case where all tokens were filtered
    if not tokens_a or not tokens_b:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    
    if union == 0:
        return 0.0
    
    jaccard_sim = float(intersection) / float(union)
    
    return jaccard_sim


def calculate_information_stagnation(
    current_text: str,
    global_token_set: set[str],
    history_texts: list[str]
) -> float:
    """
    DEPRECATED: This function is replaced by State Density in the SAUP-Flux framework.
    Kept for backward compatibility only.
    
    Calculate Information Stagnation score - measures lack of information gain velocity.
    
    Returns:
        float: Always returns 0.0 (deprecated)
    """
    logger.warning("calculate_information_stagnation is deprecated. Use calculate_state_density instead.")
    return 0.0


def calculate_tool_repetition(
    current_tool_calls: list,
    tool_call_history: list[list]
) -> float:
    """
    DEPRECATED: This function is not used in the SAUP-Flux framework.
    Kept for backward compatibility only.
    
    Calculate Tool Repetition penalty for exact duplicate tool calls.
    
    Returns:
        float: Always returns 0.0 (deprecated)
    """
    logger.warning("calculate_tool_repetition is deprecated and not used in FLUX framework.")
    return 0.0


def calculate_inquiry_drift(
    user_instruction: str,
    history_so_far: list[str]
) -> float:
    """
    DEPRECATED: This function is not used in the SAUP-Flux framework.
    Kept for backward compatibility only.
    
    Calculate Inquiry Drift (Da) - semantic distance from initial goal.
    
    Returns:
        float: Always returns 0.0 (deprecated)
    """
    logger.warning("calculate_inquiry_drift is deprecated and not used in FLUX framework.")
    return 0.0


def calculate_inference_gap(antecedent: str, consequent: str) -> float:
    """
    DEPRECATED: Replaced by calculate_semantic_flux in the SAUP-Flux framework.
    Kept for backward compatibility only.
    
    Calculate Inference Gap (Do) - semantic distance between action and observation.
    
    Returns:
        float: Always returns 0.0 (deprecated, use semantic_flux instead)
    """
    logger.warning("calculate_inference_gap is deprecated. Use calculate_semantic_flux instead.")
    return 0.0


# ============================================================================
# SAUP-D Aggregation (Final Trajectory Score)
# ============================================================================


@dataclass
class SAUPConfig:
    """
    Configuration for SAUP-Flux aggregation.
    
    The flux efficiency model measures topological efficiency of trajectories:
        Waste_i = (1/G_i) × D_i, where G_i = Φ_i / (U_i + ε)
        FLUX_Score = mean_weight × Mean(w_i × Waste_i) + max_weight × Max(w_i × Waste_i)
    
    Attributes:
        epsilon: Small constant for grounding ratio denominator (default 0.01)
        decay_lambda: Temporal decay for state density (default 0.1)
        mean_weight: Weight for mean waste in hybrid (default 0.7)
        max_weight: Weight for max waste in hybrid (default 0.3)
        flux_saturation: Saturation cap for flux (default 2.0)
    """
    epsilon: float = 0.01
    decay_lambda: float = 0.1
    mean_weight: float = 0.7
    max_weight: float = 0.3
    flux_saturation: float = 2.0


def calculate_situational_weight(
    da: Optional[float],
    do_agent: Optional[float],
    do_user: Optional[float],
    config: SAUPConfig
) -> float:
    """
    DEPRECATED: This function is not used in the SAUP-Flux framework.
    Kept for backward compatibility only.
    
    Returns:
        float: Always returns 0.0 (deprecated)
    """
    logger.warning("calculate_situational_weight is deprecated and not used in FLUX framework.")
    return 0.0


def calculate_flux_efficiency(
    step_data: list[dict],
    config: Optional[SAUPConfig] = None
) -> dict:
    """
    Calculate Gated Flux Efficiency score with Universal Innovation Detection (SAUP v4).
    
    The Universal Efficiency Gate protects "Innovative Repetition" (systematic work)
    from being penalized as waste. Uses subspace residuals to detect epistemic novelty.
    
    Formula:
        D'_i = D_i × (1 - I_i)  [Gated Density: Innovation kills density penalty]
        G_i = Φ_i / (U_i + ε)  [Grounding Ratio: Flux over uncertainty]
        E_i = min(1.0, G_i / (1 + G_i))  [Robust Efficiency: Bounded sigmoid]
        Risk_i = max(D'_i - E_i, 0)  [Turn Risk: Gated density minus efficiency]
        Risk_i = min(Risk_i, 3.0)  [Saturation cap]
        
        FLUX = 0.7 × Mean(w_i × Risk_i) + 0.3 × Max(w_i × Risk_i)
    
    Where:
        - Φ_i: Semantic flux (movement)
        - U_i: Normalized entropy (uncertainty)
        - D_i: State density (overlap)
        - I_i: Innovation score (epistemic novelty via subspace residuals)
        - G_i: Grounding ratio (efficiency of movement)
        - E_i: Bounded efficiency coefficient (prevents high-entropy bombs)
        - w_i: Logistic temporal weight (later turns weighted more)
    
    **The Innovation Gate:**
    - If I_i = 1.0 (high innovation), D'_i → 0 (density penalty disappears)
    - If I_i = 0.0 (pure repetition), D'_i = D_i (full density penalty applies)
    
    Success Pattern (Sim #4 - Systematic Search):
        - High Φ (agent moves), Moderate U (some uncertainty), High I (new entities)
        - D'_i ≈ 0 (innovation kills density), High E_i (efficient)
        - Risk ≈ 0 → Low FLUX (SUCCESS)
    
    Failure Pattern (Sim #8 - Refusal Loop):
        - Low Φ (no movement), Low U (confident), Low I (no new info)
        - D'_i = D (full density), Low E_i (inefficient)
        - Risk ≫ 0 → High FLUX (FAILURE)
    
    Args:
        step_data: List of step dictionaries containing:
                   - 'ui': normalized entropy (required)
                   - 'phi': semantic flux (required)
                   - 'density': state density (required)
                   - 'innovation': innovation score (required for v4)
        config: SAUP-Flux configuration (defaults to SAUPConfig())
        
    Returns:
        dict: {
            'flux_score': float - final trajectory score (lower is better),
            'num_steps': int,
            'mean_risk': float - average risk,
            'std_risk': float - std deviation of risk,
            'mean_grounding': float - average grounding ratio,
            'mean_efficiency': float - average robust efficiency,
            'mean_innovation': float - average innovation score,
            'mean_gated_density': float - average gated density,
            'mean_flux': float,
            'mean_density': float,
            'mean_ui': float,
            'mean_weight': float,
            'max_weighted_risk': float,
            'mean_weighted_risk': float,
            'risks': list[float],
            'groundings': list[float],
            'efficiencies': list[float],
            'weights': list[float]
        }
    """
    if config is None:
        config = SAUPConfig()
    
    if not step_data:
        return {
            'flux_score': 0.0,
            'num_steps': 0,
            'mean_risk': 0.0,
            'std_risk': 0.0,
            'mean_grounding': 0.0,
            'mean_efficiency': 0.0,
            'mean_innovation': 0.0,
            'mean_gated_density': 0.0,
            'mean_flux': 0.0,
            'mean_density': 0.0,
            'mean_ui': 0.0,
            'mean_weight': 0.0,
            'max_weighted_risk': 0.0,
            'mean_weighted_risk': 0.0,
            'risks': [],
            'groundings': [],
            'efficiencies': [],
            'weights': []
        }
    
    risks = []
    groundings = []
    efficiencies = []
    weighted_risks = []
    fluxes = []
    densities = []
    gated_densities = []
    innovations = []
    ui_values = []
    weights = []
    
    N = len(step_data)
    
    for idx, step in enumerate(step_data):
        # Extract metrics
        ui = step.get('ui', 0.0)
        phi = step.get('phi', 0.0)
        density = step.get('density', 0.0)
        innovation = step.get('innovation', 0.0)  # Universal epistemic change
        
        # Handle None values (for backward compatibility with old simulations)
        if innovation is None:
            innovation = 0.0
        
        # Apply flux saturation (outlier robustness)
        phi_saturated = config.flux_saturation * np.tanh(phi / config.flux_saturation)
        
        # Calculate Grounding Ratio: G_i = Φ_i / (U_i + ε)
        grounding = calculate_grounding_ratio(phi_saturated, ui, config.epsilon)
        
        # Calculate Robust Efficiency Coefficient: E_i = G / (1 + G)
        # This bounds the efficiency to [0, 1] preventing high-entropy bombs
        # High grounding → E approaches 1.0 (very efficient)
        # Low grounding → E approaches 0.0 (inefficient)
        efficiency = grounding / (1.0 + grounding)
        efficiency = min(1.0, efficiency)  # Clamp to [0, 1]
        
        # Apply Innovation Gate: D'_i = D_i × (1 - I_i)
        # High innovation → Gated density approaches 0 (innovation protects repetition)
        # Low innovation → Gated density = full density (repetition penalized)
        gated_density = density * (1.0 - innovation)
        
        # Calculate Turn Risk: Risk_i = D'_i - E_i
        # High gated density + Low efficiency = High risk (wasteful repetition)
        # Low gated density + High efficiency = Low risk (productive work)
        risk = max(gated_density - efficiency, 0.0)
        
        # Saturation cap: No single turn can contribute more than 3.0
        risk = min(risk, 3.0)
        
        # Calculate Logistic Temporal Weight (balanced)
        t_i = float(idx) / float(N) if N > 0 else 0.0
        k = 6.0
        c = 0.5
        
        # Sigmoid function: 1 / (1 + exp(-k×(t_i - c)))
        z = k * (t_i - c)
        if z >= 0:
            w_i = 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            w_i = exp_z / (1.0 + exp_z)
        
        # Calculate weighted risk for this step
        weighted_risk = w_i * risk
        
        risks.append(risk)
        groundings.append(grounding)
        efficiencies.append(efficiency)
        weighted_risks.append(weighted_risk)
        fluxes.append(phi)
        densities.append(density)
        gated_densities.append(gated_density)
        innovations.append(innovation)
        ui_values.append(ui)
        weights.append(w_i)
        
        logger.debug(
            f"Step {idx}: U_i={ui:.3f}, Φ_i={phi:.3f}, D_i={density:.3f}, I_i={innovation:.3f}, "
            f"G_i={grounding:.3f}, E_i={efficiency:.3f}, D'_i={gated_density:.3f}, Risk={risk:.3f}"
        )
    
    # Calculate FLUX score using MEAN-MAX HYBRID
    mean_weighted_risk = float(np.mean(weighted_risks)) if weighted_risks else 0.0
    max_weighted_risk = float(np.max(weighted_risks)) if weighted_risks else 0.0
    
    flux_score = 0.7 * mean_weighted_risk + 0.3 * max_weighted_risk
    
    # Calculate statistics
    result = {
        'flux_score': flux_score,
        'num_steps': N,
        'mean_risk': float(np.mean(risks)) if risks else 0.0,
        'std_risk': float(np.std(risks)) if risks else 0.0,
        'mean_grounding': float(np.mean(groundings)) if groundings else 0.0,
        'mean_efficiency': float(np.mean(efficiencies)) if efficiencies else 0.0,
        'mean_innovation': float(np.mean(innovations)) if innovations else 0.0,
        'mean_gated_density': float(np.mean(gated_densities)) if gated_densities else 0.0,
        'mean_flux': float(np.mean(fluxes)) if fluxes else 0.0,
        'mean_density': float(np.mean(densities)) if densities else 0.0,
        'mean_ui': float(np.mean(ui_values)) if ui_values else 0.0,
        'mean_weight': float(np.mean(weights)) if weights else 0.0,
        'max_weighted_risk': max_weighted_risk,
        'mean_weighted_risk': mean_weighted_risk,
        'risks': risks,
        'groundings': groundings,
        'efficiencies': efficiencies,
        'weights': weights
    }
    
    return result
    """
    Calculate REG (Robust Evidence Gating) score with harmonic gating.
    
    Implements the REG formula with **Harmonic Penalty** to eliminate mathematical
    outliers caused by multiplicative explosion. Failure is only certain if the
    agent is BOTH stagnant (repeating) AND incoherent (inference gap).
    
        Penalty_i = 2 × (Stagnation_i × CoherenceGap_i) / (Stagnation_i + CoherenceGap_i + ε)
        Penalty_i = min(Penalty_i, 3.0)  [Saturation cap]
        StepRisk_i = U_i + Penalty_i
        REG_score = 0.7 × Mean(w_i × StepRisk_i) + 0.3 × Max(w_i × StepRisk_i)
    
    Where:
        - Harmonic mean requires BOTH stagnation AND coherence gap to be high
        - Saturation cap prevents single-turn explosions
        - Mean-Max hybrid: Mean captures "Slow Death Spiral", Max captures "Catastrophic Break"
        - w_i = sigmoid(k × (t_i - c)) with balanced parameters (c=0.5, k=6)
        - U_i = normalized entropy for step i
        - Stagnation_i = α·Da_i (information stagnation score)
        - CoherenceGap_i = β·Do_agent_i + γ·Do_user_i
    
    Args:
        step_data: List of step dictionaries containing:
                   - 'ui': normalized entropy (required)
                   - 'da': information stagnation (optional)
                   - 'do_agent': agent coherence (optional)
                   - 'do_user': user coherence (optional)
        config: SAUP configuration (defaults to SAUPConfig())
        
    Returns:
        dict: {
            'saup_score': float - final trajectory score (REG),
            'num_steps': int - number of steps included,
            'mean_penalty': float - average harmonic penalty across steps,
            'std_penalty': float - std deviation of penalties,
            'mean_ui': float - average U_i across steps,
            'mean_stagnation': float - average stagnation across steps,
            'mean_weight': float - average temporal weight,
            'max_risk': float - maximum weighted step risk (catastrophic break),
            'mean_risk': float - mean weighted step risk (slow death spiral),
            'penalties': list[float] - Harmonic penalty for each step,
            'weights': list[float] - w_i for each step
        }
    
    Example:
        >>> steps = [
        ...     {'ui': 0.1, 'da': 0.9, 'do_agent': 0.9, 'do_user': None},  # High both → High penalty
        ...     {'ui': 0.15, 'da': 0.9, 'do_agent': 0.1, 'do_user': None}  # High stag, low Do → Low penalty
        ... ]
        >>> result = calculate_saup_score(steps)
        >>> print(f"REG Score: {result['saup_score']:.4f}")
    """
    if config is None:
        config = SAUPConfig()
    
    if not step_data:
        return {
            'saup_score': 0.0,
            'num_steps': 0,
            'mean_penalty': 0.0,
            'std_penalty': 0.0,
            'mean_ui': 0.0,
            'mean_stagnation': 0.0,
            'mean_weight': 0.0,
            'max_risk': 0.0,
            'mean_risk': 0.0,
            'penalties': [],
            'weights': []
        }
    
    # Calculate harmonic penalties and risk scores
    harmonic_penalties = []
    step_risks = []
    weighted_risks = []
    ui_values = []
    stagnation_values = []
    weights = []
    
    N = len(step_data)
    epsilon = 1e-6  # Small constant to prevent division by zero
    
    for idx, step in enumerate(step_data):
        # Extract metrics
        ui = step.get('ui', 0.0)
        da = step.get('da')
        do_agent = step.get('do_agent')
        do_user = step.get('do_user')
        
        # Handle None values (treat as 0.0)
        da_val = da if da is not None else 0.0
        do_agent_val = do_agent if do_agent is not None else 0.0
        do_user_val = do_user if do_user is not None else 0.0
        
        # Calculate STAGNATION component (scaled by alpha)
        stagnation = config.alpha * da_val
        
        # Calculate COHERENCE GAP component (Do metrics)
        coherence_gap = config.beta * do_agent_val + config.gamma * do_user_val
        
        # HARMONIC PENALTY: 2 × (A × B) / (A + B + ε)
        # Requires BOTH stagnation AND coherence gap to be high for high penalty
        # If either is low, penalty drops significantly
        numerator = 2.0 * stagnation * coherence_gap
        denominator = stagnation + coherence_gap + epsilon
        harmonic_penalty = numerator / denominator
        
        # SATURATION CAP: Prevent single-turn explosions
        # No single turn can contribute more than 3.0 to the penalty
        harmonic_penalty = min(harmonic_penalty, 3.0)
        
        # Calculate step risk: U_i + Harmonic Penalty
        step_risk = ui + harmonic_penalty
        
        # Calculate BALANCED Logistic Temporal Weight
        # w_i = sigmoid(k × (t_i - c))
        # where t_i = i/N (normalized turn index in [0, 1])
        #       c = 0.5 (center at 50% for balanced weighting)
        #       k = 6 (moderate steepness for natural ramp)
        t_i = float(idx) / float(N) if N > 0 else 0.0
        k = 6.0
        c = 0.5
        
        # Sigmoid function: 1 / (1 + exp(-k×(t_i - c)))
        z = k * (t_i - c)
        # Numerically stable sigmoid
        if z >= 0:
            w_i = 1.0 / (1.0 + np.exp(-z))
        else:
            exp_z = np.exp(z)
            w_i = exp_z / (1.0 + exp_z)
        
        # Calculate weighted risk for this step
        weighted_risk = w_i * step_risk
        
        harmonic_penalties.append(harmonic_penalty)
        step_risks.append(step_risk)
        weighted_risks.append(weighted_risk)
        ui_values.append(ui)
        stagnation_values.append(stagnation)
        weights.append(w_i)
    
    # Calculate REG score using MEAN-MAX HYBRID
    # 70% Mean (captures "Slow Death Spiral") + 30% Max (captures "Catastrophic Break")
    mean_risk = float(np.mean(weighted_risks)) if weighted_risks else 0.0
    max_risk = float(np.max(weighted_risks)) if weighted_risks else 0.0
    
    reg_score = 0.7 * mean_risk + 0.3 * max_risk
    
    # Calculate statistics
    result = {
        'saup_score': reg_score,
        'num_steps': N,
        'mean_penalty': float(np.mean(harmonic_penalties)) if harmonic_penalties else 0.0,
        'std_penalty': float(np.std(harmonic_penalties)) if harmonic_penalties else 0.0,
        'mean_ui': float(np.mean(ui_values)) if ui_values else 0.0,
        'mean_stagnation': float(np.mean(stagnation_values)) if stagnation_values else 0.0,
        'mean_weight': float(np.mean(weights)) if weights else 0.0,
        'max_risk': max_risk,
        'mean_risk': mean_risk,
        'penalties': harmonic_penalties,
        'weights': weights
    }
    
    return result


def calculate_flux_from_trajectory(
    messages: list,
    config: Optional[SAUPConfig] = None
) -> dict:
    """
    Convenience function to calculate FLUX score directly from message objects.
    
    Extracts U_i, Φ_i, D_i, and I_i metrics from message objects and calculates
    the final FLUX trajectory score with Universal Innovation Gating (v4).
    
    Args:
        messages: List of message objects (AssistantMessage, UserMessage)
                  with uncertainty, phi_score, density_score, and innovation_score attributes
        config: SAUP-Flux configuration (defaults to SAUPConfig())
        
    Returns:
        dict: FLUX metrics (same as calculate_flux_efficiency)
    
    Example:
        >>> # After running a simulation with --calculate-uncertainty
        >>> from tau2.data_model.simulation import Results
        >>> results = Results.load("simulation.json")
        >>> sim = results.simulations[0]
        >>> flux = calculate_flux_from_trajectory(sim.messages)
        >>> print(f"Trajectory FLUX Score: {flux['flux_score']:.4f}")
    """
    step_data = []
    
    for msg in messages:
        # Only process agent and user messages
        if not hasattr(msg, 'role') or msg.role not in ['assistant', 'user']:
            continue
        
        # Skip if no uncertainty data
        if not hasattr(msg, 'uncertainty') or msg.uncertainty is None:
            continue
        
        # Extract U_i
        ui = msg.uncertainty.get('normalized_entropy', 0.0)
        
        # Extract Φ_i (semantic flux)
        phi = msg.phi_score if hasattr(msg, 'phi_score') and msg.phi_score is not None else 0.0
        
        # Extract D_i (state density)
        density = msg.density_score if hasattr(msg, 'density_score') and msg.density_score is not None else 0.0
        
        # Extract I_i (innovation score) - v4 universal epistemic change
        innovation = msg.innovation_score if hasattr(msg, 'innovation_score') and msg.innovation_score is not None else 0.0
        
        step_data.append({
            'ui': ui,
            'phi': phi,
            'density': density,
            'innovation': innovation
        })
    
    return calculate_flux_efficiency(step_data, config)


# Backward compatibility aliases
calculate_saup_score = calculate_flux_efficiency
calculate_saup_from_trajectory = calculate_flux_from_trajectory

