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


def calculate_normalized_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the Normalized Entropy (single-step uncertainty) for a response.

    This metric is defined as the average token-level negative log-likelihood
    of the generated response. It measures the uncertainty in the model's
    generation for this specific turn.

    Mathematically:
        U_i = (1/|R_i|) * Σ(-log P(token_j))

    Where:
        - |R_i| is the number of tokens in the response
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
               Returns 0.0 if:
               - logprobs_object is None
               - No tokens are found
               - The content array is empty

    Example:
        >>> logprobs = {
        ...     'content': [
        ...         {'token': 'Hello', 'logprob': -0.1},
        ...         {'token': ' world', 'logprob': -0.2},
        ...     ]
        ... }
        >>> calculate_normalized_entropy(logprobs)
        0.15  # (0.1 + 0.2) / 2
    """
    # Handle None or missing logprobs
    if logprobs_object is None:
        return 0.0

    # Extract the content array
    content = logprobs_object.get("content", [])

    if not content or len(content) == 0:
        return 0.0

    # Initialize counters
    total_neg_log_likelihood = 0.0
    token_count = 0

    # Iterate through each token in the sequence
    for token_data in content:
        # Extract the logprob value
        # Note: The API returns logprob (which is log P, a negative value)
        # We need the negative log-likelihood, which is -log P (positive value)
        logprob = token_data.get("logprob")

        if logprob is not None:
            # Convert logprob to negative log-likelihood
            # logprob = log(P) (negative)
            # neg_log_likelihood = -log(P) (positive)
            neg_log_likelihood = -logprob

            total_neg_log_likelihood += neg_log_likelihood
            token_count += 1

    # Avoid division by zero
    if token_count == 0:
        return 0.0

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


def calculate_inquiry_drift(
    user_instruction: str,
    history_so_far: list[str]
) -> float:
    """
    Calculate Inquiry Drift (Da) - semantic distance from initial goal.
    
    Measures how much the conversation has drifted from the original user
    instruction/goal. This is computed by comparing the initial instruction
    with the concatenated conversation history.
    
    Formula:
        Da = SemanticDistance(user_instruction, concatenate(history_so_far))
    
    Args:
        user_instruction: The initial goal/problem description from the task
        history_so_far: List of all text strings (messages, tool calls,
                       observations) generated in the conversation up to
                       the current step
                       
    Returns:
        float: Inquiry drift score in range [0, 2] where:
               - 0.0 = conversation perfectly aligned with goal
               - ~0.5 = moderate drift
               - >1.0 = significant drift from original intent
               Returns 0.0 if calculation fails
    
    Example:
        >>> drift = calculate_inquiry_drift(
        ...     "I need to change my flight",
        ...     ["Can you help?", "Looking up flights...", "Found 3 options"]
        ... )
    """
    if not user_instruction or not history_so_far:
        return 0.0
    
    # Concatenate history into single string
    # Use newlines to separate turns for better semantic representation
    concatenated_history = "\n".join(history_so_far)
    
    if not concatenated_history.strip():
        return 0.0
    
    # Calculate semantic distance
    distance = calculate_semantic_distance(user_instruction, concatenated_history)
    
    return distance


def calculate_inference_gap(antecedent: str, consequent: str) -> float:
    """
    Calculate Inference Gap (Do) - semantic distance between action and observation.
    
    In the dual-control Tau-2 environment, this metric measures:
    - Agent Coherence: Distance between agent's tool call and its observation
    - User Coherence: Distance between agent's message and user's response
    
    Formula:
        Do = SemanticDistance(antecedent, consequent)
    
    Args:
        antecedent: What was intended/asked (agent's action or message)
        consequent: What actually happened (observation or user's response)
        
    Returns:
        float: Inference gap score in range [0, 2] where:
               - 0.0 = perfect coherence between action and outcome
               - ~0.5 = moderate gap
               - >1.0 = significant mismatch
               Returns 0.0 if calculation fails
    
    Example (Agent Coherence):
        >>> gap = calculate_inference_gap(
        ...     "Tool: get_customer_info(id='123')",
        ...     "Customer John Doe, phone: 555-1234"
        ... )
        
    Example (User Coherence):
        >>> gap = calculate_inference_gap(
        ...     "Please provide your account number",
        ...     "My number is ABC-123"
        ... )
    """
    return calculate_semantic_distance(antecedent, consequent)


# ============================================================================
# SAUP-D Aggregation (Final Trajectory Score)
# ============================================================================


@dataclass
class SAUPConfig:
    """
    Configuration for SAUP-D aggregation.
    
    The situational weight for each step is calculated as:
        W_i = alpha * Da_i + beta * Do_agent_i + gamma * Do_user_i
    
    Attributes:
        alpha: Weight for inquiry drift (Da) - default 1.0
        beta: Weight for agent coherence (Do_agent) - default 1.0
        gamma: Weight for user coherence (Do_user) - default 1.0
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0


def calculate_situational_weight(
    da: Optional[float],
    do_agent: Optional[float],
    do_user: Optional[float],
    config: SAUPConfig
) -> float:
    """
    Calculate situational weight W_i for a single step.
    
    The weight combines semantic distance metrics (Da, Do) to measure
    how "risky" or "uncertain" the situational context is at step i.
    
    Formula:
        W_i = alpha * Da_i + beta * Do_agent_i + gamma * Do_user_i
    
    Args:
        da: Inquiry drift score (None treated as 0.0)
        do_agent: Agent coherence score (None treated as 0.0)
        do_user: User coherence score (None treated as 0.0)
        config: SAUP configuration with alpha, beta, gamma weights
        
    Returns:
        float: Situational weight W_i >= 0
    
    Example:
        >>> config = SAUPConfig(alpha=1.0, beta=1.0, gamma=1.0)
        >>> w = calculate_situational_weight(0.2, 0.3, None, config)
        >>> # Returns: 1.0*0.2 + 1.0*0.3 + 1.0*0.0 = 0.5
    """
    # Handle None values (treat as 0.0)
    da_val = da if da is not None else 0.0
    do_agent_val = do_agent if do_agent is not None else 0.0
    do_user_val = do_user if do_user is not None else 0.0
    
    # Linear combination
    weight = (
        config.alpha * da_val +
        config.beta * do_agent_val +
        config.gamma * do_user_val
    )
    
    return float(weight)


def calculate_saup_score(
    step_data: list[dict],
    config: Optional[SAUPConfig] = None
) -> dict:
    """
    Calculate SAUP-D aggregation score for a trajectory using weighted RMS.
    
    Implements the core SAUP-D formula (Equation 1 from paper):
        U_trajectory = sqrt( (1/N) * sum((W_i * U_i)^2) )
    
    Where:
        - N = number of steps
        - W_i = situational weight for step i
        - U_i = normalized entropy for step i
    
    Args:
        step_data: List of step dictionaries containing:
                   - 'ui': normalized entropy (required)
                   - 'da': inquiry drift (optional)
                   - 'do_agent': agent coherence (optional)
                   - 'do_user': user coherence (optional)
        config: SAUP configuration (defaults to SAUPConfig())
        
    Returns:
        dict: {
            'saup_score': float - final trajectory score,
            'num_steps': int - number of steps included,
            'mean_weight': float - average W_i across steps,
            'std_weight': float - std deviation of weights,
            'mean_ui': float - average U_i across steps,
            'weights': list[float] - W_i for each step (for debugging)
        }
    
    Example:
        >>> steps = [
        ...     {'ui': 0.1, 'da': 0.2, 'do_agent': 0.3, 'do_user': None},
        ...     {'ui': 0.15, 'da': 0.25, 'do_agent': None, 'do_user': 0.35}
        ... ]
        >>> result = calculate_saup_score(steps)
        >>> print(f"SAUP Score: {result['saup_score']:.4f}")
    """
    if config is None:
        config = SAUPConfig()
    
    if not step_data:
        return {
            'saup_score': 0.0,
            'num_steps': 0,
            'mean_weight': 0.0,
            'std_weight': 0.0,
            'mean_ui': 0.0,
            'weights': []
        }
    
    # Calculate weights and weighted squared terms
    weights = []
    weighted_squared_terms = []
    ui_values = []
    
    for step in step_data:
        # Extract metrics
        ui = step.get('ui', 0.0)
        da = step.get('da')
        do_agent = step.get('do_agent')
        do_user = step.get('do_user')
        
        # Calculate situational weight
        w_i = calculate_situational_weight(da, do_agent, do_user, config)
        
        # Calculate weighted squared term: (W_i * U_i)^2
        weighted_term = w_i * ui
        weighted_squared = weighted_term ** 2
        
        weights.append(w_i)
        weighted_squared_terms.append(weighted_squared)
        ui_values.append(ui)
    
    # Calculate SAUP score using weighted RMS formula
    N = len(step_data)
    sum_weighted_squared = sum(weighted_squared_terms)
    saup_score = np.sqrt(sum_weighted_squared / N)
    
    # Calculate statistics
    result = {
        'saup_score': float(saup_score),
        'num_steps': N,
        'mean_weight': float(np.mean(weights)) if weights else 0.0,
        'std_weight': float(np.std(weights)) if weights else 0.0,
        'mean_ui': float(np.mean(ui_values)) if ui_values else 0.0,
        'weights': weights
    }
    
    return result


def calculate_saup_from_trajectory(
    messages: list,
    config: Optional[SAUPConfig] = None
) -> dict:
    """
    Convenience function to calculate SAUP score directly from message objects.
    
    Extracts U_i, Da, and Do metrics from message objects and calculates
    the final SAUP-D trajectory score.
    
    Args:
        messages: List of message objects (AssistantMessage, UserMessage)
                  with uncertainty, da_score, do_score, do_type attributes
        config: SAUP configuration (defaults to SAUPConfig())
        
    Returns:
        dict: SAUP metrics (same as calculate_saup_score)
    
    Example:
        >>> # After running a simulation with --calculate-uncertainty
        >>> from tau2.data_model.simulation import Results
        >>> results = Results.load("simulation.json")
        >>> sim = results.simulations[0]
        >>> saup = calculate_saup_from_trajectory(sim.messages)
        >>> print(f"Trajectory Score: {saup['saup_score']:.4f}")
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
        
        # Extract Da
        da = msg.da_score if hasattr(msg, 'da_score') else None
        
        # Extract Do (split by type)
        do_agent = None
        do_user = None
        if hasattr(msg, 'do_score') and msg.do_score is not None:
            if hasattr(msg, 'do_type'):
                if msg.do_type == 'agent_coherence':
                    do_agent = msg.do_score
                elif msg.do_type == 'user_coherence':
                    do_user = msg.do_score
        
        step_data.append({
            'ui': ui,
            'da': da,
            'do_agent': do_agent,
            'do_user': do_user
        })
    
    return calculate_saup_score(step_data, config)

