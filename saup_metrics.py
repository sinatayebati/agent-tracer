"""
SAUP Metrics Module

This module implements uncertainty quantification metrics for the SAUP
(Situation-Awareness Uncertainty Propagation) framework.

Reference: SAUP paper, Equation 3 - Normalized Entropy
"""

from typing import Optional
import numpy as np


def calculate_normalized_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the Normalized Entropy (single-step uncertainty) for a given response.
    
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
    content = logprobs_object.get('content', [])
    
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
        logprob = token_data.get('logprob')
        
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
    Calculate the standard entropy (unnormalized) for a given response.
    
    This is the total negative log-likelihood across all tokens in the response,
    without normalization by token count.
    
    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.
    
    Returns:
        float: The total entropy (sum of negative log-likelihoods).
    """
    if logprobs_object is None:
        return 0.0
    
    content = logprobs_object.get('content', [])
    
    if not content or len(content) == 0:
        return 0.0
    
    total_neg_log_likelihood = 0.0
    
    for token_data in content:
        logprob = token_data.get('logprob')
        if logprob is not None:
            total_neg_log_likelihood += (-logprob)
    
    return total_neg_log_likelihood


def calculate_token_level_uncertainties(logprobs_object: Optional[dict]) -> list[dict]:
    """
    Calculate token-level uncertainties for detailed analysis.
    
    Returns uncertainty information for each individual token in the response.
    
    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.
    
    Returns:
        list[dict]: List of dictionaries, each containing:
                   - 'token': The token string
                   - 'logprob': The log probability
                   - 'neg_log_likelihood': The negative log-likelihood (uncertainty)
                   - 'probability': The actual probability (exp(logprob))
    """
    if logprobs_object is None:
        return []
    
    content = logprobs_object.get('content', [])
    
    if not content:
        return []
    
    token_uncertainties = []
    
    for token_data in content:
        token = token_data.get('token', '')
        logprob = token_data.get('logprob')
        
        if logprob is not None:
            token_uncertainties.append({
                'token': token,
                'logprob': logprob,
                'neg_log_likelihood': -logprob,
                'probability': np.exp(logprob)
            })
    
    return token_uncertainties


def get_response_statistics(logprobs_object: Optional[dict]) -> dict:
    """
    Get comprehensive statistics about the response uncertainty.
    
    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.
    
    Returns:
        dict: Dictionary containing:
              - 'normalized_entropy': Average negative log-likelihood
              - 'total_entropy': Sum of negative log-likelihoods
              - 'token_count': Number of tokens
              - 'min_uncertainty': Minimum token uncertainty
              - 'max_uncertainty': Maximum token uncertainty
              - 'std_uncertainty': Standard deviation of uncertainties
              - 'mean_probability': Mean token probability
    """
    if logprobs_object is None:
        return {
            'normalized_entropy': 0.0,
            'total_entropy': 0.0,
            'token_count': 0,
            'min_uncertainty': 0.0,
            'max_uncertainty': 0.0,
            'std_uncertainty': 0.0,
            'mean_probability': 0.0,
        }
    
    content = logprobs_object.get('content', [])
    
    if not content:
        return {
            'normalized_entropy': 0.0,
            'total_entropy': 0.0,
            'token_count': 0,
            'min_uncertainty': 0.0,
            'max_uncertainty': 0.0,
            'std_uncertainty': 0.0,
            'mean_probability': 0.0,
        }
    
    uncertainties = []
    probabilities = []
    
    for token_data in content:
        logprob = token_data.get('logprob')
        if logprob is not None:
            uncertainties.append(-logprob)
            probabilities.append(np.exp(logprob))
    
    if not uncertainties:
        return {
            'normalized_entropy': 0.0,
            'total_entropy': 0.0,
            'token_count': 0,
            'min_uncertainty': 0.0,
            'max_uncertainty': 0.0,
            'std_uncertainty': 0.0,
            'mean_probability': 0.0,
        }
    
    return {
        'normalized_entropy': np.mean(uncertainties),
        'total_entropy': np.sum(uncertainties),
        'token_count': len(uncertainties),
        'min_uncertainty': np.min(uncertainties),
        'max_uncertainty': np.max(uncertainties),
        'std_uncertainty': np.std(uncertainties),
        'mean_probability': np.mean(probabilities),
    }

