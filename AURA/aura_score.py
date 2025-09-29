from typing import Dict, Any
import torch
from .llm_score import compute_llm_score, generate_hypothesis
from .audio_entailment import compute_audio_entailment_score


def compute_aura_score(
    question: str,
    reference: str,
    response: str,
    audio_path: str,
    openai_api_key: str,
    clap_model=None,
    w: float = 0.1,
    thresh_low: float = 0.25,
    thresh_high: float = 0.55,
    audio_input: bool = True
) -> Dict[str, Any]:
    """
    Compute the complete AURA score combining LLM-based evaluation and CLAP-based audio entailment.
    
    Args:
        question: The question about the audio
        reference: The reference/gold answer
        response: The model's predicted answer
        audio_path: Local path to audio file
        openai_api_key: OpenAI API key for LLM scoring
        clap_model: CLAP model instance (optional, will create new one if None)
        w: Weight for audio entailment score (default: 0.1)
        thresh_low: Lower threshold for CLAP scoring (default: 0.25)
        thresh_high: Upper threshold for CLAP scoring (default: 0.55)
        audio_input: Whether to use audio entailment (default: True, auto-detects CUDA)
        
    Returns:
        Dictionary containing:
        - aura_score: Final combined score
    """
    # Compute LLM score
    llm_score = compute_llm_score(question, reference, response, openai_api_key)
    
    # Normalize LLM score: (LLM-1)/2 so 3,2,1 becomes 1,0.5,0
    normalized_llm_score = (llm_score - 1) / 2
    
    # Check CUDA availability and audio_input setting
    if audio_input and torch.cuda.is_available():
        # Generate hypothesis
        hypothesis = generate_hypothesis(question, response, openai_api_key)
        
        # Compute audio entailment score
        audio_entailment_score = compute_audio_entailment_score(
            hypothesis=hypothesis,
            audio_path=audio_path,
            clap_model=clap_model,
            thresh_low=thresh_low,
            thresh_high=thresh_high
        )
        
        # Combined score: normalized_llm_score + w * audio_entailment_score
        raw_aura_score = normalized_llm_score + w * audio_entailment_score
        # Normalize to 0-1
        aura_score = (raw_aura_score + w) / (1 + 2*w)
    else:
        # Audio off mode - use only LLM score
        print("Audio off mode - using only LLM score")
        aura_score = normalized_llm_score
    
    return {
        'aura_score': aura_score
    }
