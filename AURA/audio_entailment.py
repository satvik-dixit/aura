import torch
import os
from msclap import CLAP
from torch.nn import CosineSimilarity


def cosine_similarity(input, target):
    """Calculate cosine similarity between two vectors"""
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()


def get_clap_audio_similarity(hypothesis: str, audio_path: str, clap_model: CLAP) -> float:
    """
    Compute cosine similarity between a hypothesis and an audio file using CLAP text and audio embeddings.
    """
    torch.set_default_dtype(torch.float32)
    hypothesis_emb = clap_model.get_text_embeddings([hypothesis]).to('cuda').squeeze()
    audio_emb = clap_model.get_audio_embeddings([audio_path]).to('cuda').squeeze()
    similarity = cosine_similarity(hypothesis_emb, audio_emb)
    return similarity


def compute_audio_entailment_score(
    hypothesis: str, 
    audio_path: str,
    clap_model: CLAP = None,
    thresh_low: float = 0.25,
    thresh_high: float = 0.55
) -> int:
    """
    Compute CLAP-based audio entailment score using thresholds.
    
    Args:
        hypothesis: The hypothesis about the audio
        audio_path: Local path to audio file
        clap_model: CLAP model instance (optional, will create new one if None)
        thresh_low: Lower threshold for scoring (default: 0.25)
        thresh_high: Upper threshold for scoring (default: 0.55)
        
    Returns:
        Audio entailment score:
        - -1: if CLAP similarity < thresh_low
        - 0: if thresh_low <= CLAP similarity <= thresh_high
        - 1: if CLAP similarity > thresh_high
    """
    if clap_model is None:
        clap_model = CLAP(version='2023', use_cuda=True)
    
    if not os.path.exists(audio_path):
        return 0
    
    clap_similarity = get_clap_audio_similarity(hypothesis, audio_path, clap_model)
    if clap_similarity < thresh_low:
        return -1
    elif clap_similarity <= thresh_high:
        return 0
    else:
        return 1
