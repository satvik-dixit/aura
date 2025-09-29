# AURA Score

AURA (Audio Understanding and Response Assessment) is a comprehensive evaluation metric for audio question-answering systems that combines LLM-based evaluation with CLAP-based audio entailment scoring.

## Features

- **LLM-based Evaluation**: Uses an LLM to score responses on a 1-3 scale (1=incorrect, 2=ambiguous, 3=correct)
- **Hypothesis Generation**: Uses an LLM to generate hypotheses from question and model response 
- **Audio Entailment**: Uses CLAP to compute hypothesis-audio similarity to determine entailment
- **Combined Scoring**: Integrates both components into a unified AURA score

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from AURA import compute_aura_score

# Set your OpenAI API key
openai_api_key = "your-api-key-here"

# Example data
question = "Is there a dog barking in the audio?"
reference = "yes"
response = "Yes, there is a dog barking in the audio."
audio_path = "/path/to/audio.wav"

# Compute AURA score
scores = compute_aura_score(
    question=question,
    reference=reference,
    response=response,
    audio_path=audio_path,
    openai_api_key=openai_api_key
)

print(f"AURA Score: {scores['aura_score']}")
```

## AQEval

AQEval is our evaluation set for measuring how well metrics align with human judgments for Audio QA. It combines curated subsets of ClothoAQA and OpenAQA with diverse model responses and MTurk annotations, yielding a broad benchmark spanning short to long-form AQA.

