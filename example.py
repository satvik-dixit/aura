"""
Example usage of the AURA score evaluation system.
"""

from AURA import compute_aura_score

# Set your OpenAI API key
OPENAI_API_KEY = ""

def main():

    print("=== AURA Score Evaluation ===")
    
    question = "Can you hear a crowd cheering in the audio?"
    reference = "A crowd can be cheering and clapping in the audio."
    response = "Yes"
    audio_path = "assets/crowd_cheering.wav"
    
    try:
        scores = compute_aura_score(
            question=question,
            reference=reference,
            response=response,
            audio_path=audio_path,
            openai_api_key=OPENAI_API_KEY
        )
        
        print(f"Question: {question}")
        print(f"Reference: {reference}")
        print(f"Response: {response}")
        print(f"Results:")
        print(f"  AURA Score: {scores['aura_score']:.4f}")
        
    except Exception as e:
        print(f"Error in complete evaluation: {e}")

    
if __name__ == "__main__":
    main()
