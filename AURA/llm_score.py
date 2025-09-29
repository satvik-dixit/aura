import json
from typing import Dict, Any
from openai import OpenAI


# Define all examples from the original implementation
ALL_EXAMPLES = [
    {
        "question": "Is the source of the sound indoors?",
        "reference_answer": "yes",
        "candidate_answer": "Yes, the audio suggests that the source is inside a room or building, as indicated by the consistent bouncing sounds and lack of external noise like wind.",
        "score": 3,
        "reason": "The candidate answer is correct because it is equivalent to the reference answer"
    },
    {
        "question": "What can you infer about the weather conditions when this audio was recorded?",
        "reference_answer": "There was wind blowing during the recording.",
        "candidate_answer": "The presence of wind noise suggests that it might be a breezy or windy day, possibly in an outdoor setting near water.",
        "score": 3,
        "reason": "The candidate answer is correct because it is equivalent to the reference answer"
    },
    {
        "question": "Is there a dog barking in the audio?",
        "reference_answer": "yes",
        "candidate_answer": "there is a dog barking",
        "score": 3,
        "reason": "The candidate answer is correct because it is equivalent to the reference answer"
    },
    {
        "question": "Does the motor stop?",
        "reference_answer": "No.",
        "candidate_answer": "The audio does not provide information about whether the motor stops or continues running",
        "score": 2,
        "reason": "The candidate answer is ambiguous"
    },
    {
        "question": "is the sound scary?",
        "reference_answer": "no",
        "candidate_answer": "The continuous roar of a running engine and the thunderstorm might be perceived as intense or dramatic, but it depends on individual perceptions and experiences.",
        "score": 2,
        "reason": "The candidate answer is ambiguous"
    },
    {
        "question": "What acoustic features in the audio clip indicate that a machine is running?",
        "reference_answer": "The loud sound in the audio clip indicates that a machine is running.",
        "candidate_answer": "The continuous humming sound indicates the operation of a machine, likely an engine or motor.",
        "score": 2,
        "reason": "The candidate answer is ambiguous because while it is one possible answer, it is incomplete"
    },
    {
        "question": "Are people talking?",
        "reference_answer": "no",
        "candidate_answer": "Yes, there is a faint sound of human speech in the audio.",
        "score": 1,
        "reason": "The candidate answer is incorrect because it contradicts the reference answer"
    },
    {
        "question": "Is someone coughing?",
        "reference_answer": "no",
        "candidate_answer": "Yes, a person is coughing in the audio.",
        "score": 1,
        "reason": "The candidate answer is incorrect because it contradicts the reference answer"
    },
    {
        "question": "What kind of sound do you hear in the background?",
        "reference_answer": "Bird chirping",
        "candidate_answer": "The background sound is a vehicle passing by, which suggests that the scene takes place near a road or path.",
        "score": 1,
        "reason": "The candidate answer is incorrect because it contradicts the reference answer"
    }
]


def get_prompt(question: str, reference_answer: str, predicted_answer: str) -> str:
    """Create the CoT prompt with all 9 examples."""
    base_instruction = '''You are given a question, a reference answer written by experts, and a candidate answer. Please rate the accuracy of the candidate answer for the question considering the reference answer.

Use a scale of 1-3, with 1 indicating an incorrect or irrelevant answer, 2 indicating an ambiguous or incomplete answer, and 3 indicating a correct answer.'''

    prompt = base_instruction + "\n\nHere are some examples:\n"

    # Add all 9 examples with reasoning
    for example in ALL_EXAMPLES:
        prompt += f'''
Question: {example["question"]}
Reference answer: {example["reference_answer"]}
Candidate answer: {example["candidate_answer"]}
Output: The candidate answer is {example["reason"]} and therefore the score is {example["score"]}.
'''

    prompt += f'''
Now evaluate the following: Here is the question: {question}, the reference answer is: {reference_answer}, and the candidate answer is: {predicted_answer}.
Give the rationale before rating. Format your response as a dictionary with a key "score", value either 1, 2 or 3 and a key "reason" with a string value explaining your assessment.
'''

    return prompt


def extract_score_from_text(text: str) -> int:
    """Extract score from text response."""
    # Try to parse as JSON first
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        json_str = text[start:end]
        data = json.loads(json_str)
        if "score" in data:
            return int(data["score"])

    # Look for score patterns in JSON format
    for score in [1, 2, 3]:
        if f'"score": {score}' in text or f'"score":{score}' in text:
            return score

    # Look for "score is X" patterns
    for score in [1, 2, 3]:
        if f"score is {score}" in text.lower():
            return score

    # Default fallback
    return 1


def generate_hypothesis(question: str, response: str, api_key: str) -> str:
    """
    Generate a hypothesis from question and response using GPT-4o.
    
    Args:
        question: The question about the audio
        response: The model's response
        api_key: OpenAI API key
        
    Returns:
        Generated hypothesis
    """
    prompt = f"""Given the following question and response, generate a hypothesis that combines the information from 
both. The hypothesis should be a clear, standalone statement that can be evaluated against audio
content. Ensure the hypothesis captures all relevant details, especially when the response is 
complex or detailed.

Question: Is there a dog barking in the audio?
Response: Yes, there is a dog barking loudly in the background, and you can hear the dog's paws tapping
on the ground.
Hypothesis: A dog is barking loudly in the background, and its paws can be heard tapping on the ground.

Question: What kind of vehicle can you hear?
Response: I can hear a motorcycle engine revving, and there is a high-pitched whine indicating it's a
sport motorcycle.
Hypothesis: A sport motorcycle engine is revving, producing a high-pitched whine.

Question: Are people talking?
Response: No, there are no voices or speech in the audio.
Hypothesis: No people are talking in the audio

Now, for the following:
Question: {question}
Response: {response}
Hypothesis:

Generate a hypothesis that represents what should be true in the audio based on this question-response 
pair. Return only the hypothesis statement without any prefixes or explanations."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()


def evaluate_with_gpt(client: OpenAI, prompt: str) -> int:
    """Evaluate using GPT API."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return extract_score_from_text(response.choices[0].message.content)


def compute_llm_score(question: str, reference_answer: str, predicted_answer: str, api_key: str) -> int:
    """
    Compute LLM-based score for audio QA evaluation.
    
    Args:
        question: The question about the audio
        reference_answer: The reference/gold answer
        predicted_answer: The model's predicted answer
        api_key: OpenAI API key
        
    Returns:
        Score (1-3): 1=incorrect, 2=ambiguous, 3=correct
    """
    client = OpenAI(api_key=api_key)
    prompt = get_prompt(question, reference_answer, predicted_answer)
    return evaluate_with_gpt(client, prompt)
