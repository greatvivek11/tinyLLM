import torch
import os
import json
import argparse
from .config import DEVICE, MODEL_PATH, BLOCK_SIZE, EVAL_BATCHES_PERPLEXITY
from .model import TinyLLM # Keep TinyLLM for type hinting if needed
from .llm_data import init_tokenizer, decode, load_and_process_dataset, get_batch
from .utils.llm_utils import test_gemini_connection, get_gemini_model, load_model_and_tokenizer, generate_text_from_model

@torch.no_grad()
def calculate_perplexity(model, val_data):
    """Calculates perplexity on the validation set."""
    print("\nCalculating perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0

    # Evaluate on a subset of batches for faster calculation
    for k in range(EVAL_BATCHES_PERPLEXITY):
        X, Y = get_batch('val', None, val_data) # get_batch expects train_data as first arg, but we only need val_data here
        _, loss = model(X, Y)
        total_loss += loss.item() * X.size(1) # Accumulate loss weighted by sequence length
        total_tokens += X.size(1) # Accumulate total tokens processed

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Perplexity on validation set: {perplexity.item():.2f}")
    return perplexity.item()

def generate_samples(model, num_samples=3):
    """Generates sample stories from a set of prompts."""
    print("\n--- Generating Sample Stories ---")
    prompts = [
        "Once upon a time, in a land far away,",
        "The little robot woke up and saw",
        "A curious cat followed a butterfly and found"
    ]
    
    for i in range(min(num_samples, len(prompts))):
        prompt_text = prompts[i]
        print(f"\nPrompt: {prompt_text}")
        
        generated_text = generate_text_from_model(model, prompt_text) # Use default generation parameters
        
        print("Generated Text:")
        print(generated_text)

def evaluate_with_llm_judge(story):
    """Uses an external LLM (Gemini) to judge the quality of a generated story."""
    print("\n--- Evaluating with LLM-as-a-Judge (Gemini) ---")
    
    model = get_gemini_model()
    if model is None:
        print("Skipping LLM-as-a-Judge because Gemini model could not be initialized.")
        return None

    judge_prompt = f"""
    You are an expert in evaluating children's stories. Please rate the following story on a scale of 1 to 10 for the given criteria.
    Provide your response in a JSON format.

    Story:
    "{story}"

    Criteria:
    1. Coherence: Does the story make logical sense?
    2. Creativity: Is the story original and engaging?
    3. Grammar: Is the story grammatically correct?

    JSON Response Format:
    {{
      "coherence_score": <score>,
      "creativity_score": <score>,
      "grammar_score": <score>,
      "justification": "<brief justification for your scores>"
    }}
    """

    try:
        response = model.generate_content(judge_prompt)
        
        # Print the raw response text for debugging
        print("\nRaw LLM Judge Response:")
        print(response.text)

        # Extract JSON from markdown code block if present
        json_string = response.text.strip()
        if json_string.startswith("```json") and json_string.endswith("```"):
            json_string = json_string[len("```json"): -len("```")].strip()
        
        # Attempt to parse the JSON string
        result = json.loads(json_string) 
        print("LLM Judge Evaluation:")
        print(json.dumps(result, indent=2))
        return result
    except json.JSONDecodeError as e:
        print(f"JSON decoding error from LLM-as-a-Judge: {e}")
        print("The LLM judge did not return valid JSON. Check the raw response above.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during LLM-as-a-Judge evaluation with Gemini: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluation script for TinyLLM.")
    parser.add_argument('--perplexity', action='store_true', help="Run perplexity calculation.")
    parser.add_argument('--samples', action='store_true', help="Run sample generation.")
    parser.add_argument('--judge', action='store_true', help="Run LLM-as-a-Judge evaluation.")
    parser.add_argument('--test-connection', action='store_true', help="Test Gemini API connection.")
    args = parser.parse_args()

    run_all = not (args.perplexity or args.samples or args.judge or args.test_connection)

    if args.test_connection:
        test_gemini_connection()
        return # Exit after testing connection

    model, vocab_size = load_model_and_tokenizer()
    if model is None:
        return

    if run_all or args.perplexity:
        _, val_data = load_and_process_dataset()
        calculate_perplexity(model, val_data)

    if run_all or args.samples:
        generate_samples(model)

    if run_all or args.judge:
        prompt = "A brave knight went on an adventure to"
        story = generate_text_from_model(model, prompt, max_new_tokens=150) # Use default temperature, top_p, repetition_penalty
        evaluate_with_llm_judge(story)

if __name__ == '__main__':
    main()
