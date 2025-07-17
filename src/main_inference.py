import torch
import os

from .config import DEVICE, MODEL_PATH
from .model import TinyLLM # Keep TinyLLM for type hinting if needed, but loading is now in utils
from .llm_data import init_tokenizer, encode, decode, VOCAB_SIZE
from .utils.llm_utils import get_model_size_mb, load_model_and_tokenizer, generate_text_from_model

# ASCII Art Logo for TinyLLM
TINYLLM_LOGO = r"""
  _____ _             _     _     __  __ 
 |_   _(_)_ __  _   _| |   | |   |  \/  |
   | | | | '_ \| | | | |   | |   | |\/| |
   | | | | | | | |_| | |___| |___| |  | |
   |_| |_|_| |_|\__, |_____|_____|_|  |_|
                |___/                    
"""

def display_model_card(model, model_path, vocab_size, device):
    """Displays a model card with key details and an example."""
    print("\n" + "="*50)
    print(TINYLLM_LOGO)
    print("                     TinyLLM Model Card")
    print("="*50)
    print("\nAbout:")
    print("  TinyStoriesLLM is a compact language model designed for generating short, coherent stories.")
    print("  It's trained on a small dataset to demonstrate fundamental LLM capabilities.")
    print(f"\nModel Details:")
    print(f"  Vocabulary Size: {vocab_size:,}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {num_params:,}")
    model_size_mb = get_model_size_mb(model_path)
    print(f"  Model File Size: {model_size_mb:.2f} MB")
    print(f"  Device: {device}")

    print("\nExpected Input/Output Example:")
    example_prompt = "Once upon a time, there was a little"
    print(f"  Input Prompt: \"{example_prompt}\"")
    
    generated_text = generate_text_from_model(model, example_prompt) # Use default generation parameters
    if generated_text:
        # Find the first occurrence of the prompt in the generated text and slice from there
        start_index = generated_text.find(example_prompt)
        if start_index != -1:
            generated_text_display = generated_text[start_index:]
        else:
            generated_text_display = generated_text
        print(f"  Generated Output: \"{generated_text_display}\"")
    else:
        print("  (Error generating example.)")
    print("="*50 + "\n")


def main():
    global VOCAB_SIZE # Ensure VOCAB_SIZE is accessible

    model, VOCAB_SIZE = load_model_and_tokenizer()
    if model is None:
        print("Failed to load model. Exiting inference.")
        return

    print(f"Current working directory: {os.getcwd()}")
    print(f"Expected model load path: {os.path.abspath(MODEL_PATH)}")
    
    # Display model card after successful load
    display_model_card(model, MODEL_PATH, VOCAB_SIZE, DEVICE)

    # Interactive Inference
    print("\n--- Interactive Text Generation ---")
    print("Enter a starting prompt (or type 'exit' to quit).")
    print("The model will try to complete your text.")

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == 'exit':
            break

        if not prompt:
            print("Please enter a non-empty prompt.")
            continue

        full_generated_text = generate_text_from_model(model, prompt) # Use default generation parameters
        if full_generated_text:
            print("Generated:", full_generated_text)
        else:
            print("Could not generate text for the given prompt.")

    print("Exiting text generation.")

if __name__ == '__main__':
    main()
