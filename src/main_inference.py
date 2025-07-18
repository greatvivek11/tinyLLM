import torch
import os

from .config import DEVICE, MODEL_PATH
from .model import TinyLLM # Keep TinyLLM for type hinting if needed, but loading is now in utils
from .llm_data import init_tokenizer, encode, decode, VOCAB_SIZE
from .utils.llm_utils import get_model_size_mb, load_model_and_tokenizer, generate_text_from_model, display_model_card


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
