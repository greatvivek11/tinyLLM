import os
import google.generativeai as genai
import psutil
import platform 
import math
import torch

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("pynvml not found. GPU monitoring will be disabled.")

# Import necessary components from the main project
from ..config import (
    DEVICE, MODEL_PATH, BLOCK_SIZE,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY # Import generation parameters
)
from ..model import TinyLLM
from ..llm_data import init_tokenizer, encode, decode, VOCAB_SIZE

# --- Configuration ---
# Set your Gemini API key here or load from environment variables
# For security, it's recommended to use environment variables
# Example: GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")

if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("Warning: Gemini API key is not set. LLM-as-a-Judge will not work.")
    print("Please set the GEMINI_API_KEY in src/llm_utils.py or as an environment variable.")
else:
    genai.configure(api_key=GEMINI_API_KEY) # Configure google.generativeai

def test_gemini_connection():
    """Tests the connection to the Gemini API."""
    print("\n--- Testing Gemini API Connection ---")
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("Gemini API key is not set. Cannot test connection.")
        return False
    try:
        # Attempt to list models to verify connection and authentication
        list(genai.list_models())
        print("Successfully connected to Gemini API.")
        return True
    except Exception as e:
        print(f"Failed to connect to Gemini API: {e}")
        print("Please check your API key and network connection.")
        return False

def get_resource_usage():
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    gpu_info = "N/A"

    if HAS_PYNVML and torch.cuda.is_available():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Assuming single GPU
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info = (f"GPU: {utilization.gpu}% | "
                        f"VRAM: {memory_info.used / (1024**3):.2f}/{memory_info.total / (1024**3):.2f} GB")
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as error:
            gpu_info = f"GPU Error: {error}"
    elif torch.cuda.is_available():
        gpu_info = "GPU detected, but pynvml not available for detailed monitoring."
    else:
        gpu_info = "No GPU detected."
    
    return f"CPU: {cpu_percent}% | RAM: {ram_percent}% | {gpu_info}"

def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {int(s)}s"

def get_model_size_mb(model_path):
    """Calculates the size of the model file in megabytes."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0

def load_model_and_tokenizer():
    # Initialize tokenizer
    tokenizer, vocab_size = init_tokenizer()

    # Initialize model
    model = TinyLLM(vocab_size)
    model.to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"Loading model from checkpoint: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])  # ✅ Proper load for full checkpoint
        else:
            model.load_state_dict(checkpoint)  # In case it’s a bare state_dict
    else:
        print("No checkpoint found. Returning untrained model.")

    return model, vocab_size


@torch.no_grad()
def generate_text_from_model(
    model, 
    prompt_text, 
    max_new_tokens=MAX_NEW_TOKENS, 
    temperature=TEMPERATURE, 
    top_k=TOP_K, 
    top_p=TOP_P, 
    repetition_penalty=REPETITION_PENALTY
):
    """Generates text using the provided model and prompt."""
    try:
        context_ids = encode(prompt_text)
        if not context_ids:
            print("Error: Prompt could not be encoded into tokens. Please try a different prompt.")
            return ""
        context = torch.tensor(context_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        generated_tokens = model.generate(
            context, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p, 
            repetition_penalty=repetition_penalty
        )[0].tolist()
        
        full_generated_text = decode(generated_tokens)
        return full_generated_text
    except Exception as e:
        print(f"An unexpected error occurred during text generation: {e}")
        return ""

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

def get_gemini_model(model_name=GEMINI_MODEL_NAME):
    """Returns a configured Gemini GenerativeModel instance."""
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("Gemini API key is not set. Cannot get Gemini model.")
        return None
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error getting Gemini model '{model_name}': {e}")
        return None
