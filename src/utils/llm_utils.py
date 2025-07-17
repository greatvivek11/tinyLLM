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
    """Loads the pre-trained model and tokenizer."""
    print("Loading model and tokenizer...")
    _, vocab_size = init_tokenizer()
    model = TinyLLM(vocab_size)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print("Model loaded successfully.")
            return model, vocab_size
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    else:
        print(f"No trained model found at {MODEL_PATH}.")
        return None, None

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
