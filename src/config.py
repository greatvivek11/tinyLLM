import torch
import os

# --- Configuration Parameters ---
BLOCK_SIZE = 256
BATCH_SIZE = 64
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.1

LEARNING_RATE = 5e-4
MAX_ITERS = 50000 # Increased for more training

# Learning Rate Scheduler and Gradient Clipping Parameters
WARMUP_ITERS = MAX_ITERS//100 # int: Linear LR warmup for first 1% of training
LR_DECAY_ITERS = MAX_ITERS # Should be equal to MAX_ITERS for full decay
MIN_LR = 1e-5 # Minimum learning rate after decay
GRAD_CLIP = 1.0 # Gradient clipping threshold

# --- Evaluation Parameters ---
EVAL_INTERVAL = 1000
EVAL_ITERS = 200
EVAL_BATCHES_PERPLEXITY = 100 # Number of batches to use for perplexity calculation

# --- Generation Parameters ---
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.2

CHECKPOINT_INTERVAL = 1000  # Save model every 1k steps
CHECKPOINT_PATH = "models/checkpoint.pth"
MODEL_PATH="models/tinyStoriesLLM.pth"
TRAINING_LOG_FILE_PATH = os.path.join("src", "analytics", "logs", "training_metrics.csv")


# --- Device Configuration ---
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("Using MPS (Metal Performance Shaders) for training.")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("Using CUDA (NVIDIA GPU) for training.")
else:
    DEVICE = "cpu"
    print("Using CPU for training.")

# --- Data Preparation ---
FALLBACK_TEXT = """
The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step.
To be or not to be, that is the question. All that glitters is not gold. The early bird catches the worm.
This is a longer fallback text to ensure sufficient data for training. We need enough text to fill the BLOCK_SIZE and create multiple batches.
The more diverse the text, the better the model might learn, even with simple data. In the heart of a dense forest, a tiny stream trickled over mossy stones, its water crystal clear.
Sunlight filtered through the canopy, dappling the forest floor with golden light. A gentle breeze rustled the leaves, creating a soft, whispering sound.
Nearby, a family of deer grazed peacefully, their movements graceful and serene. The air was filled with the sweet scent of pine and damp earth.
This extended text provides more than enough data to prevent the validation set from being smaller than the block size, which was the cause of the crash.
By making this text significantly longer, we ensure that both training and validation splits will have sufficient length for processing.
This is a robust solution to the problem of the dataset download failing and the fallback text being too short.
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

# --- Tokenizer Configuration ---
TOKENIZER_MODEL_NAME = 'gpt2'
VOCAB_SIZE = None # Vocab size will be determined by the pre-trained tokenizer
