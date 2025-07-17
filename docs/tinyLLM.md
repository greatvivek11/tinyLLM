# TinyLLM Project Overview

TinyLLM is a compact, character-level language model built using PyTorch, demonstrating the core concepts of a Transformer-based architecture. It's designed to be a simplified, educational implementation of a large language model, focusing on clarity and understanding rather than state-of-the-art performance.

The project aims to:
*   Provide a hands-on example of a Transformer architecture.
*   Illustrate key components like Multi-Head Self-Attention, Feed-Forward Networks, Layer Normalization, and Residual Connections.
*   Showcase a basic training loop and text generation (inference) process.
*   Incorporate modern practices like subword tokenization using Hugging Face's `transformers` and dataset loading with `datasets`.

## Project Structure and Script Elaboration

The TinyLLM project has been refactored into a modular structure for better organization and maintainability.

### New Project Structure

```
d:/Projects/tiny_llm/
├── requirements.txt
├── models/
│   └── tinystories_llm_v1.pth  (Trained model checkpoint)
├── results/
│   └── checkpoint-500/
│       └── training_args.bin
├── docs/
│   ├── notes/
│   │   ├── 1.0 AI Fundamentals.md
│   │   ├── 2.0 Configuring LLM.md
│   │   ├── 3.0 Loading and Processing Data.md
│   │   ├── 4.0 Building the LLM Architecture.md
│   │   ├── 5.0 Training the LLM.md
│   │   ├── 6.0 Running LLM Inference.md
│   │   └── 7.0 Evaluating the LLM.md
│   ├── setup.md
│   └── tinyLLM.md
├── archive/
│   └── tiny_llm.py         (Original monolithic script)
└── src/
    ├── __init__.py         # Makes 'src' a Python package
    ├── __pycache__/        # Python bytecode cache (ignored by .gitignore)
    ├── config.py           # Global configuration parameters
    ├── llm_data.py         # Data loading, tokenizer setup, encode/decode, loss estimation
    ├── main_eval.py        # Main script for evaluating the model
    ├── main_inference.py   # Main script for loading model and interactive inference
    ├── main_train.py       # Main script for training the model
    ├── model.py            # All PyTorch nn.Module classes (TinyLLM, Attention, FeedForward, TransformerBlock)
    └── utils/              # Utility functions (e.g., for LLM-as-a-Judge, resource monitoring)
```

### Script Elaboration

The core functionality is now distributed across several files within the `src/` directory:

### `src/config.py`

This file centralizes all global configuration parameters for the TinyLLM project. This includes:

*   **Model Architecture Parameters**: `BLOCK_SIZE` (now `256`), `BATCH_SIZE` (now `32`), `D_MODEL` (now `256`), `NUM_HEADS` (now `8`), `NUM_LAYERS` (now `8`), `DROPOUT`.
*   **Training Parameters**: `LEARNING_RATE`, `MAX_ITERS` (now set to `10000` for extended training), `EVAL_INTERVAL`, `EVAL_ITERS`. New parameters include `WARMUP_ITERS`, `LR_DECAY_ITERS`, `MIN_LR`, and `GRAD_CLIP` for advanced learning rate scheduling and gradient control.
*   **Learning Rate Scheduler and Gradient Clipping Parameters**: `WARMUP_ITERS`, `LR_DECAY_ITERS`, `MIN_LR`, `GRAD_CLIP`.
*   **Evaluation Parameters**: `EVAL_BATCHES_PERPLEXITY`.
*   **Generation Parameters**: `MAX_NEW_TOKENS`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `REPETITION_PENALTY`.
*   **Device Configuration**: Logic to automatically detect and set the `DEVICE` (MPS, CUDA, or CPU).
*   **Data Parameters**: `FALLBACK_TEXT` (for local data fallback) and `TOKENIZER_MODEL_NAME` (now `gpt2`).
*   **Model Path**: `MODEL_PATH` is now set to `models/tinystories_llm_v1.pth` to reflect the new naming convention and directory for trained models.

### `src/model.py`

This module defines the core neural network architecture of the TinyLLM. It contains the PyTorch `nn.Module` classes:

*   **`MultiHeadSelfAttention`**: Implements the self-attention mechanism with causal masking.
*   **`FeedForward`**: A simple two-layer feed-forward network.
*   **`TransformerBlock`**: Combines attention and feed-forward layers with Layer Normalization and residual connections.
*   **`TinyLLM`**: The main model class, integrating token and positional embeddings, a stack of Transformer blocks, and the final language modeling head.
    *   The `generate` method has been enhanced to include `temperature`, `top_k`, `top_p`, and `repetition_penalty` parameters for more controlled and diverse text generation. `temperature` scales the logits before softmax, making the output probabilities flatter (higher temperature) or sharper (lower temperature), which helps in reducing repetitive output. `top_k` limits sampling to the `k` most probable next tokens. `top_p` (nucleus sampling) selects the smallest set of tokens whose cumulative probability exceeds a threshold `p`. `repetition_penalty` discourages the model from repeating tokens.

### `src/llm_data.py`

This module encapsulates all functionalities related to data handling:

*   **Tokenizer Initialization**: `init_tokenizer()` loads the pre-trained tokenizer (`gpt2`) and sets the global `VOCAB_SIZE`. It also ensures `tokenizer.pad_token` is set to `tokenizer.eos_token` for consistent padding.
*   **Encoding/Decoding**: `encode()` and `decode()` functions convert text to token IDs and vice-versa. The `encode` function now uses `add_special_tokens=True` for better compatibility with the `gpt2` tokenizer.
*   **Batch Generation**: `get_batch()` prepares batches of data for training and evaluation, including a safeguard to ensure data is long enough for `BLOCK_SIZE`.
*   **Loss Estimation**: `estimate_loss()` evaluates the model's performance on training and validation sets.
*   **Dataset Loading and Processing**: `load_and_process_dataset()` handles fetching the `TinyStories` dataset, tokenizing it, grouping it into fixed-size blocks, and splitting it into training and validation sets. It uses `num_proc=8` (though often effectively 1 on Windows to avoid multiprocessing issues) and includes the fallback mechanism to hardcoded text if the dataset cannot be loaded.

### `src/main_train.py`

This is the dedicated script for training the TinyLLM model.

*   It imports configurations from `src.config`, model components from `src.model`, and data utilities from `src.llm_data`.
*   It orchestrates the training loop, including model instantiation, optimizer setup, and periodic loss evaluation.
*   It now incorporates a **learning rate scheduler** (`get_lr` function) for dynamic adjustment of the learning rate during training, and **gradient clipping** (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients.
*   The training loop also includes **estimated time calculations** and **resource usage logging** for better monitoring.
*   Crucially, it **only handles training** and saves the trained model's state dictionary to the `MODEL_PATH` specified in `config.py` (e.g., `models/tinystories_llm_v1.pth`). The interactive inference loop has been removed from this file.

### `src/main_inference.py`

This is the dedicated script for performing interactive text generation (inference) using a pre-trained TinyLLM model.

*   It imports necessary components from `src.config`, `src.model`, `src.llm_data`, and `src.utils.llm_utils`.
*   It handles loading the trained model from the `MODEL_PATH` specified in `config.py`. It will prompt the user if the model file is not found, guiding them to train the model first.
*   It includes a `display_model_card` function to show key model details and an example generation.
*   It contains the interactive loop where users can input prompts.
*   The `model.generate` method is called with default generation parameters (including `temperature`, `top_k`, `top_p`, and `repetition_penalty`) to produce more coherent and varied text completions.

### How to Run the Project

1.  **Install Dependencies**: Ensure all required libraries (e.g., `torch`, `transformers`, `datasets`) are installed from `requirements.txt`.
2.  **Train the Model**:
    ```bash
    python -m src.main_train
    ```
    This command will start the training process. It will train the model for `10000` steps and save the `tinystories_llm_v1.pth` file in the `models/` directory. This process will take a significant amount of time depending on your hardware.
3.  **Run Inference**:
    ```bash
    python -m src.main_inference
    ```
    Once the model is trained, this command will load the saved model and allow you to interactively generate text by providing prompts.
