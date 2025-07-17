# Setup Guide for TinyLLM

This guide provides instructions on how to set up the TinyLLM project, including its dependencies and specific considerations for different hardware configurations.

## Dependencies

The TinyLLM project relies on the following key Python libraries, as specified in `requirements.txt`:

*   **`torch`**: The core PyTorch library for building and training neural networks. This is the foundation of the TinyLLM model.
*   **`transformers`**: From Hugging Face, used for loading the pre-trained tokenizer (`bert-base-uncased`) which handles converting text into numerical tokens and vice-versa.
*   **`datasets`**: From Hugging Face, used for efficiently loading and processing large datasets, specifically the `roneneldan/TinyStories` dataset for training.
*   **`numpy`**: A fundamental package for numerical computing in Python, used for array operations.
*   **`pandas`**: A powerful data manipulation and analysis library, often used for handling tabular data.
*   **`tqdm`**: A fast, extensible progress bar for Python, used to visualize training progress.
*   **`PyYAML`**: A YAML parser and emitter for Python, often used for configuration files (though not explicitly used for config in `tiny_llm.py`, it's a common dependency in ML projects).
*   **`filelock`**: A library for file locking, often used by `transformers` or `datasets` to prevent race conditions when downloading models/datasets.
*   **`huggingface-hub`**: Client library for Hugging Face Hub, used by `transformers` and `datasets` to interact with the model and dataset hub.
*   **`tokenizers`**: Provides fast, state-of-the-art tokenizers, used by `transformers`.

## Setup Instructions

Follow these steps to get TinyLLM running on your system.

### Prerequisites

Ensure you have Python 3.8 or higher and `pip` (Python package installer) installed.

You can check your Python version with:
```bash
python --version
```

And pip version with:
```bash
pip --version
```

If you need to install Python, visit [python.org](https://www.python.org/downloads/).

### 1. Clone the Repository (if applicable)

If you haven't already, clone the TinyLLM repository:
```bash
git clone https://github.com/your-repo/tiny_llm.git # Replace with actual repo URL
cd tiny_llm
```

### 2. Install Dependencies

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

Once your virtual environment is active, install the required packages. **Pay close attention to the PyTorch installation based on your hardware.**

#### For CPU Only

If you do not have a compatible GPU or prefer to run on CPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### For Mac M1/M2/M3 (Apple Silicon with MPS)

If you have an Apple Silicon Mac, you can leverage the Metal Performance Shaders (MPS) backend for accelerated training:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/mps
pip install -r requirements.txt
```
The `tiny_llm.py` script will automatically detect and use `mps` if available.

#### For Windows/Linux with NVIDIA GPU (CUDA)

If you have an NVIDIA GPU, you can use CUDA for significant training speedup. Ensure you have the correct NVIDIA drivers installed. The specific PyTorch and CUDA version you need depends on your GPU's architecture.

Here is a consolidated table summarizing the compatibility of different PyTorch versions with NVIDIA GPU generations as of 2025:

| PyTorch Version     | CUDA Version(s) Supported | NVIDIA GPU Architectures Supported | GPU Generations / Architectures (Code Names)         | Notes                                                                    |
| :------------------ | :------------------------ | :--------------------------------- | :--------------------------------------------------- | :----------------------------------------------------------------------- |
| PyTorch ≤ 1.13      | Up to CUDA 11.x           | Up to sm_80 (Ampere) and earlier   | Pascal (sm_60), Volta (sm_70), Turing (sm_75), Ampere (sm_80) | Older releases; lack latest architecture support                         |
| PyTorch 2.0 to 2.6  | CUDA 11.7 - 12.2          | Up to sm_86 (Ada Lovelace), some Ampere | Ampere (sm_80), Ada Lovelace (sm_86)                 | Full support for 30, 40, and 50 series partially emerging                |
| PyTorch 2.7+ (stable) | CUDA 12.8                 | Includes sm_120 for Blackwell architecture (RTX 50-series) | Ada Lovelace (sm_86), Blackwell (sm_120 - RTX 50 series) | First stable release with official RTX 50 series support and cu128 builds |
| PyTorch Nightly 2.8+ | CUDA 12.8+ nightly        | Latest including enhanced sm_120 support | Latest NVIDIA GPUs including RTX 50-series           | Cutting-edge features, improved Blackwell support; some Triton caveats   |

**Explanation of GPU Architecture Code Names and Compute Capabilities:**

| NVIDIA GPU Architecture | Approx. Launch Year | CUDA Compute Capability (sm_XX) | Examples of GPUs           |
| :---------------------- | :------------------ | :------------------------------ | :------------------------- |
| Pascal                  | 2016                | sm_60                           | GTX 10-series              |
| Volta                   | 2017                | sm_70                           | Tesla V100                 |
| Turing                  | 2018                | sm_75                           | RTX 20-series              |
| Ampere                  | 2020                | sm_80                           | RTX 30-series              |
| Ada Lovelace            | 2022                | sm_86                           | RTX 40-series              |
| Blackwell               | 2024                | sm_120                          | RTX 50-series (e.g., RTX 5070 Ti) |

**Installation Command Example:**

For an RTX 5070 Ti (Blackwell architecture, `sm_120`), you would typically use a PyTorch 2.7+ stable build with CUDA 12.8 (or a nightly build for the latest features):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # For CUDA 12.8
pip install -r requirements.txt
```
**Important**: Always refer to the [official PyTorch installation page](https://pytorch.org/get-started/locally/) for the most up-to-date and precise command for your specific CUDA version and hardware.

The `tiny_llm.py` script will automatically detect and use `cuda` if available.

#### For Windows/Linux with AMD GPU (ROCm)

For AMD GPUs, PyTorch supports ROCm on Linux. Windows support for ROCm is more limited.
*   **Linux (ROCm)**: Follow the official PyTorch ROCm installation instructions: [PyTorch ROCm Installation](https://pytorch.org/get-started/locally/)
    *   After installing PyTorch with ROCm, then run:
        ```bash
        pip install -r requirements.txt
        ```
*   **Windows (AMD GPU)**: If ROCm is not an option or not supported on your Windows setup, you will likely need to fall back to the CPU-only installation method.

### 3. Run the Project

After installing dependencies, you can run the `tiny_llm.py` script:
```bash
python tiny_llm.py
```

The script will:
1.  Load a pre-trained tokenizer.
2.  Attempt to load the `roneneldan/TinyStories` dataset. If it fails (e.g., no internet, Hugging Face Hub issues), it will fall back to a hardcoded text.
3.  Initialize the TinyLLM model.
4.  Check for a pre-trained model checkpoint (`tiny_llm_model.pth`). If found, it loads it; otherwise, it starts training.
5.  After training (or loading), it enters an interactive loop where you can provide prompts and the model will generate text completions.
