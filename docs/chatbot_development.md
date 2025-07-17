# Building a Conversational AI Chatbot with TinyLLM

This document outlines the conceptual and practical steps required to evolve our `TinyLLM` from a basic next-word predictor into a conversational AI chatbot capable of understanding prompts and generating coherent stories or responses. We will cover the fine-tuning process and introduce `vLLM` for optimized inference.

## Phase 1: Base Model (Pre-training)

Our current `TinyLLM` model, trained on the `TinyStories` dataset, serves as a foundational "pre-trained" language model. In a real-world scenario, this pre-training phase would involve training a much larger model (e.g., a few billion parameters) on a massive, diverse corpus of text (like the entire internet, books, articles, etc.). The goal of this phase is for the model to learn fundamental language patterns, grammar, factual knowledge, and basic reasoning abilities.

The `TinyLLM` project, despite its small scale, mimics this initial pre-training step by teaching the model to predict the next token in a sequence based on the `TinyStories` dataset.

## Phase 2: Fine-Tuning for Conversational Ability

To transform a pre-trained language model into a conversational AI that can understand and respond to instructions (like "Create a story of Cinderella in India"), a crucial second phase called **fine-tuning** is required.

### Objective of Fine-tuning

Fine-tuning teaches the model to:
*   **Understand Instructions:** Interpret user prompts as commands or questions.
*   **Generate Relevant Responses:** Produce outputs that directly address the instruction, rather than just continuing the input text.
*   **Adopt a Conversational Style:** Learn to engage in dialogue, maintain context, and potentially exhibit a persona.

### Dataset Selection for Fine-tuning

For fine-tuning, you need datasets specifically designed for instruction-following or dialogue. These datasets typically consist of pairs of `(instruction, response)` or `(conversation history, next reply)`.

Here are some recommended datasets from Hugging Face:

1.  **For Instruction Following (General Purpose):**
    *   **[`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k):** A high-quality, human-generated dataset of ~15,000 instruction-response pairs. Excellent for teaching a model to follow various commands (e.g., summarization, brainstorming, question answering).
    *   **[`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca):** A much larger dataset derived from GPT-4 outputs, designed to replicate the instruction-following capabilities of powerful proprietary models.

2.  **For Dialogue and Chat (Conversational Flow):**
    *   **[`blended_skill_talk`](https://huggingface.co/datasets/blended_skill_talk):** Focuses on conversational skills, including empathy, personality, and knowledge integration. Useful for building more "human-like" chatbots.
    *   **[`daily_dialog`](https://huggingface.co/datasets/daily_dialog):** A dataset of everyday human conversations, good for learning natural dialogue patterns and turn-taking.

### Data Preparation for Fine-tuning

The `src/data_utils.py` module would need significant modifications to handle these new datasets. Instead of simply tokenizing raw text and grouping into blocks, you would need to:

1.  **Load the chosen dataset:** Use `load_dataset('your_chosen_dataset_name')`.
2.  **Format the data:** Convert the dataset's structure into a format suitable for instruction-tuning. A common approach is to concatenate the instruction and the desired response, often with special tokens to delineate roles (e.g., `[INST] {instruction} [/INST] {response}`).
    *   Example format for a single turn:
        ```
        "text": "User: Create a story about a brave knight.\nAssistant: Once upon a time, there was a brave knight named Sir Reginald..."
        ```
    *   For multi-turn conversations, you might concatenate turns:
        ```
        "text": "User: Hello!\nAssistant: Hi there! How can I help you today?\nUser: Tell me a story.\nAssistant: Once upon a time..."
        ```
3.  **Tokenize and Group:** Apply the tokenizer to these formatted strings. The `group_texts` function might still be useful for chunking, but the `input_ids` and `labels` would now represent the instruction-response pairs.

### Modifying the Training Script (`src/main_train.py`)

The core training loop in `src/main_train.py` would remain similar, but key considerations for fine-tuning include:

*   **Learning Rate:** Often, a smaller learning rate is used for fine-tuning compared to pre-training, as you're making smaller adjustments to an already capable model.
*   **Epochs:** Fine-tuning typically requires fewer epochs than initial pre-training.
*   **Loss Calculation:** Ensure the loss is calculated correctly over the formatted instruction-response pairs.
*   **Model Saving:** Save the fine-tuned model to a new path (e.g., `models/tinystories_chatbot_v1.pth`) to distinguish it from the base model.

## Phase 3: Optimized Inference with vLLM

Once your model is fine-tuned, `vLLM` can significantly improve its inference performance, especially for serving multiple requests concurrently.

### What is vLLM?

`vLLM` is a high-throughput and low-latency inference engine for large language models. It achieves this through:
*   **PagedAttention:** An optimized attention algorithm that manages key-value cache efficiently.
*   **Continuous Batching:** Processes requests in a continuous batch, maximizing GPU utilization.

### Installation

You can install `vLLM` via pip:
```bash
pip install vllm
```
Note: `vLLM` requires a CUDA-enabled GPU.

### Model Compatibility and Inference with vLLM

Integrating a custom PyTorch model like `TinyLLM` with `vLLM` can be complex, as `vLLM` is primarily designed for Hugging Face `transformers` models.

**General Approach:**

1.  **Save Model in Hugging Face Format:** The most straightforward way to use `vLLM` is to save your `TinyLLM` model in a format that Hugging Face `transformers` can load (e.g., by creating a `PreTrainedModel` wrapper around your `TinyLLM` and saving it with `save_pretrained`). This would involve:
    *   Defining a `TinyLLMConfig` class.
    *   Making `TinyLLM` inherit from `transformers.PreTrainedModel`.
    *   Saving the model and tokenizer using `model.save_pretrained("path/to/hf_model")` and `tokenizer.save_pretrained("path/to/hf_model")`.

2.  **Load and Serve with vLLM:**
    Once saved in the Hugging Face format, you can use `vLLM`'s `LLM` class:

    ```python
    from vllm import LLM, SamplingParams

    # Load the model from the Hugging Face compatible path
    llm = LLM(model="path/to/hf_model", trust_remote_code=True, dtype="auto")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

    prompts = [
        "Create a story about a magical forest.",
        "Tell me about a brave little mouse.",
        "Write a short tale about a talking cat."
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    ```

3.  **Serving as an API (Optional):**
    `vLLM` can also run as a local server, exposing a compatible OpenAI API endpoint:
    ```bash
    python -m vllm.entrypoints.api_server --model path/to/hf_model
    ```
    You could then send requests to `http://localhost:8000/v1/completions` from your `src/main_inference.py` or any other client.

## Challenges and Considerations

*   **Computational Resources:** Fine-tuning even a `TinyLLM` on a conversational dataset will require more computational power and time than the initial pre-training on `TinyStories`. For larger models and datasets, GPUs are essential.
*   **Model Size Limitations:** The `TinyLLM` architecture (with `D_MODEL=128`, `NUM_LAYERS=4`) is very small. While fine-tuning will improve its instruction-following, it may still struggle with complex reasoning, long-form coherent story generation, or nuanced conversational turns due to its limited capacity. It's a great learning exercise, but don't expect GPT-like performance.
*   **Data Quality:** The quality and diversity of your fine-tuning dataset are paramount. A small, high-quality dataset is often better than a large, noisy one.
*   **Prompt Engineering:** Even with a fine-tuned model, the way you phrase your prompts will significantly impact the quality of the generated stories. Experimentation is key.

This detailed guide should provide a clear roadmap for your next steps in building a conversational AI chatbot.
