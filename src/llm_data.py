import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import (
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE,
    EVAL_ITERS,
    FALLBACK_TEXT,
    TOKENIZER_MODEL_NAME,
)

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

# Prevent tokenizers from logging too much
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global tokenizer and VOCAB_SIZE (will be set during initialization)
_tokenizer = None
VOCAB_SIZE = None


def init_tokenizer():
    global _tokenizer, VOCAB_SIZE
    if _tokenizer is None:
        print(f"\nLoading pre-trained tokenizer: {TOKENIZER_MODEL_NAME}...")
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, use_fast=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")
        VOCAB_SIZE = _tokenizer.vocab_size
        print(f"Tokenizer loaded. Vocabulary Size: {VOCAB_SIZE}")
    return _tokenizer, VOCAB_SIZE


def encode(s):
    global _tokenizer
    if _tokenizer is None:
        init_tokenizer()
    return _tokenizer.encode(s, add_special_tokens=False)


def decode(l):
    global _tokenizer
    if _tokenizer is None:
        init_tokenizer()
    return _tokenizer.decode(l, skip_special_tokens=True)


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data

    if isinstance(data, torch.Tensor):  # Handle fallback text (raw tensor)
        # Safeguard against data being smaller than BLOCK_SIZE + 1
        if len(data) < BLOCK_SIZE + 1:
            print(f"Warning: '{split}' data is too short. Repeating.")
            # Repeat the data to make it long enough
            data = data.repeat((BLOCK_SIZE + 1) // len(data) + 1)
        # For raw tensor, we need to ensure we don't go out of bounds when slicing
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    else:
        ix = torch.randint(len(data), (BATCH_SIZE,))
        batch_items = [data[i.item()] for i in ix]
        x_full = torch.stack([item["input_ids"] for item in batch_items])
        x = x_full[:, :-1].contiguous()
        y = x_full[:, 1:].contiguous()

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def load_and_process_dataset():
    global VOCAB_SIZE  # Ensure VOCAB_SIZE is updated globally
    tokenizer, VOCAB_SIZE = init_tokenizer()

    try:
        print("Loading and processing dataset from Hugging Face...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")

        def tokenize_function(examples):
            return {
                "input_ids": [
                    tokenizer.encode(
                        text,
                        add_special_tokens=True,
                        max_length=BLOCK_SIZE * 2,
                        truncation=True,
                    )
                    for text in examples["text"]
                ]
            }

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=["text"],
            desc="Tokenizing...",
            load_from_cache_file=True,
        )
        print("Dataset tokenized.")

        def group_texts(examples):
            # Flatten and add EOS token between samples
            concatenated_ids = []
            for input_ids in examples["input_ids"]:
                concatenated_ids += input_ids + [tokenizer.eos_token_id] * 2
            # Truncate to multiple of BLOCK_SIZE
            total_length = (len(concatenated_ids) // BLOCK_SIZE) * BLOCK_SIZE
            concatenated_ids = concatenated_ids[:total_length]
            # Chunk into BLOCK_SIZE sequences
            input_chunks = [
                concatenated_ids[i : i + BLOCK_SIZE]
                for i in range(0, total_length, BLOCK_SIZE)
            ]
            return {"input_ids": input_chunks, "labels": input_chunks.copy()}

        processed_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=8,
            desc="Grouping texts...",
            load_from_cache_file=True,
        )
        print("Dataset processed and grouped.")

        processed_dataset.set_format(
            type="torch", columns=["input_ids", "labels"], output_all_columns=True
        )

        split_dataset = processed_dataset.train_test_split(test_size=0.1)
        train_data = split_dataset["train"]
        val_data = split_dataset["test"]

        print(f"Training data samples: {len(train_data)}")
        print(f"Validation data samples: {len(val_data)}")

        return train_data, val_data

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Falling back to hardcoded text.")
        text = FALLBACK_TEXT
        data = torch.tensor(encode(text), dtype=torch.long)
        if len(data) < BLOCK_SIZE * 2:
            data = data.repeat((BLOCK_SIZE * 2) // len(data) + 1)
        n = int(0.9 * len(data))
        return data[:n], data[n:]
