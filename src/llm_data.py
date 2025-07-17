import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import BLOCK_SIZE, BATCH_SIZE, DEVICE, EVAL_ITERS, FALLBACK_TEXT, TOKENIZER_MODEL_NAME

# Global tokenizer and VOCAB_SIZE (will be set during initialization)
tokenizer = None
VOCAB_SIZE = None

def init_tokenizer():
    global tokenizer, VOCAB_SIZE
    if tokenizer is None:
        print(f"\nLoading pre-trained tokenizer: {TOKENIZER_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set tokenizer pad_token to eos_token.")
        VOCAB_SIZE = tokenizer.vocab_size
        print(f"Tokenizer loaded. Vocabulary Size: {VOCAB_SIZE}")
    return tokenizer, VOCAB_SIZE

def encode(s):
    global tokenizer
    if tokenizer is None:
        init_tokenizer()
    return tokenizer.encode(s, add_special_tokens=False)

def decode(l):
    global tokenizer
    if tokenizer is None:
        init_tokenizer()
    return tokenizer.decode(l, skip_special_tokens=True)

def get_batch(split, train_data, val_data):
    data = train_data if split == 'train' else val_data
    
    if isinstance(data, torch.Tensor): # Handle fallback text (raw tensor)
        # Safeguard against data being smaller than BLOCK_SIZE + 1
        if len(data) < BLOCK_SIZE + 1:
            # This should not happen with the expanded fallback text, but as a safeguard:
            print(f"Warning: '{split}' data is too short to form a full batch. Repeating data.")
            # Repeat the data to make it long enough
            repeat_times = (BLOCK_SIZE + 1) // len(data) + 1
            data = data.repeat(repeat_times)

        # For raw tensor, we need to ensure we don't go out of bounds when slicing
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    else: # Handle Hugging Face Dataset (dictionary-like)
        ix = torch.randint(len(data), (BATCH_SIZE,))
        batch = [data[i.item()] for i in ix]
        x = torch.stack([item['input_ids'] for item in batch])
        y = torch.stack([item['labels'] for item in batch])
    
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def load_and_process_dataset():
    global VOCAB_SIZE # Ensure VOCAB_SIZE is updated globally
    init_tokenizer() # Initialize tokenizer and set VOCAB_SIZE

    try:
        print("Loading and processing dataset from Hugging Face...")
        dataset = load_dataset('roneneldan/TinyStories', split='train')

        def tokenize_function(examples):
            # Re-initialize tokenizer within the function for robustness in multiprocessing
            # init_tokenizer is idempotent, so it won't reload unnecessarily
            local_tokenizer, _ = init_tokenizer()
            return local_tokenizer(examples["text"], add_special_tokens=True)

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=8, # Use 1 process to avoid multiprocessing issues on Windows
            remove_columns=["text"]
        )
        print("Dataset tokenized.")

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
            result = {
                k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        processed_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            num_proc=8, # Use 1 process to avoid multiprocessing issues on Windows
        )
        print("Dataset processed and grouped.")

        processed_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
        
        split_dataset = processed_dataset.train_test_split(test_size=0.1)
        train_data = split_dataset['train']
        val_data = split_dataset['test']
        
        print(f"Training data samples: {len(train_data)}")
        print(f"Validation data samples: {len(val_data)}")
        return train_data, val_data

    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Falling back to hardcoded text.")
        text = FALLBACK_TEXT
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data
