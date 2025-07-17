import torch
import os
import time
import psutil
import platform
import math # Import math for cosine scheduler

from .config import (
    DEVICE, LEARNING_RATE, MAX_ITERS, EVAL_INTERVAL, MODEL_PATH,
    WARMUP_ITERS, LR_DECAY_ITERS, MIN_LR, GRAD_CLIP # Import new config parameters
)
from .model import TinyLLM
from .llm_data import load_and_process_dataset, get_batch, estimate_loss, init_tokenizer, VOCAB_SIZE
from .utils.llm_utils import get_resource_usage, format_time # Import moved utility functions

# Learning rate scheduler function
def get_lr(it):
    # 1) linear warmup for WARMUP_ITERS steps
    if it < WARMUP_ITERS:
        return LEARNING_RATE * it / WARMUP_ITERS
    # 2) if it > LR_DECAY_ITERS, return MIN_LR
    if it > LR_DECAY_ITERS:
        return MIN_LR
    # 3) in between, use cosine decay down to MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 1.0 to 0.0
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def main():
    global VOCAB_SIZE # Ensure VOCAB_SIZE is accessible

    # Initialize tokenizer and get VOCAB_SIZE
    _, VOCAB_SIZE = init_tokenizer()

    # Load and process dataset
    train_data, val_data = load_and_process_dataset()

    # Model Instantiation
    model = TinyLLM(VOCAB_SIZE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params:,} parameters.")
    model.to(DEVICE)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Expected model save path: {os.path.abspath(MODEL_PATH)}")

    # Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"\nStarting training on {DEVICE}...")
    start_time = time.time()
    
    # Variables for estimated time
    iter_times = []
    
    for iter in range(MAX_ITERS):
        # Determine and set the learning rate for the current iteration
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        iter_start_time = time.time()

        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed_time = time.time() - start_time
            
            # Calculate estimated time
            if iter > 0:
                avg_time_per_iter = sum(iter_times) / len(iter_times)
                estimated_total_time = avg_time_per_iter * MAX_ITERS
                estimated_remaining_time = estimated_total_time - elapsed_time
                
                print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}") # Added lr to print
                print(f"  Time Spent: {format_time(elapsed_time)}")
                print(f"  Estimated Total: {format_time(estimated_total_time)}")
                print(f"  Estimated Remaining: {format_time(estimated_remaining_time)}")
            else:
                print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}") # Added lr to print
                print(f"  Time Spent: {elapsed_time:.2f}s")

            # Resource usage
            print(f"  Resources: {get_resource_usage()}")

        xb, yb = get_batch('train', train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) # Apply gradient clipping
        optimizer.step()
        
        iter_times.append(time.time() - iter_start_time)

    print("\nTraining complete!")
    try:
        # Ensure the directory for the model path exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model to {MODEL_PATH}: {e}")

if __name__ == '__main__':
    main()
