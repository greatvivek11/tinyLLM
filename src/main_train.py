import torch
import os
import time
import psutil
import platform
import math  # Import math for cosine scheduler
import matplotlib.pyplot as plt
import csv

from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .config import *
from .model import TinyLLM
from .llm_data import *
from .utils.llm_utils import (
    get_resource_usage,
    format_time,
)

best_val_loss = float("inf")


# Calculates the learning rate based on the current iteration.
def get_lr(it, optimizer, scheduler):
    """Unified learning rate strategy:
    - Linear warmup for first `warmup_iters` steps
    - CosineAnnealingWarmRestarts thereafter (requires `scheduler.step()` outside)
    """
    if it < WARMUP_ITERS:
        lr = LEARNING_RATE * it / WARMUP_ITERS
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr
    else:
        scheduler.step()
        return scheduler.get_last_lr()[0]


# Sets up the training environment, including model, optimizer, scaler, scheduler, and checkpoint loading.
def setup_training():
    """Initializes model, optimizer, scaler, scheduler, and loads checkpoint if available."""
    global VOCAB_SIZE
    start_iter = 0

    # Initialize tokenizer and get VOCAB_SIZE
    _, VOCAB_SIZE = init_tokenizer()

    # Model Instantiation
    model = TinyLLM(VOCAB_SIZE).to(DEVICE)
    print(
        f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters."
    )

    print(f"Current working directory: {os.getcwd()}")
    print(f"Expected checkpoint path: {os.path.abspath(CHECKPOINT_PATH)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=2, eta_min=MIN_LR
    )

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
        )
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            scaler.load_state_dict(checkpoint["scaler"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_iter = checkpoint["iter"] + 1
            print(f"Resumed from iteration {start_iter}")
        else:
            print(
                f"Warning: checkpoint found at {CHECKPOINT_PATH} but missing expected keys. Starting from scratch."
            )

    return model, optimizer, scaler, scheduler, start_iter


# Loads and processes the training and validation datasets.
def load_data():
    """Loads and processes the training and validation datasets."""
    train_data, val_data = load_and_process_dataset()
    return train_data, val_data


# Calculates and prints estimated training time.
def _calculate_and_print_time_estimates(iter, losses, lr, elapsed_time, iter_times):
    """Calculates and logs estimated training time."""
    if iter_times:
        avg_time_per_iter = sum(iter_times) / len(iter_times)
        estimated_total_time = avg_time_per_iter * MAX_ITERS
        estimated_remaining_time = estimated_total_time - elapsed_time

        print(
            f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}"
        )
        print(f"  Time Spent: {format_time(elapsed_time)}")
        print(f"  Estimated Total: {format_time(estimated_total_time)}")
        print(f"  Estimated Remaining: {format_time(estimated_remaining_time)}")
    else:
        print(
            f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}"
        )
        print(f"  Time Spent: {format_time(elapsed_time)}")


# Performs evaluation and logs training progress and resource usage.
def evaluate_and_log(model, train_data, val_data, iter, lr, start_time, iter_times):
    """Performs evaluation and logs training progress and resource usage."""
    global best_val_loss
    losses = estimate_loss(model, train_data, val_data)
    elapsed_time = time.time() - start_time

    # Calculate estimated training time.
    _calculate_and_print_time_estimates(iter, losses, lr, elapsed_time, iter_times)

    # Resource usage
    print(f"  Resources: {get_resource_usage()}")

    print(f"  Best val loss so far: {best_val_loss:.4f}")

    # Save best model
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        # versioned_path = get_next_versioned_model_path(MODEL_PATH, ".pth")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  ✔️ Best model saved with val loss {best_val_loss:.4f}")

    return losses


# Saves the model, optimizer, and scaler states to a checkpoint file.
def save_checkpoint(model, scaler, optimizer, iter):
    """Saves the model checkpoint."""
    try:
        print("Saving checkpoint...")
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "scaler": scaler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iter,
            },
            CHECKPOINT_PATH,
        )
    except Exception as e:
        print(f"Error saving model to {CHECKPOINT_PATH}: {e}")


# Executes the main training loop, including forward/backward passes and updates.
def train_loop(model, optimizer, scaler, scheduler, train_data, val_data, start_iter):
    """Main training loop."""
    print(f"\nStarting training on {DEVICE}...")
    start_time = time.time()
    lrs, token_throughput, iter_times, train_losses, val_losses, perplexities, steps = (
        [],
        [],
        [],
        [],
        [],
        []
    )

    for iter in range(start_iter, MAX_ITERS):
        # Determine and set the learning rate for the current iteration
        lr = get_lr(iter, optimizer, scheduler)
        lrs.append(lr)
        iter_start_time = time.time()


        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = evaluate_and_log(
                model, train_data, val_data, iter, lr, start_time, iter_times
            )

            # Analytics purposes
            perplexity = math.exp(losses["val"])
            train_losses.append(losses["train"].item())
            val_losses.append(losses["val"].item())
            perplexities.append(perplexity)
            steps.append(iter)
            tokens_processed = (iter - start_iter + 1) * BATCH_SIZE * BLOCK_SIZE
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            token_throughput.append(tokens_per_sec)

            # Log to CSV
            with open(TRAINING_LOG_FILE_PATH, mode="a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        iter,
                        losses["train"],
                        losses["val"],
                        perplexity,
                        tokens_per_sec,
                        lr,
                    ]
                )

        xb, yb = get_dynamic_batch("train", train_data, val_data)

        # Enables mixed precision
        with autocast(device_type="cuda"):
            logits, loss = model(xb, yb)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Needed for clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Save checkpoint every N steps
        if iter % CHECKPOINT_INTERVAL == 0 or iter == MAX_ITERS - 1:
            save_checkpoint(model, scaler, optimizer, iter)

        iter_times.append(time.time() - iter_start_time)

    print("\nTraining complete!")

    # Final evaluation after training
    losses = estimate_loss(model, train_data, val_data)
    print(
        f"\nFinal training loss: {losses['train']:.4f}, validation loss: {losses['val']:.4f}"
    )

    final_perplexity = math.exp(losses["val"])
    print(f"Final Perplexity: {final_perplexity:.2f}")


# Orchestrates the entire training process.
def main():
    try:
        # Check if the file exists to decide whether to write headers
        file_exists = os.path.exists(TRAINING_LOG_FILE_PATH)
        with open(TRAINING_LOG_FILE_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "step",
                        "train_loss",
                        "val_loss",
                        "perplexity",
                        "tokens_per_sec",
                        "lr",
                    ]
                )
        model, optimizer, scaler, scheduler, start_iter = setup_training()
        train_data, val_data = load_data()
        train_loop(
            model, optimizer, scaler, scheduler, train_data, val_data, start_iter
        )
    except KeyboardInterrupt:
        print("\n⛔ Training interrupted. Saving last checkpoint...")
        save_checkpoint(model, scaler, optimizer, start_iter)


if __name__ == "__main__":
    main()
