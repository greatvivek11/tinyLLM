import matplotlib.pyplot as plt
import os
import pandas as pd # Import pandas
from ..config import TRAINING_LOG_FILE_PATH, MAX_ITERS # Import MAX_ITERS for time estimation

def plot_training_metrics(csv_file_path):
    """
    Loads training metrics from a CSV file and generates various plots.

    Args:
        csv_file_path (str): The path to the CSV file containing training metrics.
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at {csv_file_path}")
        return

    # Load metrics from CSV using pandas
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file with pandas: {e}")
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(csv_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Learning Curves
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["train_loss"], label="Train Loss")
    plt.plot(df["step"], df["val_loss"], label="Val Loss")
    plt.title("Learning Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_curves.png"))
    plt.close() # Close plot to free memory

    # Plot 2: Validation Perplexity
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["perplexity"], label="Perplexity", color='purple')
    plt.title("Validation Perplexity")
    plt.xlabel("Iteration")
    plt.ylabel("Perplexity")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "perplexity_curve.png"))
    plt.close() # Close plot to free memory

    # Plot 3: Token Throughput
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["tokens_per_sec"], label="Tokens/sec", color='green')
    plt.title("Token Throughput Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Throughput (tokens/sec)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "token_throughput.png"))
    plt.close() # Close plot to free memory

    # Plot 4: Learning Rate Schedule
    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["lr"], label="Learning Rate", color="red")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_rate_schedule.png"))
    plt.close() # Close plot to free memory

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # Example usage if run directly
    # This assumes training_metrics.csv is in src/analytics/logs/
    plot_training_metrics(TRAINING_LOG_FILE_PATH)
