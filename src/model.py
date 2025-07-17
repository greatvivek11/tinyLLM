import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import BLOCK_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT from config
from .config import BLOCK_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT, DEVICE

class MultiHeadSelfAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    This module allows the model to attend to different parts of the input sequence
    simultaneously, capturing various relationships.
    """
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads # Dimension of each head's key/query/value
        self.num_heads = num_heads

        # Linear layers for Key, Query, Value projections for all heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Output linear layer after concatenating head outputs
        self.wo = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: ensures that tokens only attend to previous tokens.
        # This is crucial for language modeling to prevent "peeking" at future tokens.
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape # Batch, Time (sequence length), Channels (d_model)

        # 1. Project inputs to Q, K, V for all heads
        # Shape: (B, T, C) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        q = self.wq(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Compute attention scores (QK^T)
        # (B, num_heads, T, d_k) @ (B, num_heads, d_k, T) -> (B, num_heads, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))

        # 3. Apply causal mask
        # Set attention scores to a very small negative number for future tokens,
        # so they become 0 after softmax.
        attn_scores = attn_scores.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))

        # 4. Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5. Multiply weights by Values (AV)
        # (B, num_heads, T, T) @ (B, num_heads, T, d_k) -> (B, num_heads, T, d_k)
        out = attn_weights @ v

        # 6. Concatenate heads and apply final linear layer
        # (B, num_heads, T, d_k) -> (B, T, num_heads * d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.wo(out))
        return out

class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward network with a ReLU activation.
    This processes information independently for each token position.
    """
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), # Expansion layer
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model), # Projection back to d_model
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    A single Transformer block, combining Multi-Head Self-Attention and a Feed-Forward network.
    Includes Layer Normalization and Residual Connections (Add & Norm).
    """
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model) # Normalizes across the feature dimension
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Residual connection 1: x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln1(x))
        # Residual connection 2: x + FeedForward(LayerNorm(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyLLM(nn.Module):
    """
    The complete Tiny Language Model.
    It comprises token embeddings, positional embeddings,
    a stack of Transformer blocks, and a final linear layer (LM head).
    """
    def __init__(self, vocab_size): # Add vocab_size as a parameter
        super().__init__()
        # Token embeddings: maps input token IDs to dense vectors.
        self.token_embedding_table = nn.Embedding(vocab_size, D_MODEL) # Use vocab_size parameter
        # Positional embeddings: adds information about the token's position in the sequence.
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, D_MODEL)
        # Stack of Transformer blocks.
        self.blocks = nn.Sequential(*[
            TransformerBlock(D_MODEL, NUM_HEADS, DROPOUT) for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(D_MODEL) # Final layer norm
        # Language Model head: projects the output of the Transformer blocks
        # back to the vocabulary size to get logits for each possible next token.
        self.lm_head = nn.Linear(D_MODEL, vocab_size) # Use vocab_size parameter

        # Initialize weights (optional, but good practice)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape # Batch size, Sequence length

        # Get token and positional embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, D_MODEL)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, D_MODEL)
        x = tok_emb + pos_emb # (B, T, D_MODEL) - Sum token and positional embeddings

        # Pass through Transformer blocks
        x = self.blocks(x) # (B, T, D_MODEL)
        x = self.ln_f(x) # (B, T, D_MODEL)

        # Project to vocabulary size to get logits
        logits = self.lm_head(x) # (B, T, VOCAB_SIZE)

        loss = None
        if targets is not None:
            # Reshape for cross-entropy
            # We need to align the logits with the targets.
            # The logit at time step `t` is a prediction for the token at `t+1`.
            # So, we should compare logits[B, t, :] with targets[B, t].
            # The way your get_batch is structured, `targets` is already idx shifted by one.
            B, T, C = logits.shape
            logits_for_loss = logits.view(B * T, C)
            targets_for_loss = targets.view(B * T)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    @torch.no_grad() # Disable gradient calculation for inference.
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2):
        """
        Generates new text based on an initial sequence (idx).
        The model predicts the next token, adds it to the sequence, and repeats.
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens (model's context window).
            # This is crucial for long generations to keep context within model's limits.
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions (logits) for the next token.
            logits, loss = self(idx_cond)
            # Focus only on the last time step (the predicted next token).
            logits = logits[:, -1, :] # (B, VOCAB_SIZE)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(idx.shape[0]): # Iterate over batch
                    for prev_token_idx in set(idx[i].tolist()): # Get unique previous tokens
                        logits[i, prev_token_idx] /= repetition_penalty

            logits = logits / temperature # Apply temperature
            
            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Optionally apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep at least one token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution to get the next token ID.
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the sampled token to the sequence.
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
