"""手写 Decoder-only GPT"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from config import (
    DROPOUT,
    LAYER_NORM_EPSILON,
    MODEL_DIR,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    N_POSITIONS,
    VOCAB_SIZE,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_positions, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(n_positions, n_positions, dtype=torch.bool))
        self.register_buffer("bias", mask.view(1, 1, n_positions, n_positions))

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.c_attn(x).split(channels, dim=2)

        q = q.view(batch, length, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~self.bias[:, :, :length, :length], float("-inf"))
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = weights @ v
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.c_proj(output))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_positions, dropout, layer_norm_epsilon):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.attn = CausalSelfAttention(n_embd, n_head, n_positions, dropout)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        n_positions=N_POSITIONS,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        layer_norm_epsilon=LAYER_NORM_EPSILON,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, n_embd),
                "wpe": nn.Embedding(n_positions, n_embd),
                "drop": nn.Dropout(dropout),
                "h": nn.ModuleList(
                    [
                        Block(n_embd, n_head, n_positions, dropout, layer_norm_epsilon)
                        for _ in range(n_layer)
                    ]
                ),
                "ln_f": nn.LayerNorm(n_embd, eps=layer_norm_epsilon),
            }
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, input_ids):
        _, length = input_ids.shape
        if length > self.n_positions:
            raise ValueError(f"序列长度 {length} 超过最大长度 {self.n_positions}")

        positions = torch.arange(length, device=input_ids.device)
        x = self.transformer.wte(input_ids) + self.transformer.wpe(positions)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        return self.lm_head(self.transformer.ln_f(x))

    @classmethod
    def from_pretrained(cls, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        model = cls()
        weights = load_file(model_dir / "model.safetensors", device="cpu")

        with torch.no_grad():
            for name, parameter in model.named_parameters():
                value = weights[name]
                if name.endswith(("c_attn.weight", "c_fc.weight", "c_proj.weight")):
                    value = value.t()
                parameter.copy_(value.float())

        return model
