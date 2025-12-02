from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from nn_utils import softmax

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.variance = 2/(out_features + in_features)
        init.trunc_normal_(self.weight, mean = 0.0, std = math.sqrt(self.variance), a=-3*self.variance, b=3*self.variance)

    def forward(self, x):
        output = torch.einsum("ki,...si->...sk",self.weight, x)
        return output


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model)))
        std = 1.0
        init.trunc_normal_(self.weight, mean = 0.0, std = std, a = -3 * std, b = 3 * std)        


    def forward(self, token_ids):
      return self.weight[token_ids, :]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        output = self._norm(x).to(in_dtype)
        return output * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):             
        assert(dim % 2 == 0)
        super().__init__()
        self.register_buffer("rope_cache", RotaryEmbedding.init_cache(theta, dim, context_length), persistent=False)

    @staticmethod
    def init_cache(theta: float, dim: int, context_length: int):
        indices = torch.arange(0, dim, 2)/dim
        freqs = theta ** -indices
        positions = torch.arange(context_length)
        freqs = torch.einsum("i,k->ik", positions, freqs)
        return torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cache = self.rope_cache[:seq_len] if token_positions is None else self.rope_cache[token_positions]
        cos = cache[..., 0]
        sin = cache[..., 1]
        x = x.reshape(*x.shape[:-1], -1, 2) #batch num_heads seq_len  d_k//2  2

        x = torch.stack((cos*x[..., 0] - sin*x[..., 1],
                         sin*x[..., 0] + cos*x[..., 1]), dim=-1) # batch seq_len d_k//2 2

        return x.flatten(-2)

class BasicsTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = RotaryEmbedding(
            context_length=context_length,
            dim=d_head,
            theta=rope_theta
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    context_length = context_length,
                    positional_encoder=self.positional_encoder,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 context_length: int, 
                 positional_encoder: RotaryEmbedding):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.attn = CausalMultiHeadSelfAttention(d_model, 
                                                 num_heads, 
                                                 context_length,
                                                 positional_encoder)

    def forward(self, x, token_positions=None):
        shortcut = x
        x = self.ln1(x)
        x = self.attn(x, token_positions)
        x = x + shortcut

        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = x + shortcut
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class CausalMultiHeadSelfAttention(nn.Module):
    def generate_mask(self, num_row, num_col):
        ones = torch.ones((num_row, num_col))
        mask = torch.triu(ones, diagonal=1)
        return mask.bool()

    def __init__(self,
                d_model: int, 
                num_heads: int, 
                context_length: int, 
                positional_encoder: RotaryEmbedding):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model//num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.register_buffer("mask", self.generate_mask(context_length, context_length), persistent=False)
        self.softmax = nn.Softmax(dim=-1)
        self.rope = positional_encoder

    def forward(self, x, token_positions):
        batch, seq_len, _ = x.shape
        queries = self.q_proj(x)
        queries = queries.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.rope(queries, token_positions)

        keys = self.k_proj(x)
        keys = keys.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.rope(keys, token_positions)

        values = self.v_proj(x)
        values = values.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        mask = self.mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(1).expand(batch, self.num_heads, seq_len, seq_len)
        attention_weight = torch.einsum("...qd,...kd->...qk", queries, keys)/math.sqrt(self.head_dim)
        attention_weight.masked_fill_(mask, -torch.inf)
        attention_score = self.softmax(attention_weight)
        contexts = torch.einsum("...qk,...kd->...qd", attention_score, values)

        contexts = contexts.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        output = self.output_proj(contexts)

        return output

def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
