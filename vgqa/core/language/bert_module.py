import sys
import json
import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertLayerNorm(nn.Module):
    """Layer normalization module for BERT."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    """Multi-head self-attention module for BERT."""

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention and return context and attention map."""
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        att_map = attention_probs.clone()
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, att_map


class BertSelfOutput(nn.Module):
    """Output projection for BERT self-attention with residual connection."""

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT attention block combining self-attention and output projection."""

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor: torch.Tensor, attention_mask=None):
        """Apply self-attention to input tensor."""
        self_output, att_map = self.self(input_tensor, input_tensor, input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, att_map


class BertIntermediate(nn.Module):
    """Feed-forward intermediate layer for BERT."""

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Output projection for BERT feed-forward layer."""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """BERT layer with self-attention and feed-forward network."""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply BERT layer with self-attention and feed-forward."""
        attention_output, att_map = self.attention(hidden_states)
        intermediate_output = self.hidden_intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, att_map
        

class BertAttention_Cross(nn.Module):
    """BERT cross-attention between query and key-value pairs."""

    def __init__(self, config):
        super(BertAttention_Cross, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention between query and key-value."""
        self_output, att_map = self.self(q, kv, kv)
        attention_output = self.output(self_output, q)
        return attention_output, att_map


class BertLayer_Cross(nn.Module):
    """BERT layer with cross-attention and feed-forward."""

    def __init__(self, config):
        super(BertLayer_Cross, self).__init__()
        self.config = config
        self.attention = BertAttention_Cross(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-attention layer with feed-forward."""
        attention_output, att_map = self.attention(q, kv)
        intermediate_output = self.hidden_intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, att_map


class BertPredictionHeadTransform(nn.Module):
    """Transform layer for BERT prediction head."""

    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = gelu
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    """Language modeling prediction head for BERT."""

    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict vocabulary distribution from hidden states."""
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


