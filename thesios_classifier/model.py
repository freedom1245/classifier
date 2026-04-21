import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TabularTraceDataset(Dataset):
    def __init__(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.categorical_inputs = categorical_inputs
        self.numeric_inputs = numeric_inputs
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.categorical_inputs[index],
            self.numeric_inputs[index],
            self.labels[index],
        )


def embedding_dim(vocab_size: int) -> int:
    return min(32, max(4, math.ceil(math.sqrt(vocab_size))))


def embedding_dim_v2(vocab_size: int) -> int:
    return min(64, max(8, math.ceil(math.sqrt(vocab_size)) * 2))


def validate_attention_settings(
    attention_dim: int, attention_heads: int, attention_layers: int
) -> None:
    if attention_dim <= 0:
        raise ValueError("attention_dim must be positive.")
    if attention_heads <= 0:
        raise ValueError("attention_heads must be positive.")
    if attention_layers <= 0:
        raise ValueError("attention_layers must be positive.")
    if attention_dim % attention_heads != 0:
        raise ValueError("attention_dim must be divisible by attention_heads.")


class SelfAttentionBlock(nn.Module):
    def __init__(self, attention_dim: int, attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(attention_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 2, attention_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attended_tokens, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.attention_norm(tokens + self.dropout(attended_tokens))
        ff_tokens = self.feed_forward(tokens)
        return self.feed_forward_norm(tokens + self.dropout(ff_tokens))


class ThesiosClassifier(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: list[int],
        numeric_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        attention_dim: int,
        attention_heads: int,
        attention_layers: int,
    ) -> None:
        super().__init__()
        validate_attention_settings(attention_dim, attention_heads, attention_layers)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim(vocab_size))
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.numeric_norm = nn.BatchNorm1d(numeric_dim) if numeric_dim > 0 else None
        self.categorical_projections = nn.ModuleList(
            [
                nn.Linear(embedding_dim(vocab_size), attention_dim)
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.numeric_projection = (
            nn.Linear(numeric_dim, attention_dim) if numeric_dim > 0 else None
        )
        self.attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(attention_dim, attention_heads, dropout)
                for _ in range(attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(attention_dim)
        self.backbone = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, categorical_inputs: torch.Tensor, numeric_inputs: torch.Tensor
    ) -> torch.Tensor:
        tokens = []
        if self.embeddings:
            embedded_tokens = [
                projection(embedding(categorical_inputs[:, index])).unsqueeze(1)
                for index, (embedding, projection) in enumerate(
                    zip(self.embeddings, self.categorical_projections)
                )
            ]
            tokens.extend(embedded_tokens)
        if numeric_inputs.size(1) > 0:
            numeric_features = (
                self.numeric_norm(numeric_inputs)
                if self.numeric_norm is not None
                else numeric_inputs
            )
            tokens.append(self.numeric_projection(numeric_features).unsqueeze(1))

        if not tokens:
            raise ValueError("Model received no input features.")

        combined_tokens = torch.cat(tokens, dim=1)
        for attention_block in self.attention_blocks:
            combined_tokens = attention_block(combined_tokens)

        pooled_features = self.output_norm(combined_tokens.mean(dim=1))
        return self.backbone(pooled_features)


class ThesiosClassifierV2(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: list[int],
        numeric_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        attention_dim: int,
        attention_heads: int,
        attention_layers: int,
    ) -> None:
        super().__init__()
        validate_attention_settings(attention_dim, attention_heads, attention_layers)

        self.numeric_dim = numeric_dim
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, embedding_dim_v2(vocab_size))
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.categorical_projections = nn.ModuleList(
            [
                nn.Linear(embedding_dim_v2(vocab_size), attention_dim)
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.numeric_norm = nn.BatchNorm1d(numeric_dim) if numeric_dim > 0 else None
        self.numeric_weight = (
            nn.Parameter(torch.empty(numeric_dim, attention_dim))
            if numeric_dim > 0
            else None
        )
        self.numeric_bias = (
            nn.Parameter(torch.zeros(numeric_dim, attention_dim))
            if numeric_dim > 0
            else None
        )
        if self.numeric_weight is not None:
            nn.init.normal_(self.numeric_weight, mean=0.0, std=0.02)

        total_feature_tokens = len(categorical_vocab_sizes) + numeric_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, attention_dim))
        self.feature_token_embeddings = nn.Parameter(
            torch.empty(1, total_feature_tokens + 1, attention_dim)
        )
        nn.init.normal_(self.feature_token_embeddings, mean=0.0, std=0.02)
        self.token_dropout = nn.Dropout(dropout)
        self.attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(attention_dim, attention_heads, dropout)
                for _ in range(attention_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(attention_dim)
        self.backbone = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim * 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, categorical_inputs: torch.Tensor, numeric_inputs: torch.Tensor
    ) -> torch.Tensor:
        tokens = []
        if self.embeddings:
            categorical_tokens = [
                projection(embedding(categorical_inputs[:, index])).unsqueeze(1)
                for index, (embedding, projection) in enumerate(
                    zip(self.embeddings, self.categorical_projections)
                )
            ]
            tokens.extend(categorical_tokens)

        if self.numeric_dim > 0:
            numeric_features = (
                self.numeric_norm(numeric_inputs)
                if self.numeric_norm is not None
                else numeric_inputs
            )
            numeric_tokens = (
                numeric_features.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
                + self.numeric_bias.unsqueeze(0)
            )
            tokens.append(numeric_tokens)

        if not tokens:
            raise ValueError("Model received no input features.")

        feature_tokens = torch.cat(tokens, dim=1)
        cls_token = self.cls_token.expand(feature_tokens.size(0), -1, -1)
        combined_tokens = torch.cat([cls_token, feature_tokens], dim=1)
        combined_tokens = combined_tokens + self.feature_token_embeddings[
            :, : combined_tokens.size(1)
        ]
        combined_tokens = self.token_dropout(combined_tokens)

        for attention_block in self.attention_blocks:
            combined_tokens = attention_block(combined_tokens)

        pooled_features = self.output_norm(combined_tokens[:, 0])
        return self.backbone(pooled_features)
