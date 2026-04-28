import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
    ) -> torch.Tensor:
        if categorical_inputs.numel() == 0:
            inputs = numeric_inputs
        elif numeric_inputs.numel() == 0:
            inputs = categorical_inputs.float()
        else:
            inputs = torch.cat([categorical_inputs.float(), numeric_inputs], dim=1)
        return self.network(inputs)


class EmbeddingMLPClassifier(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: list[int],
        numeric_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        embedding_dim: int = 16,
        numeric_projection_dim: int = 64,
    ) -> None:
        super().__init__()
        # 每个类别字段各自维护 embedding，避免把类别编号误当成连续数值。
        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(vocab_size, min(embedding_dim, max(4, vocab_size // 2)))
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.has_numeric = numeric_dim > 0
        self.numeric_projection_dim = numeric_projection_dim if self.has_numeric else 0
        self.numeric_projection = (
            nn.Sequential(
                nn.Linear(numeric_dim, numeric_projection_dim),
                nn.ReLU(),
            )
            if self.has_numeric
            else None
        )

        embedding_output_dim = sum(
            embedding.embedding_dim for embedding in self.embedding_layers
        )
        # 类别 embedding 与数值投影拼接后，再交给 MLP 做最终分类。
        fusion_dim = embedding_output_dim + self.numeric_projection_dim
        self.network = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []

        if self.embedding_layers:
            # 每一列类别特征单独嵌入，再在特征维上拼接。
            embedded_columns = [
                embedding(categorical_inputs[:, index])
                for index, embedding in enumerate(self.embedding_layers)
            ]
            features.append(torch.cat(embedded_columns, dim=1))

        if self.numeric_projection is not None:
            features.append(self.numeric_projection(numeric_inputs))

        if not features:
            raise ValueError("EmbeddingMLPClassifier requires categorical or numeric features.")

        fused_inputs = features[0] if len(features) == 1 else torch.cat(features, dim=1)
        return self.network(fused_inputs)


class AttentionTabularClassifier(nn.Module):
    def __init__(
        self,
        categorical_vocab_sizes: list[int],
        numeric_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        embedding_dim: int = 16,
        attention_dim: int = 128,
        attention_heads: int = 8,
        attention_layers: int = 2,
    ) -> None:
        super().__init__()
        # 方案 B：把每个类别特征看成一个 token，数值块也投影成一个 token。
        self.embedding_layers = nn.ModuleList(
            [
                nn.Embedding(vocab_size, min(embedding_dim, max(4, vocab_size // 2)))
                for vocab_size in categorical_vocab_sizes
            ]
        )
        self.categorical_projections = nn.ModuleList(
            [
                nn.Linear(embedding.embedding_dim, attention_dim)
                for embedding in self.embedding_layers
            ]
        )
        self.numeric_projection = (
            nn.Linear(numeric_dim, attention_dim)
            if numeric_dim > 0
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Transformer 负责建模“特征与特征之间”的交互关系。
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=attention_layers,
        )
        self.token_norm = nn.LayerNorm(attention_dim)
        self.classifier = nn.Sequential(
            nn.Linear(attention_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        categorical_inputs: torch.Tensor,
        numeric_inputs: torch.Tensor,
    ) -> torch.Tensor:
        tokens: list[torch.Tensor] = []

        if self.embedding_layers:
            for index, embedding in enumerate(self.embedding_layers):
                embedded = embedding(categorical_inputs[:, index])
                projected = self.categorical_projections[index](embedded)
                tokens.append(projected.unsqueeze(1))

        if self.numeric_projection is not None:
            numeric_token = self.numeric_projection(numeric_inputs)
            tokens.append(numeric_token.unsqueeze(1))

        if not tokens:
            raise ValueError("AttentionTabularClassifier requires categorical or numeric features.")

        # 对 token 序列做自注意力编码，再使用均值池化得到样本级表示。
        token_tensor = torch.cat(tokens, dim=1)
        encoded_tokens = self.encoder(token_tensor)
        pooled = encoded_tokens.mean(dim=1)
        normalized = self.token_norm(pooled)
        return self.classifier(normalized)


__all__ = ["MLPClassifier", "EmbeddingMLPClassifier", "AttentionTabularClassifier"]
