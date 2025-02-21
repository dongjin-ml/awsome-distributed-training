import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None

class EnhancedTransformerBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, dropout_rate=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_rate)
        
        # 중간 크기의 피드포워드 네트워크
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # Self-attention
        norm_x = self.layer_norm1(x)
        x_t = torch.transpose(norm_x, 0, 1)
        
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask).bool()
        
        attention_output, _ = self.attention(x_t, x_t, x_t, key_padding_mask=attention_mask)
        attention_output = torch.transpose(attention_output, 0, 1)
        x = x + self.dropout(attention_output)
        
        # Feed-forward
        norm_x = self.layer_norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output
        
        return x

class EnhancedModel(nn.Module):
    def __init__(self, 
                 vocab_size=30000,
                 hidden_size=768,         # 적절한 크기로 조정
                 intermediate_size=3072,   # hidden_size의 4배
                 num_layers=12,           # 적절한 수로 조정
                 num_attention_heads=12,   # hidden_size와 호환되는 수로 조정
                 max_seq_length=512,
                 dropout_rate=0.1):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(hidden_size, intermediate_size, num_attention_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # 중간 레이어 (2개로 조정)
        self.intermediate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            for _ in range(2)
        ])
        
        # 단일 pooler로 변경
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # 간소화된 classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        # Position ids
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Embeddings
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Add token type embeddings
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        x = embeddings + position_embeddings + token_type_embeddings
        x = self.dropout(x)
        
        # Transformer layers와 중간 처리
        hidden_states = []
        for transformer in self.transformer_blocks:
            x = transformer(x, attention_mask)
            hidden_states.append(x)
        
        # 중간 레이어 처리
        for layer in self.intermediate_layers:
            x = x + layer(x)  # residual connection
            hidden_states.append(x)
        
        x = self.layer_norm(x)
        
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        # Pooling
        pooled_output = self.pooler(x[:, 0])
        
        # Classification
        logits = self.classifier(pooled_output)
        
        outputs = ModelOutput(
            logits=logits,
            hidden_states=tuple(hidden_states)
        )
        
        # Loss 계산
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            outputs.loss = loss
        
        return outputs

def get_model():
    model = EnhancedModel(
        hidden_size=1280,          # BERT-large(1024)보다 크게
        intermediate_size=5120,     # hidden_size의 4배
        num_layers=18,             # BERT-large(24)보다는 작게 설정하여 메모리 확보
        num_attention_heads=20,     # hidden_size / 64 ≈ 20
        dropout_rate=0.1
    )
    return model