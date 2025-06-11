import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_len):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dim_feedforward=embed_dim*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        padding_mask = (x == 0)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.mean(dim=1) 
        x = self.fc(x)
        return x

class MambaClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, mamba_d_state, mamba_d_conv, mamba_expand, num_classes):
        super(MambaClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        mamba_config = MambaConfig(hidden_size=embed_dim, state_size=mamba_d_state, conv_kernel=mamba_d_conv, expand=mamba_expand, vocab_size=vocab_size)
        self.mamba = MambaModel(config=mamba_config)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        padding_mask = (x != 0).long()
        embedded_x = self.embedding(x)
        mamba_outputs = self.mamba(inputs_embeds=embedded_x, attention_mask=padding_mask)
        hidden_states = mamba_outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        output = self.fc(pooled_output)
        return output
        
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(self.dropout(hidden_cat))
        return output
