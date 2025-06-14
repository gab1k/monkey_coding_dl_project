import torch
import torch.nn as nn
import torch.nn.functional as F

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

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.fc(self.dropout(hidden_cat))
        return output


class SimpleMambaBlock(nn.Module):
    """
    Логика: Проекция -> 1D Свертка -> Активация -> Селективный SSM -> Выходная проекция
    """
    def __init__(self, d_model, d_state, d_conv, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=d_inner, bias=True
        )

        self.x_proj = nn.Linear(d_inner, self.d_state + self.d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(z)
        y = self.out_proj(y)
        
        return y

    def ssm(self, x):
        batch_size, seq_len, d_inner = x.shape
        
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_dbl = self.x_proj(x)
        delta, B_param, C_param = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        ys = []

        for i in range(seq_len):
            delta_i = delta[:, i, :]
            A_i = torch.exp(delta_i.unsqueeze(-1) * A)
            B_i = delta_i.unsqueeze(-1) * B_param[:, i, :].unsqueeze(1)
            
            h = A_i * h + B_i * x[:, i, :].unsqueeze(-1)
            
            y_i = (h @ C_param[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y_i)
        
        y = torch.stack(ys, dim=1)
        y = y + x * D
        
        return y


class CustomMambaClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        self.layers = nn.ModuleList(
            [SimpleMambaBlock(d_model, d_state, d_conv) for _ in range(num_layers)]
        )
        
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        
        pooled_output = x.mean(dim=1)
        return self.fc(pooled_output)
