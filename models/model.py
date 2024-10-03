import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
import math


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class ContinuousValueEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(1, d_model * 2)
        self.U = nn.Linear(d_model * 2, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.W(x.unsqueeze(2))
        out = self.tanh(out)
        out = self.U(out)
        return out


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(
            tau.unsqueeze(-1),
            self.f,
            self.out_features,
            self.w,
            self.b,
            self.w0,
            self.b0,
        )


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(
            tau.unsqueeze(-1),
            self.f,
            self.out_features,
            self.w,
            self.b,
            self.w0,
            self.b0,
        )


class VariableEmbedding(nn.Module):
    def __init__(self, d_model, num_variables):
        super().__init__()
        self.embedding = nn.Embedding(num_variables + 1, d_model)

    def forward(self, x):
        return self.embedding(x)


class Embedding(nn.Module):
    def __init__(self, d_model, num_variables):
        super().__init__()

        self.cvs_value = ContinuousValueEmbedding(d_model)
        self.cvs_time = SineActivation(1, d_model)

        self.var_embed = VariableEmbedding(d_model, num_variables + 1)
        self.d_model = d_model
        self.mask_embedding = nn.Embedding(1, d_model).to(DEVICE)
        self.class_embedding = nn.Embedding(1, d_model).to(DEVICE)

    def forward(self, encoder_input, pre_training_mask=None):
        with torch.no_grad():
            time = encoder_input[0] / 1000
        variable = encoder_input[1]
        value = encoder_input[2]

        time_embed = self.cvs_time(time)
        if pre_training_mask is not None:
            pre_training_mask = pre_training_mask.unsqueeze(-1).expand(
                -1, -1, self.d_model
            )
            mask_token = torch.tensor([[0]], dtype=torch.int64).to(DEVICE)
            mask_embed = self.mask_embedding(mask_token).to(DEVICE)
            mask_embed = mask_embed.expand(value.size(0), value.size(1), -1).cuda()
            value_embed = self.cvs_value(value)
            embeds = torch.where(pre_training_mask == 0, value_embed, mask_embed).cuda()
        else:
            embeds = self.cvs_value(value)
        var_emb = self.var_embed(variable)
        embed = time_embed + embeds + var_emb
        B = embed.size(0)
        class_embed = self.class_embedding(
            torch.tensor([[0]], dtype=torch.int64).to(DEVICE)
        )
        class_embed = class_embed.repeat(B, 1, 1)

        embed = torch.cat((embed, class_embed), dim=1)
        return embed




class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_variables, N, use_lstm=False):
        super().__init__()
        self.embedding = Embedding(d_model, num_variables)
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.encoder_blocks = nn.ModuleList(
                [nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True) for _ in range(N)]
            )
        else:
            self.encoder_blocks = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model = d_model, nhead = n_heads, dim_feedforward = d_ff,  batch_first=True) for _ in range(N)]
                # [EncoderBlock(d_model, n_heads, d_ff) for _ in range(N)]
            )
        self.N = N

    def forward(self, encoder_input, mask, pretrain_mask=None):
        time = encoder_input[0]
        variable = encoder_input[1]
        value = encoder_input[2]

        x = self.embedding((time, variable, value), pretrain_mask)

        if self.use_lstm:
            for lstm in self.encoder_blocks:
                x, _ = lstm(x)
        else:
            for block in self.encoder_blocks:
                
                causal_mask = torch.triu(torch.ones((x.size(1), x.size(1)), device=DEVICE), diagonal=1).to(torch.float32)
                x = block(x, src_mask = causal_mask, src_key_padding_mask = mask.squeeze(1), is_causal=True)
                # x = block(x, mask)
        
        return x


class BottleNeckModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_variables, N, use_lstm=False):
        super().__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, num_variables, N, use_lstm=use_lstm)

    def forward(self, x, mask, pre_training_mask=None):
        out = self.encoder(x, mask, pre_training_mask)
        return F.relu(out)


class ForecastModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        return self.fc(x).squeeze(-1)


class PredictionModel(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x, Mask):
        out = self.proj(x[:, -1, :])
        return out


class Model(nn.Module):
    def __init__(self, bottleneck_model, out_model):
        super().__init__()
        self.bottleneck_model = bottleneck_model
        self.out_model = out_model

    def forward(
        self,
        x,
        mask,
        pre_training_mask=None,
        forward_bottleneck=False,
        forward_forecast=False,
    ):
        rep = self.bottleneck_model(x, mask, pre_training_mask)
        forecast = self.out_model(rep, mask)
        return rep, forecast


class ProjectionHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        out = self.fc1(x)
        return out
