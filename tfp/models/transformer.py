import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(torch.nn.Module):
    def __init__(self, n_joints, n_inp, n_head, n_hid, n_layers, dropout=0.5, residual_velocities=False):
        super(TransformerModel, self).__init__()
        self.n_joints = n_joints
        self.n_inp = n_inp
        self.res_vel = residual_velocities
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layer = nn.TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.encoder = nn.Linear(n_joints*3, n_inp)
        self.decoder = nn.Linear(n_inp, n_joints*3)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def forward(self, src):
        batch_size = src.size(0)
        src = src.view(batch_size, -1, self.n_joints*3)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        transformer_input = self.encoder(src) * math.sqrt(self.n_inp)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_output = self.transformer(transformer_input, self.src_mask)
        output = self.decoder(transformer_output)
        if self.res_vel:
            output = output + src
        return output.view(batch_size, -1, self.n_joints, 3)
    
    def calculate_loss(self, output, target):
        loss = nn.MSELoss()
        return loss(output, target)