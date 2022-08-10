import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import MSELoss

from soft_dtw import SoftDTW


@dataclass
class PTTransformerOutput:
    decoder_output: torch.Tensor
    loss: torch.Tensor


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 maxlen: int = 250):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[:x.size(0), :]


class PTTransformerSeq2Seq(nn.Module):
    def __init__(self, hidden_size=128, embed_size=4, num_layers=4, nhead=8, device=None, loss_type=None):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.nhead = nhead

        self.loss_type = loss_type

        if self.loss_type == 'mse':
            self.criterion = MSELoss()
        elif self.loss_type == 'sdtw':
            self.criterion = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1, normalize=True)

        self.embedding = nn.Linear(embed_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, batch_first=True, dim_feedforward=hidden_size)
        self.generator = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, embed_size))
        self.pse = PositionalEncoding(hidden_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src_embd, src_mask):
        return self.transformer.encoder(src_embd, src_mask)

    def decode(self, tgt_embd, memory, tgt_mask):
        return self.transformer.decoder(tgt_embd, memory, tgt_mask)

    def forward(self, inp_tensor, tgt_tensor=None, teacher_forcing=True, max_len=100, dist_threshold=1):

        inp_tensor = inp_tensor.to(self.device)
        inp_emb = self.pse(self.embedding(inp_tensor))
        inp_len = inp_tensor.size(1)
        inp_mask = torch.zeros((inp_len, inp_len), device=self.device).type(torch.bool)

        if tgt_tensor is not None:
            # teacher forcing training

            tgt_tensor = tgt_tensor.to(self.device)
            tgt_emb = self.pse(self.embedding(tgt_tensor))
            tgt_len = tgt_tensor.size(1) - 1
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(self.device)

            # shift seq left
            outs = self.transformer(inp_emb, tgt_emb[:, :-1, :], src_mask=inp_mask, tgt_mask=tgt_mask)
            outs = self.generator(outs)

            # compute loss, shift seq right
            outs = torch.cat([tgt_tensor[:, 0:1, :], outs], dim=1)
            loss = torch.mean(self.criterion(outs, tgt_tensor)) * tgt_len  # trainer expects total loss, not per token

            return PTTransformerOutput(decoder_output=outs, loss=loss)
        else:
            # perform greedy decoding
            with torch.no_grad():
                goal = inp_tensor[:, -1, :]
                memory = self.encode(inp_emb, inp_mask).to(self.device)
                output = inp_tensor[:, 0:1, :].clone()
                for i in range(max_len - 1):
                    tgt_mask = self.generate_square_subsequent_mask(output.size(1)).to(self.device)
                    decoder_out = self.decode(self.pse(self.embedding(output)), memory, tgt_mask)
                    decoder_out = decoder_out[:, -1, :].unsqueeze(1)
                    decoder_out = self.generator(decoder_out)
                    output = torch.cat([output, decoder_out], dim=1)
                    # this only works for batch size of 1 todo; change the 1 to an arg parameter
                    if torch.linalg.norm(decoder_out[:, -1, :] - goal) <= dist_threshold:
                        break

                return PTTransformerOutput(decoder_output=output, loss=None)
