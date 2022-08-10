from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import MSELoss

from soft_dtw import SoftDTW


@dataclass
class PTGRUOutput:
    decoder_hidden: torch.Tensor
    decoder_output: torch.Tensor
    encoder_hidden: torch.Tensor
    encoder_output: torch.Tensor
    loss: torch.Tensor


class PTGRUEncoder(nn.Module):

    def __init__(self, hidden_size=128, num_layers=4, embed_size=4, device='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)


class PTGRUDecoder(nn.Module):
    def __init__(self, hidden_size=128, embed_size=4, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, embed_size))
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out, hidden = self.gru(x, hidden)
        out = self.linear(out)
        return out, hidden


class PTGRUSeq2Seq(nn.Module):
    def __init__(self, hidden_size=128, embed_size=4, num_layers=4, device=None, loss_type=None):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.loss_type = loss_type

        if self.loss_type == 'mse':
            self.criterion = MSELoss()
        elif self.loss_type == 'sdtw':
            self.criterion = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1, normalize=True)

        self.encoder = PTGRUEncoder(hidden_size=hidden_size, embed_size=embed_size, num_layers=num_layers,
                                    device=device)
        self.decoder = PTGRUDecoder(hidden_size=hidden_size, embed_size=embed_size, num_layers=num_layers)

    def forward(self, inp_tensor, tgt_tensor=None, teacher_forcing=False, max_len=100, dist_threshold=1):
        inp_tensor = inp_tensor.to(self.device)

        batch_size = inp_tensor.size(0)
        encoder_hidden = self.encoder.init_hidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(inp_tensor, encoder_hidden)

        decoder_input = inp_tensor[:, 0:1, :]
        decoder_hidden = encoder_hidden.clone()
        decoder_outputs = []
        loss = None

        if tgt_tensor is not None:
            # compute loss with target tensor
            target_length = tgt_tensor.size(1)
            tgt_tensor = tgt_tensor.to(self.device)
            loss = torch.tensor(0.).to(self.device)

            if teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length - 1):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    loss += torch.mean(self.criterion(decoder_output[:, 0:1, :], tgt_tensor[:, di + 1:di + 2, :]))
                    decoder_input = tgt_tensor[:, di + 1:di + 2, :].detach().clone()  # Teacher forcing
                    decoder_outputs.append(decoder_output)

            else:
                # Without teacher forcing: use its own predictions as the next input
                for di in range(target_length - 1):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    loss += torch.mean(self.criterion(decoder_output[:, 0:1, :], tgt_tensor[:, di + 1:di + 2, :]))
                    decoder_input = decoder_output.detach().clone()  # detach from history as input
                    decoder_outputs.append(decoder_output)
        else:
            # autoregressively generate up to length or zero vector is reached
            goal = inp_tensor[:, -1, :]
            for di in range(max_len - 1):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_input = decoder_output.detach().clone()  # detach from history as input
                decoder_outputs.append(decoder_output)
                # this only works for batch size of 1 todo; change the 1 to an arg parameter
                if torch.linalg.norm(decoder_output[:, -1, :] - goal) <= dist_threshold:
                    break
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = torch.cat([inp_tensor[:, 0:1, :], decoder_outputs], dim=1)
        return PTGRUOutput(decoder_hidden=decoder_hidden,
                           encoder_hidden=encoder_hidden,
                           decoder_output=decoder_outputs,
                           encoder_output=encoder_outputs,
                           loss=loss)
