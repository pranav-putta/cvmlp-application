import os.path
import pickle
import random
from collections import deque
from dataclasses import field

import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from mltoolkit.argparser import argclass, parse_args
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@argclass
class TrainingArguments:
    data_path: str = field(default='./trajectory_data.pkl')
    epochs: int = field(default=5)
    batch_size: int = field(default=8)
    hidden_size: int = field(default=512)
    encoder_lr: float = field(default=1e-3)
    decoder_lr: float = field(default=1e-3)
    num_layers: int = field(default=4)
    results_dir: str = field(default='./trajectory_model_outputs/')
    teacher_forcing_ratio: float = field(default=0.5)


class PTDataset(Dataset):

    def process_data(self, data):
        processed = []
        flat = []
        for pair in tqdm.tqdm(data, desc='processing data...'):
            new_pair = []
            for agent_seq in pair:
                new_pair.append([])
                for pos, rot in agent_seq:
                    heading = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_euler('xyz')
                    val = [*pos, heading[1]]
                    new_pair[-1].append(val)
                    flat.append(val)
            processed.append(new_pair)
        self.scaler.fit(flat)
        return processed

    def __init__(self, data):
        self.scaler = StandardScaler()
        self.data = self.process_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_seq, target_seq = self.data[item]
        input_seq, target_seq = self.scaler.transform(input_seq), self.scaler.transform(target_seq)
        return torch.tensor(input_seq, dtype=torch.float), torch.tensor(target_seq, dtype=torch.float)


class PTEncoder(nn.Module):

    def __init__(self, hidden_size=128, num_layers=4, embed_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class PTDecoder(nn.Module):
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


def dist(u, v):
    delta = u - v
    return torch.sqrt(delta @ delta)


def train_batch(input_tensor, target_tensor, inp_lens, tgt_lens, encoder, decoder, enc_sched, dec_sched,
                encoder_optimizer, decoder_optimizer,
                criterion,
                max_length=1000, teacher_forcing_ratio=0.5):
    batch_size = input_tensor.size(0)
    target_length = target_tensor.size(1)

    encoder_hidden = encoder.init_hidden(batch_size)

    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = torch.tensor(0.).to(device)

    _, encoder_hidden = encoder(input_tensor, encoder_hidden)

    decoder_input = input_tensor[:, 0:1, :]

    decoder_hidden = encoder_hidden.clone()

    use_teacher_forcing = True if random.uniform(0, 1) < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length - 1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output[:, 0, :], target_tensor[:, di + 1, :])
            decoder_input = target_tensor[:, di + 1:di + 2, :].detach().clone()  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length - 1):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output[:, 0, :], target_tensor[:, di + 1, :])
            decoder_input = decoder_output.detach().clone()  # detach from history as input

    if target_length > 1:
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        enc_sched.step()
        dec_sched.step()

    return loss.item() / target_length


def collate_fn(paired_batch):
    new_paired_batch = []
    lengths = []
    for batch in zip(*paired_batch):
        lengths.append([ex.size(0) for ex in batch])
        max_len = max([example.size(0) for example in batch])
        new_paired_batch.append(
            [F.pad(input=item, pad=(0, 0, 0, max_len - item.size(0)), mode='constant', value=0.) for item in batch])
    return torch.stack(new_paired_batch[0]), torch.stack(new_paired_batch[1]), lengths[0], lengths[1]


def main():
    args: TrainingArguments = parse_args(TrainingArguments, resolve_config=False)

    with open(args.data_path, 'rb') as f:
        train_data = pickle.load(f)

    train_dataset = PTDataset(train_data)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    encoder = PTEncoder(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    decoder = PTDecoder(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    enc_scheduler = optim.lr_scheduler.LinearLR(encoder_optimizer, total_iters=args.epochs * len(train_loader))
    dec_scheduler = optim.lr_scheduler.LinearLR(decoder_optimizer, total_iters=args.epochs * len(train_loader))
    criterion = MSELoss()
    running_loss = deque(maxlen=1000)
    average_losses = []
    # number of logs
    log_every = len(train_loader) // 25
    for epoch in range(args.epochs):
        for step, (input_seq, target_seq, inp_lens, tgt_lens) in tqdm.tqdm(enumerate(train_loader),
                                                                           total=len(train_loader)):
            loss = train_batch(input_seq, target_seq, inp_lens, tgt_lens, encoder,
                               decoder, enc_scheduler, dec_scheduler, encoder_optimizer,
                               decoder_optimizer, criterion, teacher_forcing_ratio=args.teacher_forcing_ratio)
            running_loss.append(loss)
            if step % log_every == 0:
                avg_loss = sum(running_loss) / len(running_loss)
                print(f'[epoch {epoch + 1}, step {step + 1}]: Loss {avg_loss}')
                if not os.path.exists(args.results_dir):
                    os.mkdir(args.results_dir)
                torch.save(encoder.state_dict(), os.path.join(args.results_dir, 'encoder.pt'))
                torch.save(decoder.state_dict(), os.path.join(args.results_dir, 'decoder.pt'))
                average_losses.append((epoch * len(train_loader) + step, avg_loss))

        plt.plot(*list(zip(*average_losses)))
        plt.title(f'Average Loss, epoch {epoch + 1}')
        plt.show()


if __name__ == '__main__':
    main()
