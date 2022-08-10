import os.path
import pickle
import random
from collections import deque
from dataclasses import field

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from dtw import dtw
from matplotlib import pyplot as plt
from mltoolkit.argparser import argclass, parse_args
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset

from gru_paired_trajectory import PTGRUSeq2Seq
from soft_dtw import SoftDTW
from transformer_paired_trajectory import PTTransformerSeq2Seq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@argclass
class TrainingArguments:
    data_path: str = field(default='./eval_trajectory_hard.pkl')
    epochs: int = field(default=3)
    batch_size: int = field(default=16)
    hidden_size: int = field(default=512)
    lr: float = field(default=1e-3)
    num_layers: int = field(default=4)
    nheads: int = field(default=8)
    results_dir: str = field(default='./trajectory_model_outputs/')
    teacher_forcing_ratio: float = field(default=1)
    num_logs: int = field(default=25)
    model_type: str = field(default='gru')
    mode: str = field(default='eval')
    dth: float = field(default=1)
    loss_type: str = field(default='mse')
    save_name: str = field(default='mse_onlytf')


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


def train_batch(input_tensor, target_tensor, model, optimizer, scheduler, teacher_forcing_ratio=0.5):
    teacher_force = random.uniform(0, 1) < teacher_forcing_ratio
    optimizer.zero_grad()
    model.zero_grad()

    out = model(input_tensor, target_tensor, teacher_force)
    loss = out.loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item() / target_tensor.size(1)


def collate_fn(paired_batch):
    new_paired_batch = []
    lengths = []
    for batch in zip(*paired_batch):
        lengths.append([ex.size(0) for ex in batch])
        max_len = max([example.size(0) for example in batch])
        new_paired_batch.append(
            [F.pad(input=item, pad=(0, 0, 0, max_len - item.size(0)), mode='constant', value=0.) for item in batch])
    return torch.stack(new_paired_batch[0]), torch.stack(new_paired_batch[1]), lengths[0], lengths[1]


def train(args: TrainingArguments):
    # load data
    with open(args.data_path, 'rb') as f:
        train_data = pickle.load(f)

    train_dataset = PTDataset(train_data)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)
    # load model
    model = None
    if args.model_type == 'gru':
        model = PTGRUSeq2Seq(hidden_size=args.hidden_size, embed_size=4, num_layers=args.num_layers, device=device,
                             loss_type=args.loss_type)
    elif args.model_type == 'transformer':
        model = PTTransformerSeq2Seq(hidden_size=args.hidden_size, embed_size=4, num_layers=args.num_layers,
                                     nhead=args.nheads, device=device, loss_type=args.loss_type)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=len(train_loader) * args.epochs)

    # logging utils
    running_loss = deque(maxlen=1000)
    average_losses = []
    log_every = len(train_loader) // args.num_logs

    # do training
    for epoch in range(args.epochs):
        for step, (input_seq, target_seq, inp_lens, tgt_lens) in tqdm.tqdm(enumerate(train_loader),
                                                                           total=len(train_loader),
                                                                           leave=True,
                                                                           position=0):
            loss = train_batch(input_seq, target_seq, model, optimizer, scheduler,
                               teacher_forcing_ratio=args.teacher_forcing_ratio)
            running_loss.append(loss)
            if step % log_every == 0:
                avg_loss = sum(running_loss) / len(running_loss)
                print(f'[epoch {epoch + 1}, step {step + 1}]: Loss {avg_loss}')
                if not os.path.exists(args.results_dir):
                    os.mkdir(args.results_dir)
                torch.save(model.state_dict(), os.path.join(args.results_dir, f'{args.model_type}_{args.save_name}.pt'))
                average_losses.append((epoch * len(train_loader) + step, avg_loss))

        plt.plot(*list(zip(*average_losses)))
        plt.title(f'Running Average Loss')
        plt.xlabel('Step')
        plt.ylabel('MSE Loss')
        plt.show()


def evaluate(args: TrainingArguments):
    args.batch_size = 1  # some evaluation things assume batch size of 1
    with open(args.data_path, 'rb') as f:
        train_data = pickle.load(f)

    eval_dataset = PTDataset(train_data)
    eval_loader = DataLoader(eval_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)
    # load model
    model = None
    if args.model_type == 'gru':
        model = PTGRUSeq2Seq(hidden_size=args.hidden_size, embed_size=4, num_layers=args.num_layers, device=device,
                             loss_type=args.loss_type)
    elif args.model_type == 'transformer':
        model = PTTransformerSeq2Seq(hidden_size=args.hidden_size, embed_size=4, num_layers=args.num_layers,
                                     nhead=args.nheads, device=device, loss_type=args.loss_type)
    model.load_state_dict(
        torch.load(os.path.join(args.results_dir, f'{args.model_type}_{args.save_name}.pt'), map_location=device))

    sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.001, normalize=False)
    ndtw_score = 0
    mse_loss = 0
    spl_score = 0
    dist = lambda x, y: torch.linalg.norm(x - y)
    for step, (input_seq, target_seq, inp_lens, tgt_lens) in tqdm.tqdm(enumerate(eval_loader),
                                                                       total=len(eval_loader)):
        out = model(input_seq, teacher_forcing=False, max_len=max(30, target_seq.size(1)), dist_threshold=args.dth)
        output = out.decoder_output

        # inverse transform
        target_shape = target_seq.shape
        output_shape = output.shape

        target_seq = target_seq.detach().clone()
        output = output.detach().clone()

        target_seq = torch.tensor(eval_dataset.scaler.inverse_transform(target_seq.squeeze()).reshape(target_shape))
        output = torch.tensor(eval_dataset.scaler.inverse_transform(output.squeeze()).reshape(output_shape))

        ndtw_score += np.exp(
            -dtw(target_seq.squeeze(), output.squeeze(), dist=dist)[0] / (args.dth * target_seq.shape[1]))
        cmp_length = min(target_seq.size(1), output.size(1))
        mse_loss += F.mse_loss(target_seq[:, :cmp_length, :], output[:, :cmp_length, :])

        # compute spl
        if torch.linalg.norm(output[:, -1, :] - target_seq[:, -1, :]) <= args.dth:
            actual_path_length = sum([torch.linalg.norm(p1 - p2) for p1, p2 in
                                      zip(target_seq[:, :-1, :], target_seq[:, 1:, :])])
            pred_path_length = sum([torch.linalg.norm(p1 - p2) for p1, p2 in
                                    zip(output[:, :-1, :], output[:, 1:, :])])
            spl_score += actual_path_length / max(actual_path_length, pred_path_length)

    mse_loss /= len(eval_loader)
    ndtw_score /= len(eval_loader)
    spl_score /= len(eval_loader)
    print(f'MSE Loss: {mse_loss}')
    print(f'NDTW Score: {ndtw_score}')
    print(f'SPL Score: {spl_score}')


def main():
    args: TrainingArguments = parse_args(TrainingArguments, resolve_config=False)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    else:
        print(f'{args.mode} not supported!')


if __name__ == '__main__':
    main()
