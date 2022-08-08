from collections import deque, namedtuple
from dataclasses import field
from random import random

import habitat
import torch.nn
from torch import optim, nn

from mltoolkit.argparser import argclass, parse_args

actions = ['MOVE_FORWARD', 'STOP', 'TURN_LEFT', 'TURN_RIGHT']
device = 'cpu'


@argclass
class PointNavArguments:
    state_size: int = field(default=12)
    hidden_size: int = field(default=64)
    max_t: int = field(default=500)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PolicyNetwork(torch.nn.Module):
    def __init__(self, hidden_states=32, n_actions=4):
        super().__init__()

        self.num_sensors = 3  # this is the number of features
        self.hidden_units = hidden_states
        self.n_actions = n_actions
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_states,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Sequential(nn.Linear(in_features=4 * self.hidden_units, out_features=self.hidden_units),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_units, 4))

    def forward(self, x, last_dist, last_heading):
        x = torch.tensor(x)
        x = x.unsqueeze(0)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        if x.shape[-1] > 0:
            _, (hn, cn) = self.lstm(x, (h0, c0))
        else:
            hn, cn = h0, c0
        last_states = torch.tensor([[[last_dist, last_heading, act]] for act in range(self.n_actions)])
        houts = []
        for i in range(last_states.shape[0]):
            out = self.lstm(last_states[i:i+1, :, :], (hn, cn))
            houts.append(out[1][0])
        houts = torch.cat(houts, dim=1)
        out = self.linear(houts.flatten(1)).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def get_screen(observation):
    return torch.from_numpy(observation['rgb']).float().permute((2, 0, 1)).unsqueeze(0)


def select_action(args, policy, state):
    pass


def train(args: PointNavArguments, env: habitat.Env, policy_net, n_actions):
    memory = ReplayMemory(10000)

    num_episodes = 1000
    steps = 0

    running_rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation = env.reset()
        last_dist, last_head = observation['pointgoal_with_gps_compass']
        state = [[]]
        for t in range(args.max_t):
            # Select and perform an action
            policy_net(state, last_dist, last_head)

            # Store the transition in memory
            # memory.push(state, action, next_state, reward)

            # Move to the next state
            # state = next_state


def main():
    env = habitat.Env(
        config=habitat.get_config("pointnav.yaml")
    )
    args: PointNavArguments = parse_args(PointNavArguments)

    policy_model = PolicyNetwork(args.hidden_size)
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-2)
    train(args, env, policy_model, 4)


if __name__ == "__main__":
    main()
