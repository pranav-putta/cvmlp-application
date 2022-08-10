import random
from collections import deque, namedtuple
from dataclasses import field
from time import sleep

import cv2
import habitat
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from habitat_sim.logging import logger
from mltoolkit.argparser import argclass, parse_args
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import dynamic_obstacle_task
import static_obstacle_task

static_obstacle_task
dynamic_obstacle_task

logger.setLevel('ERROR')
writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actions = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


def reset(env):
    return env.reset()


@argclass
class PointNavArguments:
    state_size: int = field(default=12)
    hidden_size: int = field(default=128)
    history_size: int = field(default=12)
    num_episodes: int = field(default=10000)
    max_t: int = field(default=100)
    gamma: float = field(default=1.0)
    config: str = field(default="pointnav_dynamic.yaml")


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


class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.linear = nn.Sequential(nn.Linear(3 * state_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, action_size))

    def forward(self, state):
        if len(state.size()) == 1:
            state = state[None, :]
        x = self.linear(state)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)

    def act(self, state, t):
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()

        # prohibit stopping when min state size hasn't been hit yet
        """
        if t < len(state):
            new_probs = probs.clone()
            new_probs[:, 3] = 0
            action = Categorical(new_probs).sample()
        """
        return action.item(), model.log_prob(action)


def reinforce(args: PointNavArguments, policy, optimizer, env: habitat.Env, n_episodes=1000,
              print_every=100, eps=0.1):
    scores_deque = deque(maxlen=100)
    scores = []
    success_scores = deque(maxlen=100)
    spl_scores = deque(maxlen=100)
    rwd_scores = deque(maxlen=100)

    avgs = {'success': [], 'spl': [], 'rewards': []}

    for e in range(1, n_episodes):
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda x: 0.9995  # 1 - e / n_episodes,
        )
        saved_log_probs = []
        rewards = []
        observation = reset(env)
        last_dist, heading = observation['pointgoal_with_gps_compass']
        state = [last_dist, heading, -1] * args.state_size
        update_state(args, state, float(last_dist), float(heading), float(-1))

        # Collect trajectory
        t = 0
        diff_history = deque([0. for _ in range(args.history_size)], maxlen=args.history_size)
        dists = []
        moves = []
        while t < args.max_t and not env.episode_over:
            rand = random.uniform(0, 1)
            action, log_prob = policy.act(torch.tensor(state, dtype=torch.float), t)
            # if rand < eps:
            #    action = random.randint(0, 3)
            saved_log_probs.append(log_prob)
            observation = env.step(actions[action])
            dist, heading = observation['pointgoal_with_gps_compass']
            update_state(args, state, float(dist), float(heading), float(action))

            if action == 3:  # stop pressed
                if dist <= 0.2:
                    rewards.append(100)
                else:
                    rewards.append(-(dist ** 2))
            else:
                diff = last_dist - dist
                stationary_punishment = 0.125 - torch.mean(torch.tensor(diff_history))
                if diff < 0:
                    # huge punishemnt if far away, less if closer
                    diff = diff * max(dist, 1)
                else:
                    diff = diff * 3
                rewards.append(diff - stationary_punishment)  # * torch.log(torch.tensor(max_t - t)))
                # rewards.append((diff - 0.1 * e / n_episodes) * max(10 - dist, 1) + exploration)
                last_dist = dist
                diff_history.append(diff)
            dists.append(dist)
            moves.append(action)
            t += 1
        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        metrics = env.get_metrics()
        success_scores.append(metrics['success'])
        spl_scores.append(metrics['spl'])
        rwd_scores.append(sum(rewards))

        avgs['success'].append(np.mean(success_scores).item())
        avgs['spl'].append(np.mean(spl_scores).item())
        avgs['rewards'].append(np.mean(rwd_scores).item())

        # Recalculate the total reward applying discounted factor
        discounts = [args.gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        # Calculate the loss
        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if e % print_every == 0:
            print('Episode {}\tAverage Reward: {:.2f}, Average Success: {:.2f}, Average SPL: {:.2f}'.format(e, np.mean(
                scores_deque), np.mean(success_scores), np.mean(spl_scores)))
            # print(dists)
            # print(moves)

    xs = list(range(len(avgs['success'])))

    plt.subplot(311)
    plt.plot(xs, avgs['success'])
    plt.xlabel('Step')
    plt.ylabel('Ratio Successful')
    plt.title('Success Rate')

    plt.subplot(312)
    plt.plot(xs, avgs['spl'])
    plt.xlabel('Step')
    plt.ylabel('SPL')
    plt.title('SPL')

    plt.subplot(313)
    plt.plot(xs, avgs['rewards'])
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Rewards')

    plt.show()
    return scores


def eval_epoch(args: PointNavArguments, policy, env: habitat.Env):
    observation = reset(env)
    dist, heading = observation['pointgoal_with_gps_compass']
    state = [dist, heading, -1] * args.state_size

    i = 0
    update_state(args, state, float(dist), float(heading), -1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{args.config}_video.mp4', fourcc, float(20), (256, 256), True)

    while not env.episode_over and i < 100:
        video.write(transform_rgb_bgr(observation['rgb']))
        dist, heading = observation['pointgoal_with_gps_compass']
        action, prob = policy.act(torch.tensor(state), i)
        update_state(args, state, float(dist), float(heading), float(action))

        action = actions[action]
        if action == 'STOP':
            # print(f"{i} STOP pressed, but denied")
            # action = 'MOVE_FORWARD'
            pass
        observation = env.step(action)
        sleep(0.1)
        i += 1
        print(i, action, dist, heading)
    print(env.get_metrics())
    video.release()


def update_state(args: PointNavArguments, state: list, dist, heading, action):
    state.pop(0)
    state.pop(0)
    state.pop(0)
    state += [dist / 10, heading / 3, action / 4]


def main():
    args: PointNavArguments = parse_args(PointNavArguments, resolve_config=False)
    env = habitat.Env(config=habitat.get_config(args.config))

    policy_model = Policy(state_size=args.state_size, hidden_size=args.hidden_size, action_size=len(actions))
    optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

    reinforce(args, policy_model, optimizer, env, n_episodes=args.num_episodes)
    eval_epoch(args, policy_model, env)

    writer.close()


if __name__ == "__main__":
    main()
