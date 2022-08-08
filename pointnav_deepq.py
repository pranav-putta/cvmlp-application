import math
import random
from collections import deque, namedtuple
from dataclasses import field
from time import sleep

import cv2
import habitat
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from mltoolkit.argparser import argclass, parse_args

writer = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

actions = ['MOVE_FORWARD', 'STOP', 'TURN_LEFT', 'TURN_RIGHT']


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]


@argclass
class PointNavArguments:
    screen_height: int = field(default=256)
    screen_width: int = field(default=256)
    eps_start: float = field(default=0.9)
    eps_end: float = field(default=0.05)
    eps_decay: float = field(default=200)
    batch_size: int = field(default=8)
    gamma: float = field(default=0.999)
    max_t: int = field(default=500)
    target_update: int = field(default=10)


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


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Sequential(nn.Linear(linear_input_size, 128), nn.ReLU(),
                                  nn.Linear(128, outputs))

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.flatten(1))


def eval_epoch(args: PointNavArguments, policy, env: habitat.Env):
    state = [0. for _ in range(args.state_size)]
    observation = env.reset()
    i = 0
    while not env.episode_over:
        cv2.imshow("RGB", transform_rgb_bgr(observation["rgb"]))
        dist, heading = observation['pointgoal_with_gps_compass']
        update_state(state, dist, heading)
        action, prob = policy.act(torch.tensor(state))
        action = actions[action]
        if action == 'STOP':
            print(f"{i} STOP pressed, but denied")
            action = 'MOVE_FORWARD'
        observation = env.step(action)
        sleep(0.1)
        i += 1
        print(action, dist, heading)


def update_state(state: list, dist, heading):
    state.pop(0)
    state.pop(0)
    state += [dist, heading]


def get_screen(observation):
    return torch.from_numpy(observation['rgb']).float().permute((2, 0, 1)).unsqueeze(0)


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    # display.clear_output(wait=True)
    # display.display(plt.gcf())


def train(args: PointNavArguments, env: habitat.Env):
    n_actions = len(actions)

    policy_net = DQN(args.screen_height, args.screen_width, n_actions).to(device)
    target_net = DQN(args.screen_height, args.screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    episode_durations = []

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
    memory = ReplayMemory(10000)

    num_episodes = 1000
    steps = 0

    running_rewards = []

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        observation = env.reset()
        dist, _ = observation['pointgoal_with_gps_compass']
        last_screen = get_screen(observation)
        current_screen = get_screen(observation)
        state = current_screen

        last_distance = dist
        for t in range(args.max_t):
            # Select and perform an action
            action = select_action(args, policy_net, n_actions, steps, state)
            named_action = actions[action]
            observation = env.step(named_action)

            if named_action == 'STOP':
                # distance from goal when stop is reached
                reward, _ = -(observation['pointgoal_with_gps_compass'])
                print(reward)
            else:
                # + reward if distance is lower and - if distance is higher
                reward = last_distance - observation['pointgoal_with_gps_compass'][0]
                last_distance = observation['pointgoal_with_gps_compass'][0]
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(observation)
            if not env.episode_over:
                next_state = current_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(args, memory, policy_net, target_net, optimizer)
            if env.episode_over:
                episode_durations.append(t + 1)
                running_rewards.append(reward)
                # if len(running_rewards) > 100:
                # running_rewards.pop(0)
                plot_durations(running_rewards)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())


def select_action(args: PointNavArguments, policy_net, n_actions, steps_done, state):
    sample = random.random()
    eps_threshold = args.eps_end + (args.eps_start - args.eps_end) * \
                    math.exp(-1. * steps_done / args.eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model(args: PointNavArguments, memory, policy_net, target_net, optimizer):
    if len(memory) < args.batch_size:
        return
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    env = habitat.Env(
        config=habitat.get_config("pointnav.yaml")
    )
    args: PointNavArguments = parse_args(PointNavArguments)
    train(args, env)

    writer.close()


if __name__ == "__main__":
    main()
