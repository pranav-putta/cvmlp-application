import numpy as np
from matplotlib import pyplot as plt

log = 'ppo_dynamic_log.log'

success = []
rewards = []
spl = []

with open(log) as f:
    lines = f.read().split('\n')
    for line in lines:
        if 'success' in line:
            success.append(float(line.split('success: ')[1].split(' ')[0]))
            rewards.append(float(line.split('reward: ')[1].split(' ')[0]))
            spl.append(float(line.split('spl: ')[1].split(' ')[0]))

success = np.array(success)
spl = np.array(spl)
rewards = np.array(rewards)

xs = [10 * i for i in range(len(spl))]
plt.figure(figsize=(8, 10), dpi=80)
plt.subplot(311)
plt.plot(xs, success)
plt.xlabel('Step')
plt.ylabel('Ratio Successful')
plt.yticks(np.arange(min(success), max(success)+0.5, 0.5))
plt.title('Success Rate')

plt.subplot(312)
plt.plot(xs, spl)
plt.xlabel('Step')
plt.ylabel('SPL')
plt.yticks(np.arange(min(spl), max(spl)+0.5, 0.5))
plt.title('SPL')

plt.subplot(313)
plt.plot(xs, rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.yticks(np.arange(min(rewards), max(rewards)+0.5, 0.5))
plt.title('Rewards')

plt.show()