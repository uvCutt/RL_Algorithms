import random
from collections import deque
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt


class Params:
    device = "cpu"
    env_name = "Pendulum-v1"

    # 动作空间和状态空间以及隐藏层大小
    action_space_size = 0
    state_space_size = 0
    hidden_dim = 128

    # 动作的最大最小值
    action_upper = 0
    action_lower = 0

    # 学习率和折扣率
    actor_lr = 0.001
    critic_lr = 0.001
    gamma = 0.99

    # 探索噪声率和噪声上限
    epsilon = 0.2
    noise_clip = 0.5

    # 软更新的系数
    tau = 0.005

    # 训练次数以及每次最大步长
    epochs = 1000
    episode_len = 200

    # 经验回放池大小
    replay_buffer_size = 1000

    batch_size = 64

    # 更新policy的频率以及保存频率
    policy_update_freq = 5
    save_freq = 10


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        """
        DQN贡献之一: 经验回访池, 打破数据时间相关性
        通用经验回放池，利用队列来进行维护，先入先出
        :param capacity: 经验回放池大小
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        self.buffer.append(transitions)

    def sample(self, batch_size: int) -> Tuple:
        """
        采样
        :param batch_size: 样本数
        :return: 采样结果
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, params: Params = None):
        """
        这里输出不再是各个动作的概率，而是具体的动作是多少，这里用tanh将结果限制在[-1,1]，然后再缩放到原始动作空间
        :param params: 参数
        """
        super(PolicyNet, self).__init__()
        self.params = params
        self.input_linear = nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, params.action_space_size)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = nn.Tanh()(self.output_linear(x))
        x = self.params.action_lower + (self.params.action_upper - self.params.action_lower) * x
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, params: Params = None):
        """
        这里输出的是状态和动作，输出的是这个状态动作的价值
        :param params:
        """
        super(ValueNet, self).__init__()
        self.input_linear = nn.Linear(params.state_space_size + params.action_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, out_features=1)

        self.input_linear_another = nn.Linear(params.state_space_size + params.action_space_size, params.hidden_dim)
        self.hidden_linear_another = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear_another = nn.Linear(params.hidden_dim, out_features=1)

    def forward(self, state, action):
        x = torch.cat(tensors=(state, action), dim=1)

        q = nn.ReLU()(self.input_linear(x))
        q = nn.ReLU()(self.hidden_linear(q))
        q = self.output_linear(q)

        q_another = nn.ReLU()(self.input_linear_another(x))
        q_another = nn.ReLU()(self.hidden_linear_another(q_another))
        q_another = self.output_linear_another(q_another)

        return q, q_another

    def single_q(self, state, action):
        x = torch.cat(tensors=(state, action), dim=1)
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = self.output_linear(x)
        return x


class TD3:
    def __init__(self, params: Params = None, replay_buffer: ReplayBuffer = None):
        self.params = params
        self.actor = PolicyNet(self.params).to(self.params.device)
        self.actor_target = PolicyNet(self.params).to(self.params.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.params.actor_lr)

        self.critic = ValueNet(self.params).to(self.params.device)
        self.critic_target = ValueNet(self.params).to(self.params.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.params.critic_lr)
        self.replay_buffer = replay_buffer

        self.count = 0

    def soft_update(self):
        for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.params.tau) + param.data * self.params.tau)

        for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.params.tau) + param.data * self.params.tau)

    def update(self) -> None:
        """
        Twin表示两个Q, Delay表示策略网络更新频率低于价值网络, 同时更新确实不稳定(观察A2C的奖励曲线), 目标策略平滑, 动作加了个噪声并Clip
        Critic网络输出的是状态和动作，输出的是对当前状态下这个动作的评价
        :return: None
        """
        if len(self.replay_buffer) < self.params.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(self.params.device)
        actions = torch.tensor(actions, dtype=torch.float).view(-1, 1).to(self.params.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.params.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.params.device)
        dones = torch.tensor(dones).view(-1, 1).to(self.params.device)

        # 增加均值为0的高斯噪声, 或者OU_Noise(动作精度要求比较高的任务可能表现会更好)
        with torch.no_grad():
            next_actions = self.actor(states)
            noise = torch.randn_like(actions) * self.params.epsilon
            noise = torch.clamp(noise, -self.params.noise_clip, self.params.noise_clip)
            noise_actions = torch.clamp(noise + next_actions, self.params.action_lower, self.params.action_upper)

            next_q_values = torch.min(*self.critic_target(next_states, noise_actions))
            q_targets = rewards + self.params.gamma * next_q_values * (1 - dones)

        # 计算Critic Loss并更新 这里两个Q都要更新
        cur_targets, cur_targets_another = self.critic(states, actions)
        critic_loss = nn.MSELoss()(cur_targets, q_targets) + nn.MSELoss()(cur_targets_another, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delay
        if not self.count % self.params.policy_update_freq:
            actor_loss = -torch.mean(self.critic.single_q(states, self.actor(states)))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update()

        self.count += 1

    def sample_action(self, state: np.ndarray) -> int:
        """
        获取动作
        :param state: 当前状态
        :return: 动作
        """
        state = torch.tensor(state).to(self.params.device)
        action = self.actor(state)

        return action.detach()


def smooth(data: np.ndarray, weight=0.9) -> list:
    """
    绘制平滑曲线
    :param data: 数据
    :param weight: 平滑程度
    :return: 平滑结果
    """
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def env_seed(seed: int = 1) -> None:
    """
    设定种子
    :param seed: 种子
    :return: None
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train(params: Params = None, env: gym.Env = None, agent: TD3 = None) -> None:
    """
    Off-Policy 的通用训练框架, 注意新版gym的返回值的变化
    :param params: 配置参数
    :param env: 环境
    :param agent: 智能体
    :return: None
    """
    immediately_rewards = []
    for epoch in range(params.epochs):
        rewards = []
        state, info = env.reset()

        for step in range(params.episode_len):
            action = agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)

            agent.replay_buffer.push([state, action, reward, next_state, int(done)])
            state = next_state
            rewards.append(reward)
            agent.update()
            if done:
                break
        rewards = np.array(rewards)
        immediately_rewards.append(rewards.sum())
        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 平均奖励:{np.array(immediately_rewards).mean():.2f}")
            np.save(f"./data/td3_immediately_rewards{epoch + 1}.npy", np.array(immediately_rewards))
            torch.save(agent.critic.state_dict(), f"./data/td3_critic_epoch{epoch + 1}.pt")
            torch.save(agent.actor.state_dict(), f"./data/td3_actor_epoch{epoch + 1}.pt")
    env.close()


def td3():
    params = Params()
    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.shape[0]
    params.state_space_size = env.observation_space.shape[0]
    params.action_lower = env.action_space.low[0]
    params.action_upper = env.action_space.high[0]
    params.action_space_size = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(params.replay_buffer_size)
    agent = TD3(params, replay_buffer)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/td3_immediately_rewards1000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    td3()
    plot_rewards()
