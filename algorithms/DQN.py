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
    env_name = "CartPole-v1"

    # 动作空间和状态空间以及隐藏层大小
    action_space_size = 0
    state_space_size = 0
    hidden_dim = 128
    # 学习率和折扣率
    lr = 0.0001
    gamma = 0.99

    # 探索率
    epsilon = 0.01

    # 训练次数以及每次最大步长
    epochs = 1000
    episode_len = 200

    # 经验回放池大小
    replay_buffer_size = 1000

    batch_size = 64

    # 更新target的频率以及保存频率
    target_update_freq = 5
    save_freq = 10


class MLP(nn.Module):
    def __init__(self, params: Params = None):
        """
        向量型环境状态表示, 使用全连接网络作为Q网络, 最后一层不用ReLU
        :param params: 参数
        """
        super(MLP, self).__init__()
        self.input_linear = nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, params.action_space_size)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = self.output_linear(x)
        return x


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        """
        DQN贡献之一: 经验回访池, 打破数据时间相关性
        通用经验回放池，利用队列来进行维护，先入先出
        :param capacity: 经验回放池大小
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions: list) -> None:
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
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


class DQN:
    def __init__(self, params: Params = None, model: nn.Module = None, replay_buffer: ReplayBuffer = None):
        self.params = params

        self.epsilon = self.params.epsilon

        self.policy = model.to(self.params.device)
        self.target = model.to(self.params.device)

        self.batch_size = params.batch_size
        self.replay_buffer = replay_buffer

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.params.lr)

    def update(self):
        """
        step 1. 数据转换
        step 2. 优化函数
        step 3. 损失计算
        step 4. 反向传播, 模型更新优化
        :return: None
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(states).to(self.params.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.params.device)
        rewards = torch.tensor(rewards).view(-1, 1).to(self.params.device)
        next_states = torch.tensor(next_states).to(self.params.device)
        dones = torch.tensor(dones).view(-1, 1).to(self.params.device)

        # DQN的目标函数的计算公式
        q_values = self.policy(states).gather(dim=1, index=actions)
        next_q_values, _ = torch.max(self.target(next_states), dim=1)

        # 计算预测值以及目标值
        q_targets = rewards + self.params.gamma * (next_q_values.view(-1, 1) * (1 - dones))

        # 计算均方误差
        loss = torch.mean(nn.MSELoss()(q_values, q_targets))

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample_action(self, state: np.ndarray) -> int:
        """
        根据当前状态选择一个动作, epsilon greedy策略
        :param state: 当前状态
        :return: 执行的动作
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state).to(self.params.device)
                action_values = self.policy(state)
                action = torch.argmax(action_values).item()
        else:
            action = np.random.randint(self.params.action_space_size)
        return action


def smooth(data: np.ndarray, weight=0.8) -> list:
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


def train(params: Params = None, env: gym.Env = None, agent: DQN = None) -> None:
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

        if not ((epoch + 1) % params.target_update_freq):
            agent.target.load_state_dict(agent.policy.state_dict())

        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 平均奖励:{np.array(immediately_rewards).mean():.2f}")
            np.save(f"./data/dqn_immediately_rewards{epoch + 1}.npy", np.array(immediately_rewards))
            torch.save(agent.policy.state_dict(), f"./data/dqn_policy_epoch{epoch + 1}.pt")
    env.close()


def dqn() -> None:
    """
    初始化并训练
    :return:
    """
    params = Params()

    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]

    model = MLP(params)
    replay_buffer = ReplayBuffer(params.replay_buffer_size)
    agent = DQN(params, model, replay_buffer)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/dqn_immediately_rewards1000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dqn()
    plot_rewards()
