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
    actor_lr = 0.001
    critic_lr = 0.001
    alpha_lr = 0.001
    gamma = 0.99

    # 熵的系数以及目标熵
    entropy_alpha = 0.01
    target_entropy = -1

    # 软更新的系数
    tau = 0.005

    # 训练次数以及每次最大步长
    epochs = 1000
    episode_len = 200

    # 经验回放池大小
    replay_buffer_size = 1000

    batch_size = 64

    # 更新target的频率以及保存频率
    target_update_freq = 5
    save_freq = 10


class PolicyNet(nn.Module):
    def __init__(self, params: Params = None):
        """
        Actor,策略网络，和PG的Policy网络一摸一样
        :param params: 参数
        """
        super(PolicyNet, self).__init__()
        self.input_linear = nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = nn.Linear(params.hidden_dim, params.action_space_size)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = nn.Softmax(dim=-1)(self.output_linear(x))
        return x


class ValueNet(torch.nn.Module):
    def __init__(self, params: Params = None):
        """
        这里输出不再是一维
        :param params: 参数
        """
        super(ValueNet, self).__init__()
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
        经验回放池, 打破数据时间相关性
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
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, params: Params = None, replay_buffer: ReplayBuffer = None):
        self.params = params
        self.actor = PolicyNet(self.params).to(self.params.device)

        self.critic = ValueNet(self.params).to(self.params.device)
        self.critic_another = ValueNet(self.params).to(self.params.device)

        self.critic_target = ValueNet(self.params).to(self.params.device)
        self.critic_target_another = ValueNet(self.params).to(self.params.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target_another.load_state_dict(self.critic_another.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.params.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.params.critic_lr)
        self.critic_optimizer_another = optim.Adam(self.critic_another.parameters(), lr=self.params.critic_lr)

        # 自动调整熵正则项的参数alpha, 这里取了对数,
        self.log_alpha = torch.tensor(np.log(self.params.entropy_alpha), requires_grad=True)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.params.alpha_lr)

        self.replay_buffer = replay_buffer

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones) -> torch.Tensor:
        """
        这里计算熵值并加到target中, 目的是为了增大熵值, 增加探索
        熵值计算公式: H(X) = -sigma[i = 1, i = n](log(p(xi)) * p(xi))
        :param rewards: 奖励
        :param next_states: 下一个状态
        :param dones: 中止信息
        :return: 添加了熵信息的TD Target
        """
        # 熵值的计算
        next_action_probs = self.actor(next_states)
        entropy = -torch.sum(next_action_probs * torch.log(next_action_probs + 1e-8), dim=1, keepdim=True)

        # 利用两个网络的q_value中较小的求和作为下一个
        action_values = torch.min(self.critic_target(next_states), self.critic_target_another(next_states))

        # 关键的一步, 之前输出的是状态价值, 现在输出的是每个动作的状态价值, 需要求期望
        q_value = torch.sum(next_action_probs * action_values, dim=1, keepdim=True)

        # 加入一个熵值的惩罚项, 熵越大越好
        next_value = q_value + self.log_alpha.exp() * entropy
        td_target = rewards + self.params.gamma * next_value * (1 - dones)

        return td_target

    def soft_update(self):
        for param, param_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.params.tau) + param.data * self.params.tau)

        for param, param_target in zip(self.critic_another.parameters(), self.critic_target_another.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.params.tau) + param.data * self.params.tau)

    def update(self) -> None:
        """
        跟AC不同的是, 用带信息熵的优势函数进行离线学习，用了Double DQN的思想解决过高估计, 再加上软更新的Trick
        注意这里Critic和Actor都要计算熵作为惩罚项, 最大信息熵, 增加探索性
        !!!按道理来说Softmax之后动作不存在0的情况为啥log还要加1e-8, 不加又会报错?
        想起来了, Softmax之后放缩到原动作空间,这个时候会有可能取到0
        :return: None
        """
        if len(self.replay_buffer) < self.params.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.params.batch_size)
        states = torch.tensor(states).to(self.params.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.params.device)
        rewards = torch.tensor(rewards).view(-1, 1).to(self.params.device)
        next_states = torch.tensor(next_states).to(self.params.device)
        dones = torch.tensor(dones).view(-1, 1).to(self.params.device)

        # Step 1: 计算带熵信息的TD Target
        td_target = self.calc_target(rewards, next_states, dones).detach()

        # Step 2: 计算两个Critic网络的损失
        critic_loss = torch.mean(nn.MSELoss()(self.critic(states).gather(dim=1, index=actions), td_target))
        critic_another_loss = torch.mean(nn.MSELoss()(self.critic_another(states).gather(1, actions), td_target))

        # Step 3: 两个Critic更新
        self.critic.zero_grad()
        self.critic_another.zero_grad()
        critic_loss.backward()
        critic_another_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer_another.step()

        # Step 4: Actor的计算以及更新
        action_probs = self.actor(states)

        # 加个小数1e-8避免出现概率为0的计算不稳定情况
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=1, keepdim=True)

        action_values = torch.min(self.critic_target(states), self.critic_target_another(states))
        q_value = torch.sum(action_probs * action_values, dim=1, keepdim=True)

        actor_loss = torch.mean(-self.params.entropy_alpha * entropy - q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 5: 软更新
        self.soft_update()

        # Step 6: 更新熵正则自适应项的参数alpha
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.params.target_entropy).detach())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def sample_action(self, state: np.ndarray) -> int:
        """
        根据当前状态获得各个动作的概率，然后根据这个概率建立分类分布，再用这个分布进行采样获得动作
        :param state: 当前状态
        :return: 执行的动作
        """
        state = torch.tensor(state).to(self.params.device)
        dist = torch.distributions.Categorical(self.actor(state))
        action = dist.sample()
        return action.detach().item()


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


def train(params: Params = None, env: gym.Env = None, agent: SAC = None) -> None:
    """
    Off-Policy 的通用训练框架
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
            np.save(f"./data/sac_immediately_rewards{epoch + 1}.npy", np.array(immediately_rewards))
            torch.save(agent.critic.state_dict(), f"./data/sac_critic_epoch{epoch + 1}.pt")
            torch.save(agent.actor.state_dict(), f"./data/sac_actor_epoch{epoch + 1}.pt")
    env.close()


def sac():
    params = Params()
    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]
    replay_buffer = ReplayBuffer(params.replay_buffer_size)
    agent = SAC(params, replay_buffer)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/sac_immediately_rewards1000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    sac()
    plot_rewards()
