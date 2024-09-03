import random

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
    gamma = 0.99

    # 截断的阈值
    epsilon = 0.2

    # 广义优势估计的时候利用
    lamda = 0.95

    # 训练次数以及每次最大步长
    epochs = 1000
    episode_len = 200

    # 保存频率
    save_freq = 10

    # 经验使用次数
    exp_use_times = 10


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
        Critic, 价值网络, 输入是状态, 输出的是状态价值评估, 和DQN不同, DQN输出的是不同动作的价值
        :param params:
        """
        super(ValueNet, self).__init__()
        self.input_linear = torch.nn.Linear(params.state_space_size, params.hidden_dim)
        self.hidden_linear = nn.Linear(params.hidden_dim, params.hidden_dim)
        self.output_linear = torch.nn.Linear(params.hidden_dim, 1)

    def forward(self, x):
        x = nn.ReLU()(self.input_linear(x))
        x = nn.ReLU()(self.hidden_linear(x))
        x = self.output_linear(x)
        return x


class PPO:
    def __init__(self, params: Params = None):
        self.params = params
        self.actor = PolicyNet(self.params).to(self.params.device)
        self.critic = ValueNet(self.params).to(self.params.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.params.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.params.critic_lr)

    def calc_gae(self, td_error: torch.Tensor) -> torch.Tensor:
        """
        计算广义优势估计GAE(Generalized Advantage Estimation): 可以看计算公式, 公式里的A就是第n次估计
        当gamma分别等于0和1的时候与TD和MC之间的关系
        :param td_error: 目标近似值
        :return: 广义优势估计值
        """
        gae, gae_arr = 0, []
        for target in td_error.numpy()[::-1]:
            gae = target + gae * self.params.gamma * self.params.lamda
            gae_arr.append(gae)
        gae_arr.reverse()
        return torch.tensor(np.array(gae_arr))

    def update(self, states: list, actions: list, rewards: list, next_states: list, dones: list) -> None:
        """
        PPO的优化目标涉及到重要性采样, 但只用到了上一轮策略的数据, 加入KL散度的约束或者截断, 两个策略可看做同一个, 同策略方法
        两个变种, 一个加入KL散度惩罚项Soft, 另一种是直接截断Hard
        注意理解广义优势估计GAE和TD之间的关系
        :param states: 状态
        :param actions: 动作
        :param rewards: 奖励
        :param next_states: 下一个状态
        :param dones: 是否终止
        :return: None
        """
        states = torch.tensor(np.array(states)).to(self.params.device)
        actions = torch.tensor(actions).view(-1, 1).to(self.params.device)
        rewards = torch.tensor(rewards).view(-1, 1).to(self.params.device)
        next_states = torch.tensor(np.array(next_states)).to(self.params.device)
        dones = torch.tensor(dones).view(-1, 1).to(self.params.device)

        # Step 1: TD Error (Advantage Function)
        td_target = rewards + self.params.gamma * self.critic(next_states) * (1 - dones)
        td_error = td_target - self.critic(states)

        # Step 2: 广义优势估计 (Generalized Advantage Estimation)
        gae = self.calc_gae(td_error.detach())

        # 这里先计算未使用经验的时候的策略
        pi_old = torch.log(self.actor(states).gather(dim=1, index=actions)).detach()

        for t in range(self.params.exp_use_times):
            pi_new = torch.log(self.actor(states).gather(dim=1, index=actions))
            # 新老策略之间的差距, 截断, 差距过大, 强制限制
            ratio = torch.exp(pi_new - pi_old)
            limit = torch.clamp(ratio, 1 - self.params.epsilon, 1 + self.params.epsilon) * gae

            actor_loss = torch.mean(-torch.min(ratio * gae, limit))
            critic_loss = torch.mean(nn.MSELoss()(self.critic(states), td_target.detach()))

            # 反向传播更新网络参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

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


def train(params: Params = None, env: gym.Env = None, agent: PPO = None) -> None:
    """
    On-Policy 的通用训练框架
    :param params: 配置参数
    :param env: 环境
    :param agent: 智能体
    :return: None
    """
    immediately_rewards = []
    for epoch in range(params.epochs):
        state, info = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for step in range(params.episode_len):
            action = agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(int(done))

            state = next_state
            if done:
                break

        agent.update(states, actions, rewards, next_states, dones)
        immediately_rewards.append(sum(rewards))

        if not (epoch + 1) % params.save_freq:
            print(f"回合:{epoch + 1}/{params.epochs}, 平均奖励:{np.array(immediately_rewards).mean():.2f}")
            np.save(f"./data/ppo_immediately_rewards{epoch + 1}.npy", np.array(immediately_rewards))
            torch.save(agent.critic.state_dict(), f"./data/ppo_critic_epoch{epoch + 1}.pt")
            torch.save(agent.actor.state_dict(), f"./data/ppo_actor_epoch{epoch + 1}.pt")
    env.close()


def ppo():
    params = Params()

    env = gym.make(params.env_name)
    params.action_space_size = env.action_space.n
    params.state_space_size = env.observation_space.shape[0]

    agent = PPO(params)
    train(params, env, agent)


def plot_rewards():
    data = np.load("./data/ppo_immediately_rewards1000.npy")
    plt.xlabel("episodes")
    plt.ylabel("immediately_rewards")
    plt.plot(data, label='rewards')
    plt.plot(smooth(data), label='smoothed rewards')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ppo()
    plot_rewards()
