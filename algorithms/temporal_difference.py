import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from environment.env import Env
from environment.vis import Vis
from environment.enums import TrajItems
from environment.utils import Utils


class TemporalDifference:
    """
    时序差分, 套用了随机近似理论(stochastic approximation)的RM算法框架, 其实最重要的是理解TD Target,TD Error
    TD Target 下一个时刻的状态价值在当前时刻的估计, 这里用时刻的原因是on-policy的话一般是收集trajectory来进行学习
    TD Error 当前时刻状态价值在当前时刻的估计 -下一个时刻的状态价值在当前时刻的估计 这里的结果其实就是RM算法框架下朝着最优解的迭代方向
    再就是one step sarsa 和 n step sarsa的关系, 直接用估计的值还是先实际采样一些step得到即时奖励接合估计之间的权衡
    后面在学习GAE(Generalized Advantage Estimation)广义优势估计的时候会进一步理解
    """

    def __init__(self, gamma: float = 0.9, env: Env = None, vis: Vis = None, render: bool = False):
        self.gamma = gamma
        self.env = env
        self.vis = vis
        self.render = render
        self.policy = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=float)
        self.qtable = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=float)

    def sarsa(self, lr: float = 0.01, epsilon_max: float = 0.1, epochs: int = 1000, expected: bool = False) -> None:
        """
        sarsa或expected sarsa, 之前MC的Q值更新用Trajectory的Discounted Return, 现在用TD
        :param lr: 学习率
        :param epsilon_max: 最大探索率, 实际探索率会随着迭代衰减
        :param epochs: 学习多少个回合
        :param expected: sarsa还是expected sarsa
        :return: None
        """
        self.init_policy()
        state = self.env.reset()
        action = np.random.choice(self.env.action_space, p=self.policy[state])
        total_rewards = np.zeros(shape=epochs, dtype=float)
        episode_lengths = np.zeros(shape=epochs, dtype=int)
        for epoch in tqdm(range(epochs)):
            epsilon = ((epochs - epoch) / epochs) * epsilon_max
            rewards, lengths = 0, 0
            while True:
                next_state, reward, done = self.env.step(action)
                next_action = np.random.choice(self.env.action_space, p=self.policy[next_state])

                if expected:
                    expected_qsa = 0
                    for next_action_ in self.env.action_space:
                        expected_qsa += self.policy[next_state, next_action_] * self.qtable[next_state, next_action_]
                    td_target = reward + self.gamma * expected_qsa
                else:
                    td_target = reward + self.gamma * self.qtable[next_state, next_action]
                td_error = self.qtable[state, action] - td_target
                self.qtable[state, action] = self.qtable[state, action] - lr * td_error

                other_prob = epsilon * (1 / self.env.action_space_size)
                self.policy[state] = np.ones(shape=self.env.action_space_size) * other_prob
                self.policy[state, np.argmax(self.qtable[state])] = 1 - other_prob * (self.env.action_space_size - 1)

                state = next_state
                action = next_action

                rewards += reward
                lengths += 1

                if done:
                    state = self.env.reset()
                    break

            total_rewards[epoch] = rewards
            episode_lengths[epoch] = lengths
        if self.render:
            self.show_rewards_episodes(total_rewards, episode_lengths)
            # self.vis.show_policy(self.policy)
            # self.vis.show_value(np.max(self.qtable, axis=1))
            # self.vis.show()

    def q_learning_on_policy(self, lr: float = 0.01, epsilon_max: float = 0.1, epochs: int = 10000) -> None:
        """
        这个on-policy版本跟sarsa的区别在于TD Target的计算上面, 之前是 下一个状态执行下一个动作的状态动作价值在当前时刻的估计
        现在直接就拿下一个状态状态动作价值最大的
        :param lr: 学习率
        :param epsilon_max: 最大探索率, 实际探索率会随着迭代衰减
        :param epochs: 学习多少个回合
        :return: None
        """
        self.init_policy()
        state = self.env.reset()
        action = np.random.choice(self.env.action_space, p=self.policy[state])
        total_rewards = np.zeros(shape=epochs, dtype=float)
        episode_lengths = np.zeros(shape=epochs, dtype=int)
        for epoch in tqdm(range(epochs)):
            epsilon = ((epochs - epoch) / epochs) * epsilon_max
            rewards, lengths = 0, 0
            while True:
                next_state, reward, done = self.env.step(action)
                next_action = np.random.choice(self.env.action_space, p=self.policy[next_state])

                td_target = reward + self.gamma * np.max(self.qtable[next_state])
                td_error = self.qtable[state, action] - td_target
                self.qtable[state, action] = self.qtable[state, action] - lr * td_error

                other_prob = epsilon * (1 / self.env.action_space_size)
                self.policy[state] = np.ones(shape=self.env.action_space_size) * other_prob
                self.policy[state, np.argmax(self.qtable[state])] = 1 - other_prob * (self.env.action_space_size - 1)

                state = next_state
                action = next_action

                rewards += reward
                lengths += 1

                if done:
                    state = self.env.reset()
                    break

            total_rewards[epoch] = rewards
            episode_lengths[epoch] = lengths
        if self.render:
            self.show_rewards_episodes(total_rewards, episode_lengths)
            # self.vis.show_policy(self.policy)
            # self.vis.show_value(np.max(self.qtable, axis=1))
            # self.vis.show()

    def q_learning_off_policy(self, lr: float = 0.001, epochs: int = 10000, steps: int = 500) -> None:
        """
        真正的基于价值的离线离散动作强化学习的算法, 非常重要的一个算法
        这里经验的收集保存为了文件, 第一次执行需打开重新采样, 虽然奖励曲线不好看, 但实际策略还行
        奖励曲线就应该不正常?本来就是随机策略给出的?
        可以相互交流学习
        :param lr: 学习率
        :param epochs: 学习回合数
        :param steps: 每次最大执行步数
        :return: None
        """
        # sample
        # self.init_fair_policy()
        # state = self.env.reset()
        # action = np.random.choice(self.env.action_space, p=self.policy[state])
        # trajectories = np.zeros(shape=(epochs, steps, self.env.trajectory_space_size), dtype=float)
        # for epoch in tqdm(range(epochs)):
        #     trajectories[epoch] = self.env.episode(self.policy, state, action, steps)
        # np.save("trajectories.npy", trajectories)
        trajectories = np.load("trajectories.npy")

        self.init_policy()
        total_rewards = np.zeros(shape=epochs, dtype=float)
        episode_lengths = np.zeros(shape=epochs, dtype=int)
        for epoch in tqdm(range(epochs)):
            trajectory = trajectories[epoch]
            rewards, lengths = 0, 0
            for step in range(steps):
                # offline
                state = int(trajectory[step, TrajItems.STATE.value])
                action = int(trajectory[step, TrajItems.ACTION.value])
                reward = trajectory[step, TrajItems.REWARD.value]
                next_state = int(trajectory[step, TrajItems.NEXT_STATE.value])
                if state == Utils.pos2index(*self.env.target_state, self.env.size):
                    break

                td_target = reward + self.gamma * np.max(self.qtable[next_state])
                td_error = self.qtable[state, action] - td_target
                self.qtable[state, action] = self.qtable[state, action] - lr * td_error

                self.policy[state] = np.zeros(shape=self.env.action_space_size)
                self.policy[state, np.argmax(self.qtable[state])] = 1

                rewards += reward
                lengths += 1

            total_rewards[epoch] = rewards
            episode_lengths[epoch] = lengths

        if self.render:
            # self.show_rewards_episodes(total_rewards, episode_lengths)
            self.vis.show_policy(self.policy)
            self.vis.show_value(np.max(self.qtable, axis=1))
            self.vis.show()

    @staticmethod
    def show_rewards_episodes(total_rewards: np.ndarray, episode_lengths: np.ndarray) -> None:
        """
        绘制奖励曲线
        :param total_rewards:
        :param episode_lengths:
        :return:
        """
        plt.clf()
        fig = plt.subplot(2, 1, 1)
        xs = range(total_rewards.size)
        ys = total_rewards
        fig.plot(xs, ys)
        plt.xticks(range(total_rewards.size, 10))
        plt.xlabel("epoch")
        plt.ylabel("total_rewards")
        fig.set_title("total_rewards per epoch")
        fig = plt.subplot(2, 1, 2)
        xs = range(episode_lengths.size)
        ys = episode_lengths
        fig.plot(xs, ys)
        plt.xticks(range(total_rewards.size, 10))
        plt.xlabel("epoch")
        plt.ylabel("episode_lengths")
        fig.set_title("episode_lengths per epoch")
        plt.show()
        plt.pause(100)

    def init_policy(self) -> None:
        """
        随机初始化策略
        :return: None
        """
        random_action = np.random.randint(self.env.action_space_size, size=self.env.state_space_size)
        for state, action in enumerate(random_action):
            self.policy[state, action] = 1

    def init_fair_policy(self) -> None:
        """
        没有先验知识的情况下, 人人平等的策略
        :return: None
        """
        self.policy.fill(1 / self.env.action_space_size)


if __name__ == "__main__":
    start_state = [0, 0]
    target_state = [2, 3]
    forbid = [[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]]
    model = TemporalDifference(vis=Vis(start_state=start_state, target_state=target_state, forbid=forbid),
                               env=Env(start_state=start_state, target_state=target_state, forbid=forbid),
                               render=True)
    model.sarsa(expected=True)
    # model.q_learning_on_policy()
    # model.q_learning_off_policy()
