import numpy as np
from tqdm import tqdm

from environment.env import Env
from environment.vis import Vis
from environment.enums import TrajItems


class MonteCarlo:
    """
        蒙特卡洛是近似Qsa值的一种方法, 用多条实际轨迹的Discounted Return的平均值来代替真正的Qsa
        同时, 原本PE过程计算的是状态价值, 然后PI过程用这个值来更新策略, 为啥MC会近似Qsa而不是状态价值呢
        是因为展开PI过程会发现, 实际是sigma(pi(a | s) * sigma(p(r | s, a) * r)), 其中Qsa的计算需要状态价值和环境模型p(r | s, a)
        所以直接近似Qsa才能进行PI
    """
    def __init__(self, gamma: float = 0.9, env: Env = None, vis: Vis = None, render: bool = False):
        self.gamma = gamma
        self.env = env
        self.vis = vis
        self.render = render
        self.policy = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=float)
        self.qtable = np.zeros(shape=self.env.state_space_size, dtype=float)

    def mc_basic(self, steps: int = 30, epochs: int = 100, trajectory_numbers: int = 1) -> None:
        """
        基本的mc, 遍历每个状态动作对, 并从每个状态动作队采样多个trajectory, 用trajectory的平局奖励作为q(s,a)
        注意这里采样的多条轨迹是确定且一致的, 因为在某个策略下, 下一个状态要采取动作只有一个概率为1, 其他为0
        :param steps: trajectory的长度
        :param epochs: 迭代次数
        :param trajectory_numbers: 每个状态动作对采集的trajectory的数量,这里设置为1
        :return: None
        """
        self.init_policy()
        for _ in tqdm(range(epochs)):
            for state in self.env.state_space:
                qsa = np.zeros(shape=self.env.action_space_size, dtype=float)
                for action in self.env.action_space:
                    gs = np.zeros(shape=trajectory_numbers, dtype=float)
                    for traj_index in range(trajectory_numbers):
                        traj = self.env.episode(self.policy, state, action, steps)[::-1, :]
                        for step in range(steps):
                            gs[traj_index] = traj[step, TrajItems.REWARD.value] + self.gamma * gs[traj_index]
                    qsa[action] = gs.mean()
                self.policy[state] = np.zeros(shape=self.env.action_space_size)
                self.policy[state, np.argmax(qsa)] = 1
                self.qtable[state] = np.max(qsa)
        if self.render:
            self.vis.show_policy(self.policy)
            self.vis.show_value(self.qtable)
            self.vis.show()

    def mc_exploring_starts(self, steps: int = 30, epochs: int = 100) -> None:
        """
        为了保证每个状态动作对都访问到, 使用了遍历。这个算法整体来说只能说提高了数据利用率, 效果很差
        :param steps: trajectory的长度
        :param epochs: 迭代次数
        :return: None
        """
        self.init_policy()
        returns = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=float)
        nums = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=int)
        for _ in tqdm(range(epochs)):
            for state in self.env.state_space:
                qsa = np.zeros(shape=self.env.action_space_size, dtype=float)
                for action in self.env.action_space:
                    traj = self.env.episode(self.policy, state, action, steps)[::-1, :]
                    g = 0
                    for step in range(steps):
                        g = traj[step, TrajItems.REWARD.value] + self.gamma * g
                        traj_state = int(traj[step, TrajItems.STATE.value])
                        traj_action = int(traj[step, TrajItems.ACTION.value])

                        returns[traj_state, traj_action] += g
                        nums[traj_state, traj_action] += 1
                        qsa[traj_action] = returns[traj_state, traj_action] / nums[traj_state, traj_action]

                        self.policy[traj_state] = np.zeros(shape=self.env.action_space_size)
                        self.policy[traj_state, np.argmax(qsa)] = 1

                        self.qtable[traj_state] = np.max(qsa)
        if self.render:
            self.vis.show_policy(self.policy)
            self.vis.show_value(self.qtable)
            self.vis.show()

    def mc_epsilon_greedy(self, steps: int = 200, epochs: int = 2000, epsilon: float = 0.1) -> None:
        """
        非傻贪婪,实际上效果和上面一个一样, 很差, 但是这种思想很重要
        :param steps: trajectory的长度
        :param epochs: 迭代次数
        :param epsilon: 探索率
        :return: None
        """
        self.init_policy()
        returns = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=float)
        nums = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=int)

        # for each episode, do
        for _ in tqdm(range(epochs)):
            state = np.random.choice(self.env.state_space)
            action = np.random.choice(self.env.action_space)

            qsa = np.zeros(shape=self.env.action_space_size, dtype=float)

            traj = self.env.episode(self.policy, state, action, steps)[::-1, :]
            g = 0
            for step in range(steps):
                g = traj[step, TrajItems.REWARD.value] + self.gamma * g

                traj_state = int(traj[step, TrajItems.STATE.value])
                traj_action = int(traj[step, TrajItems.ACTION.value])

                returns[traj_state, traj_action] += g

                nums[traj_state, traj_action] += 1
                qsa[traj_action] = returns[traj_state, traj_action] / nums[traj_state, traj_action]

                other_probability = epsilon * (1 / self.env.action_space_size)
                self.policy[traj_state] = np.ones(shape=self.env.action_space_size) * other_probability
                self.policy[traj_state, np.argmax(qsa)] = 1 - other_probability * (self.env.action_space_size - 1)

                self.qtable[traj_state] = np.max(qsa)

        if self.render:
            self.vis.show_policy(self.policy)
            self.vis.show_value(self.qtable)
            self.vis.show()

    def init_policy(self) -> None:
        """
        随机初始化策略
        :return: None
        """
        random_action = np.random.randint(self.env.action_space_size, size=self.env.state_space_size)
        for state, action in enumerate(random_action):
            self.policy[state, action] = 1


if __name__ == "__main__":
    start_state = [0, 0]
    target_state = [2, 3]
    forbid = [[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]]
    model = MonteCarlo(vis=Vis(target_state=target_state, forbid=forbid),
                       env=Env(target_state=target_state, forbid=forbid),
                       render=True)
    # model.mc_basic()
    # model.mc_exploring_starts()
    model.mc_epsilon_greedy()
