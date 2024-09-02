import numpy as np

from environment.env import Env
from environment.vis import Vis


class DynamicProgramming:
    """
    动态规划的两个方法, 实际都为Truncated Policy Iteration, 具体代码尽量复刻伪代码的逻辑
    """

    def __init__(self, gamma: float = 0.9, env: Env = None, vis: Vis = None, render: bool = False):
        self.gamma = gamma
        self.env = env
        self.vis = vis
        self.render = render
        self.policy = np.zeros(shape=(self.env.state_space_size, self.env.action_space_size), dtype=int)
        self.qtable = np.zeros(shape=self.env.state_space_size, dtype=float)

    def value_iteration(self, threshold: float = 0.01) -> None:
        """
        计算每个状态动作对的状态动作价值，然后每个状态选择最大的值对应的动作作为自己的策略，并将值作为自己的状态价值
        根据Contraction Mapping Theorem, qsa的计算公式满足该理论要求，通过迭代不断优化全局状态价值，并找到对应的最优策略
        :param threshold: 迭代结束的阈值，前后两次迭代后的全局状态价值的欧氏距离相差小于该阈值时代表优化空间已经不大，结束优化
        :return: None
        """
        differ = np.inf
        while differ > threshold:
            kth_qtable = self.qtable.copy()
            for state in self.env.state_space:
                qsa = np.zeros(shape=self.env.action_space_size, dtype=float)
                for action in self.env.action_space:
                    qsa[action] = self.calculate_qvalue(state, action)
                self.policy[state] = np.zeros(shape=self.env.action_space_size)
                self.policy[state, np.argmax(qsa)] = 1
                self.qtable[state] = np.max(qsa)
            differ = np.linalg.norm(kth_qtable - self.qtable, ord=1)
        if self.render:
            self.vis.show_policy(self.policy)
            self.vis.show_value(self.qtable)
            self.vis.show()

    def policy_iteration(self, policy_threshold: float = 0.01, value_threshold: float = 0.01, steps: int = 10) -> None:
        """
        step 1:从初始策略开始，求解该策略对应的全局状态价值(在这个过程中本来要无穷次迭代得到真正的状态价值，但实际会设置阈值，截断策略迭代算法)
        step 2:拿到第K次迭代对应的策略求解出的全局状态价值之后，利用该价值作为初始值，再进行全局状态价值优化以及策略优化
        这个过程其实相较于值迭代比较难理解

        Q1:In the policy evaluation step, how to get the state value vπk by solving the Bellman equation?
        A1:x=f(x)这种满足Contraction Mapping Theorem的迭代求解方式(也可以解析解matrix vector form，但是涉及矩阵逆运算会很慢O(n^3))
        Q2*:In the policy improvement step, why is the new policy πk+1 better than πk?
        A2:直观上不是很好理解就得利用数学工具了，赵老师原著Chapter4.P73页对比了前后两次迭代证明了Vπk - Vπk+1 < 0
        Q3*:Why can this algorithm finally converge to an optimal policy?
        A3:Chapter4.P75页不仅证明了能达到最优，而且引入这种PE过程会收敛得更快，证明了Vπk>Vk，同一个迭代timing，策略迭代状态价值更接近最优

        :param policy_threshold: 策略阈值
        :param value_threshold: 全局状态价值阈值
        :param steps: 截断的最大迭代次数，只用阈值也行，但这样更方便说明
        :return: None
        """
        policy_differ = np.inf
        self.init_policy()
        while policy_differ > policy_threshold:
            kth_policy = self.policy.copy()
            # step 1: policy evaluation
            value_differ = np.inf
            while value_differ > value_threshold and steps > 0:
                steps -= 1
                kth_qtable = self.qtable.copy()
                for state in self.env.state_space:
                    state_value = 0
                    for action in self.env.action_space:
                        state_value += self.policy[state, action] * self.calculate_qvalue(state, action)
                    self.qtable[state] = state_value
                value_differ = np.linalg.norm(kth_qtable - self.qtable, ord=1)
            # step 2: policy improvement 相当于上面的PE给下面提供了一个初始状态(对应策略)，之前值迭代的时候是全0为初始值
            value_differ = np.inf
            while value_differ > value_threshold:
                kth_qtable = self.qtable.copy()
                for state in self.env.state_space:
                    qsa = np.zeros(shape=self.env.action_space_size, dtype=float)
                    for action in self.env.action_space:
                        qsa[action] = self.calculate_qvalue(state, action)
                    self.policy[state] = np.zeros(shape=self.env.action_space_size)
                    self.policy[state, np.argmax(qsa)] = 1
                    self.qtable[state] = np.max(qsa)
                value_differ = np.linalg.norm(kth_qtable - self.qtable, ord=1)
            policy_differ = np.linalg.norm(kth_policy - self.policy, ord=1)
        if self.render:
            self.vis.show_policy(self.policy)
            self.vis.show_value(self.qtable)
            self.vis.show()

    def init_policy(self) -> None:
        """
        之前值迭代可以不用初始化，因为只对policy进行了更新，现在策略迭代得初始化，因为首先就要利用policy进行PE
        :return: None
        """
        random_action = np.random.randint(self.env.action_space_size, size=self.env.state_space_size)
        for state, action in enumerate(random_action):
            self.policy[state, action] = 1

    def calculate_qvalue(self, state: int, action: int) -> float:
        """
        计算状态动作价值函数的元素展开式, 这里就能理解为什么环境模型为什么是这样的数据结构
        :param state: 当前状态
        :param action: 当前动作
        :return: 当前的状态动作价值
        """
        qvalue = 0
        # immediately reward: sigma(r * p(r | s, a))
        for reward_type in range(self.env.reward_space_size):
            qvalue += self.env.reward_space[reward_type] * self.env.rewards_model[state, action, reward_type]
        # next state expected reward : sigma(vk(s') * p(s' | s, a))
        for next_state in range(self.env.state_space_size):
            qvalue += self.gamma * self.env.states_model[state, action, next_state] * self.qtable[next_state]
        return qvalue


if __name__ == "__main__":
    start_state = [0, 0]
    target_state = [2, 3]
    forbid = [[2, 2], [2, 1], [1, 1], [3, 3], [1, 3], [1, 4]]
    model = DynamicProgramming(vis=Vis(target_state=target_state, forbid=forbid),
                               env=Env(target_state=target_state, forbid=forbid),
                               render=True)
    model.value_iteration()
    # model.policy_iteration()
