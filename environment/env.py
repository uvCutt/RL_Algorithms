import numpy as np

from typing import Tuple
from environment.utils import Utils
from environment.enums import RewardType
from environment.enums import TrajItems


class Env:
    # 动作: 不动, 上, 下, 左, 右
    actions = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
    action_mapper = np.array([np.array(action) for action in actions])

    def __init__(self, size: int = 5, forbid: list = None, target_state: list = None, start_state: list = None):
        """
        环境
        动作，状态，奖励，环境模型
        :param size: 地图大小
        """
        self.size = size
        # 初始状态与目标状态
        self.start_state = start_state
        self.target_state = target_state

        # 禁止区域
        self.forbid = forbid

        # 动作空间
        self.action_space_size = len(self.actions)
        self.action_space = np.arange(self.action_space_size)

        # 状态空间: 每个格子, 左到右, 上到下拉成一维的
        self.state_space_size = self.size * self.size
        self.state_space = np.arange(self.state_space_size)

        # 奖励设定: 禁止区域扣10分,到达终点0分, 走路-1但因为gamma的存在, 路径越长, 奖励越低
        self.reward_space = np.array([-1, 0, -10, -10])
        self.reward_space_size = 4

        # 环境模型: 任意的s跟a对应的p(r|s,a)与p(s'|s,a)
        self.rewards_model = None
        self.states_model = None
        self.init_model()

        # 轨迹空间大小
        self.trajectory_space_size = len(TrajItems.__members__)

        # 交互相关
        self.state = None
        self.done = False
        self.info = None

    def init_model(self) -> None:
        """
        初始化环境模型p(r|s,a) p(s''|s,a)
        :return: None
        """
        states_model_shape = (self.state_space_size, self.action_space_size, self.state_space_size)
        rewards_model_shape = (self.state_space_size, self.action_space_size, self.reward_space_size)
        self.states_model = np.zeros(shape=states_model_shape, dtype=float)
        self.rewards_model = np.zeros(shape=rewards_model_shape, dtype=float)

        for state in self.state_space:
            for action in self.action_space:
                next_state_pos, inside = self.next_state_pos(state, action)
                if not inside:
                    reward_type = RewardType.OUTSIDE
                else:
                    if Utils.arr_equal(next_state_pos, self.target_state):
                        reward_type = RewardType.TARGET
                    elif Utils.arr_contains(self.forbid, next_state_pos):
                        reward_type = RewardType.FORBID
                    else:
                        reward_type = RewardType.NORMAL
                # 前状态state采取当前动作action转移到next_state的概率为1
                self.states_model[state, action, Utils.pos2index(*next_state_pos, self.size)] = 1
                # 当前状态state采取当前动作action获得该种奖励类型reward_type的概率为1
                self.rewards_model[state, action, reward_type.value] = 1

    def next_state_pos(self, state: int, action: int) -> Tuple[list, bool]:
        """
        在当前状态根据动作获取下一个状态
        :param state: 当前状态
        :param action: 当前动作
        :return: 下一个状态(越界返回当前状态)的坐标; 执行当前动作后是否还在地图内
        """
        pos = np.array(Utils.index2pos(state, self.size))
        next_pos = pos + self.action_mapper[action]

        inside = bool((0 <= next_pos[0] <= self.size - 1) and (0 <= next_pos[1] <= self.size - 1))

        next_state_pos = [*next_pos] if inside else [*pos]

        return next_state_pos, inside

    def episode(self, policy: np.ndarray, state: int, action: int, steps: int) -> np.ndarray:
        """
        根据当前策略从当前状态以及当前动作出发, 生成一个trajectory
        :param policy: 当前策略
        :param state: 当前状态
        :param action: 当前动作
        :param steps: 轨迹长度
        :return: 轨迹
        """
        # 存的是state, action, reward, next_state, next_action --> sarsa
        trajectory = np.zeros(shape=(steps, self.trajectory_space_size), dtype=float)
        next_state, next_action = state, action
        for step in range(steps):
            state, action = next_state, next_action

            # 获取概率为1的奖励的具体值
            reward_type = np.where(self.rewards_model[state, action] == 1)
            reward = self.reward_space[reward_type].item()

            next_state_pos, _ = self.next_state_pos(state, action)
            next_state = Utils.pos2index(*next_state_pos, self.size)

            next_action = np.random.choice(self.action_space, p=policy[next_state])

            trajectory[step] = np.array([state, action, reward, next_state, next_action])

        return trajectory

    def reset(self) -> int:
        self.done = False
        self.state = Utils.pos2index(*self.start_state, self.size)
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        这里的环境根据动作直接从环境模型中获取对应的奖励, 然后再计算下一个状态, 再判断是否结束
        :param action: 当前执行的动作
        :return: 下一个状态, 当前状态执行当前动作的即时奖励, 是否到达目标格子(是否终止)
        """
        reward_type = np.where(self.rewards_model[self.state, action] == 1)
        reward = self.reward_space[reward_type].item()

        next_state_pos, _ = self.next_state_pos(self.state, action)
        next_state = Utils.pos2index(*next_state_pos, self.size)
        self.state = next_state

        if self.state == Utils.pos2index(*self.target_state, self.size):
            self.done = True

        return self.state, reward, self.done
