from enum import Enum


class RewardType(Enum):
    """
    四种类型: 正常的格子, 目标格子, 禁止格子, 越边界
    """
    NORMAL = 0
    TARGET = 1
    FORBID = 2
    OUTSIDE = 3


class TrajItems(Enum):
    """
    SARSA,按需所取
    """
    STATE = 0
    ACTION = 1
    REWARD = 2
    NEXT_STATE = 3
    NEXT_ACTION = 4
