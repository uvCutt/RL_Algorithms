import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from environment.utils import Utils
from environment.env import Env


class Vis:
    """
    用于算法结果可视化, 目前比较简陋, 可自行调整样式
    """

    def __init__(self, size: int = 5, forbid: list = None, target_state: list = None, start_state: list = None):
        self.size = size
        self.forbid = forbid
        self.target_state = target_state
        self.start_state = start_state

        self.fig = plt.figure(figsize=(self.size, self.size))
        self.ax = plt.gca()
        self.ax.xaxis.set_ticks(range(self.size + 1))
        self.ax.yaxis.set_ticks(range(self.size + 1))
        self.ax.invert_yaxis()
        self.rect_width = 1
        self.init()

    def init(self) -> None:
        """
        这里绘制的是初始的格子: 正常格子, 禁止区域, 开始位置, 目标位置
        :return: None
        """
        for pos in range(self.size * self.size):
            xy = [*Utils.index2pos(pos, self.size)]
            self.draw_rect(xy, "#cccccc", fill=False, alpha=0.2)

        for forbid in self.forbid:
            self.draw_rect(forbid, "#DC143C")

        if self.start_state:
            self.draw_rect(self.start_state, "#00FF7F")
        self.draw_rect(self.target_state, "#00FF7F")

    def draw_rect(self, pos: list, color: str, fill: bool = True, alpha: float = 1.0) -> None:
        """
        绘制正方形格子
        :param pos: 位置
        :param color: 颜色
        :param fill: 是否填充
        :param alpha: 透明度
        :return: None
        """
        self.ax.add_patch(patches.Rectangle(
            xy=(pos[0], pos[1]),
            width=self.rect_width,
            height=self.rect_width,
            facecolor=color,
            fill=fill,
            alpha=alpha
        ))

    def draw_arrow(self, pos: int, direction: [list, np.ndarray], color: str) -> None:
        """
        绘制表示策略的箭头
        :param pos: 位置
        :param color: 颜色
        :param direction: 箭头朝向
        :return: None
        """
        arrow_offset = self.rect_width / 2
        x, y = Utils.index2pos(pos, self.size)
        self.ax.add_patch(patches.Arrow(
            x=x + arrow_offset,
            y=y + arrow_offset,
            dx=direction[0],
            dy=direction[1],
            color=color,
            width=0.2,
            linewidth=0.5
        ))

    def draw_circle(self, pos: int, color: str, radius: float) -> None:
        """
        绘制圆形,当策略为呆在原地的时候绘制一个圆形
        :param pos: 圆心
        :param color: 颜色
        :param radius: 半径
        :return: None
        """
        circle_offset = self.rect_width / 2
        x, y = Utils.index2pos(pos, self.size)
        self.ax.add_patch(patches.Circle(
            xy=(x + circle_offset, y + circle_offset),
            radius=radius,
            facecolor=color,
            linewidth=1,
        ))

    def draw_text(self, pos: int, text: str) -> None:
        """
        绘制一些文本信息
        :param pos: 绘制的位置
        :param text: 文本信息
        :return: None
        """
        circle_offset = self.rect_width / 4
        x, y = Utils.index2pos(pos, self.size)
        self.ax.text(x + circle_offset, y + circle_offset, text, size=10, ha='center', va='center')

    def show_policy(self, policy: np.ndarray) -> None:
        """
        绘制策略,用箭头和圆形表示
        :param policy: 策略
        :return: None
        """
        for state, action in enumerate(policy):
            action = np.argmax(action)
            direction = Env.action_mapper[action] * 0.4
            if action:
                self.draw_arrow(state, direction, "green")
            else:
                self.draw_circle(state, "green", 0.06)

    def show_value(self, values: np.ndarray):
        """
        绘制状态价值, 绘制具体的状态价值文本信息
        :param values: 状态价值
        :return: None
        """
        for state, value in enumerate(values):
            self.draw_text(state, str(round(value, 1)))

    def show(self):
        self.fig.show()
        plt.pause(100)
