class Utils:
    def __init__(self):
        pass

    @staticmethod
    def index2pos(pos: int, size: int) -> tuple:
        """
        将一维序列对应下标pos的转换到边长为size的二维矩形内的坐标xy
        :param pos: 一维序列对应下标
        :param size: 矩形边长
        :return: 二维矩阵内的坐标xy
        """
        x, y = pos // size, pos % size
        return x, y

    @staticmethod
    def pos2index(x: int, y: int, size: int) -> int:
        """
        边长为size的二维矩形内的坐标xy转换到一维序列对应下标pos
        :param x: x
        :param y: y
        :param size: 矩形边长
        :return: 一维序列对应下标
        """
        pos = x * size + y
        return pos

    @staticmethod
    def arr_equal(a: list, b: list) -> bool:
        """
        判断两个列表是否相等
        :param a: 列表a
        :param b: 列表b
        :return: 是否相等
        """
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    @staticmethod
    def arr_contains(high_dim: list, low_dim: list) -> bool:
        """
        判断一个一位列表是否是另一个二维列表的子列表
        :param high_dim: 二维列表
        :param low_dim: 一维列表
        :return: 是否是
        """
        for arr in high_dim:
            if Utils.arr_equal(low_dim, arr):
                return True
        return False
