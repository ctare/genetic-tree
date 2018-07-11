import numpy as np
import deap

def to_d(bits):
    result = 0
    for b in bits:
        result <<= 1
        result += b
    return result

def grow(fn, pos, length=10, angle=0, cut=0.8, n=2, min_length=1, angle_d=lambda x, n: (x + 10, x - 10)):
    x, y = pos
    if length > min_length:
        rad = np.radians(angle)
        nx = x + np.cos(rad) * length
        ny = y + np.sin(rad) * length
        fn((x, y), (nx, ny))
        for i in angle_d(angle, n):
            grow(fn, (nx, ny), length * cut, i, cut, n, min_length, angle_d)

def grow_(fn, pos, length=10, angle=90, cut=0.8, n=2, min_length=1, angle_d=lambda x, n: (x + 10, x - 10)):
    x, y = pos
    if length > min_length:
        rad = np.radians(angle)
        nx = x + np.cos(rad) * length
        ny = y + np.sin(rad) * length
        fn((x, y), (nx, ny))
        for i in angle_d(angle, 1):
            grow_(fn, (nx, ny), length * cut, i, cut, n, min_length, angle_d)


class Evaluator(pylink.CellCalculator):
    def __init__(self, n, l, **tree_args):
        super().__init__()
        self.tree_args = tree_args
        self.l = l
        self.bit_n = n * l

    def _calc_nodes(self, itr):
        nodes = []
        for s in zip(*[itr] * self.l):
            nodes.append(to_d(s))
        return nodes

    def _get_calc_angle(self, nodes):
        self.now = 0
        def _calc_angle(angle, n):
            angles = []
            for i in range(n):
                angles.append(angle + nodes[self.now])
                self.now += 1
            return angles
        return _calc_angle

    def get_cell_size(self):
        return self.bit_n

    def fitness_function(self, itr):
        points = []
        def leaves(f, t):
            points.append(f)

        nodes = self._calc_nodes(itr)
        grow(leaves, (0, 0), angle=90, angle_d=self._get_calc_angle(nodes), **self.tree_args)

        points = np.asarray(points)
        # fvalue = 1 / (np.prod(np.var(points, axis=0)) / 100) * np.sum(points[..., 1])
        fvalue = np.sum(points[..., 1])
        return int(fvalue)

    def eval(self, itr):
        pass

