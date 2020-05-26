from scipy.optimize import fsolve
import math
from numpy.linalg import inv

def equations(p):
    x, y = p
    return x + y ** 2 - 4, math.exp(x) + x * y - 3


if __name__ == '__main__':
    m = [2, 2]

    A = np.diag([2, 1, 1, 4])

    B = [np.diag([2, 1, 1, 2]),
         np.diag([1, 2, 2, 1])]

    Q = [np.diag([2, 1, 2, 2]),
         np.diag([1, 2, 3, 4])]

    R = [np.diag([100, 200, 100, 200]),
         np.diag([100, 300, 200, 400])]

    coupled_R = [np.diag([100, 200, 100, 200]),
                 np.diag([100, 300, 200, 400])]

    equations = []

    x, y = fsolve(equations, (1, 1))
    print(y)
