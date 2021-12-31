import numpy as np

from ContinuousPyDiffGame import ContinuousPyDiffGame
from DiscretePyDiffGame import DiscretePyDiffGame

A = np.array([[-2, 1],
              [1, 4]])
B = [np.array([[1, 0],
               [0, 1]]),
     np.array([[0],
               [1]]),
     np.array([[1],
               [0]])]
Q = [np.array([[1, 0],
               [0, 1]]),
     np.array([[1, 0],
               [0, 10]]),
     np.array([[10, 0],
               [0, 1]])
     ]
R = [np.array([[100, 0],
               [0, 200]]),
     np.array([[5]]),
     np.array([[7]])]

x_0 = np.array([10, 20])
T_f = 5
P_f = [np.array([[10, 0],
                 [0, 20]]),
       np.array([[30, 0],
                 [0, 40]]),
       np.array([[50, 0],
                 [0, 60]])]

for L in [int(1000 / i) for i in range(1, 5)]:
    continuous_game = ContinuousPyDiffGame(A=A,
                                           B=B,
                                           Q=Q,
                                           R=R,
                                           x_0=x_0,
                                           L=L,
                                           # P_f=P_f,
                                           # T_f=T_f
                                           )
    continuous_game.solve_game_and_simulate_state_space()
    discrete_game = DiscretePyDiffGame(A=A,
                                       B=B,
                                       Q=Q,
                                       R=R,
                                       x_0=x_0,
                                       L=L
                                       # P_f=P_f,
                                       # T_f=T_f,
                                       # debug=True
                                       )
    discrete_game.solve_game_and_simulate_state_space()

    D_l_C = discrete_game < continuous_game
    print('#' * 10 + f' delta_T: {L} ' + '#' * 10 + f'\nD<C: {D_l_C}')

    print(f'D_cost: {discrete_game.get_costs()}\nC_cost: {continuous_game.get_costs()}')