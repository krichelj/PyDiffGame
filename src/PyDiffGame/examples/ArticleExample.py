import numpy as np
from PyDiffGame import ContinuousPyDiffGame, DiscretePyDiffGame

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

continuous_game = ContinuousPyDiffGame(A=A,
                                       B=B,
                                       Q=Q,
                                       R=R,
                                       x_0=x_0,
                                       P_f=P_f,
                                       T_f=T_f,)
continuous_game.solve_game_and_plot_state_space()


# discrete_game = DiscretePyDiffGame(A=A,
#                                    B=B,
#                                    Q=Q,
#                                    R=R,
#                                    x_0=x_0,
#                                    # P_f=P_f,
#                                    # T_f=T_f,
#                                    data_points=data_points,
#                                    show_legend=show_legend)
# discrete_game.solve_game_and_plot_state_space()