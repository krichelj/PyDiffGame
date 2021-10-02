import numpy as np
from scipy.integrate import odeint
from PyDiffGame import PyDiffGame, ContinuousPyDiffGame


class OligopolisticCompetition(ContinuousPyDiffGame):
    def __init__(self,
                 N: int,
                 r: float,
                 s: float,
                 beta: np.array,
                 c: np.array,
                 p: float,
                 x_0: np.array = None,
                 T_f: float = 5,
                 epsilon: float = PyDiffGame._epsilon_default):
        self.__r = r
        self.__s = s
        self.__beta = beta
        self.__c = c
        self.__p = p

        A = np.array([[- 0.5 * self.__r - self.__s * (1 + np.sum(beta)), self.__s * (self.__p +
                                                                                     np.inner(self.__beta, self.__c))],
                      [0, - 0.5 * self.__r]])
        B = [np.array([[- self.__s * beta_i],
                       [0]]) for beta_i in self.__beta]
        Q = [0.5 * np.array([[-1, c_i],
                             [c_i, -c_i ** 2]]) for c_i in self.__c]
        R = [np.array([0.5])] * N

        super().__init__(A=A,
                         B=B,
                         Q=Q,
                         R=R,
                         cl=True,
                         x_0=x_0,
                         T_f=T_f,
                         epsilon=epsilon)

        self.__check_input()

    def __check_input(self):
        """
        Input checking method

        Raises
        ------
        Case-specific errors
        """

        if self.__s <= 0:
            raise ValueError('s must be positive')
        if self.__p <= 0:
            raise ValueError('p must be positive')
        if len(self.__beta) != self._N:
            raise ValueError('beta must be of length N')
        if len(self.__c) != self._N:
            raise ValueError('c must be of length N')

    def get_steady_state_price(self):
        numerator_sum = 0
        denominator_sum = 0

        P_f = [(self._P_f[i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n) for i in range(self._N)]

        for i in range(self._N):
            beta_i = self.__beta[i]
            c_i = self.__c[i]
            P_f_i = P_f[i]
            k_i = P_f_i[0, 0]
            l_i = P_f_i[0, 1]

            numerator_sum += beta_i * (2 * self.__s * l_i - c_i)
            denominator_sum += beta_i * (1 + 2 * self.__s * k_i)

        p_s = (self.__p - numerator_sum) / (1 + denominator_sum)

        return p_s

    def simulate_p(self):
        t = len(self._P) - 1

        def p_t_diff_eqn(p_t: np.array, _: float) -> np.array:
            """
            odeint State Variables Solver function

            Parameters
            ----------
            p_t: numpy 1-d array of shape(n)
                Current integrated state variables
            _: float
                Current time

            Returns
            ----------
            dx_t_dt: numpy 1-d array of shape(n)
                Current calculated value for the time-derivative of the state variable vector x_t
            """

            nonlocal t

            numerator_sum = 0
            denominator_sum = 0

            P_t = [(self._P[t][i * self._P_size:(i + 1) * self._P_size]).reshape(self._n, self._n)
                   for i in range(self._N)]

            for i in range(self._N):
                beta_i = self.__beta[i]
                c_i = self.__c[i]
                P_t_i = P_t[i]
                k_i = P_t_i[0, 0]
                l_i = P_t_i[0, 1]

                numerator_sum += beta_i * (2 * self.__s * l_i - c_i)
                denominator_sum += beta_i * (1 + 2 * self.__s * k_i)

            dp_t_dt = - self.__s * (1 + denominator_sum) * p_t + self.__s * (self.__p - numerator_sum)

            if t:
                t -= 1

            return dp_t_dt

        p_0 = self._x_0[0]
        p = odeint(func=p_t_diff_eqn,
                   y0=p_0,
                   t=self._forward_time)

        self._plot(t=self._forward_time,
                   mat=p,
                   is_P=False,
                   title=f"p_t")
        # self._plot(t=self._forward_time,
        #            mat=[x[1] for x in self._x],
        #            is_P=False,
        #            title=f"p_t")


r = 0.1
s = 2
p = 85
x_0 = np.array([80, 1])
T_f = 5
epsilon = 10 ** (-3)

for N in [2, 4, 10]:
    beta = [1] * N
    c = [1] * N
    oc = OligopolisticCompetition(N=N,
                                  r=r,
                                  s=s,
                                  beta=beta,
                                  c=c,
                                  p=p,
                                  x_0=x_0,
                                  T_f=T_f,
                                  epsilon=epsilon
                                  )
    oc.solve_game_and_plot_state_space()
    # oc.get_steady_state_price()
    oc.simulate_p()