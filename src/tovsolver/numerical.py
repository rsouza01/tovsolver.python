# tovsolver - Tolman-Oppenheimer-Volkoff equation solver
# Copyright (C) 2015 Rodrigo Souza <rsouza01@gmail.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
# 02110-1301, USA.

import numpy as np
from collections import namedtuple


class RungeKuttaParameters(namedtuple('RungeKutta4',
                                      'first_element last_element rk_steps initial_conditions derivatives verbose')):
    """
    Named tuple that represents the RK parameters
    """
    pass


class RungeKutta(object):
    """ Runge Kutta implementation. """

    def __init__(self, rk_parameters):
        """
        Class constructor.

        Keyword arguments:
        rk_parameters -- named tuple of type RungeKuttaParameters.
        """

        self.__derivatives = rk_parameters.derivatives
        self.__initial_conditions = rk_parameters.initial_conditions
        self.__verbose = rk_parameters.verbose

        self.__first_element = rk_parameters.first_element
        self.__last_element = rk_parameters.last_element
        self.__total_steps = rk_parameters.rk_steps

        self.__h = (self.__last_element - self.__first_element)/self.__total_steps

        self.__array_t = np.linspace(self.__first_element, self.__last_element, self.__total_steps)

    def run(self):
        """
        Runs the method based on parameters passed in the constructor.
        """
        h = self.__h

        t = self.__first_element

        ws = self.__initial_conditions

        N = self.__total_steps

        m = len(self.__derivatives)

        stop_condition_reached = False

        # print("(h, t, ws, N, m)= ({}, {}, {}, {}, {})".format(h, t, ws, N, m))

        try:

            for i in range(1, N + 1):

                # k1
                k1 = []
                for j in range(0, m):
                    k1.append(h * self.__derivatives[j](t, ws))
                # print("k1 = {}".format(k1))

                # k2
                ws_k2 = []
                for j in range(0, m):
                    ws_k2.append(ws[j] + 0.5 * k1[j])
                k2 = []
                for j in range(0, m):
                    k2.append(h * self.__derivatives[j](t + 0.5*h, ws_k2))
                # print("k2 = {}".format(k2))

                # k3
                ws_k3 = []
                for j in range(0, m):
                    ws_k3.append(ws[j] + 0.5 * k2[j])
                k3 = []
                for j in range(0, m):
                    k3.append(h * self.__derivatives[j](t + 0.5*h, ws_k3))
                # print("k3 = {}".format(k3))

                # k4
                ws_k4 = []
                for j in range(0, m):
                    ws_k4.append(ws[j] + k3[j])
                k4 = []
                for j in range(0, m):
                    k4.append(h * self.__derivatives[j](t + h, ws_k4))
                # print("k4 = {}".format(k4))

                for j in range(0, m):
                    ws[j] += (k1[j] + 2.*k2[j] + 2.*k3[j] + k4[j])/6.

                self.perform_calculations(t, ws)

                t = self.__first_element + i * h

                if self.must_stop(t, ws):
                    # print("Stop condition reached.")
                    stop_condition_reached = True
                    break
        except ValueError:
            stop_condition_reached = True

        # TODO: Return interpolated functions.

        return [ws, stop_condition_reached]




    def run2(self):
        """
        Runs the method based on parameters passed in the constructor.
        """

        m = len(self.__derivatives)
        print("m = {}".format(m))

        ws = self.__initial_conditions
        print("ws = {}".format(ws))

        t = self.__first_element
        print("t = {}".format(t))

        h = self.__h
        print("h = {}".format(h))

        stop_condition_reached = False

        k1 = [0*x for x in range(0, m)]
        k2 = [0*x for x in range(0, m)]
        k3 = [0*x for x in range(0, m)]
        k4 = [0*x for x in range(0, m)]

        # print("k1=%s, k2=%s, k3=%s, k4=%s" % (k1, k2, k3, k4))
        # print("Type ws_0 = {}".format(type(ws[0])))
        # print("Type ws_1 = {}".format(type(ws[1])))
        # print("h = {}".format(h))

        # print("****************************************************\n")

        ws_k2 = [0*x for x in range(0, m)]
        ws_k3 = [0*x for x in range(0, m)]
        ws_k4 = [0*x for x in range(0, m)]

        # print(self.__array_t)
        try:
            for i in range(1, self.__total_steps):

                # print(t)

                # print("ws = {}".format(str(ws)))
                # K1
                for j in range(0, m):
                    k1[j] = h * self.__derivatives[j](t, ws)
                # print("k1 = {}".format(str(k1)))

                # K2
                for k in range(0, m):
                    ws_k2[k] = ws[k] + 0.5*k1[k]
                for j in range(0, m):
                    k2[j] = h * self.__derivatives[j](t+0.5*h, ws_k2)
                # print("k2 = {}".format(str(k2)))

                # K3
                for k in range(0, m):
                    ws_k3[k] = ws[k] + 0.5*k2[k]
                for j in range(0, m):
                    k3[j] = h * self.__derivatives[j](t+0.5*h, ws_k3)
                # print("k3 = {}".format(str(k3)))

                # K4
                for k in range(0, m):
                    ws_k4[k] = ws[k] + k3[k]
                for j in range(0, m):
                    k4[j] = h * self.__derivatives[j](t+h, ws_k4)
                # print("k4 = {}".format(str(k4)))

                for j in range(0, m):
                    ws[j] += (k1[j] + 2.*k2[j] + 2.*k3[j] + k4[j])/6.
                # print("ws = {}".format(str(ws)))

                t = self.__first_element + i * h

                if self.__verbose:
                    print("(i, t, m, p) = (%d, %f, %e, %e)" % (i, t, ws[0], ws[1]))

                # print("(i, t, m, p) = (%d, %f, %e, %e)" % (i, t, ws[0], ws[1]))

                self.perform_calculations(t, ws)

                if self.must_stop(t, ws):
                    # print("Stop condition reached.")
                    stop_condition_reached = True
                    break

                # print("****************************************************\n")
        except ValueError:
            stop_condition_reached = True

        # TODO: Return interpolated functions.

        return [ws, stop_condition_reached]

    def must_stop(self, eta, ws):
        """
        Stop condition. Must override.

        Keyword arguments:
        eta -- eta parameter(parametrization).
        ws  -- Differential equations evaluated at eta.
        """

        return False

    def perform_calculations(self, eta, ws):
        """
        If there is another calculations that must be performed, it must be
        in this method. Must override.

        Keyword arguments:
        eta -- eta parameter(parametrization).
        ws  -- Differential equations evaluated at eta.
        """
        pass
