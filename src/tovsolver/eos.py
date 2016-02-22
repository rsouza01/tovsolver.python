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


import csv
import numpy

from collections import namedtuple
from scipy import interpolate
import cgs_constants as const
# import atomic_constants as const


import matplotlib.pyplot as plt


ENERGY_DENSITY_INDEX = 0
MASS_DENSITY_INDEX = 0
PRESSURE_INDEX = 1
BARYONIC_NUMBER_INDEX = 2


class EoSValue(namedtuple('EoSValue', 'energy pressure baryonic_number')):
    """
    Named tuple that represents an EoS value
    """
    pass


class EoS:

    def __init__(self, filename, central_energy_density, verbose=False):

        self.__filename = filename

        self.__central_energy_density = central_energy_density

        loader = EoSLoader(self.__filename, central_energy_density)

        eos_list = loader.load_eos_file()

        interp = EoSInterpolation(eos_list)

        self.__energy_from_pressure_function = \
            interp.interpolate_spline_energy_from_pressure(plot_fit=verbose)

        self.__pressure_from_energy_function = \
            interp.interpolate_spline_pressure_from_energy(plot_fit=verbose)

    def energy_from_pressure(self, pressure):

        # print("energy_from_pressure(%f)" % (pressure))

        return self.__energy_from_pressure_function(pressure).item(0)

    def pressure_from_energy(self, energy):

        # print("pressure_from_energy(%f)" % (energy))

        return self.__pressure_from_energy_function(energy).item(0)


class EoSLoader:
    """ EoS Loader. """

    __eos_list = []

    def __init__(self, filename, central_energy_density=1):

        self.__filename = filename
        self.__central_energy_density = central_energy_density

        # print("self.__central_energy_density = {}".format(self.__central_energy_density))

    def get_eos_list(self):
        return self.__eos_list

    def load_eos_file(self):

        with open(self.__filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row[0].startswith('#'):

                    # Dimensionless values
                    energy_density = float(row[MASS_DENSITY_INDEX])\
                                     * const.LIGHT_SPEED_SQUARED/self.__central_energy_density

                    pressure = float(row[PRESSURE_INDEX])/self.__central_energy_density

                    baryonic_number = float(row[BARYONIC_NUMBER_INDEX])

                    eos_value = EoSValue(energy_density, pressure, baryonic_number)

                    self.__eos_list.append(eos_value)

        # print(self.__eos_list)
        # firstColumn = [row[0] for row in self.__eosList]
        # print("OK")

        return self.__eos_list


class EoSInterpolation:
    """ EoS Interpolation. """

    def __init__(self, eos_list):

        self.__eos_list = eos_list

        self.__energy_values = numpy.asarray(
            [row[MASS_DENSITY_INDEX] for row in self.__eos_list],  dtype=numpy.float32)

        # print(self.__energy_values)

        self.__pressure_values = numpy.asarray(
            [row[PRESSURE_INDEX] for row in self.__eos_list],  dtype=numpy.float32)

        # print(self.__pressure_values)

        self.__baryonicNumberValues = numpy.asarray(
            [row[BARYONIC_NUMBER_INDEX] for row in self.__eos_list],  dtype=numpy.float32)

    def interpolate_spline_energy_from_pressure(self, plot_fit=False):

        return self.interpolate_spline(self.__pressure_values[::-1], self.__energy_values[::-1], plot_fit)

    def interpolate_spline_pressure_from_energy(self, plot_fit=False):

        return self.interpolate_spline(self.__energy_values[::-1], self.__pressure_values[::-1], plot_fit)

    @staticmethod
    def interpolate_spline(x_values, y_values, plot_fit=False):

        fc = interpolate.interp1d(x_values, y_values)

        if plot_fit:
            plt.figure()
            plt.plot(x_values,
                     fc(x_values), 'x',
                     x_values, y_values)
            plt.legend(['True', 'Cubic Spline'])
            plt.ylabel("Y")
            plt.xlabel("X")
            plt.show()

        return fc

